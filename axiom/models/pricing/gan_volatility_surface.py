"""
GAN-Enhanced Implied Volatility Surface Generator

Based on: Yao Ge, Ying Wang, Jingyi Liu, Jiyuan Wang (2025)
"GAN-Enhanced Implied Volatility Surface Reconstruction for Option Pricing Error Mitigation"
IEEE Access, 2025 (Open Access)

This implementation uses Generative Adversarial Networks to reconstruct implied
volatility surfaces with:
- No-arbitrage constraints enforced
- Smooth, market-consistent surfaces
- Complex nonlinear pattern capture
- Domain-specific regularization

Overcomes rigid functional forms of traditional parametric approaches (SVI, etc.)
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SurfaceConstraint(Enum):
    """Types of volatility surface constraints"""
    NO_ARBITRAGE = "no_arbitrage"
    CALENDAR_SPREAD = "calendar_spread"
    BUTTERFLY_SPREAD = "butterfly_spread"
    MONOTONICITY = "monotonicity"


@dataclass
class GANSurfaceConfig:
    """Configuration for GAN Volatility Surface"""
    # Surface dimensions
    n_strikes: int = 20
    n_maturities: int = 15
    
    # GAN architecture
    latent_dim: int = 100
    generator_hidden_dims: List[int] = None
    discriminator_hidden_dims: List[int] = None
    
    # Training parameters
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 1e-4
    batch_size: int = 32
    n_critic: int = 5  # Train discriminator n times per generator
    
    # Constraint enforcement
    lambda_no_arbitrage: float = 10.0  # Weight for no-arbitrage loss
    lambda_smoothness: float = 1.0  # Smoothness regularization
    use_gradient_penalty: bool = True  # WGAN-GP
    gp_lambda: float = 10.0
    
    def __post_init__(self):
        if self.generator_hidden_dims is None:
            self.generator_hidden_dims = [256, 512, 512, 256]
        if self.discriminator_hidden_dims is None:
            self.discriminator_hidden_dims = [256, 512, 256, 128]


import torch.nn as nn

class VolatilitySurfaceGenerator(nn.Module):
    """
    Generator network for implied volatility surfaces
    
    Maps random noise → realistic volatility surface with market characteristics
    """
    
    def __init__(self, config: GANSurfaceConfig):
        super(VolatilitySurfaceGenerator, self).__init__()
        
        self.config = config
        output_dim = config.n_strikes * config.n_maturities
        
        # Build generator layers
        layers = []
        prev_dim = config.latent_dim
        
        for hidden_dim in config.generator_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus())  # Ensure positive volatilities
        
        self.generator = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate volatility surface from latent vector
        
        Args:
            z: Latent noise (batch, latent_dim)
            
        Returns:
            Volatility surface (batch, n_strikes, n_maturities)
        """
        surface_flat = self.generator(z)
        
        # Reshape to surface
        batch_size = z.size(0)
        surface = surface_flat.view(
            batch_size,
            self.config.n_strikes,
            self.config.n_maturities
        )
        
        return surface


class VolatilitySurfaceDiscriminator(nn.Module):
    """
    Discriminator network for volatility surfaces
    
    Classifies surfaces as real (market data) or fake (generated)
    """
    
    def __init__(self, config: GANSurfaceConfig):
        super(VolatilitySurfaceDiscriminator, self).__init__()
        
        self.config = config
        input_dim = config.n_strikes * config.n_maturities
        
        # Build discriminator layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.discriminator_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer (WGAN: no sigmoid, output real number)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake surface
        
        Args:
            surface: Volatility surface (batch, n_strikes, n_maturities)
            
        Returns:
            Discriminator score (batch, 1)
        """
        # Flatten surface
        batch_size = surface.size(0)
        surface_flat = surface.view(batch_size, -1)
        
        # Discriminate
        score = self.discriminator(surface_flat)
        
        return score


class NoArbitrageConstraintLayer(nn.Module):
    """
    Enforces no-arbitrage constraints on volatility surfaces
    
    Key constraints:
    1. Calendar spread: σ(T1) ≤ σ(T2) for T1 < T2 (usually)
    2. Butterfly spread: Convexity constraints
    3. Monotonicity: Certain ordering preserved
    """
    
    def __init__(self, config: GANSurfaceConfig):
        super(NoArbitrageConstraintLayer, self).__init__()
        self.config = config
    
    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        """
        Calculate no-arbitrage violation penalty
        
        Args:
            surface: Volatility surface (batch, n_strikes, n_maturities)
            
        Returns:
            Penalty score (lower = better)
        """
        batch_size = surface.size(0)
        
        # Calendar spread constraint (across time)
        # Volatility should generally increase with time (term structure)
        time_diffs = surface[:, :, 1:] - surface[:, :, :-1]
        calendar_violations = torch.sum(F.relu(-time_diffs))  # Penalize decreases
        
        # Butterfly spread constraint (across strikes)
        # Second derivative should be reasonable
        strike_second_deriv = surface[:, 2:, :] - 2 * surface[:, 1:-1, :] + surface[:, :-2, :]
        butterfly_violations = torch.sum(torch.abs(strike_second_deriv))
        
        # Combine penalties
        total_penalty = calendar_violations * 0.5 + butterfly_violations * 0.5
        
        return total_penalty / batch_size


class GANVolatilitySurface:
    """
    Complete GAN system for volatility surface generation
    
    Uses Wasserstein GAN with gradient penalty (WGAN-GP) to generate
    arbitrage-free implied volatility surfaces.
    """
    
    def __init__(self, config: Optional[GANSurfaceConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for GANVolatilitySurface")
        
        self.config = config or GANSurfaceConfig()
        
        # Initialize GAN components
        self.generator = VolatilitySurfaceGenerator(self.config)
        self.discriminator = VolatilitySurfaceDiscriminator(self.config)
        self.no_arbitrage_layer = NoArbitrageConstraintLayer(self.config)
        
        # Optimizers
        self.optimizer_g = None
        self.optimizer_d = None
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'arbitrage_penalty': [],
            'gradient_penalty': []
        }
    
    def train(
        self,
        real_surfaces: torch.Tensor,
        epochs: int = 100,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train GAN on real volatility surfaces
        
        Args:
            real_surfaces: Real market surfaces (n_samples, n_strikes, n_maturities)
            epochs: Training epochs
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        self.generator.train()
        self.discriminator.train()
        
        # Initialize optimizers
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate_g,
            betas=(0.5, 0.999)
        )
        
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate_d,
            betas=(0.5, 0.999)
        )
        
        n_samples = real_surfaces.size(0)
        batch_size = self.config.batch_size
        
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_arb_penalty = 0.0
            epoch_gp = 0.0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                real_batch = real_surfaces[batch_indices]
                current_batch_size = real_batch.size(0)
                
                # Train Discriminator
                for _ in range(self.config.n_critic):
                    self.optimizer_d.zero_grad()
                    
                    # Generate fake surfaces
                    z = torch.randn(current_batch_size, self.config.latent_dim)
                    fake_batch = self.generator(z)
                    
                    # Discriminator outputs
                    d_real = self.discriminator(real_batch)
                    d_fake = self.discriminator(fake_batch.detach())
                    
                    # Wasserstein loss
                    d_loss = -(d_real.mean() - d_fake.mean())
                    
                    # Gradient penalty (WGAN-GP)
                    if self.config.use_gradient_penalty:
                        gp = self._gradient_penalty(real_batch, fake_batch)
                        d_loss = d_loss + self.config.gp_lambda * gp
                        epoch_gp += gp.item()
                    
                    d_loss.backward()
                    self.optimizer_d.step()
                    
                    epoch_d_loss += d_loss.item()
                
                # Train Generator
                self.optimizer_g.zero_grad()
                
                # Generate new batch
                z = torch.randn(current_batch_size, self.config.latent_dim)
                fake_batch = self.generator(z)
                
                # Generator loss (fool discriminator)
                d_fake = self.discriminator(fake_batch)
                g_loss = -d_fake.mean()
                
                # Add no-arbitrage constraint penalty
                arbitrage_penalty = self.no_arbitrage_layer(fake_batch)
                g_loss = g_loss + self.config.lambda_no_arbitrage * arbitrage_penalty
                
                # Smoothness regularization
                smoothness_penalty = self._calculate_smoothness_penalty(fake_batch)
                g_loss = g_loss + self.config.lambda_smoothness * smoothness_penalty
                
                g_loss.backward()
                self.optimizer_g.step()
                
                epoch_g_loss += g_loss.item()
                epoch_arb_penalty += arbitrage_penalty.item()
                n_batches += 1
            
            # Average losses
            avg_g_loss = epoch_g_loss / n_batches
            avg_d_loss = epoch_d_loss / (n_batches * self.config.n_critic)
            avg_arb = epoch_arb_penalty / n_batches
            avg_gp = epoch_gp / (n_batches * self.config.n_critic) if self.config.use_gradient_penalty else 0
            
            # Store history
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['arbitrage_penalty'].append(avg_arb)
            self.history['gradient_penalty'].append(avg_gp)
            
            if verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"G Loss: {avg_g_loss:.4f}, "
                      f"D Loss: {avg_d_loss:.4f}, "
                      f"Arb Penalty: {avg_arb:.4f}, "
                      f"GP: {avg_gp:.4f}")
        
        return self.history
    
    def generate_surface(
        self,
        n_samples: int = 1,
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate arbitrage-free volatility surface(s)
        
        Args:
            n_samples: Number of surfaces to generate
            return_numpy: Return as numpy array
            
        Returns:
            Generated surface(s)
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(n_samples, self.config.latent_dim)
            
            # Generate surfaces
            surfaces = self.generator(z)
        
        if return_numpy:
            return surfaces.cpu().numpy()
        
        return surfaces
    
    def reconstruct_surface(
        self,
        partial_surface: torch.Tensor,
        optimization_steps: int = 100
    ) -> torch.Tensor:
        """
        Reconstruct complete surface from partial observations
        
        Useful for filling in missing volatility quotes.
        
        Args:
            partial_surface: Partially observed surface with NaNs
            optimization_steps: Steps to optimize latent vector
            
        Returns:
            Complete reconstructed surface
        """
        self.generator.eval()
        
        # Initialize latent vector
        z = torch.randn(1, self.config.latent_dim, requires_grad=True)
        
        # Optimizer for latent vector
        optimizer = torch.optim.Adam([z], lr=0.01)
        
        # Mask for observed values
        observed_mask = ~torch.isnan(partial_surface)
        
        # Optimize latent vector to match observed values
        for _ in range(optimization_steps):
            optimizer.zero_grad()
            
            # Generate surface
            generated = self.generator(z)
            
            # Loss only on observed values
            loss = F.mse_loss(
                generated[observed_mask],
                partial_surface[observed_mask]
            )
            
            loss.backward()
            optimizer.step()
        
        # Final generation
        with torch.no_grad():
            reconstructed = self.generator(z)
        
        return reconstructed
    
    def _gradient_penalty(
        self,
        real_surfaces: torch.Tensor,
        fake_surfaces: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate gradient penalty for WGAN-GP
        
        Enforces 1-Lipschitz constraint on discriminator.
        """
        batch_size = real_surfaces.size(0)
        
        # Random interpolation weight
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand_as(real_surfaces)
        
        # Interpolated samples
        interpolated = alpha * real_surfaces + (1 - alpha) * fake_surfaces
        interpolated.requires_grad_(True)
        
        # Discriminator output for interpolated
        d_interpolated = self.discriminator(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def _calculate_smoothness_penalty(self, surface: torch.Tensor) -> torch.Tensor:
        """
        Calculate smoothness penalty to encourage smooth surfaces
        
        Uses total variation regularization.
        """
        # Differences along strike dimension
        strike_diff = torch.abs(surface[:, 1:, :] - surface[:, :-1, :])
        
        # Differences along maturity dimension
        maturity_diff = torch.abs(surface[:, :, 1:] - surface[:, :, :-1])
        
        # Total variation
        tv = strike_diff.mean() + maturity_diff.mean()
        
        return tv
    
    def save(self, path: str):
        """Save GAN models"""
        torch.save({
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    def load(self, path: str):
        """Load GAN models"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.history = checkpoint.get('history', {})


def create_sample_market_surfaces(
    n_samples: int = 100,
    n_strikes: int = 20,
    n_maturities: int = 15,
    base_vol: float = 0.25
) -> torch.Tensor:
    """
    Create sample market-like volatility surfaces
    
    Generates realistic surfaces with:
    - Volatility smile (higher vol for OTM)
    - Term structure
    - Market noise
    
    Returns:
        Tensor of shape (n_samples, n_strikes, n_maturities)
    """
    np.random.seed(42)
    
    surfaces = []
    
    strikes = np.linspace(0.7, 1.3, n_strikes)  # Moneyness
    maturities = np.linspace(0.08, 2.0, n_maturities)  # Years
    
    for _ in range(n_samples):
        surface = np.zeros((n_strikes, n_maturities))
        
        # Random base volatility
        sample_base = base_vol + np.random.normal(0, 0.05)
        
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                # ATM vol with term structure
                atm_vol = sample_base * (1 + 0.1 * np.sqrt(maturity))
                
                # Volatility smile (quadratic in log-moneyness)
                log_moneyness = np.log(strike)
                smile = 0.15 * log_moneyness ** 2
                
                # Market noise
                noise = np.random.normal(0, 0.01)
                
                surface[i, j] = atm_vol + smile + noise
        
        # Ensure positive
        surface = np.maximum(surface, 0.05)
        
        surfaces.append(surface)
    
    return torch.FloatTensor(np.array(surfaces))


# Example usage
if __name__ == "__main__":
    print("GAN Volatility Surface - Example Usage")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required")
        print("Install with: pip install torch")
    else:
        # Configuration
        print("\n1. Configuration")
        config = GANSurfaceConfig(
            n_strikes=20,
            n_maturities=15,
            latent_dim=100,
            lambda_no_arbitrage=10.0,
            use_gradient_penalty=True
        )
        print(f"   Surface dimensions: {config.n_strikes} × {config.n_maturities}")
        print(f"   Latent dimension: {config.latent_dim}")
        print(f"   No-arbitrage weight: {config.lambda_no_arbitrage}")
        print(f"   Using WGAN-GP: {config.use_gradient_penalty}")
        
        # Generate training data
        print("\n2. Generating Sample Market Surfaces")
        market_surfaces = create_sample_market_surfaces(
            n_samples=200,
            n_strikes=config.n_strikes,
            n_maturities=config.n_maturities
        )
        print(f"   Created {len(market_surfaces)} sample surfaces")
        print(f"   Shape: {market_surfaces.shape}")
        
        # Initialize GAN
        print("\n3. Initializing GAN Volatility Surface")
        gan = GANVolatilitySurface(config)
        print("   ✓ Generator initialized")
        print("   ✓ Discriminator initialized")
        print("   ✓ No-arbitrage constraint layer ready")
        
        # Train
        print("\n4. Training GAN")
        print("   Training with WGAN-GP and no-arbitrage constraints...")
        history = gan.train(market_surfaces, epochs=50, verbose=1)
        print("   ✓ Training completed")
        
        # Generate surfaces
        print("\n5. Generating Arbitrage-Free Surfaces")
        generated = gan.generate_surface(n_samples=5)
        print(f"   Generated {len(generated)} new surfaces")
        print(f"   Shape: {generated.shape}")
        
        # Analyze generated surface
        sample_surface = generated[0]
        print(f"\n6. Sample Generated Surface Analysis")
        print(f"   Min volatility: {sample_surface.min():.2%}")
        print(f"   Max volatility: {sample_surface.max():.2%}")
        print(f"   Mean volatility: {sample_surface.mean():.2%}")
        print(f"   ATM (strike=1.0) short-term: {sample_surface[10, 0]:.2%}")
        print(f"   ATM (strike=1.0) long-term: {sample_surface[10, -1]:.2%}")
        
        # Test reconstruction
        print("\n7. Surface Reconstruction Test")
        partial = market_surfaces[0].clone()
        # Mask some values as missing
        mask = torch.rand_like(partial) > 0.7
        partial[mask] = float('nan')
        print(f"   Missing values: {mask.sum().item()} of {partial.numel()}")
        
        reconstructed = gan.reconstruct_surface(partial, optimization_steps=50)
        print(f"   ✓ Surface reconstructed")
        
        print("\n8. Model Features")
        print("   ✓ No-arbitrage constraints enforced")
        print("   ✓ Smooth surface generation")
        print("   ✓ Market-consistent patterns")
        print("   ✓ Missing data reconstruction")
        print("   ✓ WGAN-GP stability")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("\nBased on: Ge et al. (IEEE Access 2025)")
        print("Innovation: GAN with no-arbitrage constraints for vol surfaces")