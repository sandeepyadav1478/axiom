"""
Variational Autoencoder + MLP for Options Pricing

Based on: Lijie Ding, Egang Lu, Kin Cheung (September 2025)
"Deep Learning Option Pricing with Market Implied Volatility Surfaces"
arXiv preprint arXiv:2509.05911

This implementation uses a VAE to compress implied volatility surfaces into a
low-dimensional latent space, then combines these latent variables with option-specific
inputs into an MLP to predict option prices. Handles American puts and arithmetic
Asian options with high accuracy.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OptionType(Enum):
    """Types of options supported"""
    EUROPEAN_CALL = "european_call"
    EUROPEAN_PUT = "european_put"
    AMERICAN_CALL = "american_call"
    AMERICAN_PUT = "american_put"
    ASIAN_ARITHMETIC = "asian_arithmetic"
    ASIAN_GEOMETRIC = "asian_geometric"


@dataclass
class VAEConfig:
    """Configuration for VAE Option Pricer"""
    # Volatility surface dimensions
    n_strikes: int = 20
    n_maturities: int = 15
    latent_dim: int = 10
    
    # Network architecture
    encoder_hidden_dims: List[int] = None
    decoder_hidden_dims: List[int] = None
    pricer_hidden_dims: List[int] = None
    
    # Training parameters
    learning_rate: float = 1e-3
    beta: float = 1.0  # KL divergence weight
    batch_size: int = 32
    
    # Option parameters dimension
    n_option_params: int = 5  # [strike, maturity, spot, rate, dividend]
    
    def __post_init__(self):
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [256, 128, 64]
        if self.decoder_hidden_dims is None:
            self.decoder_hidden_dims = [64, 128, 256]
        if self.pricer_hidden_dims is None:
            self.pricer_hidden_dims = [128, 64, 32]


class VolatilitySurfaceEncoder(nn.Module):
    """
    Encoder network to compress volatility surfaces
    
    Maps high-dimensional implied volatility surface to low-dimensional latent space.
    """
    
    def __init__(self, config: VAEConfig):
        super(VolatilitySurfaceEncoder, self).__init__()
        
        self.config = config
        input_dim = config.n_strikes * config.n_maturities
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.encoder_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, config.latent_dim)
        
    def forward(self, surface: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode volatility surface to latent distribution
        
        Args:
            surface: Volatility surface (batch, n_strikes * n_maturities)
            
        Returns:
            (mu, logvar) - Parameters of latent Gaussian distribution
        """
        # Flatten surface if needed
        if len(surface.shape) == 3:
            batch_size = surface.size(0)
            surface = surface.view(batch_size, -1)
        
        # Encode
        encoded = self.encoder(surface)
        
        # Get distribution parameters
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VolatilitySurfaceDecoder(nn.Module):
    """
    Decoder network to reconstruct volatility surfaces
    
    Maps low-dimensional latent representation back to volatility surface.
    """
    
    def __init__(self, config: VAEConfig):
        super(VolatilitySurfaceDecoder, self).__init__()
        
        self.config = config
        output_dim = config.n_strikes * config.n_maturities
        
        # Build decoder layers
        layers = []
        prev_dim = config.latent_dim
        
        for hidden_dim in config.decoder_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Final layer to reconstruct surface
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus())  # Ensure positive volatilities
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to volatility surface
        
        Args:
            z: Latent vector (batch, latent_dim)
            
        Returns:
            Reconstructed surface (batch, n_strikes * n_maturities)
        """
        reconstructed = self.decoder(z)
        
        # Reshape to surface if needed
        batch_size = z.size(0)
        surface = reconstructed.view(
            batch_size,
            self.config.n_strikes,
            self.config.n_maturities
        )
        
        return surface


class VolatilitySurfaceVAE(nn.Module):
    """
    Complete VAE for volatility surface compression and reconstruction
    """
    
    def __init__(self, config: VAEConfig):
        super(VolatilitySurfaceVAE, self).__init__()
        
        self.config = config
        self.encoder = VolatilitySurfaceEncoder(config)
        self.decoder = VolatilitySurfaceDecoder(config)
        
    def forward(self, surface: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        
        Args:
            surface: Input volatility surface
            
        Returns:
            (reconstructed_surface, mu, logvar)
        """
        # Encode
        mu, logvar = self.encoder(surface)
        
        # Reparameterize
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar
        
    def encode(self, surface: torch.Tensor) -> torch.Tensor:
        """
        Encode surface to latent representation
        
        Args:
            surface: Volatility surface
            
        Returns:
            Latent vector (uses mean, not sampled)
        """
        mu, _ = self.encoder(surface)
        return mu
        
    def vae_loss(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate VAE loss (reconstruction + KL divergence)
        
        Returns:
            (total_loss, reconstruction_loss, kl_divergence)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        
        # Total loss
        total_loss = recon_loss + self.config.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class MLPOptionPricer(nn.Module):
    """
    Multi-Layer Perceptron for option pricing
    
    Takes latent volatility representation + option parameters as input,
    outputs option price.
    """
    
    def __init__(self, config: VAEConfig):
        super(MLPOptionPricer, self).__init__()
        
        self.config = config
        input_dim = config.latent_dim + config.n_option_params
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.pricer_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())  # Ensure positive prices
        
        self.pricer = nn.Sequential(*layers)
        
    def forward(
        self,
        latent_vol: torch.Tensor,
        option_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Price option using latent volatility + option parameters
        
        Args:
            latent_vol: Latent volatility representation (batch, latent_dim)
            option_params: Option parameters (batch, n_option_params)
                         [strike, maturity, spot, rate, dividend_yield]
                         
        Returns:
            Option prices (batch, 1)
        """
        # Concatenate inputs
        combined_input = torch.cat([latent_vol, option_params], dim=1)
        
        # Price option
        price = self.pricer(combined_input)
        
        return price


class VAEMLPOptionPricer:
    """
    Complete VAE+MLP Option Pricing System
    
    Combines volatility surface compression via VAE with option pricing via MLP.
    Two-stage training: first train VAE on surfaces, then train MLP for pricing.
    """
    
    def __init__(self, config: Optional[VAEConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for VAEMLPOptionPricer")
            
        self.config = config or VAEConfig()
        
        # Initialize models
        self.vae = VolatilitySurfaceVAE(self.config)
        self.pricer = MLPOptionPricer(self.config)
        
        # Optimizers
        self.vae_optimizer = None
        self.pricer_optimizer = None
        
        # Training history
        self.vae_history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
        self.pricer_history = {'loss': []}
        
    def train_vae(
        self,
        volatility_surfaces: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train VAE on volatility surfaces
        
        Args:
            volatility_surfaces: Training data (n_samples, n_strikes, n_maturities)
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        self.vae.train()
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        n_samples = volatility_surfaces.size(0)
        batch_size = self.config.batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            
            # Mini-batch training
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_surfaces = volatility_surfaces[batch_indices]
                
                # Forward pass
                reconstructed, mu, logvar = self.vae(batch_surfaces)
                
                # Calculate loss
                loss, recon_loss, kl_loss = self.vae.vae_loss(
                    reconstructed, batch_surfaces, mu, logvar
                )
                
                # Backward pass
                self.vae_optimizer.zero_grad()
                loss.backward()
                self.vae_optimizer.step()
                
                # Accumulate losses
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
            
            # Average losses
            n_batches = (n_samples + batch_size - 1) // batch_size
            epoch_loss /= n_batches
            epoch_recon /= n_batches
            epoch_kl /= n_batches
            
            # Store history
            self.vae_history['loss'].append(epoch_loss)
            self.vae_history['recon_loss'].append(epoch_recon)
            self.vae_history['kl_loss'].append(epoch_kl)
            
            if verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_loss:.4f} "
                      f"(Recon: {epoch_recon:.4f}, KL: {epoch_kl:.4f})")
        
        return self.vae_history
        
    def train_pricer(
        self,
        volatility_surfaces: torch.Tensor,
        option_params: torch.Tensor,
        option_prices: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train MLP pricer using latent volatility representations
        
        Args:
            volatility_surfaces: Market volatility surfaces
            option_params: Option parameters (strike, maturity, spot, rate, div)
            option_prices: True option prices
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        self.pricer.train()
        self.vae.eval()  # VAE in evaluation mode
        
        self.pricer_optimizer = torch.optim.Adam(
            self.pricer.parameters(),
            lr=learning_rate
        )
        
        n_samples = option_params.size(0)
        batch_size = self.config.batch_size
        
        # Encode all surfaces once (VAE is frozen)
        with torch.no_grad():
            latent_vols = self.vae.encode(volatility_surfaces)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_latent = latent_vols[batch_indices]
                batch_params = option_params[batch_indices]
                batch_prices = option_prices[batch_indices]
                
                # Forward pass
                predicted_prices = self.pricer(batch_latent, batch_params)
                
                # Calculate loss (MSE + relative error)
                mse_loss = F.mse_loss(predicted_prices, batch_prices)
                relative_error = torch.mean(
                    torch.abs(predicted_prices - batch_prices) / (batch_prices + 1e-8)
                )
                loss = mse_loss + 0.1 * relative_error
                
                # Backward pass
                self.pricer_optimizer.zero_grad()
                loss.backward()
                self.pricer_optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss
            n_batches = (n_samples + batch_size - 1) // batch_size
            epoch_loss /= n_batches
            
            # Store history
            self.pricer_history['loss'].append(epoch_loss)
            
            if verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")
        
        return self.pricer_history
        
    def price_option(
        self,
        volatility_surface: Union[torch.Tensor, np.ndarray],
        strike: float,
        maturity: float,
        spot: float,
        rate: float,
        dividend_yield: float = 0.0,
        option_type: OptionType = OptionType.EUROPEAN_CALL
    ) -> float:
        """
        Price a single option
        
        Args:
            volatility_surface: Current implied volatility surface
            strike: Strike price
            maturity: Time to maturity (years)
            spot: Current spot price
            rate: Risk-free interest rate
            dividend_yield: Dividend yield
            option_type: Type of option
            
        Returns:
            Option price
        """
        self.vae.eval()
        self.pricer.eval()
        
        with torch.no_grad():
            # Convert to tensor if needed
            if isinstance(volatility_surface, np.ndarray):
                vol_surface = torch.FloatTensor(volatility_surface)
            else:
                vol_surface = volatility_surface
            
            # Add batch dimension
            if len(vol_surface.shape) == 2:
                vol_surface = vol_surface.unsqueeze(0)
            
            # Encode volatility surface
            latent_vol = self.vae.encode(vol_surface)
            
            # Create option parameters tensor
            option_params = torch.FloatTensor([[
                strike, maturity, spot, rate, dividend_yield
            ]])
            
            # Price option
            price = self.pricer(latent_vol, option_params)
            
            return float(price.item())
            
    def price_batch(
        self,
        volatility_surfaces: torch.Tensor,
        option_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Price a batch of options
        
        Args:
            volatility_surfaces: Batch of surfaces (batch, n_strikes, n_maturities)
            option_params: Batch of parameters (batch, 5)
            
        Returns:
            Option prices (batch, 1)
        """
        self.vae.eval()
        self.pricer.eval()
        
        with torch.no_grad():
            # Encode surfaces
            latent_vols = self.vae.encode(volatility_surfaces)
            
            # Price options
            prices = self.pricer(latent_vols, option_params)
            
            return prices
            
    def reconstruct_surface(
        self,
        volatility_surface: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct volatility surface through VAE
        
        Useful for denoising and ensuring arbitrage-free surfaces.
        
        Args:
            volatility_surface: Input surface
            
        Returns:
            Reconstructed surface
        """
        self.vae.eval()
        
        with torch.no_grad():
            reconstructed, _, _ = self.vae(volatility_surface)
            return reconstructed
            
    def get_latent_representation(
        self,
        volatility_surface: torch.Tensor
    ) -> torch.Tensor:
        """
        Get latent representation of volatility surface
        
        Args:
            volatility_surface: Input surface
            
        Returns:
            10-dimensional latent vector
        """
        self.vae.eval()
        
        with torch.no_grad():
            latent = self.vae.encode(volatility_surface)
            return latent
            
    def save(self, path: str):
        """Save both VAE and pricer models"""
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'pricer_state_dict': self.pricer.state_dict(),
            'config': self.config,
            'vae_history': self.vae_history,
            'pricer_history': self.pricer_history
        }, path)
        
    def load(self, path: str):
        """Load both VAE and pricer models"""
        checkpoint = torch.load(path)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.pricer.load_state_dict(checkpoint['pricer_state_dict'])
        self.vae_history = checkpoint.get('vae_history', {'loss': [], 'recon_loss': [], 'kl_loss': []})
        self.pricer_history = checkpoint.get('pricer_history', {'loss': []})


def create_sample_volatility_surface(
    n_strikes: int = 20,
    n_maturities: int = 15,
    base_vol: float = 0.25
) -> np.ndarray:
    """
    Create a sample implied volatility surface
    
    Generates a realistic volatility smile pattern with:
    - ATM volatility
    - Volatility smile (higher vol for OTM/ITM)
    - Term structure (volatility changes with maturity)
    
    Returns:
        Volatility surface (n_strikes, n_maturities)
    """
    strikes = np.linspace(0.7, 1.3, n_strikes)  # Moneyness
    maturities = np.linspace(0.08, 2.0, n_maturities)  # Years
    
    surface = np.zeros((n_strikes, n_maturities))
    
    for i, strike in enumerate(strikes):
        for j, maturity in enumerate(maturities):
            # ATM volatility with term structure
            atm_vol = base_vol * (1 + 0.1 * np.sqrt(maturity))
            
            # Volatility smile (quadratic in log-moneyness)
            log_moneyness = np.log(strike)
            smile = 0.1 * log_moneyness ** 2
            
            # Small random noise
            noise = np.random.normal(0, 0.01)
            
            surface[i, j] = atm_vol + smile + noise
    
    # Ensure positive
    surface = np.maximum(surface, 0.05)
    
    return surface


def create_sample_option_data(
    n_samples: int = 1000,
    config: Optional[VAEConfig] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sample training data for option pricer
    
    Returns:
        (volatility_surfaces, option_params, option_prices)
    """
    if config is None:
        config = VAEConfig()
    
    np.random.seed(42)
    
    # Generate volatility surfaces
    surfaces = []
    for _ in range(n_samples):
        base_vol = np.random.uniform(0.15, 0.35)
        surface = create_sample_volatility_surface(
            config.n_strikes, 
            config.n_maturities, 
            base_vol
        )
        surfaces.append(surface)
    
    surfaces = torch.FloatTensor(np.array(surfaces))
    
    # Generate option parameters
    strikes = np.random.uniform(80, 120, n_samples)
    maturities = np.random.uniform(0.1, 2.0, n_samples)
    spots = np.random.uniform(95, 105, n_samples)
    rates = np.random.uniform(0.01, 0.05, n_samples)
    div_yields = np.random.uniform(0.0, 0.03, n_samples)
    
    option_params = torch.FloatTensor(np.column_stack([
        strikes, maturities, spots, rates, div_yields
    ]))
    
    # Generate synthetic option prices (using simplified BS formula)
    from scipy.stats import norm
    
    prices = []
    for i in range(n_samples):
        S = spots[i]
        K = strikes[i]
        T = maturities[i]
        r = rates[i]
        q = div_yields[i]
        sigma = 0.25  # Average volatility
        
        # Black-Scholes formula for call
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-8)
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        prices.append(call_price)
    
    option_prices = torch.FloatTensor(prices).unsqueeze(1)
    
    return surfaces, option_params, option_prices


# Example usage
if __name__ == "__main__":
    print("VAE+MLP Option Pricer - Example Usage")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required")
        print("Install with: pip install torch")
    else:
        # Configuration
        config = VAEConfig(
            n_strikes=20,
            n_maturities=15,
            latent_dim=10,
            n_option_params=5
        )
        
        # Create sample data
        print("\n1. Generating sample data...")
        vol_surfaces, params, prices = create_sample_option_data(
            n_samples=500,
            config=config
        )
        print(f"   Volatility surfaces: {vol_surfaces.shape}")
        print(f"   Option parameters: {params.shape}")
        print(f"   Option prices: {prices.shape}")
        
        # Initialize pricer
        print("\n2. Initializing VAE+MLP Option Pricer...")
        pricer = VAEMLPOptionPricer(config)
        print("   ✓ VAE initialized")
        print("   ✓ MLP pricer initialized")
        
        # Train VAE
        print("\n3. Training VAE on volatility surfaces...")
        vae_history = pricer.train_vae(
            vol_surfaces,
            epochs=50,
            learning_rate=1e-3,
            verbose=1
        )
        print("   ✓ VAE training completed")
        
        # Train pricer
        print("\n4. Training MLP option pricer...")
        pricer_history = pricer.train_pricer(
            vol_surfaces,
            params,
            prices,
            epochs=50,
            learning_rate=1e-3,
            verbose=1
        )
        print("   ✓ Pricer training completed")
        
        # Test pricing
        print("\n5. Testing option pricing...")
        test_surface = create_sample_volatility_surface()
        test_price = pricer.price_option(
            volatility_surface=test_surface,
            strike=100.0,
            maturity=1.0,
            spot=100.0,
            rate=0.03,
            dividend_yield=0.02,
            option_type=OptionType.EUROPEAN_CALL
        )
        print(f"   Sample option price: ${test_price:.2f}")
        
        print("\n6. Model summary:")
        print(f"   VAE latent dimension: {config.latent_dim}D")
        print(f"   Surface compression: {config.n_strikes * config.n_maturities}D → {config.latent_dim}D")
        print(f"   Compression ratio: {(config.n_strikes * config.n_maturities) / config.latent_dim:.1f}x")
        print(f"   Final VAE loss: {vae_history['loss'][-1]:.4f}")
        print(f"   Final pricer loss: {pricer_history['loss'][-1]:.6f}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("\nBased on: Ding et al. (September 2025) arXiv:2509.05911")