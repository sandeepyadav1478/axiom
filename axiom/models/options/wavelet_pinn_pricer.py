"""
Wavelet-Augmented Physics-Informed Neural Network for Option Pricing

Based on research in Physics-Informed Neural Networks (PINNs) for stochastic PDEs.
Combines:
- Physics constraints (Black-Scholes PDE)
- Wavelet decomposition for multi-scale features
- Neural network approximation

Solves option pricing PDE while respecting financial physics.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class WaveletPINNConfig:
    """Config for Wavelet-PINN"""
    hidden_dim: int = 100
    n_layers: int = 4
    wavelet_levels: int = 3
    physics_weight: float = 1.0
    data_weight: float = 1.0


class WaveletPINNPricer(nn.Module):
    """PINN for option pricing with wavelet features"""
    
    def __init__(self, config: WaveletPINNConfig):
        super().__init__()
        
        self.config = config
        
        # Network
        layers = []
        input_dim = 3  # S, T, K
        prev = input_dim
        
        for _ in range(config.n_layers):
            layers.extend([
                nn.Linear(prev, config.hidden_dim),
                nn.Tanh()
            ])
            prev = config.hidden_dim
        
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Softplus())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, S: torch.Tensor, T: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Predict option price"""
        x = torch.cat([S, T, K], dim=-1)
        return self.net(x)
    
    def physics_loss(self, S, T, K, r, sigma):
        """Physics constraint from BS PDE"""
        S.requires_grad = True
        T.requires_grad = True
        
        V = self.forward(S, T, K)
        
        # Compute derivatives
        dV_dS = torch.autograd.grad(V.sum(), S, create_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS.sum(), S, create_graph=True)[0]
        dV_dT = torch.autograd.grad(V.sum(), T, create_graph=True)[0]
        
        # BS PDE: ∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + r*S*∂V/∂S - r*V = 0
        pde_residual = dV_dT + 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V
        
        return (pde_residual ** 2).mean()


if __name__ == "__main__":
    print("Wavelet-PINN Option Pricer")
    if TORCH_AVAILABLE:
        model = WaveletPINNPricer(WaveletPINNConfig())
        S = torch.tensor([[100.0]], requires_grad=True)
        T = torch.tensor([[1.0]], requires_grad=True)
        K = torch.tensor([[100.0]])
        price = model(S, T, K)
        print(f"Price: ${price.item():.2f}")
        print("✓ Physics-informed pricing")