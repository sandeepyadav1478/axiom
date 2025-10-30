"""
Black-Scholes-ANN Hybrid Option Pricing Model

Based on: Milad Shahvaroughi Farahani, Shiva Babaei, Amirhossein Estahani (2024)
"Black-Scholes-Artificial Neural Network: A novel option pricing model"
International Journal of Financial, Accounting, and Management (IJFAM)

Combines theoretical Black-Scholes with ANN corrections for better accuracy.
Tested on real Khodro automobile company options (Tehran).

Most accurate estimation with lowest standard deviation among 8 models tested.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class BSANNConfig:
    """Config for BS-ANN Hybrid"""
    hidden_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 1e-3


class ANNCorrectionNetwork(nn.Module):
    """ANN that learns corrections to Black-Scholes"""
    
    def __init__(self, config: BSANNConfig):
        super().__init__()
        
        layers = []
        input_dim = 6  # S, K, T, r, sigma, BS_price
        prev_dim = input_dim
        
        for _ in range(config.n_layers):
            layers.extend([
                nn.Linear(prev_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = config.hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict correction to BS price"""
        return self.network(x)


class BSANNHybridPricer:
    """
    Hybrid Black-Scholes + ANN Pricer
    
    Uses theoretical BS as baseline, ANN learns market corrections
    """
    
    def __init__(self, config: Optional[BSANNConfig] = None):
        if not TORCH_AVAILABLE or not SCIPY_AVAILABLE:
            raise ImportError("PyTorch and scipy required")
        
        self.config = config or BSANNConfig()
        self.correction_network = ANNCorrectionNetwork(self.config)
        self.optimizer = None
    
    def _black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """Calculate theoretical BS price"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-8)
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100):
        """Train ANN correction"""
        self.correction_network.train()
        self.optimizer = torch.optim.Adam(
            self.correction_network.parameters(),
            lr=self.config.learning_rate
        )
        
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Predict correction
            correction = self.correction_network(X)
            
            # BS prices are in X[:, 5]
            bs_prices = X[:, 5:6]
            hybrid_prices = bs_prices + correction
            
            # Loss
            loss = criterion(hybrid_prices, y)
            
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
    
    def price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """Price option with hybrid model"""
        # BS baseline
        bs_price = self._black_scholes(S, K, T, r, sigma, option_type)
        
        # ANN correction
        self.correction_network.eval()
        with torch.no_grad():
            x = torch.FloatTensor([[S, K, T, r, sigma, bs_price]])
            correction = self.correction_network(x).item()
        
        return bs_price + correction


if __name__ == "__main__":
    print("BS-ANN Hybrid - Example")
    
    if TORCH_AVAILABLE and SCIPY_AVAILABLE:
        pricer = BSANNHybridPricer()
        price = pricer.price(100, 100, 1.0, 0.03, 0.25)
        print(f"Hybrid price: ${price:.2f}")
        print("✓ Combines theory + ML")