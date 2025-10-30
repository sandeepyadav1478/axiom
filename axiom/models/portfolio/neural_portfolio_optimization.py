"""
Neural Portfolio Optimization

Direct neural network approach to portfolio optimization.
Learns optimal allocation patterns from historical data.
"""

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NeuralPortfolioOptimizer(nn.Module):
    """Neural network for portfolio optimization"""
    
    def __init__(self, n_assets: int = 10, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_assets * 5, hidden_dim),  # 5 features per asset
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )
    
    def forward(self, market_data):
        """Predict optimal weights"""
        return self.net(market_data)


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        model = NeuralPortfolioOptimizer()
        data = torch.randn(1, 50)
        weights = model(data)
        print(f"Weights: {weights.shape}")