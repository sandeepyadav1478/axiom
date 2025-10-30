"""
LSTM Stock Price Factor Model for Corporate Default

Based on: Z. Lin (2025)
"Enterprise Loan Default Prediction Based on Neural Network LSTM Model—Stock Price Factor Model"
International Conference on AI and Machine Learning, 2025

Uses stock price movements as leading indicators for corporate default risk.
Bridges equity and credit markets.
"""

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LSTMStockFactorCredit(nn.Module):
    """LSTM using stock prices to predict corporate defaults"""
    
    def __init__(self, seq_len: int = 30, hidden_dim: int = 64):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=5,  # OHLCV
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, stock_data):
        """Predict default from stock prices"""
        lstm_out, _ = self.lstm(stock_data)
        final_state = lstm_out[:, -1, :]
        default_prob = self.fc(final_state)
        return default_prob


if __name__ == "__main__":
    print("LSTM Stock Factor Credit - 2025")
    if TORCH_AVAILABLE:
        model = LSTMStockFactorCredit()
        stock_data = torch.randn(1, 30, 5)
        prob = model(stock_data)
        print(f"Default probability: {prob.item():.2%}")
        print("✓ Stock prices → credit risk")