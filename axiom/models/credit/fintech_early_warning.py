"""
CNN-LSTM Early Warning for Internet Financial Enterprises

Based on: Z. Xia (2024)
"Early warning of credit risk of internet financial enterprises based on CNN-LSTM model"
Procedia Computer Science, 2024, Elsevier

Specialized for fintech companies with early warning capabilities.
LSTM captures time trends, CNN extracts spatial patterns.
"""

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FintechEarlyWarning(nn.Module):
    """Early warning system for fintech credit risk"""
    
    def __init__(self, seq_len: int = 12, n_features: int = 15):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Early warning prediction"""
        # CNN features
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        
        # LSTM temporal
        lstm_out, _ = self.lstm(x)
        
        # Classify
        prob = self.classifier(lstm_out[:, -1, :])
        return prob


if __name__ == "__main__":
    print("Fintech Early Warning - Elsevier 2024")
    if TORCH_AVAILABLE:
        model = FintechEarlyWarning()
        data = torch.randn(1, 12, 15)
        risk = model(data)
        print(f"Risk: {risk.item():.2%}")
        print("✓ Early warning for fintech")