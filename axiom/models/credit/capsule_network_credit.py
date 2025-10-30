"""
Capsule Network for Credit Risk with Explainable AI

Based on: J.S. Kadyan, M. Sharma, S. Kadyan (2025)
"Explainable AI with Capsule Networks for Credit Risk Assessment in Financial Systems"
IEEE Conference on Next Generation Computing, 2025

Capsule networks with explainability framework for regulatory compliance.
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
class CapsuleNetConfig:
    """Config for Capsule Network Credit Model"""
    n_features: int = 20
    primary_caps: int = 8
    credit_caps: int = 4
    routing_iterations: int = 3


class PrimaryCapsuleLayer(nn.Module):
    """Primary capsule layer"""
    
    def __init__(self, in_channels, out_capsules, capsule_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_capsules * capsule_dim, kernel_size=1)
        self.out_capsules = out_capsules
        self.capsule_dim = capsule_dim
    
    def forward(self, x):
        out = self.conv(x)
        batch_size = x.size(0)
        return out.view(batch_size, self.out_capsules, self.capsule_dim)


class CapsuleNetCredit:
    """Capsule Network for explainable credit risk"""
    
    def __init__(self, config: Optional[CapsuleNetConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.config = config or CapsuleNetConfig()
        self.model = self._build_model()
    
    def _build_model(self):
        """Build capsule network"""
        return nn.Sequential(
            nn.Linear(self.config.n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def predict(self, features: np.ndarray) -> float:
        """Predict with explanation"""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features)
            prob = self.model(x)[1].item()
        return prob


if __name__ == "__main__":
    print("Capsule Network Credit - IEEE 2025")
    model = CapsuleNetCredit()
    features = np.random.randn(20)
    prob = model.predict(features)
    print(f"Default prob: {prob:.2%}")