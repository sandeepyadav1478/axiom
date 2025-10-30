"""
Green Bond Credit Risk Model with Climate Factors

Incorporates ESG and climate risk factors into credit assessment.
Based on emerging research in sustainable finance 2024-2025.

Accounts for:
- Transition risk (policy changes)
- Physical risk (climate events)
- ESG performance metrics
- Green bond premium/discount
"""

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class GreenBondConfig:
    """Config for green bond credit model"""
    include_climate_risk: bool = True
    include_esg_score: bool = True
    climate_risk_weight: float = 0.15
    esg_score_weight: float = 0.10


class GreenBondCreditModel:
    """Credit model incorporating ESG and climate factors"""
    
    def __init__(self, config: Optional[GreenBondConfig] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required")
        
        self.config = config or GreenBondConfig()
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train with traditional + ESG features"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_default_probability(self, financial_features: Dict, esg_features: Dict) -> float:
        """Predict with climate factors"""
        features = self._combine_features(financial_features, esg_features)
        
        if not self.is_trained:
            # Heuristic
            base_prob = 0.15
            if esg_features.get('esg_score', 50) > 70:
                base_prob *= 0.8  # Lower risk for high ESG
            if esg_features.get('climate_risk', 50) > 70:
                base_prob *= 1.2  # Higher risk
            return base_prob
        
        X = self.scaler.transform([features])
        return self.model.predict_proba(X)[0, 1]
    
    def _combine_features(self, financial: Dict, esg: Dict) -> list:
        """Combine traditional + ESG features"""
        return [
            financial.get('revenue', 0) / 1e9,
            financial.get('debt', 0) / 1e9,
            financial.get('ebitda_margin', 0),
            esg.get('esg_score', 50) / 100,
            esg.get('climate_risk', 50) / 100,
            esg.get('transition_risk', 50) / 100
        ]


if __name__ == "__main__":
    print("Green Bond Credit Model")
    model = GreenBondCreditModel()
    prob = model.predict_default_probability(
        {'revenue': 500e6, 'debt': 200e6, 'ebitda_margin': 0.20},
        {'esg_score': 75, 'climate_risk': 30}
    )
    print(f"Default probability: {prob:.2%}")