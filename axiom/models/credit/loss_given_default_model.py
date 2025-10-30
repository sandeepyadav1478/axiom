"""Loss Given Default (LGD) Model"""
import numpy as np

class LossGivenDefaultModel:
    def predict_lgd(self, exposure_at_default, recovery_rate, collateral_value):
        """Calculate loss given default"""
        loss = exposure_at_default - min(exposure_at_default, collateral_value * recovery_rate)
        lgd = loss / exposure_at_default if exposure_at_default > 0 else 0.0
        return max(0.0, min(1.0, lgd))
    
    def calculate_expected_loss(self, exposure, default_prob, lgd):
        """Calculate expected loss"""
        return exposure * default_prob * lgd

if __name__ == "__main__":
    model = LossGivenDefaultModel()
    lgd = model.predict_lgd(100000, 0.40, 80000)
    el = model.calculate_expected_loss(100000, 0.15, lgd)
    print(f"LGD: {lgd:.0%}, EL: ${el:,.0f}")