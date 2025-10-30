"""Recovery Rate Prediction for Credit"""
import numpy as np

class RecoveryRatePredictor:
    def predict_recovery(self, loan_characteristics):
        base_rate = 0.40
        if loan_characteristics.get('secured'):
            base_rate += 0.25
        if loan_characteristics.get('senior'):
            base_rate += 0.15
        collateral_ratio = loan_characteristics.get('collateral_value', 0) / loan_characteristics.get('loan_amount', 1)
        base_rate += min(0.20, collateral_ratio * 0.20)
        return min(0.95, max(0.10, base_rate))

if __name__ == "__main__":
    predictor = RecoveryRatePredictor()
    recovery = predictor.predict_recovery({'secured': True, 'senior': True, 'collateral_value': 80000, 'loan_amount': 100000})
    print(f"Recovery: {recovery:.0%}")