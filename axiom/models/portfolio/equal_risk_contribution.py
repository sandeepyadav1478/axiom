"""Equal Risk Contribution Portfolio"""
import numpy as np
from scipy.optimize import minimize

class EqualRiskContribution:
    def optimize(self, cov_matrix):
        n = cov_matrix.shape[0]
        
        def risk_budget_objective(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            marginal_contrib = cov_matrix @ w
            risk_contrib = w * marginal_contrib / port_vol
            target = port_vol / n
            return np.sum((risk_contrib - target)**2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0.01, 0.30) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else x0

if __name__ == "__main__":
    cov = np.random.rand(5,5); cov = cov @ cov.T
    erc = EqualRiskContribution()
    w = erc.optimize(cov)
    print(f"ERC weights: {w}")