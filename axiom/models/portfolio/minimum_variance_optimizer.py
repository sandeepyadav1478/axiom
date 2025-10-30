"""Minimum Variance Portfolio Optimizer"""
import numpy as np
from scipy.optimize import minimize

class MinimumVarianceOptimizer:
    def optimize(self, returns_data, max_weight=0.25):
        cov = np.cov(returns_data.T)
        n = cov.shape[0]
        
        def objective(w):
            return w @ cov @ w
        
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else x0

if __name__ == "__main__":
    returns = np.random.randn(100, 5) * 0.01
    opt = MinimumVarianceOptimizer()
    weights = opt.optimize(returns)
    print(f"Min var weights: {weights}")