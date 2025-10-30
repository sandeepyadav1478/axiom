"""Maximum Diversification Portfolio"""
import numpy as np
from scipy.optimize import minimize

class MaximumDiversification:
    def optimize(self, returns_data):
        cov = np.cov(returns_data.T)
        vols = np.sqrt(np.diag(cov))
        n = len(vols)
        
        def diversification_ratio(w):
            port_vol = np.sqrt(w @ cov @ w)
            weighted_vol = w @ vols
            return -weighted_vol / port_vol
        
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0, 0.25) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(diversification_ratio, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else x0

if __name__ == "__main__":
    returns = np.random.randn(100, 5) * 0.01
    md = MaximumDiversification()
    w = md.optimize(returns)
    print(f"Max div: {w}")