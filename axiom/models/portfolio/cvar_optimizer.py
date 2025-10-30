"""CVaR (Conditional Value at Risk) Portfolio Optimization"""
import numpy as np
import cvxpy as cp

class CVaROptimizer:
    def optimize(self, returns_data, alpha=0.95, target_return=None):
        T, n = returns_data.shape
        w = cp.Variable(n)
        z = cp.Variable(T)
        zeta = cp.Variable()
        
        # CVaR optimization
        cvar = zeta + (1/(T*(1-alpha))) * cp.sum(z)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= 0.25,
            z >= 0,
            z >= -returns_data @ w - zeta
        ]
        
        if target_return:
            constraints.append(returns_data.mean(axis=0) @ w >= target_return)
        
        prob = cp.Problem(cp.Minimize(cvar), constraints)
        prob.solve()
        
        return w.value if w.value is not None else np.ones(n)/n

if __name__ == "__main__":
    returns = np.random.randn(100, 5) * 0.01
    opt = CVaROptimizer()
    w = opt.optimize(returns)
    print(f"CVaR weights: {w}")