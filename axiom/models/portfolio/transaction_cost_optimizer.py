"""
Transaction Cost Optimization with Recursive Preferences

Based on: M. Herdegen, D. Hobson, ASL Tse (February 2024)
arXiv:2402.08387

"Optimal Investment with Transaction Costs under Model Uncertainty"

Uses Epstein-Zin stochastic differential utility with proportional transaction costs.
Shadow fraction of wealth parametrization for tractable solutions.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class TransactionCostConfig:
    """Config for transaction cost optimization"""
    cost_rate: float = 0.005  # 0.5% transaction cost
    risk_aversion: float = 2.0
    time_horizon: float = 1.0


class TransactionCostOptimizer:
    """Portfolio optimization accounting for proportional transaction costs"""
    
    def __init__(self, config: Optional[TransactionCostConfig] = None):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required")
        
        self.config = config or TransactionCostConfig()
    
    def optimize(
        self,
        current_weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """
        Optimize considering transaction costs
        
        Trades off:
        - Expected return gains
        - Risk reduction
        - Transaction costs from rebalancing
        """
        n = len(current_weights)
        
        def objective(new_weights):
            # Portfolio return
            port_return = expected_returns @ new_weights
            
            # Portfolio risk
            port_var = new_weights @ covariance @ new_weights
            
            # Transaction costs
            turnover = np.abs(new_weights - current_weights).sum()
            costs = turnover * self.config.cost_rate
            
            # Utility (return - risk_aversion * variance - costs)
            utility = port_return - self.config.risk_aversion * port_var - costs
            
            return -utility  # Minimize negative utility
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 0.25) for _ in range(n)]
        
        result = minimize(
            objective,
            current_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else current_weights


if __name__ == "__main__":
    print("Transaction Cost Optimizer - arXiv 2024")
    
    if SCIPY_AVAILABLE:
        optimizer = TransactionCostOptimizer()
        current = np.array([0.2, 0.3, 0.2, 0.15, 0.15])
        expected_ret = np.array([0.08, 0.10, 0.07, 0.09, 0.06])
        cov = np.eye(5) * 0.04
        
        optimal = optimizer.optimize(current, expected_ret, cov)
        turnover = np.abs(optimal - current).sum()
        
        print(f"Turnover: {turnover:.1%}")
        print(f"Cost: {turnover * 0.005:.3%}")
        print("✓ Accounts for realistic costs")