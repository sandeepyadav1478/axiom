"""
Multi-Asset Option Pricing

For options on multiple underlyings:
- Basket options
- Best-of options
- Worst-of options
- Rainbow options

Uses Monte Carlo with correlation modeling.
"""

import numpy as np

class MultiAssetOptionPricer:
    """Price options on multiple assets"""
    
    def __init__(self, n_assets: int = 2):
        self.n_assets = n_assets
    
    def price_basket_option(
        self,
        spots: np.ndarray,
        strike: float,
        time_to_mat: float,
        rate: float,
        vols: np.ndarray,
        correlation: np.ndarray,
        weights: np.ndarray,
        n_sims: int = 10000
    ) -> float:
        """Price basket option using MC"""
        
        # Cholesky decomposition for correlated paths
        L = np.linalg.cholesky(correlation)
        
        # Simulate paths
        Z = np.random.randn(n_sims, self.n_assets)
        Z_corr = Z @ L.T
        
        # Terminal prices
        S_T = spots * np.exp(
            (rate - 0.5 * vols**2) * time_to_mat +
            vols * np.sqrt(time_to_mat) * Z_corr
        )
        
        # Basket value
        basket_values = S_T @ weights
        
        # Payoff
        payoffs = np.maximum(basket_values - strike, 0)
        
        # Price
        price = np.exp(-rate * time_to_mat) * payoffs.mean()
        
        return price


if __name__ == "__main__":
    pricer = MultiAssetOptionPricer(n_assets=2)
    price = pricer.price_basket_option(
        spots=np.array([100, 100]),
        strike=100,
        time_to_mat=1.0,
        rate=0.03,
        vols=np.array([0.2, 0.25]),
        correlation=np.array([[1.0, 0.5], [0.5, 1.0]]),
        weights=np.array([0.5, 0.5]),
        n_sims=10000
    )
    print(f"Basket option price: ${price:.2f}")