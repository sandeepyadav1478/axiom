"""Asian Options Pricer - Arithmetic and Geometric Averaging"""
import numpy as np
from scipy.stats import norm

class AsianOptionPricer:
    def price_geometric_asian_call(self, S, K, T, r, sigma, n_steps=252):
        """Geometric average Asian call (closed form)"""
        sigma_g = sigma / np.sqrt(3)
        mu_g = (r - 0.5 * sigma**2) / 2 + (r + sigma_g**2 / 2)
        
        d1 = (np.log(S/K) + (mu_g + 0.5*sigma_g**2)*T) / (sigma_g*np.sqrt(T))
        d2 = d1 - sigma_g*np.sqrt(T)
        
        return S*np.exp((mu_g - r)*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    def price_arithmetic_asian_call(self, S, K, T, r, sigma, n_steps=252, n_sims=10000):
        """Arithmetic average Asian call (Monte Carlo)"""
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S
        
        for t in range(1, n_steps + 1):
            Z = np.random.randn(n_sims)
            paths[:, t] = paths[:, t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        
        avg_prices = paths.mean(axis=1)
        payoffs = np.maximum(avg_prices - K, 0)
        price = np.exp(-r*T) * payoffs.mean()
        return price

if __name__ == "__main__":
    pricer = AsianOptionPricer()
    geom = pricer.price_geometric_asian_call(100, 100, 1.0, 0.03, 0.25)
    arith = pricer.price_arithmetic_asian_call(100, 100, 1.0, 0.03, 0.25, n_sims=5000)
    print(f"Geometric: ${geom:.2f}, Arithmetic: ${arith:.2f}")