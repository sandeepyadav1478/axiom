"""Compound Options - Options on Options"""
import numpy as np
from scipy.stats import norm

class CompoundOptions:
    def price_call_on_call(self, S, K1, K2, T1, T2, r, sigma):
        """Call option on a call option"""
        rho = np.sqrt(T1 / T2)
        
        d1 = (np.log(S/K2) + (r + 0.5*sigma**2)*T2) / (sigma*np.sqrt(T2))
        d2 = d1 - sigma*np.sqrt(T2)
        d3 = (np.log(S/K2) + (r + 0.5*sigma**2)*T1) / (sigma*np.sqrt(T1))
        d4 = d3 - sigma*np.sqrt(T1)
        
        # Bivariate normal CDF (simplified)
        from scipy.stats import multivariate_normal
        
        cov = np.array([[1, rho], [rho, 1]])
        M = multivariate_normal(cov=cov)
        
        value = S * M.cdf([d3, d1]) - K2*np.exp(-r*T2) * M.cdf([d4, d2]) - K1*np.exp(-r*T1)
        
        return max(0, value)

if __name__ == "__main__":
    pricer = CompoundOptions()
    price = pricer.price_call_on_call(100, 5, 100, 0.5, 1.0, 0.03, 0.25)
    print(f"Compound: ${price:.2f}")