"""Lookback Options - Floating and Fixed Strike"""
import numpy as np

class LookbackOptions:
    def price_floating_strike_call(self, S, T, r, sigma, n_sims=10000):
        """Floating strike lookback call (payoff: S_T - min(S))"""
        dt = T / 252
        paths = np.zeros((n_sims, 253))
        paths[:, 0] = S
        
        for t in range(1, 253):
            Z = np.random.randn(n_sims)
            paths[:, t] = paths[:, t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        
        mins = paths.min(axis=1)
        payoffs = np.maximum(paths[:, -1] - mins, 0)
        return np.exp(-r*T) * payoffs.mean()

if __name__ == "__main__":
    pricer = LookbackOptions()
    price = pricer.price_floating_strike_call(100, 1.0, 0.03, 0.25, n_sims=5000)
    print(f"Lookback: ${price:.2f}")