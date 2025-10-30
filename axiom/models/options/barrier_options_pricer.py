"""Barrier Options Pricer - Knock-in/Knock-out"""
import numpy as np
from scipy.stats import norm

class BarrierOptionPricer:
    def price_down_and_out_call(self, S, K, H, T, r, sigma):
        """Down-and-out call option"""
        if S <= H:
            return 0.0
        
        lambda_val = (r + 0.5 * sigma**2) / (sigma**2)
        y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lambda_val * sigma * np.sqrt(T)
        x1 = np.log(S / H) / (sigma * np.sqrt(T)) + lambda_val * sigma * np.sqrt(T)
        y1 = np.log(H / S) / (sigma * np.sqrt(T)) + lambda_val * sigma * np.sqrt(T)
        
        # Vanilla call price
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        vanilla = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        # Barrier adjustment
        barrier_adj = S * (H/S)**(2*lambda_val) * norm.cdf(y) - K*np.exp(-r*T)*(H/S)**(2*lambda_val-2)*norm.cdf(y-sigma*np.sqrt(T))
        
        return vanilla - barrier_adj

if __name__ == "__main__":
    pricer = BarrierOptionPricer()
    price = pricer.price_down_and_out_call(100, 100, 90, 1.0, 0.03, 0.25)
    print(f"Barrier option: ${price:.2f}")