"""Chooser Options - Choose call or put at future date"""
import numpy as np
from scipy.stats import norm

class ChooserOptions:
    def price_chooser(self, S, K, T_choose, T_expiry, r, sigma):
        """Price chooser option (can choose call or put at T_choose)"""
        # At T_choose, holder picks max(Call, Put)
        # Analytical formula exists
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_expiry) / (sigma*np.sqrt(T_expiry))
        d2 = d1 - sigma*np.sqrt(T_expiry)
        
        y1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_choose) / (sigma*np.sqrt(T_choose))
        y2 = y1 - sigma*np.sqrt(T_choose)
        
        call_value = S*norm.cdf(d1) - K*np.exp(-r*T_expiry)*norm.cdf(d2)
        put_value = K*np.exp(-r*T_expiry)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        chooser_value = call_value*norm.cdf(y1) + put_value*norm.cdf(-y1)
        
        return chooser_value

if __name__ == "__main__":
    pricer = ChooserOptions()
    price = pricer.price_chooser(100, 100, 0.5, 1.0, 0.03, 0.25)
    print(f"Chooser: ${price:.2f}")