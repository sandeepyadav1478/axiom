"""
GJR-GARCH VaR for Asymmetric Volatility

Based on: Giot & Laurent (2003)
"Value-at-risk for long and short trading positions"
Journal of Applied Econometrics

GJR-GARCH captures leverage effects:
- Negative returns increase volatility MORE than positive returns
- Critical for downside risk (VaR)
- Model: σ²ₜ = ω + α*ε²ₜ₋₁ + γ*I{εₜ₋₁<0}*ε²ₜ₋₁ + β*σ²ₜ₋₁

Where γ > 0 captures asymmetry (leverage effect).

Better than standard GARCH for VaR applications.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class GJRGARCHConfig:
    """Config for GJR-GARCH VaR"""
    p: int = 1  # GARCH lag
    o: int = 1  # Asymmetry lag (GJR component)
    q: int = 1  # ARCH lag
    dist: str = 'studentst'  # Student's t for fat tails


class GJRGARCHVaR:
    """
    GJR-GARCH VaR for asymmetric volatility
    
    Captures leverage effects critical for downside risk.
    """
    
    def __init__(self, config: Optional[GJRGARCHConfig] = None):
        if not ARCH_AVAILABLE or not SCIPY_AVAILABLE:
            raise ImportError("arch and scipy required")
        
        self.config = config or GJRGARCHConfig()
        self.model = None
        self.fitted_model = None
    
    def fit(self, returns: np.ndarray):
        """
        Fit GJR-GARCH model
        
        Args:
            returns: Historical returns
        """
        # Create GJR-GARCH model
        self.model = arch_model(
            returns * 100,  # Scale to percentage
            vol='GARCH',
            p=self.config.p,
            o=self.config.o,  # This makes it GJR-GARCH
            q=self.config.q,
            dist=self.config.dist
        )
        
        # Fit model
        self.fitted_model = self.model.fit(disp='off', show_warning=False)
        
        return {
            'omega': self.fitted_model.params['omega'],
            'alpha': self.fitted_model.params.get('alpha[1]', 0),
            'gamma': self.fitted_model.params.get('gamma[1]', 0),  # Leverage effect
            'beta': self.fitted_model.params.get('beta[1]', 0)
        }
    
    def forecast_volatility(self, horizon: int = 1) -> float:
        """
        Forecast conditional volatility
        
        Args:
            horizon: Days ahead
            
        Returns:
            Forecasted volatility
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted")
        
        # Forecast
        forecast = self.fitted_model.forecast(horizon=horizon)
        
        # Extract volatility forecast
        vol_forecast = np.sqrt(forecast.variance.iloc[-1, 0]) / 100  # Convert back from %
        
        return vol_forecast
    
    def calculate_var(
        self,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        Calculate VaR using GJR-GARCH forecast
        
        Args:
            confidence_level: Confidence level
            horizon: Time horizon in days
            
        Returns:
            VaR estimate
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted")
        
        # Forecast volatility
        vol_forecast = self.forecast_volatility(horizon)
        
        # Get distribution
        if self.config.dist == 'studentst':
            # Use Student's t quantile (fitted df)
            df = self.fitted_model.params.get('nu', 5)
            from scipy.stats import t
            z_score = t.ppf(1 - confidence_level, df)
        else:
            # Normal
            z_score = norm.ppf(1 - confidence_level)
        
        # VaR = volatility * z_score
        var = -z_score * vol_forecast * np.sqrt(horizon)
        
        return var


if __name__ == "__main__":
    print("GJR-GARCH VaR - Research Implementation")
    print("=" * 60)
    
    if ARCH_AVAILABLE and SCIPY_AVAILABLE:
        # Sample data with leverage effect
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        
        # Add leverage effect (negative returns → higher vol)
        for i in range(1, len(returns)):
            if returns[i-1] < -0.02:
                returns[i] *= 1.5
        
        # Fit GJR-GARCH
        gjr = GJRGARCHVaR()
        params = gjr.fit(returns)
        
        print("\nGJR-GARCH Parameters:")
        print(f"  ω (omega): {params['omega']:.6f}")
        print(f"  α (alpha): {params['alpha']:.6f}")
        print(f"  γ (gamma): {params['gamma']:.6f}  ← Leverage effect")
        print(f"  β (beta): {params['beta']:.6f}")
        
        # Forecast
        vol_forecast = gjr.forecast_volatility(horizon=1)
        print(f"\nVolatility Forecast: {vol_forecast:.2%}")
        
        # VaR
        var_95 = gjr.calculate_var(0.95)
        var_99 = gjr.calculate_var(0.99)
        
        print(f"\nGJR-GARCH VaR:")
        print(f"  95% VaR: {var_95:.2%}")
        print(f"  99% VaR: {var_99:.2%}")
        
        print("\n✓ GJR-GARCH from research implemented")
        print("Captures leverage effects for better downside risk")