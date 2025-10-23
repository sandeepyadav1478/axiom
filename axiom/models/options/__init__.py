"""
Axiom Options Pricing Models
=============================

Institutional-grade options pricing and analysis tools:
- Black-Scholes-Merton model (European options)
- Binomial tree model (American options)
- Monte Carlo simulation (Exotic options)
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility solver (Newton-Raphson)
- Real-time options chain analysis

All models are optimized for <10ms execution time with institutional-grade accuracy.
"""

from .black_scholes import (
    BlackScholesModel,
    OptionType,
    calculate_option_price,
    calculate_call_price,
    calculate_put_price,
)
from .greeks import (
    GreeksCalculator,
    Greeks,
    calculate_greeks,
    calculate_delta,
    calculate_gamma,
    calculate_vega,
    calculate_theta,
    calculate_rho,
)
from .implied_vol import (
    ImpliedVolatilitySolver,
    calculate_implied_volatility,
    newton_raphson_iv,
)

__all__ = [
    # Black-Scholes
    "BlackScholesModel",
    "OptionType",
    "calculate_option_price",
    "calculate_call_price",
    "calculate_put_price",
    # Greeks
    "GreeksCalculator",
    "Greeks",
    "calculate_greeks",
    "calculate_delta",
    "calculate_gamma",
    "calculate_vega",
    "calculate_theta",
    "calculate_rho",
    # Implied Volatility
    "ImpliedVolatilitySolver",
    "calculate_implied_volatility",
    "newton_raphson_iv",
]