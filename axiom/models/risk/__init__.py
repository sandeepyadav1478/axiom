"""
Quantitative Risk Models for Trading and Portfolio Management

Provides industry-standard risk measurement and management tools:
- Value at Risk (VaR) - Portfolio loss estimation
- Expected Shortfall (ES/CVaR) - Tail risk measurement
- Risk decomposition and attribution
- Backtesting and model validation

Designed for:
- Quantitative traders and hedge funds
- Risk managers and compliance teams
- Portfolio managers
- Institutional investors
"""

from .var_models import (
    # Core VaR classes
    VaRMethod,
    ConfidenceLevel,
    VaRResult,
    ParametricVaR,
    HistoricalSimulationVaR,
    MonteCarloVaR,
    VaRCalculator,
    
    # Portfolio functions
    calculate_portfolio_var,
    calculate_marginal_var,
    calculate_component_var,
    
    # Convenience functions
    quick_var,
    regulatory_var
)

__all__ = [
    # Enums and data classes
    "VaRMethod",
    "ConfidenceLevel",
    "VaRResult",
    
    # VaR calculation classes
    "ParametricVaR",
    "HistoricalSimulationVaR",
    "MonteCarloVaR",
    "VaRCalculator",
    
    # Portfolio VaR functions
    "calculate_portfolio_var",
    "calculate_marginal_var",
    "calculate_component_var",
    
    # Convenience functions
    "quick_var",
    "regulatory_var"
]