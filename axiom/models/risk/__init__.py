"""
Quantitative Risk Models for Trading and Portfolio Management

Provides industry-standard risk measurement and management tools:
- Value at Risk (VaR) - Portfolio loss estimation
- Expected Shortfall (ES/CVaR) - Tail risk measurement
- Extreme Value Theory (EVT) - Tail risk modeling
- Regime-Switching VaR - Adaptive risk estimation
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

from .evt_var import (
    # EVT VaR classes
    GPDParameters,
    EVTVaR,
    GARCHEVTVaR,
    
    # Convenience functions
    calculate_evt_var,
    calculate_garch_evt_var
)

from .regime_switching_var import (
    # Regime-Switching classes
    RegimeParameters,
    HMMModel,
    RegimeSwitchingVaR,
    
    # Convenience functions
    calculate_regime_switching_var
)

# Lazy imports for optional advanced models
def get_cnn_lstm_credit_model():
    """Lazy import of CNNLSTMCreditPredictor (requires torch, sklearn)"""
    from .cnn_lstm_credit_model import CNNLSTMCreditPredictor, CreditModelConfig
    return CNNLSTMCreditPredictor, CreditModelConfig

def get_ensemble_credit_model():
    """Lazy import of EnsembleCreditModel (requires xgboost, lightgbm, sklearn)"""
    from .ensemble_credit_model import EnsembleCreditModel, EnsembleConfig
    return EnsembleCreditModel, EnsembleConfig

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
    "regulatory_var",
    
    # EVT VaR
    "GPDParameters",
    "EVTVaR",
    "GARCHEVTVaR",
    "calculate_evt_var",
    "calculate_garch_evt_var",
    
    # Regime-Switching VaR
    "RegimeParameters",
    "HMMModel",
    "RegimeSwitchingVaR",
    "calculate_regime_switching_var"
]