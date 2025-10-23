"""
Time Series Models for Algorithmic Trading

Implements ARIMA, GARCH, and EWMA models for price forecasting,
volatility estimation, and trend analysis.

Quick Start:
    >>> from axiom.models.time_series import ARIMAModel, GARCHModel, EWMAModel
    >>> 
    >>> # Fit ARIMA model
    >>> arima = ARIMAModel()
    >>> arima.fit(price_data)
    >>> forecast = arima.forecast(horizon=5)
    >>> 
    >>> # Fit GARCH model for volatility
    >>> garch = GARCHModel(order=(1, 1))
    >>> garch.fit(returns_data)
    >>> vol_forecast = garch.forecast(horizon=10)
    >>> 
    >>> # Calculate EWMA
    >>> ewma = EWMAModel(decay_factor=0.94)
    >>> ewma.fit(returns_data)
    >>> trend = ewma.detect_trend()
"""

# Base classes
from .base_model import (
    BaseTimeSeriesModel,
    ModelType,
    ForecastHorizon,
    TimeSeriesData,
    ForecastResult,
    ModelDiagnostics
)

# ARIMA Model
from .arima import (
    ARIMAModel,
    ARIMAOrder
)

# GARCH Model
from .garch import (
    GARCHModel,
    GARCHOrder,
    VolatilityForecast,
    fit_garch
)

# EWMA Model
from .ewma import (
    EWMAModel,
    EWMAResult,
    calculate_ewma,
    calculate_ewma_volatility
)

# Configuration
from .config import (
    TradingFrequency,
    RiskProfile,
    ARIMAConfig,
    GARCHConfig,
    EWMAConfig,
    TimeSeriesConfig,
    DEFAULT_CONFIG,
    INTRADAY_CONFIG,
    SWING_CONFIG,
    POSITION_CONFIG,
    RISK_MANAGEMENT_CONFIG
)

# Utilities
from .utils import (
    prepare_returns,
    check_stationarity,
    calculate_acf,
    calculate_pacf,
    detect_seasonality,
    calculate_rolling_statistics,
    detect_outliers,
    calculate_information_criteria,
    split_train_test,
    calculate_forecast_accuracy,
    detrend,
    normalize_series
)


__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseTimeSeriesModel",
    "ModelType",
    "ForecastHorizon",
    "TimeSeriesData",
    "ForecastResult",
    "ModelDiagnostics",
    
    # ARIMA
    "ARIMAModel",
    "ARIMAOrder",
    
    # GARCH
    "GARCHModel",
    "GARCHOrder",
    "VolatilityForecast",
    "fit_garch",
    
    # EWMA
    "EWMAModel",
    "EWMAResult",
    "calculate_ewma",
    "calculate_ewma_volatility",
    
    # Configuration
    "TradingFrequency",
    "RiskProfile",
    "ARIMAConfig",
    "GARCHConfig",
    "EWMAConfig",
    "TimeSeriesConfig",
    "DEFAULT_CONFIG",
    "INTRADAY_CONFIG",
    "SWING_CONFIG",
    "POSITION_CONFIG",
    "RISK_MANAGEMENT_CONFIG",
    
    # Utilities
    "prepare_returns",
    "check_stationarity",
    "calculate_acf",
    "calculate_pacf",
    "detect_seasonality",
    "calculate_rolling_statistics",
    "detect_outliers",
    "calculate_information_criteria",
    "split_train_test",
    "calculate_forecast_accuracy",
    "detrend",
    "normalize_series",
]