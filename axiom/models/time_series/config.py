"""
Configuration Settings for Time Series Models

Provides default configurations and parameter presets for algorithmic trading.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum


class TradingFrequency(Enum):
    """Trading data frequency."""
    MINUTE = "1min"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"


class RiskProfile(Enum):
    """Risk profile for parameter selection."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class ARIMAConfig:
    """ARIMA model configuration."""
    
    max_p: int = 5  # Maximum AR order
    max_d: int = 2  # Maximum differencing
    max_q: int = 5  # Maximum MA order
    default_order: Optional[Tuple[int, int, int]] = None  # Default (p,d,q)
    trend: str = 'c'  # 'n', 'c', 't', 'ct'
    max_iter: int = 100
    seasonal: bool = False
    
    @classmethod
    def for_price_forecasting(cls) -> 'ARIMAConfig':
        """Configuration optimized for price forecasting."""
        return cls(
            max_p=3,
            max_d=1,
            max_q=3,
            trend='c',
            seasonal=False
        )
    
    @classmethod
    def for_returns_forecasting(cls) -> 'ARIMAConfig':
        """Configuration optimized for returns forecasting."""
        return cls(
            max_p=5,
            max_d=0,
            max_q=5,
            trend='c',
            seasonal=False
        )


@dataclass
class GARCHConfig:
    """GARCH model configuration."""
    
    order: Tuple[int, int] = (1, 1)  # (p, q)
    mean_model: str = 'constant'  # 'zero', 'constant', 'ar'
    distribution: str = 'normal'  # 'normal', 'studentt', 'ged'
    max_iter: int = 1000
    
    @classmethod
    def riskmetrics(cls) -> 'GARCHConfig':
        """RiskMetrics-style configuration (essentially GARCH(1,1))."""
        return cls(
            order=(1, 1),
            mean_model='zero',
            distribution='normal'
        )
    
    @classmethod
    def for_high_frequency(cls) -> 'GARCHConfig':
        """Configuration for high-frequency data."""
        return cls(
            order=(1, 1),
            mean_model='constant',
            distribution='studentt'  # Fat tails
        )
    
    @classmethod
    def for_volatility_clustering(cls) -> 'GARCHConfig':
        """Configuration when strong volatility clustering detected."""
        return cls(
            order=(2, 2),  # Higher order for persistence
            mean_model='constant',
            distribution='normal'
        )


@dataclass
class EWMAConfig:
    """EWMA model configuration."""
    
    decay_factor: Optional[float] = None
    span: Optional[int] = None
    half_life: Optional[int] = None
    adjust: bool = True
    calculate_volatility: bool = True
    
    @classmethod
    def riskmetrics_daily(cls) -> 'EWMAConfig':
        """RiskMetrics configuration for daily data."""
        return cls(decay_factor=0.94)
    
    @classmethod
    def riskmetrics_monthly(cls) -> 'EWMAConfig':
        """RiskMetrics configuration for monthly data."""
        return cls(decay_factor=0.97)
    
    @classmethod
    def for_trend_following(cls, frequency: TradingFrequency) -> 'EWMAConfig':
        """Configuration for trend following strategies."""
        span_map = {
            TradingFrequency.MINUTE: 60,
            TradingFrequency.HOURLY: 24,
            TradingFrequency.DAILY: 20,
            TradingFrequency.WEEKLY: 10,
            TradingFrequency.MONTHLY: 6
        }
        return cls(span=span_map.get(frequency, 20))
    
    @classmethod
    def for_mean_reversion(cls) -> 'EWMAConfig':
        """Configuration for mean reversion strategies."""
        return cls(span=10)  # Faster response


@dataclass
class TimeSeriesConfig:
    """Master configuration for all time series models."""
    
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    ewma: EWMAConfig = field(default_factory=EWMAConfig)
    
    # General settings
    confidence_level: float = 0.95
    forecast_horizon: int = 5
    periods_per_year: int = 252  # Trading days
    
    @classmethod
    def for_intraday_trading(cls) -> 'TimeSeriesConfig':
        """Configuration for intraday trading."""
        return cls(
            arima=ARIMAConfig(max_p=3, max_d=0, max_q=3),
            garch=GARCHConfig.for_high_frequency(),
            ewma=EWMAConfig(span=60),
            forecast_horizon=10,
            periods_per_year=252 * 6.5  # Trading hours
        )
    
    @classmethod
    def for_swing_trading(cls) -> 'TimeSeriesConfig':
        """Configuration for swing trading (multi-day holds)."""
        return cls(
            arima=ARIMAConfig.for_price_forecasting(),
            garch=GARCHConfig(order=(1, 1)),
            ewma=EWMAConfig(span=20),
            forecast_horizon=5,
            periods_per_year=252
        )
    
    @classmethod
    def for_position_trading(cls) -> 'TimeSeriesConfig':
        """Configuration for position trading (weeks to months)."""
        return cls(
            arima=ARIMAConfig(max_p=5, max_d=1, max_q=5),
            garch=GARCHConfig(order=(2, 1)),
            ewma=EWMAConfig(span=50),
            forecast_horizon=20,
            periods_per_year=252
        )
    
    @classmethod
    def for_risk_management(cls, profile: RiskProfile) -> 'TimeSeriesConfig':
        """Configuration optimized for risk management."""
        if profile == RiskProfile.CONSERVATIVE:
            return cls(
                garch=GARCHConfig.riskmetrics(),
                ewma=EWMAConfig.riskmetrics_daily(),
                confidence_level=0.99,
                forecast_horizon=10
            )
        elif profile == RiskProfile.MODERATE:
            return cls(
                garch=GARCHConfig(order=(1, 1)),
                ewma=EWMAConfig.riskmetrics_daily(),
                confidence_level=0.95,
                forecast_horizon=5
            )
        else:  # AGGRESSIVE
            return cls(
                garch=GARCHConfig(order=(1, 1)),
                ewma=EWMAConfig(span=10),
                confidence_level=0.90,
                forecast_horizon=3
            )


# Default configurations
DEFAULT_CONFIG = TimeSeriesConfig()
INTRADAY_CONFIG = TimeSeriesConfig.for_intraday_trading()
SWING_CONFIG = TimeSeriesConfig.for_swing_trading()
POSITION_CONFIG = TimeSeriesConfig.for_position_trading()
RISK_MANAGEMENT_CONFIG = TimeSeriesConfig.for_risk_management(RiskProfile.MODERATE)


# Export
__all__ = [
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
    "RISK_MANAGEMENT_CONFIG"
]