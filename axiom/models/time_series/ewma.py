"""
EWMA Model - Exponentially Weighted Moving Average

Implements EWMA for trend analysis and volatility estimation in algorithmic trading.

Features:
- Trend detection and smoothing
- Volatility estimation (RiskMetrics™ approach)
- Adaptive decay parameter selection
- Real-time updating for live trading
- Integration with risk models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats

from .base_model import (
    BaseTimeSeriesModel,
    ModelType,
    TimeSeriesData,
    ForecastResult,
    ModelDiagnostics
)
from axiom.models.base.mixins import PerformanceMixin, ValidationMixin, LoggingMixin
from axiom.config.model_config import TimeSeriesConfig, get_config
from axiom.core.logging.axiom_logger import timeseries_logger


@dataclass
class EWMAResult:
    """EWMA calculation result."""
    
    ewma_values: np.ndarray  # EWMA series
    ewma_volatility: Optional[np.ndarray] = None  # EWMA volatility
    decay_factor: float = 0.94
    center_of_mass: Optional[float] = None
    half_life: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ewma_values": self.ewma_values.tolist(),
            "ewma_volatility": self.ewma_volatility.tolist() if self.ewma_volatility is not None else None,
            "decay_factor": self.decay_factor,
            "center_of_mass": self.center_of_mass,
            "half_life": self.half_life
        }


class EWMAModel(BaseTimeSeriesModel, PerformanceMixin, ValidationMixin, LoggingMixin):
    """
    EWMA Model for Trend Analysis and Volatility Estimation.
    
    EWMA assigns exponentially decreasing weights to past observations:
    - Recent data has more influence
    - Smooth trend estimation
    - Fast adaptation to regime changes
    
    EWMA formula:
    EWMAₜ = λ × EWMAₜ₋₁ + (1 - λ) × Xₜ
    
    Where λ is the decay factor (0 < λ < 1)
    
    RiskMetrics™ uses λ = 0.94 for daily volatility
    λ = 0.97 for monthly volatility
    
    Ideal for:
    - Trend following strategies
    - Volatility estimation (alternative to GARCH)
    - Risk management systems
    - Real-time trading signals
    """
    
    def __init__(
        self,
        decay_factor: Optional[float] = None,
        span: Optional[int] = None,
        half_life: Optional[int] = None,
        com: Optional[float] = None,
        adjust: bool = True,
        config: Optional[TimeSeriesConfig] = None
    ):
        """
        Initialize EWMA model.
        
        Specify exactly one of: decay_factor, span, half_life, or com
        
        Args:
            decay_factor: Decay factor λ (0 < λ < 1). RiskMetrics uses 0.94
            span: Span parameter (N). λ = 2/(N+1)
            half_life: Half-life parameter (H). λ = 1 - exp(log(0.5)/H)
            com: Center of mass (α). λ = 1/(1+α)
            adjust: Whether to use adjustment factor for initial observations
            config: TimeSeriesConfig for customization
        """
        super().__init__(ModelType.EWMA)
        self.config = config or get_config().time_series
        self.logger = timeseries_logger
        self.enable_logging = True
        
        # Determine decay factor from one of the parameters, or use config default
        self.decay_factor = self._calculate_decay_factor(
            decay_factor, span, half_life, com
        )
        if self.decay_factor == 0.94 and decay_factor is None and span is None and half_life is None and com is None:
            # Use config default if no parameters specified
            self.decay_factor = self.config.ewma_decay_factor
        
        self.adjust = adjust
        
        # Calculated parameters
        self.span = self._decay_to_span(self.decay_factor)
        self.half_life = self._decay_to_half_life(self.decay_factor)
        self.com = self._decay_to_com(self.decay_factor)
        
        # Fitted data
        self.ewma_mean: Optional[np.ndarray] = None
        self.ewma_variance: Optional[np.ndarray] = None
        self.ewma_volatility: Optional[np.ndarray] = None
    
    def fit(
        self,
        data: Union[np.ndarray, pd.Series, TimeSeriesData],
        use_returns: bool = False,
        calculate_volatility: bool = True
    ) -> 'EWMAModel':
        """
        Fit EWMA model to data.
        
        Args:
            data: Time series data (prices or returns)
            use_returns: If True, treats data as returns. If False, calculates returns
            calculate_volatility: Whether to calculate EWMA volatility
        
        Returns:
            Self for method chaining
        """
        with self.track_time("EWMA calculation"):
            # Prepare data
            ts_data = self._prepare_data(data)
            self.training_data = ts_data
            
            # Validate data size
            if len(ts_data.values) < self.config.ewma_min_periods:
                self.log_warning(
                    "Insufficient data for reliable EWMA estimation",
                    observations=len(ts_data.values),
                    recommended_minimum=self.config.ewma_min_periods
                )
            
            # Calculate returns if needed
            if use_returns:
                values = ts_data.values
            else:
                # Calculate log returns
                values = ts_data.get_returns(log_returns=True)
            
            # Calculate EWMA mean
            self.ewma_mean = self._calculate_ewma(values)
        
            # Calculate EWMA volatility if requested
            if calculate_volatility:
                # Calculate squared deviations (centered or not)
                if use_returns:
                    # For returns, use squared returns for volatility
                    squared_values = values ** 2
                else:
                    # For prices converted to returns
                    squared_values = values ** 2
                
                # EWMA variance
                self.ewma_variance = self._calculate_ewma(squared_values)
                self.ewma_volatility = np.sqrt(self.ewma_variance)
            
            self.is_fitted = True
            self.model_params = {
                "decay_factor": float(self.decay_factor),
                "span": float(self.span),
                "half_life": float(self.half_life),
                "com": float(self.com),
                "adjust": self.adjust
            }
            
            # Calculate diagnostics
            if self.ewma_mean is not None and len(self.ewma_mean) > 0:
                self.residuals = values[1:] - self.ewma_mean[:-1]
                self.diagnostics = self.calculate_diagnostics(
                    values[1:], self.ewma_mean[:-1]
                )
            
            self.log_calculation_end("EWMA calculation", self.model_params, 0)
            
            return self
    
    def forecast(
        self,
        horizon: int = 1,
        confidence_level: float = 0.95,
        last_value: Optional[float] = None
    ) -> ForecastResult:
        """
        Forecast using EWMA.
        
        For EWMA, the forecast is simply the last EWMA value for all horizons
        (constant forecast). Confidence intervals widen with horizon.
        
        Args:
            horizon: Forecast horizon
            confidence_level: Confidence level for intervals
            last_value: Optional last observed value for updating
        
        Returns:
            ForecastResult with forecasts
        """
        self._check_fitted()
        
        # Last EWMA value
        if last_value is not None:
            forecast_value = self._update_ewma(self.ewma_mean[-1], last_value)
        else:
            forecast_value = self.ewma_mean[-1]
        
        # Constant forecast
        forecasts = np.full(horizon, forecast_value)
        
        # Calculate confidence intervals
        if self.ewma_volatility is not None:
            last_vol = self.ewma_volatility[-1]
            
            # Widen intervals with horizon (sqrt rule)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            widening_factor = np.sqrt(np.arange(1, horizon + 1))
            
            lower_ci = forecasts - z_score * last_vol * widening_factor
            upper_ci = forecasts + z_score * last_vol * widening_factor
            
            confidence_intervals = (lower_ci, upper_ci)
        else:
            # Simple interval based on historical volatility
            hist_std = np.std(self.residuals) if self.residuals is not None else 0.01
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            widening_factor = np.sqrt(np.arange(1, horizon + 1))
            
            lower_ci = forecasts - z_score * hist_std * widening_factor
            upper_ci = forecasts + z_score * hist_std * widening_factor
            
            confidence_intervals = (lower_ci, upper_ci)
        
        return ForecastResult(
            forecast=forecasts,
            confidence_intervals=confidence_intervals,
            confidence_level=confidence_level,
            horizon=horizon,
            model_type=ModelType.EWMA,
            fitted_values=self.ewma_mean,
            residuals=self.residuals,
            model_params=self.model_params,
            metrics=self.diagnostics.to_dict() if self.diagnostics else {}
        )
    
    def get_volatility_forecast(
        self,
        horizon: int = 1,
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> np.ndarray:
        """
        Get EWMA volatility forecast.
        
        Args:
            horizon: Forecast horizon
            annualize: Whether to annualize volatility
            periods_per_year: Trading periods per year
        
        Returns:
            Volatility forecasts
        """
        self._check_fitted()
        
        if self.ewma_volatility is None:
            raise ValueError("Volatility not calculated. Call fit() with calculate_volatility=True")
        
        # Last volatility (constant forecast)
        last_vol = self.ewma_volatility[-1]
        vol_forecast = np.full(horizon, last_vol)
        
        # Annualize if requested
        if annualize:
            vol_forecast = vol_forecast * np.sqrt(periods_per_year)
        
        return vol_forecast
    
    def update(
        self,
        new_value: float,
        update_volatility: bool = True
    ) -> Tuple[float, Optional[float]]:
        """
        Update EWMA with new observation (for real-time trading).
        
        Args:
            new_value: New observation
            update_volatility: Whether to update volatility
        
        Returns:
            Tuple of (new_ewma_mean, new_ewma_volatility)
        """
        self._check_fitted()
        
        # Update mean
        new_ewma_mean = self._update_ewma(self.ewma_mean[-1], new_value)
        
        # Update volatility
        new_ewma_volatility = None
        if update_volatility and self.ewma_variance is not None:
            squared_value = new_value ** 2
            new_variance = self._update_ewma(self.ewma_variance[-1], squared_value)
            new_ewma_volatility = np.sqrt(new_variance)
        
        return new_ewma_mean, new_ewma_volatility
    
    def detect_trend(
        self,
        lookback: int = 20,
        threshold: float = 0.0
    ) -> Dict:
        """
        Detect trend using EWMA.
        
        Args:
            lookback: Lookback period for trend strength
            threshold: Threshold for trend detection
        
        Returns:
            Dictionary with trend information
        """
        self._check_fitted()
        
        # Calculate trend slope
        recent_ewma = self.ewma_mean[-lookback:]
        x = np.arange(len(recent_ewma))
        slope = np.polyfit(x, recent_ewma, 1)[0]
        
        # Trend direction
        if slope > threshold:
            trend = "uptrend"
        elif slope < -threshold:
            trend = "downtrend"
        else:
            trend = "sideways"
        
        # Trend strength (normalized slope)
        trend_strength = abs(slope) / (np.std(recent_ewma) + 1e-8)
        
        return {
            "trend": trend,
            "slope": float(slope),
            "strength": float(trend_strength),
            "current_ewma": float(self.ewma_mean[-1]),
            "lookback_period": lookback
        }
    
    def get_trading_signals(
        self,
        price_data: Optional[np.ndarray] = None,
        fast_span: Optional[int] = None,
        slow_span: Optional[int] = None
    ) -> Dict:
        """
        Generate trading signals using dual EWMA crossover.
        
        Args:
            price_data: Price data (if None, uses training data)
            fast_span: Fast EWMA span (defaults to config)
            slow_span: Slow EWMA span (defaults to config)
        
        Returns:
            Dictionary with trading signals
        """
        # Use config defaults if not specified
        fast_span = fast_span or self.config.ewma_fast_span
        slow_span = slow_span or self.config.ewma_slow_span
        
        if price_data is None:
            if self.training_data is None:
                raise ValueError("No data available for signal generation")
            price_data = self.training_data.values
        
        # Calculate fast and slow EWMA
        fast_ewma = self._calculate_ewma_with_span(price_data, fast_span)
        slow_ewma = self._calculate_ewma_with_span(price_data, slow_span)
        
        # Detect crossovers
        diff = fast_ewma - slow_ewma
        
        # Signal: 1 for buy, -1 for sell, 0 for hold
        signal = np.zeros(len(diff))
        signal[diff > 0] = 1  # Fast above slow: bullish
        signal[diff < 0] = -1  # Fast below slow: bearish
        
        # Detect crossover points
        crossovers = np.diff(np.sign(diff))
        buy_signals = np.where(crossovers > 0)[0] + 1
        sell_signals = np.where(crossovers < 0)[0] + 1
        
        return {
            "fast_ewma": fast_ewma,
            "slow_ewma": slow_ewma,
            "signal": signal,
            "current_signal": int(signal[-1]),
            "buy_crossovers": buy_signals.tolist(),
            "sell_crossovers": sell_signals.tolist(),
            "current_position": "long" if signal[-1] > 0 else ("short" if signal[-1] < 0 else "neutral")
        }
    
    def _calculate_ewma(self, values: np.ndarray) -> np.ndarray:
        """
        Calculate EWMA series.
        
        Args:
            values: Input values
        
        Returns:
            EWMA series
        """
        n = len(values)
        ewma = np.zeros(n)
        ewma[0] = values[0]
        
        if self.adjust:
            # Adjusted EWMA (pandas default)
            for t in range(1, n):
                weight_sum = (1 - self.decay_factor ** (t + 1)) / (1 - self.decay_factor)
                weighted_sum = sum(
                    self.decay_factor ** (t - i) * values[i]
                    for i in range(t + 1)
                )
                ewma[t] = weighted_sum / weight_sum
        else:
            # Simple recursive EWMA
            for t in range(1, n):
                ewma[t] = self.decay_factor * ewma[t-1] + (1 - self.decay_factor) * values[t]
        
        return ewma
    
    def _calculate_ewma_with_span(self, values: np.ndarray, span: int) -> np.ndarray:
        """Calculate EWMA with specific span."""
        decay = 2.0 / (span + 1)
        
        ewma = np.zeros(len(values))
        ewma[0] = values[0]
        
        for t in range(1, len(values)):
            ewma[t] = decay * values[t] + (1 - decay) * ewma[t-1]
        
        return ewma
    
    def _update_ewma(self, last_ewma: float, new_value: float) -> float:
        """Update EWMA with single new observation."""
        return self.decay_factor * last_ewma + (1 - self.decay_factor) * new_value
    
    @staticmethod
    def _calculate_decay_factor(
        decay: Optional[float],
        span: Optional[int],
        half_life: Optional[int],
        com: Optional[float]
    ) -> float:
        """
        Calculate decay factor from one of the parameterizations.
        
        Only one should be specified.
        """
        count = sum(x is not None for x in [decay, span, half_life, com])
        
        if count == 0:
            # Default: RiskMetrics for daily data
            return 0.94
        elif count > 1:
            raise ValueError("Specify only one of: decay_factor, span, half_life, or com")
        
        if decay is not None:
            if not 0 < decay < 1:
                raise ValueError("decay_factor must be between 0 and 1")
            return decay
        
        if span is not None:
            if span < 1:
                raise ValueError("span must be >= 1")
            return 2.0 / (span + 1)
        
        if half_life is not None:
            if half_life <= 0:
                raise ValueError("half_life must be positive")
            return 1 - np.exp(np.log(0.5) / half_life)
        
        if com is not None:
            if com < 0:
                raise ValueError("com must be non-negative")
            return 1.0 / (1 + com)
        
        return 0.94  # Default
    
    @staticmethod
    def _decay_to_span(decay: float) -> float:
        """Convert decay factor to span."""
        # decay = 2 / (span + 1)
        # Therefore: span = (2 / decay) - 1
        return (2.0 / decay) - 1
    
    @staticmethod
    def _decay_to_half_life(decay: float) -> float:
        """Convert decay factor to half-life."""
        return np.log(0.5) / np.log(decay)
    
    @staticmethod
    def _decay_to_com(decay: float) -> float:
        """Convert decay factor to center of mass."""
        return (1.0 / (1 - decay)) - 1


# Convenience functions
def calculate_ewma(
    data: Union[np.ndarray, pd.Series],
    decay_factor: float = 0.94
) -> np.ndarray:
    """
    Quick EWMA calculation.
    
    Args:
        data: Input data
        decay_factor: Decay factor (default: 0.94 for RiskMetrics)
    
    Returns:
        EWMA series
    """
    model = EWMAModel(decay_factor=decay_factor)
    model.fit(data, use_returns=True, calculate_volatility=False)
    return model.ewma_mean


def calculate_ewma_volatility(
    returns: Union[np.ndarray, pd.Series],
    decay_factor: float = 0.94,
    annualize: bool = True,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    Calculate EWMA volatility (RiskMetrics approach).
    
    Args:
        returns: Return series
        decay_factor: Decay factor (default: 0.94 for daily)
        annualize: Whether to annualize
        periods_per_year: Trading periods per year
    
    Returns:
        EWMA volatility series
    """
    model = EWMAModel(decay_factor=decay_factor)
    model.fit(returns, use_returns=True, calculate_volatility=True)
    
    volatility = model.ewma_volatility
    if annualize:
        volatility = volatility * np.sqrt(periods_per_year)
    
    return volatility


# Export
__all__ = [
    "EWMAModel",
    "EWMAResult",
    "calculate_ewma",
    "calculate_ewma_volatility"
]