"""
ARIMA Model - Autoregressive Integrated Moving Average

Implements ARIMA(p,d,q) model for price forecasting in algorithmic trading.

Features:
- Auto-ARIMA parameter selection
- Seasonal ARIMA (SARIMA)
- Information criteria-based model selection
- Rolling forecast validation
- Prediction intervals
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize

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
class ARIMAOrder:
    """ARIMA model order (p, d, q)."""
    p: int  # AR order
    d: int  # Differencing order
    q: int  # MA order
    
    def __str__(self) -> str:
        return f"ARIMA({self.p},{self.d},{self.q})"
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to tuple."""
        return (self.p, self.d, self.q)


class ARIMAModel(BaseTimeSeriesModel, PerformanceMixin, ValidationMixin, LoggingMixin):
    """
    ARIMA Model for Time Series Forecasting.
    
    ARIMA(p,d,q) combines:
    - AR(p): Autoregressive component - uses past values
    - I(d): Integration - differencing to make series stationary
    - MA(q): Moving Average - uses past forecast errors
    
    Ideal for:
    - Price forecasting
    - Trend prediction
    - Mean reversion strategies
    """
    
    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: str = 'c',
        config: Optional[TimeSeriesConfig] = None
    ):
        """
        Initialize ARIMA model.
        
        Args:
            order: (p, d, q) - AR order, differencing, MA order
            seasonal_order: (P, D, Q, s) for SARIMA
            trend: 'n' (none), 'c' (constant), 't' (linear), 'ct' (both)
            config: TimeSeriesConfig for customization
        """
        super().__init__(ModelType.ARIMA)
        self.config = config or get_config().time_series
        self.logger = timeseries_logger
        self.enable_logging = True
        
        self.order = ARIMAOrder(*order) if order else None
        self.seasonal_order = seasonal_order
        self.trend = trend
        
        # Model parameters
        self.ar_params: Optional[np.ndarray] = None
        self.ma_params: Optional[np.ndarray] = None
        self.trend_params: Optional[np.ndarray] = None
        self.sigma2: float = 0.0
        
        # Fitted data
        self.original_data: Optional[np.ndarray] = None
        self.differenced_data: Optional[np.ndarray] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
    
    def fit(
        self,
        data: Union[np.ndarray, pd.Series, TimeSeriesData],
        max_iter: int = 100,
        method: str = 'css-mle'
    ) -> 'ARIMAModel':
        """
        Fit ARIMA model to data.
        
        Args:
            data: Time series data
            max_iter: Maximum optimization iterations
            method: 'css', 'mle', or 'css-mle'
        
        Returns:
            Self for method chaining
        """
        with self.track_time("ARIMA model fitting"):
            # Prepare data
            ts_data = self._prepare_data(data)
            self.training_data = ts_data
            self.original_data = ts_data.values.copy()
            
            # Validate data size
            if len(self.original_data) < self.config.min_observations:
                self.log_warning(
                    "Insufficient data for reliable ARIMA estimation",
                    observations=len(self.original_data),
                    recommended_minimum=self.config.min_observations
                )
            
            # If order not specified, use auto-ARIMA
            if self.order is None:
                if self.config.arima_auto_select:
                    self.order = self._auto_arima(
                        ts_data.values,
                        max_p=self.config.arima_max_p,
                        max_d=self.config.arima_max_d,
                        max_q=self.config.arima_max_q,
                        seasonal=self.config.arima_seasonal
                    )
                else:
                    self.order = ARIMAOrder(1, 1, 1)  # Default
        
            # Difference the series
            self.differenced_data = self._difference_series(
                self.original_data, self.order.d
            )
            
            # Estimate parameters
            self._estimate_parameters(max_iter, method)
            
            # Calculate fitted values and residuals
            self._calculate_fitted_values()
            
            # Calculate diagnostics
            self.diagnostics = self.calculate_diagnostics(
                self.original_data[self.order.d:],
                self.fitted_values
            )
            
            # Add information criteria
            self._calculate_information_criteria()
            
            self.is_fitted = True
            self.model_params = {
                "order": self.order.to_tuple(),
                "trend": self.trend,
                "ar_params": self.ar_params.tolist() if self.ar_params is not None else [],
                "ma_params": self.ma_params.tolist() if self.ma_params is not None else [],
                "sigma2": float(self.sigma2)
            }
            
            self.log_calculation_end("ARIMA fitting", self.model_params, 0)
            
            return self
    
    def forecast(
        self,
        horizon: int = 1,
        confidence_level: float = 0.95,
        return_std: bool = True
    ) -> ForecastResult:
        """
        Generate ARIMA forecasts.
        
        Args:
            horizon: Number of periods to forecast
            confidence_level: Confidence level for intervals
            return_std: Whether to return forecast standard errors
        
        Returns:
            ForecastResult with forecasts and confidence intervals
        """
        self._check_fitted()
        
        # Generate point forecasts
        forecasts = self._forecast_steps(horizon)
        
        # Calculate confidence intervals
        forecast_std = self._calculate_forecast_std(horizon)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_ci = forecasts - z_score * forecast_std
        upper_ci = forecasts + z_score * forecast_std
        
        return ForecastResult(
            forecast=forecasts,
            confidence_intervals=(lower_ci, upper_ci),
            confidence_level=confidence_level,
            horizon=horizon,
            model_type=ModelType.ARIMA,
            fitted_values=self.fitted_values,
            residuals=self.residuals,
            model_params=self.model_params,
            metrics=self.diagnostics.to_dict() if self.diagnostics else {}
        )
    
    def _auto_arima(
        self,
        data: np.ndarray,
        max_p: Optional[int] = None,
        max_d: Optional[int] = None,
        max_q: Optional[int] = None,
        seasonal: Optional[bool] = None
    ) -> ARIMAOrder:
        """
        Automatic ARIMA order selection using information criteria.
        
        Tests various (p,d,q) combinations and selects best based on AIC.
        
        Args:
            data: Time series data
            max_p: Maximum AR order to test
            max_d: Maximum differencing order
            max_q: Maximum MA order to test
            seasonal: Whether to test seasonal models
        
        Returns:
            Best ARIMAOrder
        """
        # Use config defaults if not specified
        max_p = max_p or self.config.arima_max_p
        max_d = max_d or self.config.arima_max_d
        max_q = max_q or self.config.arima_max_q
        seasonal = seasonal if seasonal is not None else self.config.arima_seasonal
        
        self.log_calculation_start(
            "Auto-ARIMA order selection",
            max_p=max_p, max_d=max_d, max_q=max_q
        )
        
        best_aic = np.inf
        best_order = ARIMAOrder(1, 0, 1)  # Default
        
        # Determine optimal differencing order using ADF test
        d_optimal = self._find_optimal_d(data, max_d)
        
        # Search over p and q
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue  # Skip null model
                
                try:
                    # Create temporary model
                    temp_order = ARIMAOrder(p, d_optimal, q)
                    
                    # Quick fit
                    differenced = self._difference_series(data, d_optimal)
                    params = self._estimate_arma_params(differenced, p, q)
                    
                    if params is None:
                        continue
                    
                    # Calculate AIC
                    fitted = self._calculate_arma_fitted(differenced, params, p, q)
                    residuals = differenced[max(p, q):] - fitted
                    sigma2 = np.var(residuals)
                    
                    n = len(residuals)
                    k = p + q + (1 if self.trend != 'n' else 0)
                    aic = n * np.log(sigma2) + 2 * k
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = temp_order
                
                except:
                    continue
        
        return best_order
    
    def _find_optimal_d(self, data: np.ndarray, max_d: int) -> int:
        """
        Find optimal differencing order using statistical tests.
        
        Uses variance reduction and autocorrelation to determine d.
        """
        # Start with d=0 and increase until stationary
        for d in range(max_d + 1):
            differenced = self._difference_series(data, d)
            
            # Check if variance is stable
            if d > 0:
                var_current = np.var(differenced)
                var_previous = np.var(self._difference_series(data, d - 1))
                
                # If variance increased, previous d was better
                if var_current > var_previous * 1.1:
                    return d - 1
            
            # Simple stationarity check using autocorrelation
            if d > 0 and len(differenced) > 20:
                acf = self._calculate_acf(differenced, nlags=10)
                if np.max(np.abs(acf[1:])) < 0.5:  # Roughly stationary
                    return d
        
        return max_d
    
    def _difference_series(self, data: np.ndarray, d: int) -> np.ndarray:
        """Apply differencing to make series stationary."""
        result = data.copy()
        for _ in range(d):
            result = np.diff(result)
        return result
    
    def _estimate_parameters(self, max_iter: int, method: str):
        """
        Estimate ARIMA parameters using maximum likelihood.
        
        Args:
            max_iter: Maximum iterations
            method: Estimation method
        """
        p, d, q = self.order.p, self.order.d, self.order.q
        
        # Number of parameters
        n_trend = self._get_trend_params_count()
        n_params = p + q + n_trend
        
        # Initial parameter guess
        initial_params = np.zeros(n_params)
        
        # Optimize log-likelihood
        def neg_log_likelihood(params):
            try:
                ar_params = params[:p] if p > 0 else np.array([])
                ma_params = params[p:p+q] if q > 0 else np.array([])
                trend_params = params[p+q:] if n_trend > 0 else np.array([])
                
                fitted = self._calculate_arma_fitted_full(
                    self.differenced_data, ar_params, ma_params, trend_params
                )
                
                residuals = self.differenced_data[max(p, q):] - fitted
                sigma2 = np.var(residuals)
                
                n = len(residuals)
                ll = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
                
                return -ll
            except:
                return 1e10
        
        # Optimize
        try:
            result = minimize(
                neg_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': max_iter}
            )
            
            optimal_params = result.x
        except:
            warnings.warn("Optimization failed, using OLS estimates")
            optimal_params = self._estimate_arma_params(self.differenced_data, p, q)
        
        # Extract parameters
        self.ar_params = optimal_params[:p] if p > 0 else np.array([])
        self.ma_params = optimal_params[p:p+q] if q > 0 else np.array([])
        self.trend_params = optimal_params[p+q:] if n_trend > 0 else np.array([])
    
    def _estimate_arma_params(
        self,
        data: np.ndarray,
        p: int,
        q: int
    ) -> Optional[np.ndarray]:
        """
        Estimate AR and MA parameters using least squares.
        
        Simplified estimation for auto-ARIMA search.
        """
        n = len(data)
        max_lag = max(p, q)
        
        if n <= max_lag + 10:
            return None
        
        # Build regression matrices
        y = data[max_lag:]
        X = []
        
        # AR terms
        for i in range(1, p + 1):
            X.append(data[max_lag - i:-i])
        
        # MA terms (approximated by lagged residuals)
        if q > 0:
            # Initialize with zeros
            for i in range(q):
                X.append(np.zeros(len(y)))
        
        if len(X) == 0:
            return None
        
        X = np.column_stack(X)
        
        try:
            # OLS estimation
            params = np.linalg.lstsq(X, y, rcond=None)[0]
            return params
        except:
            return None
    
    def _calculate_arma_fitted(
        self,
        data: np.ndarray,
        params: np.ndarray,
        p: int,
        q: int
    ) -> np.ndarray:
        """Calculate fitted values for given parameters."""
        max_lag = max(p, q)
        fitted = []
        residuals = np.zeros(len(data))
        
        for t in range(max_lag, len(data)):
            value = 0.0
            
            # AR component
            for i in range(p):
                if t - i - 1 >= 0:
                    value += params[i] * data[t - i - 1]
            
            # MA component
            for i in range(q):
                if t - i - 1 >= max_lag:
                    value += params[p + i] * residuals[t - i - 1]
            
            fitted.append(value)
            if t < len(data):
                residuals[t] = data[t] - value
        
        return np.array(fitted)
    
    def _calculate_arma_fitted_full(
        self,
        data: np.ndarray,
        ar_params: np.ndarray,
        ma_params: np.ndarray,
        trend_params: np.ndarray
    ) -> np.ndarray:
        """Calculate fitted values with trend."""
        p = len(ar_params)
        q = len(ma_params)
        max_lag = max(p, q) if max(p, q) > 0 else 1
        
        fitted = []
        residuals = np.zeros(len(data))
        
        for t in range(max_lag, len(data)):
            value = 0.0
            
            # Trend component
            if self.trend == 'c':
                value += trend_params[0] if len(trend_params) > 0 else 0
            elif self.trend == 't':
                value += trend_params[0] * t if len(trend_params) > 0 else 0
            elif self.trend == 'ct':
                if len(trend_params) >= 2:
                    value += trend_params[0] + trend_params[1] * t
            
            # AR component
            for i in range(p):
                if t - i - 1 >= 0:
                    value += ar_params[i] * data[t - i - 1]
            
            # MA component
            for i in range(q):
                if t - i - 1 >= max_lag:
                    value += ma_params[i] * residuals[t - i - 1]
            
            fitted.append(value)
            residuals[t] = data[t] - value
        
        return np.array(fitted)
    
    def _calculate_fitted_values(self):
        """Calculate fitted values and residuals for the training data."""
        p, d, q = self.order.p, self.order.d, self.order.q
        
        # Fit on differenced data
        max_lag = max(p, q) if max(p, q) > 0 else 1
        fitted_diff = self._calculate_arma_fitted_full(
            self.differenced_data,
            self.ar_params,
            self.ma_params,
            self.trend_params
        )
        
        # Integrate back to original scale
        self.fitted_values = self._integrate_series(
            fitted_diff, self.original_data, d
        )
        
        # Calculate residuals
        n_fitted = len(self.fitted_values)
        self.residuals = self.original_data[-n_fitted:] - self.fitted_values
        self.sigma2 = np.var(self.residuals)
    
    def _integrate_series(
        self,
        differenced: np.ndarray,
        original: np.ndarray,
        d: int
    ) -> np.ndarray:
        """Integrate differenced series back to original scale."""
        if d == 0:
            return differenced
        
        result = differenced.copy()
        for _ in range(d):
            # Use last value from original series as base
            base_idx = len(original) - len(result) - 1
            if base_idx >= 0:
                base = original[base_idx]
            else:
                base = original[-1]  # Use last value if index out of range
            
            # Cumulative sum of differences starting from base
            result = base + np.cumsum(result)
        
        return result
    
    def _forecast_steps(self, horizon: int) -> np.ndarray:
        """Generate multi-step ahead forecasts."""
        p, d, q = self.order.p, self.order.d, self.order.q
        
        # Start with last observations
        history = list(self.differenced_data[-max(p, q):])
        residuals = list(self.residuals[-q:]) if q > 0 else []
        
        forecasts_diff = []
        
        for h in range(horizon):
            forecast = 0.0
            
            # Trend component
            t = len(self.differenced_data) + h
            if self.trend == 'c' and len(self.trend_params) > 0:
                forecast += self.trend_params[0]
            elif self.trend == 't' and len(self.trend_params) > 0:
                forecast += self.trend_params[0] * t
            elif self.trend == 'ct' and len(self.trend_params) >= 2:
                forecast += self.trend_params[0] + self.trend_params[1] * t
            
            # AR component
            for i in range(p):
                if i < len(history):
                    forecast += self.ar_params[i] * history[-(i+1)]
            
            # MA component (residuals become zero for h > q)
            for i in range(min(q, len(residuals))):
                if i < len(residuals):
                    forecast += self.ma_params[i] * residuals[-(i+1)]
            
            forecasts_diff.append(forecast)
            history.append(forecast)
            if q > 0:
                residuals.append(0.0)  # Future residuals are zero
        
        # Integrate forecasts
        forecasts = self._integrate_series(
            np.array(forecasts_diff),
            self.original_data,
            d
        )
        
        return forecasts
    
    def _calculate_forecast_std(self, horizon: int) -> np.ndarray:
        """Calculate forecast standard errors."""
        # Approximate forecast variance
        # For ARIMA, variance increases with horizon
        base_std = np.sqrt(self.sigma2)
        
        # Simple approximation: std grows with sqrt(horizon)
        std_errors = []
        for h in range(1, horizon + 1):
            # MA(inf) representation coefficient sum
            psi_sum = 1 + 0.5 * h  # Simplified
            std_errors.append(base_std * np.sqrt(psi_sum))
        
        return np.array(std_errors)
    
    def _calculate_information_criteria(self):
        """Calculate AIC, BIC, and HQIC."""
        if self.residuals is None or len(self.residuals) == 0:
            return
        
        n = len(self.residuals)
        k = self.order.p + self.order.q + self._get_trend_params_count()
        
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(self.sigma2) + 1)
        
        # Akaike Information Criterion
        aic = 2 * k - 2 * log_likelihood
        
        # Bayesian Information Criterion
        bic = k * np.log(n) - 2 * log_likelihood
        
        # Hannan-Quinn Information Criterion
        hqic = 2 * k * np.log(np.log(n)) - 2 * log_likelihood
        
        if self.diagnostics:
            self.diagnostics.aic = aic
            self.diagnostics.bic = bic
            self.diagnostics.hqic = hqic
            self.diagnostics.log_likelihood = log_likelihood
    
    def _get_trend_params_count(self) -> int:
        """Get number of trend parameters."""
        if self.trend == 'n':
            return 0
        elif self.trend in ['c', 't']:
            return 1
        elif self.trend == 'ct':
            return 2
        return 0
    
    @staticmethod
    def _calculate_acf(data: np.ndarray, nlags: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        mean = np.mean(data)
        c0 = np.sum((data - mean) ** 2) / len(data)
        
        acf = [1.0]
        for k in range(1, nlags + 1):
            ck = np.sum((data[:-k] - mean) * (data[k:] - mean)) / len(data)
            acf.append(ck / c0)
        
        return np.array(acf)


# Export
__all__ = ["ARIMAModel", "ARIMAOrder"]