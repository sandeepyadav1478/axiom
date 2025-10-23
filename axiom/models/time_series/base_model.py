"""
Base Time Series Model for Algorithmic Trading

Provides common interface and utilities for all time series models.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class ModelType(Enum):
    """Time series model types."""
    ARIMA = "arima"
    GARCH = "garch"
    EWMA = "ewma"


class ForecastHorizon(Enum):
    """Standard forecast horizons."""
    SHORT_TERM = 5  # 5 periods ahead
    MEDIUM_TERM = 20  # 20 periods ahead
    LONG_TERM = 60  # 60 periods ahead


@dataclass
class TimeSeriesData:
    """Container for time series data."""
    
    values: Union[np.ndarray, pd.Series]
    dates: Optional[pd.DatetimeIndex] = None
    symbol: Optional[str] = None
    frequency: str = "D"  # Daily by default
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process data."""
        if isinstance(self.values, pd.Series):
            self.dates = self.values.index
            self.values = self.values.values
        
        if self.dates is None:
            # Generate default date range
            self.dates = pd.date_range(
                end=datetime.now(),
                periods=len(self.values),
                freq=self.frequency
            )
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.values, index=self.dates)
    
    def get_returns(self, log_returns: bool = False) -> np.ndarray:
        """Calculate returns from price series."""
        if log_returns:
            return np.log(self.values[1:] / self.values[:-1])
        else:
            return np.diff(self.values) / self.values[:-1]


@dataclass
class ForecastResult:
    """Base forecast result."""
    
    forecast: np.ndarray  # Point forecasts
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lower, upper)
    confidence_level: float = 0.95
    horizon: int = 1
    model_type: ModelType = ModelType.ARIMA
    fitted_values: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    model_params: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "forecast": self.forecast.tolist() if isinstance(self.forecast, np.ndarray) else self.forecast,
            "horizon": self.horizon,
            "model_type": self.model_type.value,
            "confidence_level": self.confidence_level,
            "model_params": self.model_params,
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }
        
        if self.confidence_intervals:
            result["confidence_intervals"] = {
                "lower": self.confidence_intervals[0].tolist(),
                "upper": self.confidence_intervals[1].tolist()
            }
        
        if self.fitted_values is not None:
            result["fitted_values"] = self.fitted_values.tolist()
        
        return result
    
    def get_forecast_dataframe(self, dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """Convert forecast to DataFrame."""
        df = pd.DataFrame({
            'forecast': self.forecast
        }, index=dates if dates is not None else range(len(self.forecast)))
        
        if self.confidence_intervals:
            df['lower_ci'] = self.confidence_intervals[0]
            df['upper_ci'] = self.confidence_intervals[1]
        
        return df


@dataclass
class ModelDiagnostics:
    """Model diagnostic statistics."""
    
    aic: Optional[float] = None  # Akaike Information Criterion
    bic: Optional[float] = None  # Bayesian Information Criterion
    hqic: Optional[float] = None  # Hannan-Quinn Information Criterion
    mse: Optional[float] = None  # Mean Squared Error
    rmse: Optional[float] = None  # Root Mean Squared Error
    mae: Optional[float] = None  # Mean Absolute Error
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    log_likelihood: Optional[float] = None
    residual_stats: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "aic": self.aic,
            "bic": self.bic,
            "hqic": self.hqic,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "log_likelihood": self.log_likelihood,
            "residual_stats": self.residual_stats
        }


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series models.
    
    All time series models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, model_type: ModelType):
        """
        Initialize base time series model.
        
        Args:
            model_type: Type of time series model
        """
        self.model_type = model_type
        self.is_fitted = False
        self.training_data: Optional[TimeSeriesData] = None
        self.model_params: Dict = {}
        self.diagnostics: Optional[ModelDiagnostics] = None
    
    @abstractmethod
    def fit(self, data: Union[np.ndarray, pd.Series, TimeSeriesData], **kwargs) -> 'BaseTimeSeriesModel':
        """
        Fit the model to training data.
        
        Args:
            data: Time series data
            **kwargs: Additional model-specific parameters
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def forecast(self, horizon: int = 1, **kwargs) -> ForecastResult:
        """
        Generate forecasts.
        
        Args:
            horizon: Number of periods to forecast
            **kwargs: Additional forecast parameters
        
        Returns:
            ForecastResult with predictions and confidence intervals
        """
        pass
    
    def _prepare_data(self, data: Union[np.ndarray, pd.Series, TimeSeriesData]) -> TimeSeriesData:
        """
        Prepare and validate input data.
        
        Args:
            data: Input data in various formats
        
        Returns:
            TimeSeriesData object
        """
        if isinstance(data, TimeSeriesData):
            return data
        elif isinstance(data, pd.Series):
            return TimeSeriesData(values=data)
        elif isinstance(data, np.ndarray):
            return TimeSeriesData(values=data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def calculate_diagnostics(self, actual: np.ndarray, fitted: np.ndarray) -> ModelDiagnostics:
        """
        Calculate model diagnostic statistics.
        
        Args:
            actual: Actual values
            fitted: Fitted values
        
        Returns:
            ModelDiagnostics object
        """
        # Align arrays if different lengths (can happen with differencing)
        min_len = min(len(actual), len(fitted))
        actual = actual[-min_len:]
        fitted = fitted[-min_len:]
        
        residuals = actual - fitted
        
        # Basic error metrics
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs(residuals / np.where(actual != 0, actual, 1))) * 100
        
        # Residual statistics
        residual_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "skewness": float(self._calculate_skewness(residuals)),
            "kurtosis": float(self._calculate_kurtosis(residuals))
        }
        
        return ModelDiagnostics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            mape=mape,
            residual_stats=residual_stats
        )
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
    
    def _check_fitted(self):
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forecasting. Call fit() first.")
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model_type.value,
            "is_fitted": self.is_fitted,
            "model_params": self.model_params,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "training_data_size": len(self.training_data.values) if self.training_data else None
        }


# Export all components
__all__ = [
    "ModelType",
    "ForecastHorizon",
    "TimeSeriesData",
    "ForecastResult",
    "ModelDiagnostics",
    "BaseTimeSeriesModel"
]