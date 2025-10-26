"""
ARCH/GARCH Volatility Models Wrapper

This module provides a wrapper around the arch library for production-grade volatility modeling.
The arch library offers comprehensive ARCH/GARCH implementations with multiple variants and
distribution options, used extensively in financial risk management.

Features:
- GARCH, EGARCH, TGARCH, and more variants
- Multiple distributions (Normal, Student-t, Skewed-t, GED)
- Volatility forecasting
- Model diagnostics and tests
- Parameter estimation with multiple methods

Models Supported:
- GARCH: Generalized Autoregressive Conditional Heteroskedasticity
- EGARCH: Exponential GARCH (asymmetric effects)
- TGARCH/GJR-GARCH: Threshold GARCH (leverage effects)
- APARCH: Asymmetric Power ARCH
- FIGARCH: Fractionally Integrated GARCH
- ARCH: Basic ARCH model
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import get_config, LibraryAvailability

logger = logging.getLogger(__name__)

# Check if arch is available
ARCH_AVAILABLE = LibraryAvailability.check_library('arch')

if ARCH_AVAILABLE:
    from arch import arch_model
    from arch.univariate import (
        GARCH,
        EGARCH,
        FIGARCH,
        APARCH,
        HARCH,
        Normal,
        StudentsT,
        SkewStudent,
        GeneralizedError,
    )


class VolatilityModel(Enum):
    """Volatility model types."""
    GARCH = "GARCH"
    EGARCH = "EGARCH"
    FIGARCH = "FIGARCH"
    APARCH = "APARCH"
    HARCH = "HARCH"
    ARCH = "ARCH"


class Distribution(Enum):
    """Error distribution types."""
    NORMAL = "normal"
    STUDENTS_T = "t"
    SKEWED_T = "skewt"
    GED = "ged"  # Generalized Error Distribution


@dataclass
class GARCHResult:
    """Result from GARCH model estimation."""
    model_name: str
    parameters: Dict[str, float]
    conditional_volatility: np.ndarray
    standardized_residuals: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    fitted_values: np.ndarray
    forecast: Optional[Dict] = None


@dataclass
class VolatilityForecast:
    """Volatility forecast result."""
    mean: np.ndarray
    variance: np.ndarray
    residual_variance: np.ndarray
    horizon: int


class ArchGARCH:
    """Wrapper for arch library GARCH models.
    
    This class provides a production-ready interface to GARCH volatility models
    with support for multiple model variants and distributions.
    
    Example:
        >>> garch = ArchGARCH()
        >>> returns = pd.Series([...])  # Return series
        >>> result = garch.fit_garch(
        ...     returns,
        ...     p=1, q=1,
        ...     model_type=VolatilityModel.GARCH,
        ...     dist=Distribution.STUDENTS_T
        ... )
        >>> print(f"AIC: {result.aic:.2f}")
        >>> # Forecast volatility
        >>> forecast = garch.forecast_volatility(result, horizon=10)
    """
    
    def __init__(self):
        """Initialize the arch GARCH wrapper."""
        if not ARCH_AVAILABLE:
            raise ImportError(
                "arch library is not available. Install it with: pip install arch"
            )
        
        self.config = get_config()
        self._fitted_model = None
        logger.info("arch GARCH wrapper initialized")
    
    def fit_garch(
        self,
        returns: Union[pd.Series, np.ndarray],
        p: int = 1,
        q: int = 1,
        o: int = 0,
        power: float = 2.0,
        model_type: VolatilityModel = VolatilityModel.GARCH,
        dist: Distribution = Distribution.NORMAL,
        mean: str = 'Constant',
        rescale: bool = True
    ) -> GARCHResult:
        """Fit a GARCH model to return series.
        
        Args:
            returns: Return series (percentage or log returns)
            p: Order of symmetric innovation (GARCH terms)
            q: Order of lagged volatility (ARCH terms)
            o: Order of asymmetric innovation (for EGARCH, TGARCH)
            power: Power for APARCH model (1=AVGARCH, 2=GARCH)
            model_type: Type of volatility model
            dist: Error distribution
            mean: Mean model ('Constant', 'Zero', 'ARX')
            rescale: Rescale returns to percentage if needed
            
        Returns:
            GARCHResult with fitted parameters and diagnostics
        """
        try:
            # Convert to pandas Series if needed
            if isinstance(returns, np.ndarray):
                returns = pd.Series(returns)
            
            # Rescale if needed (arch works better with percentage returns)
            if rescale and returns.abs().mean() < 1:
                returns = returns * 100
                logger.debug("Rescaled returns to percentage scale")
            
            # Create model
            if model_type == VolatilityModel.GARCH:
                model = arch_model(
                    returns,
                    mean=mean,
                    vol='GARCH',
                    p=p,
                    q=q,
                    dist=dist.value,
                    rescale=False  # We handle rescaling
                )
            elif model_type == VolatilityModel.EGARCH:
                model = arch_model(
                    returns,
                    mean=mean,
                    vol='EGARCH',
                    p=p,
                    o=o,
                    q=q,
                    dist=dist.value,
                    rescale=False
                )
            elif model_type == VolatilityModel.FIGARCH:
                model = arch_model(
                    returns,
                    mean=mean,
                    vol='FIGARCH',
                    p=p,
                    q=q,
                    dist=dist.value,
                    rescale=False
                )
            elif model_type == VolatilityModel.APARCH:
                model = arch_model(
                    returns,
                    mean=mean,
                    vol='GARCH',
                    p=p,
                    o=o,
                    q=q,
                    power=power,
                    dist=dist.value,
                    rescale=False
                )
            elif model_type == VolatilityModel.HARCH:
                model = arch_model(
                    returns,
                    mean=mean,
                    vol='HARCH',
                    lags=[1, 5, 22],  # Common lags for HARCH
                    dist=dist.value,
                    rescale=False
                )
            elif model_type == VolatilityModel.ARCH:
                model = arch_model(
                    returns,
                    mean=mean,
                    vol='GARCH',
                    p=0,
                    q=q,
                    dist=dist.value,
                    rescale=False
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Fit model
            fitted = model.fit(disp='off', show_warning=False)
            self._fitted_model = fitted
            
            # Extract results
            result = GARCHResult(
                model_name=f"{model_type.value}({p},{q})",
                parameters=fitted.params.to_dict(),
                conditional_volatility=fitted.conditional_volatility.values,
                standardized_residuals=fitted.std_resid.values,
                log_likelihood=fitted.loglikelihood,
                aic=fitted.aic,
                bic=fitted.bic,
                fitted_values=fitted.fittedvalues.values
            )
            
            if self.config.log_library_usage:
                logger.info(
                    f"Fitted {model_type.value}({p},{q}) model: "
                    f"AIC={result.aic:.2f}, BIC={result.bic:.2f}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            raise
    
    def forecast_volatility(
        self,
        result: Optional[GARCHResult] = None,
        horizon: int = 1,
        start: Optional[int] = None,
        reindex: bool = True
    ) -> VolatilityForecast:
        """Forecast conditional volatility.
        
        Args:
            result: GARCHResult from fit_garch (uses last fitted if None)
            horizon: Forecast horizon (number of steps ahead)
            start: Starting index for forecast (None = end of sample)
            reindex: Reindex forecast to original index
            
        Returns:
            VolatilityForecast with mean and variance forecasts
        """
        if self._fitted_model is None:
            raise ValueError("No fitted model available. Call fit_garch first.")
        
        try:
            # Generate forecast
            forecast = self._fitted_model.forecast(
                horizon=horizon,
                start=start,
                reindex=reindex
            )
            
            vol_forecast = VolatilityForecast(
                mean=forecast.mean.values[-1] if hasattr(forecast.mean, 'values') else forecast.mean,
                variance=forecast.variance.values[-1] if hasattr(forecast.variance, 'values') else forecast.variance,
                residual_variance=forecast.residual_variance.values[-1] if hasattr(forecast.residual_variance, 'values') else forecast.residual_variance,
                horizon=horizon
            )
            
            if self.config.log_library_usage:
                logger.debug(f"Generated {horizon}-step volatility forecast")
            
            return vol_forecast
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            raise
    
    def rolling_forecast(
        self,
        returns: Union[pd.Series, np.ndarray],
        window: int,
        p: int = 1,
        q: int = 1,
        model_type: VolatilityModel = VolatilityModel.GARCH,
        dist: Distribution = Distribution.NORMAL,
        horizon: int = 1,
        refit: bool = False
    ) -> pd.DataFrame:
        """Generate rolling volatility forecasts.
        
        Args:
            returns: Return series
            window: Rolling window size
            p: GARCH p parameter
            q: GARCH q parameter
            model_type: Type of volatility model
            dist: Error distribution
            horizon: Forecast horizon
            refit: If True, refit model at each step; if False, use expanding window
            
        Returns:
            DataFrame with rolling forecasts
        """
        try:
            if isinstance(returns, np.ndarray):
                returns = pd.Series(returns)
            
            forecasts = []
            dates = []
            
            for i in range(window, len(returns)):
                if refit:
                    # Use fixed window
                    train_data = returns.iloc[i-window:i]
                else:
                    # Use expanding window
                    train_data = returns.iloc[:i]
                
                # Fit model
                result = self.fit_garch(
                    train_data,
                    p=p,
                    q=q,
                    model_type=model_type,
                    dist=dist
                )
                
                # Forecast
                vol_forecast = self.forecast_volatility(result, horizon=horizon)
                
                forecasts.append(vol_forecast.variance[0])
                dates.append(returns.index[i] if hasattr(returns, 'index') else i)
            
            forecast_df = pd.DataFrame({
                'forecast': forecasts,
                'realized': returns.iloc[window:].values ** 2  # Squared returns as proxy for realized variance
            }, index=dates)
            
            if self.config.log_library_usage:
                logger.info(f"Generated {len(forecasts)} rolling forecasts")
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error in rolling forecast: {e}")
            raise
    
    def compare_models(
        self,
        returns: Union[pd.Series, np.ndarray],
        models: List[Tuple[VolatilityModel, int, int]],
        dist: Distribution = Distribution.NORMAL
    ) -> pd.DataFrame:
        """Compare multiple GARCH model specifications.
        
        Args:
            returns: Return series
            models: List of (model_type, p, q) tuples to compare
            dist: Error distribution
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for model_type, p, q in models:
            try:
                result = self.fit_garch(
                    returns,
                    p=p,
                    q=q,
                    model_type=model_type,
                    dist=dist
                )
                
                results.append({
                    'model': f"{model_type.value}({p},{q})",
                    'log_likelihood': result.log_likelihood,
                    'aic': result.aic,
                    'bic': result.bic,
                    'num_params': len(result.parameters)
                })
            except Exception as e:
                logger.warning(f"Error fitting {model_type.value}({p},{q}): {e}")
        
        comparison_df = pd.DataFrame(results)
        
        if not comparison_df.empty:
            # Sort by AIC (lower is better)
            comparison_df = comparison_df.sort_values('aic')
            
            if self.config.log_library_usage:
                best_model = comparison_df.iloc[0]['model']
                logger.info(f"Best model by AIC: {best_model}")
        
        return comparison_df
    
    def calculate_var(
        self,
        result: GARCHResult,
        confidence_level: float = 0.95,
        portfolio_value: float = 1.0
    ) -> np.ndarray:
        """Calculate Value at Risk from GARCH volatility.
        
        Args:
            result: GARCHResult from fit_garch
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            portfolio_value: Portfolio value for VaR calculation
            
        Returns:
            Array of VaR values over time
        """
        # Get z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        var = -z_score * result.conditional_volatility * portfolio_value / 100
        
        if self.config.log_library_usage:
            logger.debug(f"Calculated {confidence_level:.0%} VaR")
        
        return var
    
    def diagnostic_tests(self) -> Dict[str, float]:
        """Run diagnostic tests on fitted model.
        
        Returns:
            Dictionary with test statistics
        """
        if self._fitted_model is None:
            raise ValueError("No fitted model available")
        
        diagnostics = {}
        
        try:
            # Ljung-Box test on standardized residuals
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            resid = self._fitted_model.std_resid.dropna()
            lb_test = acorr_ljungbox(resid, lags=[10], return_df=False)
            diagnostics['ljung_box_stat'] = lb_test[0][0]
            diagnostics['ljung_box_pvalue'] = lb_test[1][0]
            
            # Ljung-Box test on squared standardized residuals
            lb_test_sq = acorr_ljungbox(resid**2, lags=[10], return_df=False)
            diagnostics['ljung_box_sq_stat'] = lb_test_sq[0][0]
            diagnostics['ljung_box_sq_pvalue'] = lb_test_sq[1][0]
            
            if self.config.log_library_usage:
                logger.debug("Ran diagnostic tests on GARCH model")
            
        except Exception as e:
            logger.warning(f"Error running diagnostic tests: {e}")
        
        return diagnostics


def check_arch_availability() -> bool:
    """Check if arch library is available.
    
    Returns:
        True if arch is available, False otherwise
    """
    return ARCH_AVAILABLE


def estimate_simple_garch(
    returns: Union[pd.Series, np.ndarray],
    p: int = 1,
    q: int = 1
) -> Optional[GARCHResult]:
    """Convenience function to estimate a simple GARCH(p,q) model.
    
    Args:
        returns: Return series
        p: GARCH p parameter
        q: GARCH q parameter
        
    Returns:
        GARCHResult or None if estimation fails
    """
    if not ARCH_AVAILABLE:
        logger.error("arch library not available")
        return None
    
    try:
        garch = ArchGARCH()
        return garch.fit_garch(returns, p=p, q=q)
    except Exception as e:
        logger.error(f"Error estimating GARCH model: {e}")
        return None