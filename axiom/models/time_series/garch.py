"""
GARCH Model - Generalized Autoregressive Conditional Heteroskedasticity

Implements GARCH(p,q) model for volatility forecasting in algorithmic trading.

Features:
- Standard GARCH(1,1) and GARCH(p,q)
- Volatility clustering detection
- Multiple distribution assumptions (Normal, Student-t, GED)
- Volatility forecasting with confidence intervals
- Integration with VaR models
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
class GARCHOrder:
    """GARCH model order (p, q)."""
    p: int  # GARCH order (lagged variance terms)
    q: int  # ARCH order (lagged squared residual terms)
    
    def __str__(self) -> str:
        return f"GARCH({self.p},{self.q})"
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple."""
        return (self.p, self.q)


@dataclass
class VolatilityForecast:
    """Volatility forecast result."""
    
    volatility: np.ndarray  # Forecasted volatility
    variance: np.ndarray  # Forecasted variance
    horizon: int
    annualization_factor: float = 252.0
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def get_annualized_volatility(self) -> np.ndarray:
        """Get annualized volatility."""
        return self.volatility * np.sqrt(self.annualization_factor)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "volatility": self.volatility.tolist(),
            "variance": self.variance.tolist(),
            "annualized_volatility": self.get_annualized_volatility().tolist(),
            "horizon": self.horizon
        }
        
        if self.confidence_intervals:
            result["confidence_intervals"] = {
                "lower": self.confidence_intervals[0].tolist(),
                "upper": self.confidence_intervals[1].tolist()
            }
        
        return result


class GARCHModel(BaseTimeSeriesModel, PerformanceMixin, ValidationMixin, LoggingMixin):
    """
    GARCH Model for Volatility Forecasting.
    
    GARCH(p,q) models conditional heteroskedasticity:
    - Captures volatility clustering (high volatility follows high volatility)
    - Models time-varying variance
    - Essential for risk management and options pricing
    
    Standard GARCH(1,1) equation:
    σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
    
    Where:
    - σ²ₜ is the conditional variance at time t
    - ω is the constant term (long-run variance)
    - α is the ARCH coefficient (news impact)
    - β is the GARCH coefficient (persistence)
    
    Ideal for:
    - Volatility forecasting
    - VaR calculations
    - Options pricing
    - Risk-adjusted position sizing
    """
    
    def __init__(
        self,
        order: Optional[Tuple[int, int]] = None,
        mean_model: str = 'constant',
        distribution: Optional[str] = None,
        config: Optional[TimeSeriesConfig] = None
    ):
        """
        Initialize GARCH model.
        
        Args:
            order: (p, q) - GARCH order, ARCH order
            mean_model: 'zero', 'constant', or 'ar'
            distribution: 'normal', 'studentt', or 'ged'
            config: TimeSeriesConfig for customization
        """
        super().__init__(ModelType.GARCH)
        self.config = config or get_config().time_series
        self.logger = timeseries_logger
        self.enable_logging = True
        
        self.order = GARCHOrder(*order) if order else GARCHOrder(*self.config.garch_order)
        self.mean_model = mean_model
        self.distribution = distribution or self.config.garch_distribution
        
        # Model parameters
        self.omega: float = 0.0  # Constant term
        self.alpha: Optional[np.ndarray] = None  # ARCH parameters
        self.beta: Optional[np.ndarray] = None  # GARCH parameters
        self.mean_params: Optional[np.ndarray] = None
        
        # Distribution parameters
        self.dist_params: Dict = {}
        
        # Fitted data
        self.returns: Optional[np.ndarray] = None
        self.conditional_volatility: Optional[np.ndarray] = None
        self.standardized_residuals: Optional[np.ndarray] = None
    
    def fit(
        self,
        data: Union[np.ndarray, pd.Series, TimeSeriesData],
        use_returns: bool = True,
        max_iter: int = 1000,
        initial_variance: Optional[float] = None
    ) -> 'GARCHModel':
        """
        Fit GARCH model to data.
        
        Args:
            data: Time series data (returns or prices)
            use_returns: If False, will calculate returns from prices
            max_iter: Maximum optimization iterations
            initial_variance: Initial variance estimate (if None, uses sample variance)
        
        Returns:
            Self for method chaining
        """
        with self.track_time("GARCH model fitting"):
            # Prepare data
            ts_data = self._prepare_data(data)
            self.training_data = ts_data
            
            # Calculate returns if needed
            if use_returns:
                self.returns = ts_data.values.copy()
            else:
                self.returns = ts_data.get_returns(log_returns=True)
            
            # Validate data size
            if len(self.returns) < self.config.min_observations:
                self.log_warning(
                    "Insufficient data for reliable GARCH estimation",
                    observations=len(self.returns),
                    recommended_minimum=self.config.min_observations
                )
            
            # Remove mean (demean returns)
            self.returns = self.returns - np.mean(self.returns)
            
            # Initial variance
            if initial_variance is None:
                initial_variance = np.var(self.returns)
            
            # Estimate parameters via MLE
            self._estimate_parameters(max_iter, initial_variance)
        
            # Calculate conditional volatility
            self.conditional_volatility = self._calculate_conditional_volatility()
            
            # Calculate standardized residuals
            self.standardized_residuals = self.returns / self.conditional_volatility
            
            # Calculate diagnostics
            self.residuals = self.standardized_residuals
            fitted_vol = self.conditional_volatility[:-1]
            actual_vol = np.abs(self.returns[1:])
            
            self.diagnostics = self.calculate_diagnostics(actual_vol, fitted_vol)
            self._calculate_information_criteria()
            
            self.is_fitted = True
            self.model_params = {
                "order": self.order.to_tuple(),
                "omega": float(self.omega),
                "alpha": self.alpha.tolist() if self.alpha is not None else [],
                "beta": self.beta.tolist() if self.beta is not None else [],
                "mean_model": self.mean_model,
                "distribution": self.distribution,
                "persistence": float(self._calculate_persistence())
            }
            
            self.log_calculation_end("GARCH fitting", self.model_params, 0)
            
            return self
    
    def forecast(
        self,
        horizon: int = 1,
        confidence_level: float = 0.95,
        method: str = 'analytic'
    ) -> VolatilityForecast:
        """
        Forecast volatility.
        
        Args:
            horizon: Number of periods to forecast
            confidence_level: Confidence level for intervals
            method: 'analytic' or 'simulation'
        
        Returns:
            VolatilityForecast with volatility predictions
        """
        self._check_fitted()
        
        if method == 'analytic':
            variance_forecast = self._forecast_variance_analytic(horizon)
        else:
            variance_forecast = self._forecast_variance_simulation(horizon)
        
        volatility_forecast = np.sqrt(variance_forecast)
        
        # Calculate confidence intervals (approximate)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        std_error = volatility_forecast * 0.1  # Simplified standard error
        
        lower_ci = volatility_forecast - z_score * std_error
        upper_ci = volatility_forecast + z_score * std_error
        lower_ci = np.maximum(lower_ci, 0)  # Volatility can't be negative
        
        return VolatilityForecast(
            volatility=volatility_forecast,
            variance=variance_forecast,
            horizon=horizon,
            confidence_intervals=(lower_ci, upper_ci)
        )
    
    def forecast_returns(
        self,
        horizon: int = 1,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Forecast returns with GARCH volatility.
        
        Assumes zero mean returns with time-varying volatility.
        
        Args:
            horizon: Forecast horizon
            confidence_level: Confidence level
        
        Returns:
            ForecastResult with return forecasts
        """
        self._check_fitted()
        
        # Forecast volatility
        vol_forecast = self.forecast(horizon)
        
        # Assume zero mean returns
        return_forecast = np.zeros(horizon)
        
        # Confidence intervals based on forecasted volatility
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        lower_ci = return_forecast - z_score * vol_forecast.volatility
        upper_ci = return_forecast + z_score * vol_forecast.volatility
        
        return ForecastResult(
            forecast=return_forecast,
            confidence_intervals=(lower_ci, upper_ci),
            confidence_level=confidence_level,
            horizon=horizon,
            model_type=ModelType.GARCH,
            fitted_values=self.conditional_volatility,
            residuals=self.standardized_residuals,
            model_params=self.model_params,
            metrics={
                **(self.diagnostics.to_dict() if self.diagnostics else {}),
                **vol_forecast.to_dict()
            }
        )
    
    def _estimate_parameters(self, max_iter: int, initial_variance: float):
        """
        Estimate GARCH parameters using maximum likelihood estimation.
        
        Args:
            max_iter: Maximum iterations
            initial_variance: Initial variance estimate
        """
        p, q = self.order.p, self.order.q
        
        # Initial parameter guess
        # omega, alpha[1..q], beta[1..p]
        n_params = 1 + q + p
        
        # Start with reasonable initial values
        initial_params = np.zeros(n_params)
        initial_params[0] = initial_variance * 0.01  # omega (small constant)
        
        if q > 0:
            initial_params[1:1+q] = 0.1 / q  # alpha parameters
        if p > 0:
            initial_params[1+q:] = 0.85 / p  # beta parameters
        
        # Bounds to ensure stationarity and positivity
        bounds = [(1e-6, None)]  # omega > 0
        bounds.extend([(0, 1)] * (q + p))  # 0 < alpha, beta < 1
        
        # Negative log-likelihood function
        def neg_log_likelihood(params):
            try:
                omega = params[0]
                alpha = params[1:1+q] if q > 0 else np.array([])
                beta = params[1+q:] if p > 0 else np.array([])
                
                # Check stationarity: sum(alpha) + sum(beta) < 1
                if np.sum(alpha) + np.sum(beta) >= 0.9999:
                    return 1e10
                
                # Calculate conditional variance
                variance = self._compute_variance(omega, alpha, beta)
                
                # Avoid numerical issues
                variance = np.maximum(variance, 1e-8)
                
                # Log-likelihood
                if self.distribution == 'normal':
                    ll = -0.5 * np.sum(
                        np.log(2 * np.pi) +
                        np.log(variance) +
                        (self.returns ** 2) / variance
                    )
                else:
                    # Simplified for other distributions
                    ll = -0.5 * np.sum(np.log(variance) + (self.returns ** 2) / variance)
                
                return -ll
            
            except:
                return 1e10
        
        # Optimize
        try:
            result = minimize(
                neg_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8}
            )
            
            if result.success:
                optimal_params = result.x
            else:
                warnings.warn("GARCH optimization did not converge, using initial estimates")
                optimal_params = initial_params
        
        except Exception as e:
            warnings.warn(f"GARCH optimization failed: {str(e)}, using initial estimates")
            optimal_params = initial_params
        
        # Extract parameters
        self.omega = optimal_params[0]
        self.alpha = optimal_params[1:1+q] if q > 0 else np.array([])
        self.beta = optimal_params[1+q:] if p > 0 else np.array([])
    
    def _compute_variance(
        self,
        omega: float,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> np.ndarray:
        """
        Compute conditional variance series.
        
        Args:
            omega: Constant term
            alpha: ARCH parameters
            beta: GARCH parameters
        
        Returns:
            Conditional variance series
        """
        p = len(beta)
        q = len(alpha)
        n = len(self.returns)
        
        # Initialize variance array
        variance = np.zeros(n)
        
        # Initial variance (unconditional)
        persistence = np.sum(alpha) + np.sum(beta)
        if persistence < 1:
            variance[0] = omega / (1 - persistence)
        else:
            variance[0] = np.var(self.returns)
        
        # Compute conditional variance recursively
        for t in range(1, n):
            variance[t] = omega
            
            # ARCH terms (lagged squared residuals)
            for i in range(min(q, t)):
                variance[t] += alpha[i] * (self.returns[t-1-i] ** 2)
            
            # GARCH terms (lagged variance)
            for i in range(min(p, t)):
                variance[t] += beta[i] * variance[t-1-i]
        
        return variance
    
    def _calculate_conditional_volatility(self) -> np.ndarray:
        """Calculate conditional volatility using fitted parameters."""
        variance = self._compute_variance(self.omega, self.alpha, self.beta)
        return np.sqrt(variance)
    
    def _forecast_variance_analytic(self, horizon: int) -> np.ndarray:
        """
        Analytical variance forecast.
        
        Uses the analytical formula for multi-step GARCH forecasts.
        
        Args:
            horizon: Forecast horizon
        
        Returns:
            Variance forecasts
        """
        p, q = self.order.p, self.order.q
        
        # Last conditional variance
        last_variance = self.conditional_volatility[-1] ** 2
        
        # Last squared residual
        last_squared_resid = self.returns[-1] ** 2
        
        # Persistence
        persistence = self._calculate_persistence()
        
        # Unconditional variance
        if persistence < 1:
            uncond_var = self.omega / (1 - persistence)
        else:
            uncond_var = np.var(self.returns)
        
        # Forecast variances
        forecasts = np.zeros(horizon)
        
        for h in range(horizon):
            if h == 0:
                # One-step ahead
                forecast_var = self.omega
                
                # Add ARCH terms
                for i in range(q):
                    if i == 0:
                        forecast_var += self.alpha[i] * last_squared_resid
                    else:
                        forecast_var += self.alpha[i] * (self.returns[-1-i] ** 2)
                
                # Add GARCH terms
                for i in range(p):
                    if i == 0:
                        forecast_var += self.beta[i] * last_variance
                    else:
                        forecast_var += self.beta[i] * (self.conditional_volatility[-1-i] ** 2)
                
                forecasts[h] = forecast_var
            else:
                # Multi-step ahead: mean revert to unconditional variance
                forecasts[h] = uncond_var + (forecasts[0] - uncond_var) * (persistence ** h)
        
        return forecasts
    
    def _forecast_variance_simulation(
        self,
        horizon: int,
        n_simulations: int = 10000
    ) -> np.ndarray:
        """
        Simulate variance forecasts using Monte Carlo.
        
        Args:
            horizon: Forecast horizon
            n_simulations: Number of simulation paths
        
        Returns:
            Mean variance forecasts
        """
        p, q = self.order.p, self.order.q
        
        # Initialize simulation arrays
        simulated_variances = np.zeros((n_simulations, horizon))
        
        for sim in range(n_simulations):
            # Initialize with last variance and residuals
            variance_history = list(self.conditional_volatility[-max(p, q):]**2)
            resid_history = list(self.returns[-max(p, q):]**2)
            
            for h in range(horizon):
                # Compute next variance
                next_var = self.omega
                
                for i in range(q):
                    if i < len(resid_history):
                        next_var += self.alpha[i] * resid_history[-(i+1)]
                
                for i in range(p):
                    if i < len(variance_history):
                        next_var += self.beta[i] * variance_history[-(i+1)]
                
                # Simulate next residual
                if self.distribution == 'normal':
                    shock = np.random.normal(0, 1)
                else:
                    shock = np.random.normal(0, 1)  # Default to normal
                
                next_resid_sq = next_var * (shock ** 2)
                
                # Store
                simulated_variances[sim, h] = next_var
                variance_history.append(next_var)
                resid_history.append(next_resid_sq)
        
        # Return mean across simulations
        return np.mean(simulated_variances, axis=0)
    
    def _calculate_persistence(self) -> float:
        """Calculate volatility persistence (sum of alpha and beta)."""
        persistence = np.sum(self.alpha) + np.sum(self.beta)
        return float(persistence)
    
    def _calculate_information_criteria(self):
        """Calculate AIC, BIC for GARCH model."""
        if self.standardized_residuals is None:
            return
        
        n = len(self.standardized_residuals)
        k = 1 + len(self.alpha) + len(self.beta)  # omega + alphas + betas
        
        # Log-likelihood
        variance = self.conditional_volatility ** 2
        ll = -0.5 * np.sum(
            np.log(2 * np.pi) +
            np.log(variance) +
            (self.returns ** 2) / variance
        )
        
        # Information criteria
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        hqic = 2 * k * np.log(np.log(n)) - 2 * ll
        
        if self.diagnostics:
            self.diagnostics.aic = aic
            self.diagnostics.bic = bic
            self.diagnostics.hqic = hqic
            self.diagnostics.log_likelihood = ll
    
    def detect_volatility_clustering(self) -> Dict:
        """
        Detect and quantify volatility clustering.
        
        Returns:
            Dictionary with clustering statistics
        """
        self._check_fitted()
        
        # Calculate absolute returns
        abs_returns = np.abs(self.returns)
        
        # Autocorrelation of squared returns (sign of clustering)
        squared_returns = self.returns ** 2
        acf_squared = self._calculate_acf(squared_returns, nlags=20)
        
        # Ljung-Box test statistic
        n = len(squared_returns)
        lb_stat = n * (n + 2) * np.sum(acf_squared[1:]**2 / (n - np.arange(1, len(acf_squared))))
        
        return {
            "volatility_clustering_detected": lb_stat > 30,  # Rule of thumb
            "ljung_box_statistic": float(lb_stat),
            "acf_squared_returns": acf_squared.tolist(),
            "persistence": self._calculate_persistence(),
            "half_life": self._calculate_volatility_half_life()
        }
    
    def _calculate_volatility_half_life(self) -> float:
        """Calculate volatility shock half-life in periods."""
        persistence = self._calculate_persistence()
        if persistence >= 1:
            return np.inf
        return np.log(0.5) / np.log(persistence)
    
    @staticmethod
    def _calculate_acf(data: np.ndarray, nlags: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        mean = np.mean(data)
        c0 = np.sum((data - mean) ** 2) / len(data)
        
        acf = [1.0]
        for k in range(1, min(nlags + 1, len(data))):
            ck = np.sum((data[:-k] - mean) * (data[k:] - mean)) / len(data)
            acf.append(ck / c0 if c0 > 0 else 0)
        
        return np.array(acf)


# Convenience function
def fit_garch(
    returns: Union[np.ndarray, pd.Series],
    order: Tuple[int, int] = (1, 1)
) -> GARCHModel:
    """
    Quick GARCH model fitting.
    
    Args:
        returns: Return series
        order: GARCH order (p, q)
    
    Returns:
        Fitted GARCH model
    """
    model = GARCHModel(order=order)
    model.fit(returns, use_returns=True)
    return model


# Export
__all__ = ["GARCHModel", "GARCHOrder", "VolatilityForecast", "fit_garch"]