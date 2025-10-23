"""
Reusable Mixins for Common Financial Model Functionality
=========================================================

Mixins provide reusable, composable functionality that can be mixed into
any financial model class, implementing DRY principles by eliminating
code duplication across models.

Each mixin provides a specific category of functionality:
- MonteCarloMixin: Monte Carlo simulation logic
- NumericalMethodsMixin: Numerical solvers (Newton-Raphson, bisection, etc.)
- PerformanceMixin: Performance tracking and benchmarking
- ValidationMixin: Common input validation logic
- LoggingMixin: Enhanced logging capabilities
"""

import numpy as np
from scipy import optimize
from scipy.stats import norm
from typing import Callable, Optional, Tuple, Any, Dict, List
import time
from contextlib import contextmanager

from axiom.core.logging.axiom_logger import get_logger


class MonteCarloMixin:
    """
    Reusable Monte Carlo simulation logic.
    
    Provides variance reduction techniques and standard simulation patterns
    used across multiple models (options, VaR, credit risk, etc.).
    """
    
    def run_monte_carlo_simulation(
        self,
        n_paths: int,
        n_steps: int,
        spot_price: float,
        volatility: float,
        risk_free_rate: float,
        time_to_expiry: float,
        dividend_yield: float = 0.0,
        variance_reduction: Optional[str] = "antithetic",
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Run geometric Brownian motion simulation.
        
        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            spot_price: Initial price
            volatility: Volatility (annualized)
            risk_free_rate: Risk-free rate
            time_to_expiry: Time horizon
            dividend_yield: Dividend yield
            variance_reduction: Variance reduction technique
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_steps + 1) with price paths
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = time_to_expiry / n_steps
        drift = (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) * dt
        diffusion = volatility * np.sqrt(dt)
        
        # Generate random numbers
        if variance_reduction == "antithetic":
            # Antithetic variates: use pairs (Z, -Z)
            n_half = n_paths // 2
            Z_half = np.random.standard_normal((n_half, n_steps))
            Z = np.vstack([Z_half, -Z_half])[:n_paths]
        elif variance_reduction == "importance":
            # Importance sampling (shift mean)
            Z = np.random.standard_normal((n_paths, n_steps))
            Z += 0.5  # Shift towards positive outcomes
        elif variance_reduction == "stratified":
            # Stratified sampling
            Z = np.zeros((n_paths, n_steps))
            for i in range(n_steps):
                u = (np.arange(n_paths) + np.random.rand(n_paths)) / n_paths
                Z[:, i] = norm.ppf(u)
        else:
            # Standard Monte Carlo
            Z = np.random.standard_normal((n_paths, n_steps))
        
        # Generate price paths
        log_returns = drift + diffusion * Z
        log_price_paths = np.zeros((n_paths, n_steps + 1))
        log_price_paths[:, 0] = np.log(spot_price)
        log_price_paths[:, 1:] = log_price_paths[:, 0:1] + np.cumsum(log_returns, axis=1)
        
        price_paths = np.exp(log_price_paths)
        
        return price_paths
    
    def calculate_confidence_interval(
        self,
        values: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for Monte Carlo results.
        
        Args:
            values: Array of simulated values
            confidence_level: Confidence level (0 to 1)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        mean = np.mean(values)
        std_error = np.std(values) / np.sqrt(len(values))
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = mean - z_score * std_error
        upper_bound = mean + z_score * std_error
        
        return mean, lower_bound, upper_bound


class NumericalMethodsMixin:
    """
    Reusable numerical solution methods.
    
    Provides standard numerical algorithms used across models:
    - Newton-Raphson iteration
    - Bisection method
    - Brent's method
    - Gradient descent
    """
    
    def newton_raphson(
        self,
        func: Callable[[float], float],
        derivative: Callable[[float], float],
        initial_guess: float,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> Tuple[float, bool, int]:
        """
        Newton-Raphson root finding.
        
        Args:
            func: Function to find root of
            derivative: Derivative of function
            initial_guess: Starting point
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (solution, converged, iterations)
        """
        x = initial_guess
        
        for i in range(max_iterations):
            f_x = func(x)
            
            # Check convergence
            if abs(f_x) < tolerance:
                return x, True, i + 1
            
            # Newton-Raphson update
            f_prime_x = derivative(x)
            if abs(f_prime_x) < 1e-10:
                # Derivative too small, method fails
                return x, False, i + 1
            
            x = x - f_x / f_prime_x
        
        # Max iterations reached
        return x, False, max_iterations
    
    def bisection(
        self,
        func: Callable[[float], float],
        lower_bound: float,
        upper_bound: float,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> Tuple[float, bool, int]:
        """
        Bisection method for root finding.
        
        Args:
            func: Function to find root of
            lower_bound: Lower bound of search interval
            upper_bound: Upper bound of search interval
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (solution, converged, iterations)
        """
        a, b = lower_bound, upper_bound
        f_a, f_b = func(a), func(b)
        
        # Check if root exists in interval
        if f_a * f_b > 0:
            return (a + b) / 2, False, 0
        
        for i in range(max_iterations):
            c = (a + b) / 2
            f_c = func(c)
            
            # Check convergence
            if abs(f_c) < tolerance or (b - a) / 2 < tolerance:
                return c, True, i + 1
            
            # Update interval
            if f_a * f_c < 0:
                b, f_b = c, f_c
            else:
                a, f_a = c, f_c
        
        return (a + b) / 2, False, max_iterations
    
    def brent(
        self,
        func: Callable[[float], float],
        lower_bound: float,
        upper_bound: float,
        tolerance: float = 1e-6
    ) -> Tuple[float, bool]:
        """
        Brent's method for root finding (combines bisection, secant, and inverse quadratic).
        
        Args:
            func: Function to find root of
            lower_bound: Lower bound
            upper_bound: Upper bound
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (solution, converged)
        """
        try:
            result = optimize.brentq(
                func,
                lower_bound,
                upper_bound,
                xtol=tolerance,
                full_output=True
            )
            return result[0], result[1].converged
        except ValueError:
            # No root in interval
            return (lower_bound + upper_bound) / 2, False
    
    def solve_optimization_problem(
        self,
        objective: Callable,
        constraints: List[Dict],
        bounds: Tuple,
        method: str = 'SLSQP',
        initial_guess: Optional[np.ndarray] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, bool, str]:
        """
        Solve constrained optimization problem using scipy.optimize.
        
        Eliminates duplication across portfolio and risk models by providing
        a unified interface for numerical optimization.
        
        Args:
            objective: Objective function to minimize
            constraints: List of constraint dictionaries
            bounds: Bounds for each variable (tuple or list of tuples)
            method: Optimization method ('SLSQP', 'trust-constr', etc.)
            initial_guess: Initial guess for solution
            options: Additional solver options
            
        Returns:
            Tuple of (solution, success, message)
            
        Raises:
            ValueError: If optimization fails critically
        """
        if options is None:
            options = {'maxiter': 1000, 'ftol': 1e-9}
        
        try:
            result = optimize.minimize(
                objective,
                x0=initial_guess,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options
            )
            
            return result.x, result.success, result.message
            
        except Exception as e:
            # Return initial guess on failure
            if initial_guess is not None:
                return initial_guess, False, f"Optimization error: {str(e)}"
            else:
                raise ValueError(f"Optimization failed: {str(e)}")


class PerformanceMixin:
    """
    Performance tracking and benchmarking utilities.
    
    Provides standardized performance measurement across all models.
    """
    
    @contextmanager
    def track_time(self, operation_name: str):
        """
        Context manager for tracking execution time.
        
        Usage:
            with self.track_time("calculation"):
                result = expensive_operation()
        
        Args:
            operation_name: Name of the operation being tracked
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            if hasattr(self, 'logger') and hasattr(self, 'enable_logging'):
                if self.enable_logging:
                    self.logger.debug(
                        f"{operation_name} completed",
                        execution_time_ms=round(execution_time_ms, 3)
                    )
    
    def benchmark_against_target(
        self,
        func: Callable,
        target_ms: float,
        *args,
        **kwargs
    ) -> Tuple[Any, float, bool]:
        """
        Benchmark function execution against target time.
        
        Args:
            func: Function to benchmark
            target_ms: Target execution time in milliseconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time_ms, met_target)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        met_target = execution_time_ms <= target_ms
        
        return result, execution_time_ms, met_target


class ValidationMixin:
    """
    Common input validation logic.
    
    Provides reusable validation methods for common parameter types
    used across financial models.
    """
    
    def validate_positive(self, value: float, name: str):
        """Validate that value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    def validate_non_negative(self, value: float, name: str):
        """Validate that value is non-negative."""
        if value < 0:
            raise ValueError(f"{name} cannot be negative, got {value}")
    
    def validate_probability(self, value: float, name: str):
        """Validate that value is a valid probability (0 to 1)."""
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
    def validate_confidence_level(self, value: float):
        """Validate confidence level."""
        if not 0 < value < 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {value}")
    
    def validate_weights(self, weights: np.ndarray, tolerance: float = 1e-6):
        """Validate portfolio weights sum to 1."""
        if not np.isclose(np.sum(weights), 1.0, atol=tolerance):
            raise ValueError(f"Weights must sum to 1.0, got {np.sum(weights):.6f}")
        if np.any(weights < -tolerance):
            raise ValueError("Weights cannot be negative")
    
    def validate_array_shape(
        self,
        array: np.ndarray,
        expected_shape: Tuple[int, ...],
        name: str
    ):
        """Validate array shape."""
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} shape must be {expected_shape}, got {array.shape}"
            )
    
    def validate_finite(self, value: float, name: str):
        """Validate that value is finite (not NaN or inf)."""
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")


class LoggingMixin:
    """
    Enhanced logging capabilities.
    
    Provides structured logging methods for financial calculations.
    """
    
    def log_calculation_start(self, operation: str, **params):
        """Log calculation start with parameters."""
        if hasattr(self, 'logger') and hasattr(self, 'enable_logging'):
            if self.enable_logging:
                self.logger.info(
                    f"{operation} started",
                    **{k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
                )
    
    def log_calculation_end(self, operation: str, result: Any, execution_time_ms: float):
        """Log calculation completion."""
        if hasattr(self, 'logger') and hasattr(self, 'enable_logging'):
            if self.enable_logging:
                self.logger.info(
                    f"{operation} completed",
                    execution_time_ms=round(execution_time_ms, 3)
                )
    
    def log_warning(self, message: str, **context):
        """Log warning with context."""
        if hasattr(self, 'logger'):
            self.logger.warning(message, **context)
    
    def log_validation_error(self, parameter: str, value: Any, constraint: str):
        """Log validation error."""
        if hasattr(self, 'logger'):
            self.logger.error(
                f"Validation failed for {parameter}",
                value=value,
                constraint=constraint
            )


__all__ = [
    "MonteCarloMixin",
    "NumericalMethodsMixin",
    "PerformanceMixin",
    "ValidationMixin",
    "LoggingMixin",
]