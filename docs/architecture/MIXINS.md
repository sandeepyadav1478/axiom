# Mixins Architecture

**Reusable Functionality Through Composition**

## Overview

Mixins provide composable, reusable functionality that can be mixed into any financial model class. Instead of duplicating code across models, common functionality is implemented once in mixins and inherited where needed.

**Location**: [`axiom/models/base/mixins.py`](../../axiom/models/base/mixins.py)

## Available Mixins

| Mixin | Purpose | Use Cases |
|-------|---------|-----------|
| [`MonteCarloMixin`](../../axiom/models/base/mixins.py:27) | Monte Carlo simulation | Options pricing, VaR, credit risk |
| [`NumericalMethodsMixin`](../../axiom/models/base/mixins.py:126) | Numerical solvers | Implied volatility, optimization |
| [`PerformanceMixin`](../../axiom/models/base/mixins.py:255) | Performance tracking | All models needing benchmarks |
| [`ValidationMixin`](../../axiom/models/base/mixins.py:314) | Input validation | All models requiring validation |
| [`LoggingMixin`](../../axiom/models/base/mixins.py:367) | Enhanced logging | All models needing structured logs |

## MonteCarloMixin

**Location**: [`axiom/models/base/mixins.py:27`](../../axiom/models/base/mixins.py:27)

Provides Monte Carlo simulation logic shared across multiple models.

### Features

1. **Geometric Brownian Motion Simulation**
2. **Variance Reduction Techniques**
3. **Confidence Interval Calculation**

### Usage

```python
from axiom.models.base.mixins import MonteCarloMixin
from axiom.models.base.base_model import BasePricingModel

class MonteCarloOptionPricer(BasePricingModel, MonteCarloMixin):
    """Option pricer using Monte Carlo with variance reduction."""
    
    def price(self, S, K, T, r, sigma, n_paths=10000):
        # Use inherited Monte Carlo simulation
        price_paths = self.run_monte_carlo_simulation(
            n_paths=n_paths,
            n_steps=int(T * 252),
            spot_price=S,
            volatility=sigma,
            risk_free_rate=r,
            time_to_expiry=T,
            variance_reduction="antithetic"  # Built-in variance reduction
        )
        
        # Calculate payoff
        terminal_prices = price_paths[:, -1]
        payoffs = np.maximum(terminal_prices - K, 0)
        
        # Use inherited confidence interval calculation
        mean_price, lower, upper = self.calculate_confidence_interval(
            payoffs * np.exp(-r * T),
            confidence_level=0.95
        )
        
        return mean_price
```

### Variance Reduction Techniques

```python
# Antithetic variates (default, recommended)
paths = self.run_monte_carlo_simulation(
    n_paths=10000,
    variance_reduction="antithetic"
)
# Uses pairs (Z, -Z) to reduce variance

# Importance sampling
paths = self.run_monte_carlo_simulation(
    n_paths=10000,
    variance_reduction="importance"
)
# Shifts distribution towards important regions

# Stratified sampling
paths = self.run_monte_carlo_simulation(
    n_paths=10000,
    variance_reduction="stratified"
)
# Divides sample space into strata

# Standard Monte Carlo (no variance reduction)
paths = self.run_monte_carlo_simulation(
    n_paths=10000,
    variance_reduction=None
)
```

### Confidence Intervals

```python
# Calculate confidence interval for any Monte Carlo result
simulated_values = np.array([...])  # Your MC results

mean, lower, upper = self.calculate_confidence_interval(
    simulated_values,
    confidence_level=0.95
)

print(f"Mean: ${mean:,.2f}")
print(f"95% CI: [${lower:,.2f}, ${upper:,.2f}]")
```

## NumericalMethodsMixin

**Location**: [`axiom/models/base/mixins.py:126`](../../axiom/models/base/mixins.py:126)

Provides standard numerical solution methods shared across models.

### Features

1. **Newton-Raphson Iteration**
2. **Bisection Method**
3. **Brent's Method**

### Usage

```python
from axiom.models.base.mixins import NumericalMethodsMixin
from axiom.models.base.base_model import BasePricingModel

class ImpliedVolatilityCalculator(BasePricingModel, NumericalMethodsMixin):
    """Calculate implied volatility using numerical methods."""
    
    def calculate_implied_vol(self, market_price, S, K, T, r):
        # Define objective function
        def objective(sigma):
            bs_price = self._black_scholes_price(S, K, T, r, sigma)
            return bs_price - market_price
        
        # Define derivative (vega)
        def vega(sigma):
            return self._black_scholes_vega(S, K, T, r, sigma)
        
        # Use inherited Newton-Raphson
        implied_vol, converged, iterations = self.newton_raphson(
            func=objective,
            derivative=vega,
            initial_guess=0.20,  # 20% initial guess
            tolerance=1e-6,
            max_iterations=100
        )
        
        if not converged:
            # Fall back to bisection if Newton-Raphson fails
            implied_vol, converged, iterations = self.bisection(
                func=objective,
                lower_bound=0.01,
                upper_bound=2.0,
                tolerance=1e-6
            )
        
        return implied_vol if converged else None
```

### Newton-Raphson Method

```python
# Fast convergence when derivative is available
solution, converged, iterations = self.newton_raphson(
    func=lambda x: x**2 - 4,        # f(x) = x² - 4
    derivative=lambda x: 2*x,        # f'(x) = 2x
    initial_guess=1.0,
    tolerance=1e-6,
    max_iterations=100
)
# Solution: 2.0, Converged: True, Iterations: ~5
```

### Bisection Method

```python
# Guaranteed convergence but slower
solution, converged, iterations = self.bisection(
    func=lambda x: x**2 - 4,
    lower_bound=0.0,
    upper_bound=5.0,
    tolerance=1e-6,
    max_iterations=100
)
# Solution: 2.0, Converged: True, Iterations: ~20
```

### Brent's Method

```python
# Combines best of bisection, secant, and inverse quadratic
solution, converged = self.brent(
    func=lambda x: x**2 - 4,
    lower_bound=0.0,
    upper_bound=5.0,
    tolerance=1e-6
)
# Fastest convergence with guaranteed bracket
```

## PerformanceMixin

**Location**: [`axiom/models/base/mixins.py:255`](../../axiom/models/base/mixins.py:255)

Provides performance tracking and benchmarking utilities.

### Features

1. **Execution Time Tracking**
2. **Benchmark Against Targets**
3. **Context Manager for Timing**

### Usage

```python
from axiom.models.base.mixins import PerformanceMixin
from axiom.models.base.base_model import BaseRiskModel

class PerformanceTrackedVaR(BaseRiskModel, PerformanceMixin):
    """VaR model with performance tracking."""
    
    def calculate_risk(self, **kwargs):
        # Track specific operations
        with self.track_time("data_validation"):
            self.validate_inputs(**kwargs)
        
        with self.track_time("var_calculation"):
            var_value = self._calculate_var(**kwargs)
        
        with self.track_time("cvar_calculation"):
            cvar_value = self._calculate_cvar(**kwargs)
        
        # Benchmark against target
        result, time_ms, met_target = self.benchmark_against_target(
            func=self._full_calculation,
            target_ms=10.0,  # Target: <10ms
            **kwargs
        )
        
        if not met_target:
            self.logger.warning(
                f"Performance target missed: {time_ms:.2f}ms > 10.0ms"
            )
        
        return result
```

### Context Manager

```python
class MyModel(BaseRiskModel, PerformanceMixin):
    def calculate(self):
        # Automatic timing with context manager
        with self.track_time("monte_carlo_simulation"):
            paths = self._run_simulation(n_paths=10000)
        
        with self.track_time("payoff_calculation"):
            payoffs = self._calculate_payoffs(paths)
        
        # Logs automatically:
        # "monte_carlo_simulation completed, execution_time_ms=45.3"
        # "payoff_calculation completed, execution_time_ms=2.1"
```

### Benchmarking

```python
# Benchmark function against target time
result, execution_ms, met_target = self.benchmark_against_target(
    func=expensive_calculation,
    target_ms=100.0,
    arg1=value1,
    arg2=value2
)

if met_target:
    print(f"✓ Performance target met: {execution_ms:.2f}ms < 100ms")
else:
    print(f"✗ Performance target missed: {execution_ms:.2f}ms > 100ms")
```

## ValidationMixin

**Location**: [`axiom/models/base/mixins.py:314`](../../axiom/models/base/mixins.py:314)

Provides common input validation logic used across all models.

### Features

1. **Positivity Checks**
2. **Probability Validation**
3. **Weight Validation**
4. **Array Shape Validation**
5. **Finite Value Checks**

### Usage

```python
from axiom.models.base.mixins import ValidationMixin
from axiom.models.base.base_model import BaseRiskModel

class ValidatedVaRModel(BaseRiskModel, ValidationMixin):
    """VaR model with comprehensive validation."""
    
    def validate_inputs(
        self,
        portfolio_value,
        returns,
        confidence_level,
        **kwargs
    ):
        # Use inherited validation methods
        self.validate_positive(portfolio_value, "portfolio_value")
        self.validate_probability(confidence_level, "confidence_level")
        self.validate_finite(portfolio_value, "portfolio_value")
        
        # Validate returns array
        if len(returns) < 100:
            raise ValueError("Minimum 100 observations required")
        
        return True
```

### Validation Methods

```python
class MyModel(BaseFinancialModel, ValidationMixin):
    def validate_inputs(self, **kwargs):
        # Validate positive values
        self.validate_positive(kwargs["spot_price"], "spot_price")
        self.validate_positive(kwargs["volatility"], "volatility")
        
        # Validate non-negative values
        self.validate_non_negative(kwargs["time_to_expiry"], "time_to_expiry")
        
        # Validate probabilities (0 to 1)
        self.validate_probability(kwargs["default_prob"], "default_prob")
        
        # Validate confidence level (0 to 1, exclusive)
        self.validate_confidence_level(kwargs["confidence"])
        
        # Validate portfolio weights
        self.validate_weights(kwargs["weights"])
        
        # Validate array shapes
        self.validate_array_shape(
            kwargs["returns"],
            expected_shape=(252,),
            name="returns"
        )
        
        # Validate finite values
        self.validate_finite(kwargs["price"], "price")
        
        return True
```

## LoggingMixin

**Location**: [`axiom/models/base/mixins.py:367`](../../axiom/models/base/mixins.py:367)

Provides enhanced, structured logging capabilities.

### Features

1. **Calculation Start/End Logging**
2. **Warning Logging**
3. **Validation Error Logging**
4. **Structured Context**

### Usage

```python
from axiom.models.base.mixins import LoggingMixin
from axiom.models.base.base_model import BaseRiskModel

class WellLoggedModel(BaseRiskModel, LoggingMixin):
    """Model with comprehensive logging."""
    
    def calculate_risk(self, portfolio_value, returns, confidence_level):
        # Log calculation start with parameters
        self.log_calculation_start(
            "VaR calculation",
            portfolio_value=portfolio_value,
            confidence_level=confidence_level,
            n_observations=len(returns)
        )
        
        # Perform calculation
        start_time = time.perf_counter()
        var_value = self._calculate_var(returns, confidence_level)
        execution_time = (time.perf_counter() - start_time) * 1000
        
        # Check for warnings
        if len(returns) < 252:
            self.log_warning(
                "Low sample size",
                n_observations=len(returns),
                recommended=252
            )
        
        # Log calculation completion
        self.log_calculation_end(
            "VaR calculation",
            result=var_value,
            execution_time_ms=execution_time
        )
        
        return var_value
    
    def validate_inputs(self, **kwargs):
        try:
            # Your validation
            pass
        except ValueError as e:
            self.log_validation_error(
                parameter="confidence_level",
                value=kwargs.get("confidence_level"),
                constraint="must be between 0 and 1"
            )
            raise
```

## Combining Multiple Mixins

Models can inherit from multiple mixins:

```python
from axiom.models.base.base_model import BasePricingModel
from axiom.models.base.mixins import (
    MonteCarloMixin,
    NumericalMethodsMixin,
    PerformanceMixin,
    ValidationMixin,
    LoggingMixin
)

class AdvancedOptionPricer(
    BasePricingModel,
    MonteCarloMixin,
    NumericalMethodsMixin,
    PerformanceMixin,
    ValidationMixin,
    LoggingMixin
):
    """Fully-featured option pricer with all mixins."""
    
    def price(self, S, K, T, r, sigma, method="monte_carlo"):
        # Logging from LoggingMixin
        self.log_calculation_start("option_pricing", method=method)
        
        # Validation from ValidationMixin
        self.validate_positive(S, "spot_price")
        self.validate_positive(sigma, "volatility")
        
        # Performance tracking from PerformanceMixin
        with self.track_time(f"{method}_pricing"):
            if method == "monte_carlo":
                # Monte Carlo from MonteCarloMixin
                price = self._mc_price(S, K, T, r, sigma)
            elif method == "implied_vol":
                # Numerical methods from NumericalMethodsMixin
                price = self._newton_price(S, K, T, r, sigma)
        
        return price
```

## Benefits of Mixins

### 1. Code Reuse

**Without Mixins** (code duplication):
```python
class VaRModel:
    def newton_raphson(self, func, derivative, initial_guess):
        # 50 lines of Newton-Raphson code
        pass

class ImpliedVolModel:
    def newton_raphson(self, func, derivative, initial_guess):
        # 50 lines of SAME Newton-Raphson code
        pass
```

**With Mixins** (DRY):
```python
class VaRModel(BaseRiskModel, NumericalMethodsMixin):
    # Newton-Raphson inherited automatically
    pass

class ImpliedVolModel(BasePricingModel, NumericalMethodsMixin):
    # Newton-Raphson inherited automatically
    pass
```

### 2. Composability

Mix and match functionality as needed:
```python
# Simple model - just base functionality
class SimpleVaR(BaseRiskModel):
    pass

# Model with validation
class ValidatedVaR(BaseRiskModel, ValidationMixin):
    pass

# Model with validation + performance tracking
class OptimizedVaR(BaseRiskModel, ValidationMixin, PerformanceMixin):
    pass

# Full-featured model - all mixins
class EnterpriseVaR(
    BaseRiskModel,
    MonteCarloMixin,
    ValidationMixin,
    PerformanceMixin,
    LoggingMixin
):
    pass
```

### 3. Maintainability

Fix bugs or add features in one place:
```python
# Update MonteCarloMixin with new variance reduction technique
# → Automatically available in:
#   - MonteCarloVaR
#   - MonteCarloOptionPricer
#   - CreditVaRModel
#   - All other models using MonteCarloMixin
```

## Creating Custom Mixins

```python
# Custom mixin for your specific needs
class RiskMetricsMixin:
    """Mixin providing risk metrics calculations."""
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.03):
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns)
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

# Use custom mixin
class PerformanceAnalyzer(BasePortfolioModel, RiskMetricsMixin):
    def analyze_performance(self, returns):
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd = self.calculate_max_drawdown(returns)
        
        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd
        }
```

## Best Practices

### 1. Keep Mixins Focused

Each mixin should have a single, clear purpose:
```python
# ✓ Good - focused mixin
class MonteCarloMixin:
    """Monte Carlo simulation functionality."""
    pass

# ✗ Bad - too many responsibilities
class HelperMixin:
    """Monte Carlo, validation, logging, and everything else."""
    pass
```

### 2. Don't Depend on Specific Base Classes

Mixins should work with any base class:
```python
# ✓ Good - works with any model
class ValidationMixin:
    def validate_positive(self, value, name):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

# ✗ Bad - assumes specific base class
class ValidationMixin:
    def validate_positive(self, value, name):
        self.specific_base_method()  # Assumes BaseRiskModel
```

### 3. Use Method Naming Conventions

Prefix mixin methods to avoid conflicts:
```python
# ✓ Good - clear naming
class PerformanceMixin:
    def track_time(self, operation):
        pass
    
    def benchmark_against_target(self, func, target_ms):
        pass

# ✗ Bad - generic names may conflict
class PerformanceMixin:
    def track(self, operation):  # Too generic
        pass
```

## See Also

- [`BASE_CLASSES.md`](BASE_CLASSES.md) - Base class hierarchy
- [`FACTORY_PATTERN.md`](FACTORY_PATTERN.md) - Model creation patterns
- [`CONFIGURATION_SYSTEM.md`](CONFIGURATION_SYSTEM.md) - Configuration management

---

**Last Updated**: 2025-10-23  
**Version**: 2.0.0