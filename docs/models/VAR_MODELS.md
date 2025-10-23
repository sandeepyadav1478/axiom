# Value at Risk (VaR) Models - Complete Reference

**Production-ready VaR calculation with <10ms execution time**

## Overview

Three industry-standard VaR methodologies for institutional risk management:

| Method | Description | Performance | Best For |
|--------|-------------|-------------|----------|
| **Parametric** | Normal distribution assumption | <1ms | Large, liquid portfolios |
| **Historical Simulation** | Empirical distribution | <5ms | Non-normal distributions |
| **Monte Carlo** | Simulated scenarios | <10ms (10K paths) | Complex derivatives |

## Mathematical Framework

### Parametric VaR

**Formula:**
```
VaR_α = Portfolio_Value × σ × N⁻¹(α) × √Δt

where:
σ = Portfolio volatility (annualized)
N⁻¹(α) = Inverse normal CDF at confidence level α
Δt = Time horizon (in years)
```

**95% Confidence:** N⁻¹(0.95) = 1.645
**99% Confidence:** N⁻¹(0.99) = 2.326

### Historical VaR

**Method:**
1. Sort historical returns in ascending order
2. Select (1-α)% percentile
3. Scale to portfolio value

```python
VaR = Portfolio_Value × percentile(returns, 1-α)
```

### Monte Carlo VaR

**Simulation:**
```
r_t = μΔt + σ√Δt × Z
where Z ~ N(0,1)

VaR = Portfolio_Value × percentile(simulated_returns, 1-α)
```

## Quick Start

### Modern Factory Pattern API (Recommended)

```python
from axiom.models.base.factory import ModelFactory, ModelType
from axiom.config.model_config import VaRConfig

# Method 1: Use default configuration
var_model = ModelFactory.create(ModelType.HISTORICAL_VAR)
result = var_model.calculate_risk(
    portfolio_value=1_000_000,
    returns=portfolio_returns,
    confidence_level=0.95,
    time_horizon=1
)

print(f"VaR (95%, 1-day): ${result.value['var']:,.2f}")
print(f"CVaR/ES: ${result.value['cvar']:,.2f}")
print(f"Execution time: {result.metadata.execution_time_ms:.2f}ms")

# Method 2: Use configuration profiles
from axiom.config.model_config import ModelConfig

# Basel III compliant VaR
basel_config = ModelConfig.for_basel_iii_compliance()
basel_var = ModelFactory.create(ModelType.PARAMETRIC_VAR, config=basel_config.var)

# High performance VaR (speed-optimized)
fast_config = ModelConfig.for_high_performance()
fast_var = ModelFactory.create(ModelType.MONTE_CARLO_VAR, config=fast_config.var)

# High precision VaR (accuracy-optimized)
precise_config = ModelConfig.for_high_precision()
precise_var = ModelFactory.create(ModelType.MONTE_CARLO_VAR, config=precise_config.var)

# Method 3: Custom configuration
custom_config = VaRConfig(
    default_confidence_level=0.99,
    default_method="monte_carlo",
    default_simulations=50000,
    parallel_mc=True,
    cache_results=True
)
custom_var = ModelFactory.create(ModelType.MONTE_CARLO_VAR, config=custom_config)
```

### Legacy API (Still Supported)

```python
from axiom.models.risk import VaRCalculator, VaRMethod

# Initialize calculator
calculator = VaRCalculator()

# Historical VaR
var_result = calculator.calculate_var(
    portfolio_value=1_000_000,
    returns=portfolio_returns,  # NumPy array or pandas Series
    method=VaRMethod.HISTORICAL,
    confidence_level=0.95,
    time_horizon=1  # days
)

print(f"VaR (95%, 1-day): ${var_result.var_amount:,.2f}")
print(f"CVaR/ES: ${var_result.expected_shortfall:,.2f}")
print(f"Execution time: {var_result.execution_time_ms:.2f}ms")
```

## Advanced Features

### Conditional VaR (CVaR / Expected Shortfall)

```python
# Automatically calculated with all methods
cvar = var_result.expected_shortfall

# Coherent risk measure (satisfies subadditivity)
print(f"Expected loss given breach: ${cvar:,.2f}")
```

### Multi-Period VaR

```python
# 10-day VaR using square-root-of-time rule
var_10day = calculator.calculate_var(
    portfolio_value=1_000_000,
    returns=returns,
    method=VaRMethod.PARAMETRIC,
    confidence_level=0.95,
    time_horizon=10
)

# Note: VaR_T = VaR_1 × √T (for parametric only)
```

### Component VaR

```python
# Calculate VaR contribution by asset
component_var = calculator.calculate_component_var(
    portfolio_value=1_000_000,
    asset_returns=asset_returns_matrix,  # Each column = asset
    weights=portfolio_weights,
    method=VaRMethod.HISTORICAL
)

for asset, contribution in component_var.items():
    print(f"{asset}: ${contribution:,.2f}")
```

## Configuration

```python
VAR_CONFIG = {
    "default_confidence_level": 0.95,
    "default_time_horizon": 1,  # days
    "default_method": "historical",
    
    # Parametric
    "assume_normal": True,
    "use_ewma_volatility": False,
    "ewma_lambda": 0.94,
    
    # Historical Simulation
    "min_observations": 252,  # 1 year
    "bootstrap_enabled": False,
    
    # Monte Carlo
    "default_simulations": 10000,
    "variance_reduction": "antithetic",
    "random_seed": None,
    
    # Performance
    "cache_results": True,
    "parallel_mc": True
}
```

## Performance Benchmarks

| Method | Portfolio Size | Execution Time | Status |
|--------|---------------|----------------|--------|
| Parametric | Any | <1ms | ✅ |
| Historical | 1000 observations | <5ms | ✅ |
| Monte Carlo | 10K simulations | <10ms | ✅ |

## Basel Committee Requirements

✅ 99% confidence level (10-day VaR)
✅ Minimum 1 year historical data
✅ Backtesting capability
✅ Stress testing integration

## References

- Basel Committee (1996). "Amendment to the Capital Accord"
- J.P. Morgan (1994). "RiskMetrics Technical Document"
- Artzner et al. (1999). "Coherent Measures of Risk"

---

**Last Updated**: 2025-10-23  
**Version**: 1.0.0