# Configuration Guide - Axiom Investment Banking Analytics

**Complete guide to configuring and customizing all Axiom financial models**

## Table of Contents

- [Overview](#overview)
- [Configuration System](#configuration-system)
- [Environment Variables](#environment-variables)
- [Configuration Profiles](#configuration-profiles)
- [Model-Specific Configuration](#model-specific-configuration)
- [Runtime Configuration](#runtime-configuration)
- [Best Practices](#best-practices)

## Overview

Axiom uses a centralized, type-safe configuration system that enables:

- **Zero code changes** for customization
- **Environment variable** overrides
- **Configuration profiles** for common scenarios (Basel III, High Performance, etc.)
- **Runtime updates** without restarts
- **Type safety** with dataclass validation
- **JSON import/export** for configuration management

## Configuration System

### Basic Usage

```python
from axiom.config.model_config import get_config, ModelConfig

# Get global configuration (singleton)
config = get_config()

# Access model-specific configs
options_config = config.options
credit_config = config.credit
var_config = config.var
portfolio_config = config.portfolio
ts_config = config.time_series

# Use in models
from axiom.models.options import BlackScholesModel

model = BlackScholesModel()
price = model.price(
    spot_price=100,
    strike_price=105,
    time_to_expiry=0.5,
    risk_free_rate=options_config.default_risk_free_rate,  # From config
    volatility=0.25
)
```

### Configuration Hierarchy

```
ModelConfig (Master)
├── OptionsConfig
│   ├── Black-Scholes parameters
│   ├── Binomial tree settings
│   ├── Monte Carlo configuration
│   ├── Greeks calculation
│   └── Implied volatility solver
├── CreditConfig
│   ├── PD estimation
│   ├── LGD modeling
│   ├── EAD calculation
│   ├── Credit VaR
│   └── Portfolio risk
├── VaRConfig
│   ├── Parametric VaR
│   ├── Historical simulation
│   └── Monte Carlo VaR
├── PortfolioConfig
│   ├── Optimization methods
│   ├── Constraints
│   ├── Efficient frontier
│   └── Rebalancing
└── TimeSeriesConfig
    ├── ARIMA
    ├── GARCH
    └── EWMA
```

## Environment Variables

### Options Pricing

```bash
# Black-Scholes
export OPTIONS_RISK_FREE_RATE=0.05
export OPTIONS_DIVIDEND_YIELD=0.0

# Monte Carlo
export OPTIONS_MC_PATHS=10000
export OPTIONS_VARIANCE_REDUCTION=antithetic  # antithetic, importance, stratified

# Implied Volatility
export OPTIONS_IV_MAX_ITER=100
export OPTIONS_IV_TOLERANCE=0.000001
```

### Credit Risk

```bash
# Basel III Compliance
export CREDIT_BASEL_CONFIDENCE=0.999  # 99.9%
export CREDIT_DOWNTURN_MULTIPLIER=1.25
export CREDIT_CAPITAL_APPROACH=ADVANCED_IRB  # SA_CR, FIRB, AIRB

# Performance
export CREDIT_ENABLE_CACHING=true
export CREDIT_PARALLEL_PROCESSING=true
export CREDIT_MAX_WORKERS=4
```

### VaR Models

```bash
export VAR_CONFIDENCE=0.95  # 95% confidence level
export VAR_METHOD=historical  # parametric, historical, monte_carlo
export VAR_MIN_OBS=252  # Minimum 1 year of data
export VAR_MC_SIMULATIONS=10000
```

### Portfolio Optimization

```bash
export PORTFOLIO_RISK_FREE_RATE=0.03
export PORTFOLIO_METHOD=max_sharpe
export PORTFOLIO_LONG_ONLY=true
export PORTFOLIO_REBALANCE_THRESHOLD=0.05  # 5%
```

### Time Series

```bash
export TS_EWMA_LAMBDA=0.94  # RiskMetrics standard
export TS_FORECAST_HORIZON=5
export TS_MIN_OBSERVATIONS=100
```

### Loading from Environment

```python
from axiom.config.model_config import ModelConfig

# Automatically loads all environment variables
config = ModelConfig.from_env()
```

## Configuration Profiles

### Basel III Compliance

```python
from axiom.config.model_config import ModelConfig

# Complete Basel III compliant configuration
config = ModelConfig.for_basel_iii_compliance()

# Key settings:
# - 99.9% confidence level
# - 10-day VaR horizon
# - Downturn LGD (1.25x multiplier)
# - Advanced IRB capital approach
# - Through-the-Cycle PDs
```

### High Performance (Speed Optimized)

```python
config = ModelConfig.for_high_performance()

# Optimizations:
# - Reduced Monte Carlo paths (5,000 instead of 10,000)
# - Fewer binomial steps (50 instead of 100)
# - Caching enabled
# - Parallel processing enabled
# - 3-5x faster with minimal accuracy loss
```

### High Precision (Accuracy Optimized)

```python
config = ModelConfig.for_high_precision()

# Enhancements:
# - 100,000 Monte Carlo paths
# - 500 binomial steps
# - 1e-10 precision for Black-Scholes
# - More rigorous convergence criteria
# - Slower but maximum accuracy
```

### Trading Style Profiles

```python
from axiom.config.model_config import TimeSeriesConfig, TradingStyle

# Intraday trading (high frequency)
config = TimeSeriesConfig.for_intraday_trading()
# - Fast EWMA (λ=0.99)
# - Short forecast horizon (1 period)
# - Minimal observations (50)

# Swing trading (multi-day)
config = TimeSeriesConfig.for_swing_trading()
# - Standard EWMA (λ=0.94, RiskMetrics)
# - Medium forecast horizon (5 periods)
# - Standard observations (100)

# Position trading (weeks to months)
config = TimeSeriesConfig.for_position_trading()
# - Slow EWMA (λ=0.88)
# - Long forecast horizon (20 periods)
# - More observations (252)
```

### Risk Management Profiles

```python
from axiom.config.model_config import TimeSeriesConfig, RiskProfile

# Conservative (low risk tolerance)
config = TimeSeriesConfig.for_risk_management(RiskProfile.CONSERVATIVE)
# - 99% confidence intervals
# - 10-day forecast horizon
# - Less reactive to market changes

# Moderate (balanced)
config = TimeSeriesConfig.for_risk_management(RiskProfile.MODERATE)
# - 95% confidence intervals
# - 5-day forecast horizon
# - Standard RiskMetrics parameters

# Aggressive (high risk tolerance)
config = TimeSeriesConfig.for_risk_management(RiskProfile.AGGRESSIVE)
# - 90% confidence intervals
# - 3-day forecast horizon
# - More reactive to market changes
```

## Model-Specific Configuration

### Options Pricing

```python
from axiom.config.model_config import OptionsConfig

config = OptionsConfig(
    # Black-Scholes
    default_risk_free_rate=0.05,
    default_dividend_yield=0.02,
    
    # Binomial Tree
    binomial_steps_default=200,  # More accuracy
    
    # Monte Carlo
    monte_carlo_paths_default=50000,  # Higher precision
    variance_reduction="importance",  # Advanced variance reduction
    monte_carlo_seed=42,  # Reproducible results
    
    # Implied Volatility
    iv_solver_method="brent",  # Alternative solver
    iv_max_iterations=200,
    iv_tolerance=1e-8,
    
    # Performance
    enable_logging=True,
    cache_results=True
)
```

### Credit Risk

```python
from axiom.config.model_config import CreditConfig

config = CreditConfig(
    # PD Configuration
    default_confidence_level=0.99,
    basel_confidence_level=0.999,
    pd_approach="kmv_merton",
    pit_to_ttc_weight=0.8,  # More weight on rating-based TTC
    
    # LGD Configuration
    default_recovery_rate=0.45,  # 45% recovery
    downturn_multiplier=1.3,  # Stricter than Basel minimum
    use_downturn_lgd=True,
    collateral_haircut=0.25,  # Conservative haircut
    
    # Credit VaR
    cvar_approach="monte_carlo",
    monte_carlo_scenarios=50000,  # High precision
    variance_reduction="antithetic",
    correlation_method="t_copula",  # Fat tails
    
    # Performance
    parallel_processing=True,
    max_workers=8  # Use more cores
)
```

### VaR Models

```python
from axiom.config.model_config import VaRConfig

config = VaRConfig(
    # General
    default_confidence_level=0.99,
    default_time_horizon=10,  # 10-day VaR for Basel
    default_method="monte_carlo",
    
    # Parametric
    assume_normal=False,  # Use historical distribution
    use_ewma_volatility=True,  # RiskMetrics volatility
    ewma_lambda=0.94,
    
    # Historical Simulation
    min_observations=500,  # 2 years of data
    bootstrap_enabled=True,
    bootstrap_iterations=2000,
    
    # Monte Carlo
    default_simulations=100000,  # High precision
    variance_reduction="stratified",
    random_seed=None,  # Different each time
    
    # Performance
    parallel_mc=True,
    max_workers=8
)
```

### Portfolio Optimization

```python
from axiom.config.model_config import PortfolioConfig

config = PortfolioConfig(
    # Optimization
    default_risk_free_rate=0.025,  # Current market rate
    optimization_method="risk_parity",
    
    # Constraints
    long_only=True,
    fully_invested=True,
    min_weight=0.02,  # Minimum 2% per asset
    max_weight=0.30,  # Maximum 30% per asset
    sector_limits={
        "Technology": 0.40,
        "Healthcare": 0.30,
        "Financials": 0.30
    },
    
    # Black-Litterman
    bl_tau=0.025,  # Lower uncertainty
    bl_confidence_method="meucci",
    
    # Rebalancing
    rebalance_threshold=0.03,  # Tighter threshold
    transaction_costs=0.0005,  # 5 bps
    min_trade_size=1000.0,
    
    # Performance
    cache_covariance=True,
    use_shrinkage=True,  # Ledoit-Wolf shrinkage
    shrinkage_target="constant_variance"
)
```

### Time Series

```python
from axiom.config.model_config import TimeSeriesConfig

config = TimeSeriesConfig(
    # ARIMA
    arima_auto_select=True,
    arima_ic="bic",  # Bayesian IC for model selection
    arima_max_p=8,
    arima_max_d=2,
    arima_max_q=8,
    arima_seasonal=True,
    arima_m=12,  # Monthly seasonality
    
    # GARCH
    garch_order=(2, 2),  # GARCH(2,2) for more complex dynamics
    garch_distribution="t",  # Student-t for fat tails
    garch_vol_target=0.20,  # Target 20% annualized vol
    
    # EWMA
    ewma_decay_factor=0.96,  # Custom decay
    ewma_fast_span=8,
    ewma_slow_span=21,
    
    # General
    min_observations=200,
    confidence_level=0.99,
    forecast_horizon=10
)
```

## Runtime Configuration

### Update Configuration at Runtime

```python
from axiom.config.model_config import get_config, set_config

# Get current config
config = get_config()

# Modify specific values
config.options.monte_carlo_paths_default = 20000
config.var.default_confidence_level = 0.99

# Apply changes globally
set_config(config)

# All subsequent model instances use new configuration
```

### Temporary Configuration Changes

```python
from axiom.config.model_config import get_config, set_config, reset_config

# Save current config
original_config = get_config()

try:
    # Create temporary config
    temp_config = ModelConfig.for_high_performance()
    set_config(temp_config)
    
    # Run models with temporary config
    result = run_analysis()
    
finally:
    # Restore original config
    set_config(original_config)
```

### Configuration Context Manager

```python
from contextlib import contextmanager
from axiom.config.model_config import get_config, set_config

@contextmanager
def temporary_config(config):
    """Context manager for temporary configuration."""
    original = get_config()
    try:
        set_config(config)
        yield config
    finally:
        set_config(original)

# Usage
with temporary_config(ModelConfig.for_high_precision()):
    # This block uses high precision config
    result = calculate_option_price(...)
    
# Original config restored automatically
```

## Configuration Files

### Save Configuration to File

```python
from axiom.config.model_config import ModelConfig

# Create configuration
config = ModelConfig.for_basel_iii_compliance()

# Save to JSON file
config.save_to_file("config/basel_iii.json")
```

### Load Configuration from File

```python
# Load from file
config = ModelConfig.from_file("config/basel_iii.json")

# Apply globally
set_config(config)
```

### Example Configuration File

```json
{
  "options": {
    "default_risk_free_rate": 0.05,
    "default_dividend_yield": 0.0,
    "monte_carlo_paths_default": 10000,
    "variance_reduction": "antithetic"
  },
  "credit": {
    "default_confidence_level": 0.99,
    "basel_confidence_level": 0.999,
    "downturn_multiplier": 1.25,
    "capital_approach": "ADVANCED_IRB"
  },
  "var": {
    "default_confidence_level": 0.95,
    "default_time_horizon": 1,
    "default_method": "historical"
  },
  "portfolio": {
    "default_risk_free_rate": 0.03,
    "optimization_method": "max_sharpe",
    "long_only": true
  },
  "time_series": {
    "ewma_decay_factor": 0.94,
    "forecast_horizon": 5
  }
}
```

## Best Practices

### 1. Use Environment Variables for Deployment

```bash
# Production .env file
OPTIONS_RISK_FREE_RATE=0.045
CREDIT_BASEL_CONFIDENCE=0.999
VAR_CONFIDENCE=0.99
PORTFOLIO_RISK_FREE_RATE=0.03
```

### 2. Use Profiles for Standard Scenarios

```python
# Instead of manual configuration
config = ModelConfig.for_basel_iii_compliance()  # ✅ Good

# Avoid manual field-by-field setup for common scenarios
config = ModelConfig()  # ❌ Less optimal for Basel III
config.credit.basel_confidence_level = 0.999
config.var.default_confidence_level = 0.999
# ... many more settings
```

### 3. Document Custom Configurations

```python
def get_company_config() -> ModelConfig:
    """
    Company-specific configuration.
    
    - Conservative risk parameters
    - 99.5% confidence levels
    - Enhanced validation
    """
    config = ModelConfig()
    config.var.default_confidence_level = 0.995
    config.credit.downturn_multiplier = 1.4
    return config
```

### 4. Version Control Configuration Files

```bash
# Store standard configurations in version control
config/
├── basel_iii.json
├── high_performance.json
├── high_precision.json
├── production.json
└── development.json
```

### 5. Validate Configuration

```python
from axiom.config.model_config import ModelConfig

config = ModelConfig()

# Validate ranges
assert 0.0 <= config.var.default_confidence_level <= 1.0
assert config.options.monte_carlo_paths_default > 0
assert config.credit.downturn_multiplier >= 1.0
```

## Environment-Specific Configuration

### Development

```python
# Fast iteration, detailed logging
config = ModelConfig.for_high_performance()
config.options.enable_logging = True
config.credit.enable_caching = True
```

### Testing

```python
# Reproducible results, moderate precision
config = ModelConfig()
config.options.monte_carlo_seed = 42
config.var.random_seed = 42
config.credit.monte_carlo_scenarios = 1000  # Faster tests
```

### Production

```python
# Basel III compliant, high performance
config = ModelConfig.for_basel_iii_compliance()
config.options.cache_results = True
config.credit.parallel_processing = True
config.var.parallel_mc = True
```

---

**Last Updated**: 2025-10-23  
**Version**: 1.0.0  
**See Also**: [Model Config API](api/model_config.md)