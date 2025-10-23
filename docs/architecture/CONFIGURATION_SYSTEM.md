# Configuration System Architecture

**Unified Configuration for All Financial Models**

## Overview

Axiom's configuration system provides centralized, type-safe configuration management for all financial models with 47+ parameters organized into 5 categories. The system supports multiple configuration sources (environment variables, files, presets) and enables runtime customization without code changes.

**Location**: [`axiom/config/model_config.py`](../../axiom/config/model_config.py)

## Core Principles

1. **DRY (Don't Repeat Yourself)**: Single source of truth for all configuration
2. **Type Safety**: Dataclass-based configuration with validation
3. **Environment Flexibility**: Load from env vars, files, or programmatic presets
4. **Profile Support**: Pre-configured profiles for common scenarios
5. **Runtime Updates**: Modify configuration without restarting

## Configuration Hierarchy

```python
ModelConfig (master container)
├── OptionsConfig (13 parameters)
│   ├── Black-Scholes settings
│   ├── Binomial tree settings
│   ├── Monte Carlo settings
│   ├── Greeks calculation settings
│   └── Implied volatility settings
│
├── CreditConfig (17 parameters)
│   ├── PD (Probability of Default) settings
│   ├── LGD (Loss Given Default) settings
│   ├── EAD (Exposure at Default) settings
│   ├── Credit VaR settings
│   └── Portfolio risk settings
│
├── VaRConfig (13 parameters)
│   ├── General VaR settings
│   ├── Parametric VaR settings
│   ├── Historical simulation settings
│   └── Monte Carlo VaR settings
│
├── PortfolioConfig (15 parameters)
│   ├── Optimization settings
│   ├── Constraint settings
│   ├── Efficient frontier settings
│   ├── Risk parity settings
│   └── Rebalancing settings
│
└── TimeSeriesConfig (14 parameters)
    ├── ARIMA settings
    ├── GARCH settings
    ├── EWMA settings
    └── General forecasting settings
```

## Loading Strategies

### Strategy 1: Default Configuration

```python
from axiom.config.model_config import get_config

# Get global configuration with defaults
config = get_config()

# Access specific config sections
var_config = config.var
ts_config = config.time_series
```

### Strategy 2: From Environment Variables

```bash
# Set in .env or shell
export VAR_CONFIDENCE=0.99
export VAR_METHOD=monte_carlo
export TS_EWMA_LAMBDA=0.94
export PORTFOLIO_RISK_FREE_RATE=0.03
```

```python
from axiom.config.model_config import ModelConfig

# Automatically loads from environment
config = ModelConfig.from_env()

# Individual config sections can also load from env
from axiom.config.model_config import VaRConfig
var_config = VaRConfig.from_env()
```

### Strategy 3: From JSON File

```python
# Create configuration file
config_data = {
    "var": {
        "default_confidence_level": 0.99,
        "default_method": "monte_carlo",
        "default_simulations": 50000,
        "parallel_mc": true
    },
    "time_series": {
        "ewma_decay_factor": 0.94,
        "forecast_horizon": 5
    }
}

# Save to file
import json
with open("my_config.json", "w") as f:
    json.dump(config_data, f)

# Load from file
config = ModelConfig.from_file("my_config.json")

# Or use the convenience method on ModelConfig
config.save_to_file("saved_config.json")
```

### Strategy 4: Configuration Profiles

```python
from axiom.config.model_config import ModelConfig

# Basel III compliance profile
basel_config = ModelConfig.for_basel_iii_compliance()
# - VaR: 99.9% confidence, 10-day horizon
# - Credit: Downturn LGD, Advanced IRB

# High performance profile (speed-optimized)
fast_config = ModelConfig.for_high_performance()
# - Monte Carlo: 5K paths
# - Binomial: 50 steps
# - Caching enabled

# High precision profile (accuracy-optimized)
precise_config = ModelConfig.for_high_precision()
# - Monte Carlo: 100K paths
# - Binomial: 500 steps
# - High precision tolerance
```

### Strategy 5: Trading Style Presets

```python
from axiom.config.model_config import TimeSeriesConfig

# Intraday trading (high reactivity)
intraday = TimeSeriesConfig.for_intraday_trading()
# - EWMA decay: 0.99 (very reactive)
# - Fast/slow spans: 5/15
# - Forecast horizon: 1

# Swing trading (multi-day)
swing = TimeSeriesConfig.for_swing_trading()
# - EWMA decay: 0.94 (RiskMetrics)
# - Fast/slow spans: 12/26
# - Forecast horizon: 5

# Position trading (weeks to months)
position = TimeSeriesConfig.for_position_trading()
# - EWMA decay: 0.88 (less reactive)
# - Fast/slow spans: 26/52
# - Forecast horizon: 20
```

### Strategy 6: Risk Management Profiles

```python
from axiom.config.model_config import TimeSeriesConfig, RiskProfile

# Conservative risk management
conservative = TimeSeriesConfig.for_risk_management(RiskProfile.CONSERVATIVE)
# - EWMA decay: 0.88
# - Confidence: 99%
# - Forecast horizon: 10

# Aggressive risk management
aggressive = TimeSeriesConfig.for_risk_management(RiskProfile.AGGRESSIVE)
# - EWMA decay: 0.99
# - Confidence: 90%
# - Forecast horizon: 3

# Moderate (default)
moderate = TimeSeriesConfig.for_risk_management(RiskProfile.MODERATE)
# - EWMA decay: 0.94
# - Confidence: 95%
# - Forecast horizon: 5
```

## Configuration Sections

### VaRConfig

**Location**: [`axiom/config/model_config.py:155`](../../axiom/config/model_config.py:155)

```python
from axiom.config.model_config import VaRConfig

var_config = VaRConfig(
    # General
    default_confidence_level=0.95,
    default_time_horizon=1,  # days
    default_method="historical",
    
    # Parametric
    assume_normal=True,
    use_ewma_volatility=False,
    ewma_lambda=0.94,
    
    # Historical Simulation
    min_observations=252,
    bootstrap_enabled=False,
    bootstrap_iterations=1000,
    
    # Monte Carlo
    default_simulations=10000,
    max_simulations=1000000,
    variance_reduction="antithetic",
    random_seed=None,
    
    # Performance
    cache_results=True,
    parallel_mc=True,
    max_workers=4
)
```

**Environment Variables**:
```bash
VAR_CONFIDENCE=0.95
VAR_METHOD=historical
VAR_MIN_OBS=252
```

### TimeSeriesConfig

**Location**: [`axiom/config/model_config.py:251`](../../axiom/config/model_config.py:251)

```python
from axiom.config.model_config import TimeSeriesConfig

ts_config = TimeSeriesConfig(
    # ARIMA
    arima_auto_select=True,
    arima_ic="aic",  # aic, bic, hqic
    arima_max_p=5,
    arima_max_d=2,
    arima_max_q=5,
    arima_seasonal=False,
    arima_m=12,
    
    # GARCH
    garch_order=(1, 1),
    garch_distribution="normal",  # normal, t, ged
    garch_vol_target=None,
    garch_rescale=True,
    
    # EWMA
    ewma_decay_factor=0.94,
    ewma_min_periods=30,
    ewma_fast_span=12,
    ewma_slow_span=26,
    
    # General
    min_observations=100,
    confidence_level=0.95,
    forecast_horizon=5,
    
    # Performance
    cache_models=True,
    parallel_fitting=False
)
```

**Environment Variables**:
```bash
TS_EWMA_LAMBDA=0.94
TS_FORECAST_HORIZON=5
```

### PortfolioConfig

**Location**: [`axiom/config/model_config.py:199`](../../axiom/config/model_config.py:199)

```python
from axiom.config.model_config import PortfolioConfig

portfolio_config = PortfolioConfig(
    # Optimization
    default_risk_free_rate=0.03,
    periods_per_year=252,
    optimization_method="max_sharpe",
    
    # Constraints
    long_only=True,
    fully_invested=True,
    min_weight=0.0,
    max_weight=1.0,
    sector_limits={},  # e.g., {"tech": 0.3, "finance": 0.2}
    
    # Efficient Frontier
    frontier_points=100,
    target_return_range="auto",
    
    # Risk Parity
    risk_parity_max_iter=1000,
    risk_parity_tolerance=1e-6,
    
    # Black-Litterman
    bl_tau=0.05,
    bl_confidence_method="idzorek",
    
    # Rebalancing
    rebalance_threshold=0.05,
    transaction_costs=0.001,
    min_trade_size=100.0,
    
    # Performance
    cache_covariance=True,
    use_shrinkage=False,
    shrinkage_target="constant_correlation"
)
```

**Environment Variables**:
```bash
PORTFOLIO_RISK_FREE_RATE=0.03
PORTFOLIO_METHOD=max_sharpe
PORTFOLIO_LONG_ONLY=true
```

### CreditConfig

**Location**: [`axiom/config/model_config.py:93`](../../axiom/config/model_config.py:93)

```python
from axiom.config.model_config import CreditConfig

credit_config = CreditConfig(
    # PD Configuration
    default_confidence_level=0.99,
    basel_confidence_level=0.999,
    pd_approach="kmv_merton",
    pit_to_ttc_weight=0.7,
    
    # LGD Configuration
    default_recovery_rate=0.40,
    downturn_multiplier=1.25,
    use_downturn_lgd=True,
    collateral_haircut=0.20,
    
    # EAD Configuration
    default_ccf=0.75,
    sa_ccr_alpha=1.4,
    calculate_pfe=True,
    
    # Credit VaR
    cvar_approach="monte_carlo",
    monte_carlo_scenarios=10000,
    variance_reduction="antithetic",
    correlation_method="gaussian",
    
    # Portfolio Risk
    concentration_threshold=0.10,
    enable_diversification_benefit=True,
    capital_approach="ADVANCED_IRB",
    
    # Performance
    enable_caching=True,
    parallel_processing=True,
    max_workers=4
)

# Basel III preset
basel_credit = CreditConfig.for_basel_iii()
```

**Environment Variables**:
```bash
CREDIT_BASEL_CONFIDENCE=0.999
CREDIT_DOWNTURN_MULTIPLIER=1.25
CREDIT_CAPITAL_APPROACH=ADVANCED_IRB
```

### OptionsConfig

**Location**: [`axiom/config/model_config.py:41`](../../axiom/config/model_config.py:41)

```python
from axiom.config.model_config import OptionsConfig

options_config = OptionsConfig(
    # Black-Scholes
    default_risk_free_rate=0.05,
    default_dividend_yield=0.0,
    black_scholes_precision=1e-6,
    
    # Binomial Tree
    binomial_steps_default=100,
    binomial_max_steps=1000,
    binomial_convergence_threshold=0.01,
    
    # Monte Carlo
    monte_carlo_paths_default=10000,
    monte_carlo_max_paths=1000000,
    monte_carlo_seed=None,
    variance_reduction="antithetic",
    
    # Greeks
    greeks_delta=0.01,
    greeks_precision=1e-6,
    greeks_calculate_all=True,
    
    # Implied Volatility
    iv_solver_method="newton_raphson",
    iv_max_iterations=100,
    iv_tolerance=1e-6,
    iv_initial_guess_method="brenner_subrahmanyam",
    iv_constant_guess=0.25,
    
    # Logging and performance
    enable_logging=True,
    enable_performance_tracking=True,
    cache_results=False
)
```

**Environment Variables**:
```bash
OPTIONS_RISK_FREE_RATE=0.05
OPTIONS_DIVIDEND_YIELD=0.0
OPTIONS_MC_PATHS=10000
OPTIONS_VARIANCE_REDUCTION=antithetic
```

## Runtime Configuration Updates

### Global Configuration

```python
from axiom.config.model_config import get_config, set_config

# Get current global config
config = get_config()

# Modify
config.var.default_confidence_level = 0.99
config.time_series.forecast_horizon = 10

# Set as new global config
set_config(config)

# All new models will use updated config
from axiom.models.base.factory import ModelFactory, ModelType
var_model = ModelFactory.create(ModelType.HISTORICAL_VAR)
# Uses confidence_level = 0.99 from updated config
```

### Model-Specific Configuration

```python
# Create model with specific config
custom_config = VaRConfig(default_confidence_level=0.99)
model = ModelFactory.create(ModelType.HISTORICAL_VAR, config=custom_config)

# Update model config at runtime
model.update_config({"default_simulations": 50000})

# Get current config
current_config = model.get_config()
print(current_config)
```

## Real-World Examples

### Example 1: Multi-Environment Setup

```python
# development.json
{
    "var": {
        "default_simulations": 1000,  # Fast for dev
        "cache_results": true
    }
}

# production.json
{
    "var": {
        "default_simulations": 100000,  # Accurate for prod
        "parallel_mc": true,
        "cache_results": true
    }
}

# Load based on environment
import os
env = os.getenv("ENVIRONMENT", "development")
config = ModelConfig.from_file(f"{env}.json")
```

### Example 2: A/B Testing Configurations

```python
# Test different VaR configurations
configs = {
    "conservative": VaRConfig(
        default_confidence_level=0.999,
        default_simulations=100000
    ),
    "moderate": VaRConfig(
        default_confidence_level=0.95,
        default_simulations=10000
    ),
    "aggressive": VaRConfig(
        default_confidence_level=0.90,
        default_simulations=5000
    )
}

results = {}
for name, config in configs.items():
    model = ModelFactory.create(ModelType.MONTE_CARLO_VAR, config=config)
    result = model.calculate_risk(
        portfolio_value=1_000_000,
        returns=returns_data
    )
    results[name] = result.value["var"]

# Compare
for name, var in results.items():
    print(f"{name:15s}: ${var:>12,.0f}")
```

### Example 3: Dynamic Configuration Based on Market Conditions

```python
from axiom.config.model_config import TimeSeriesConfig

def get_market_adjusted_config(volatility_regime: str) -> TimeSeriesConfig:
    """Adjust configuration based on market volatility."""
    
    if volatility_regime == "high":
        # High volatility - more reactive
        return TimeSeriesConfig(
            ewma_decay_factor=0.99,
            forecast_horizon=3,
            confidence_level=0.99
        )
    elif volatility_regime == "low":
        # Low volatility - less reactive
        return TimeSeriesConfig(
            ewma_decay_factor=0.88,
            forecast_horizon=10,
            confidence_level=0.95
        )
    else:
        # Normal volatility - standard
        return TimeSeriesConfig.for_swing_trading()

# Use dynamic config
current_regime = detect_volatility_regime(market_data)
config = get_market_adjusted_config(current_regime)
model = ModelFactory.create(ModelType.EWMA, config=config)
```

### Example 4: Configuration Inheritance

```python
# Base configuration for all trading strategies
base_config = TimeSeriesConfig(
    min_observations=100,
    confidence_level=0.95,
    cache_models=True
)

# Extend for specific strategy
intraday_config = TimeSeriesConfig(
    **base_config.to_dict(),
    ewma_decay_factor=0.99,
    forecast_horizon=1,
    ewma_fast_span=5,
    ewma_slow_span=15
)

# Another strategy inheriting base
swing_config = TimeSeriesConfig(
    **base_config.to_dict(),
    ewma_decay_factor=0.94,
    forecast_horizon=5,
    ewma_fast_span=12,
    ewma_slow_span=26
)
```

## Configuration Validation

### Automatic Validation

```python
from axiom.config.model_config import VaRConfig

# Valid configuration
config = VaRConfig(default_confidence_level=0.95)  # ✓

# Invalid configuration will raise error at creation time
try:
    bad_config = VaRConfig(default_confidence_level=1.5)  # ✗ > 1.0
except ValueError as e:
    print(f"Validation error: {e}")
```

### Custom Validation

```python
class ValidatedVaRConfig(VaRConfig):
    """VaR config with additional validation."""
    
    def __post_init__(self):
        """Validate after initialization."""
        if self.default_simulations < 1000:
            raise ValueError("Minimum 1000 simulations required")
        
        if self.max_simulations < self.default_simulations:
            raise ValueError("max_simulations must be >= default_simulations")
        
        if self.default_confidence_level < 0.90:
            raise ValueError("Minimum 90% confidence required")
```

## Configuration Serialization

### To Dictionary

```python
config = ModelConfig.for_basel_iii_compliance()

# Convert to dictionary
config_dict = config.to_dict()
# {
#     "options": {...},
#     "credit": {...},
#     "var": {...},
#     "portfolio": {...},
#     "time_series": {...}
# }

# Access specific section
var_dict = config.var.to_dict()
```

### To JSON

```python
# Convert to JSON string
config_json = config.to_json()
print(config_json)

# Save to file
config.save_to_file("basel_iii_config.json")

# Load from file
loaded_config = ModelConfig.from_file("basel_iii_config.json")
```

## Best Practices

### 1. Use Configuration Profiles for Common Scenarios

```python
# ✓ Good - use built-in profiles
config = ModelConfig.for_basel_iii_compliance()

# ✗ Bad - manually setting all parameters
config = ModelConfig()
config.var.default_confidence_level = 0.999
config.var.default_time_horizon = 10
# ... many more settings
```

### 2. Load from Environment in Production

```python
# ✓ Good - flexible deployment
config = ModelConfig.from_env()

# ✗ Bad - hard-coded values
config = ModelConfig()
config.var.default_confidence_level = 0.95  # Hard-coded
```

### 3. Use Type-Safe Configuration Classes

```python
# ✓ Good - type safe
from axiom.config.model_config import VaRConfig
config = VaRConfig(default_confidence_level=0.95)

# ✗ Bad - dictionary with no validation
config = {"default_confidence_level": "0.95"}  # Wrong type!
```

### 4. Version Your Configuration Files

```bash
# Git-track configuration files
configs/
├── basel_iii.json
├── high_performance.json
├── high_precision.json
└── production.json
```

### 5. Document Custom Configurations

```python
# ✓ Good - well-documented custom config
class HedgeFundVaRConfig(VaRConfig):
    """
    VaR configuration for hedge fund operations.
    
    Uses 99.9% confidence for regulatory compliance
    and 100K simulations for accuracy.
    
    Suitable for: Hedge funds, prop trading desks
    Not suitable for: Retail portfolios
    """
    def __init__(self):
        super().__init__(
            default_confidence_level=0.999,
            default_simulations=100000,
            parallel_mc=True
        )
```

## Configuration Testing

```python
import pytest
from axiom.config.model_config import ModelConfig, VaRConfig

def test_basel_iii_config():
    """Test Basel III configuration profile."""
    config = ModelConfig.for_basel_iii_compliance()
    
    # Verify VaR settings
    assert config.var.default_confidence_level == 0.999
    assert config.var.default_time_horizon == 10
    assert config.var.min_observations == 252
    
    # Verify credit settings
    assert config.credit.basel_confidence_level == 0.999
    assert config.credit.use_downturn_lgd == True

def test_config_from_env(monkeypatch):
    """Test loading configuration from environment."""
    # Set environment variables
    monkeypatch.setenv("VAR_CONFIDENCE", "0.99")
    monkeypatch.setenv("VAR_METHOD", "monte_carlo")
    
    # Load config
    config = VaRConfig.from_env()
    
    # Verify
    assert config.default_confidence_level == 0.99
    assert config.default_method == "monte_carlo"

def test_config_serialization():
    """Test configuration serialization."""
    # Create config
    original = VaRConfig(default_confidence_level=0.99)
    
    # Convert to dict and back
    config_dict = original.to_dict()
    restored = VaRConfig(**config_dict)
    
    # Verify
    assert restored.default_confidence_level == original.default_confidence_level
```

## See Also

- [`BASE_CLASSES.md`](BASE_CLASSES.md) - Base class hierarchy
- [`MIXINS.md`](MIXINS.md) - Reusable functionality mixins
- [`FACTORY_PATTERN.md`](FACTORY_PATTERN.md) - Model creation with configuration injection

---

**Last Updated**: 2025-10-23  
**Version**: 2.0.0