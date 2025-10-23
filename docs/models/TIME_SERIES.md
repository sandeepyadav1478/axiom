# Time Series Models - Complete Reference

**Production-ready ARIMA, GARCH, and EWMA with configuration system**

## Overview

Three essential time series methodologies for financial forecasting and risk analysis:

| Model | Purpose | Performance | Best For |
|-------|---------|-------------|----------|
| **ARIMA** | Price forecasting | <50ms | Trend prediction, mean reversion |
| **GARCH** | Volatility forecasting | <100ms | VaR, option pricing, risk management |
| **EWMA** | Trend following | <10ms | Real-time signals, momentum strategies |

## Mathematical Framework

### ARIMA(p,d,q) - AutoRegressive Integrated Moving Average

**Formula:**
```
(1 - φ₁L - φ₂L² - ... - φₚLᵖ)(1-L)ᵈy_t = (1 + θ₁L + θ₂L² + ... + θ_qLᵍ)ε_t

where:
p = AR order (past values)
d = Integration order (differencing)
q = MA order (past errors)
L = Lag operator
ε_t = White noise
```

### GARCH(p,q) - Generalized AutoRegressive Conditional Heteroskedasticity

**Formula:**
```
r_t = μ + ε_t,  ε_t = σ_t × z_t,  z_t ~ N(0,1)

σ_t² = ω + Σ(α_i × ε²_{t-i}) + Σ(β_j × σ²_{t-j})

where:
p = ARCH order (past squared residuals)
q = GARCH order (past variances)
ω, α, β = Model parameters
```

### EWMA - Exponentially Weighted Moving Average

**Formula:**
```
σ_t² = λ × σ²_{t-1} + (1-λ) × r²_{t-1}

where:
λ = Decay factor (0.94 for RiskMetrics)
σ_t² = Variance forecast at time t
r_{t-1} = Return at time t-1
```

## Quick Start

### Modern Factory Pattern API (Recommended)

```python
from axiom.models.base.factory import ModelFactory, ModelType
from axiom.config.model_config import TimeSeriesConfig
import numpy as np

# Create ARIMA model with default config
arima_model = ModelFactory.create(ModelType.ARIMA)
result = arima_model.calculate(
    data=price_data,
    forecast_horizon=5
)

print(f"Forecast: {result.value['forecast']}")
print(f"Confidence intervals: {result.value['confidence_intervals']}")
print(f"Execution time: {result.metadata.execution_time_ms:.2f}ms")

# Create GARCH model with custom config
garch_config = TimeSeriesConfig(
    garch_order=(1, 1),
    garch_distribution="t",  # Student-t for fat tails
    forecast_horizon=10
)
garch_model = ModelFactory.create(ModelType.GARCH, config=garch_config)

# Create EWMA model
ewma_model = ModelFactory.create(ModelType.EWMA)
ewma_result = ewma_model.calculate(
    data=returns_data,
    decay_factor=0.94
)
```

### Trading Style Presets

```python
from axiom.config.model_config import TimeSeriesConfig

# Intraday trading (high reactivity)
intraday_config = TimeSeriesConfig.for_intraday_trading()
# - EWMA decay: 0.99 (very reactive)
# - Min periods: 10
# - Forecast horizon: 1
# - Fast EWMA span: 5, Slow: 15

# Swing trading (multi-day holds)
swing_config = TimeSeriesConfig.for_swing_trading()
# - EWMA decay: 0.94 (RiskMetrics standard)
# - Min periods: 30
# - Forecast horizon: 5
# - Fast EWMA span: 12, Slow: 26

# Position trading (weeks to months)
position_config = TimeSeriesConfig.for_position_trading()
# - EWMA decay: 0.88 (less reactive)
# - Min periods: 60
# - Forecast horizon: 20
# - Fast EWMA span: 26, Slow: 52

# Use with factory
model = ModelFactory.create(ModelType.EWMA, config=intraday_config)
```

### Risk Management Profiles

```python
from axiom.config.model_config import TimeSeriesConfig, RiskProfile

# Conservative risk management
conservative_config = TimeSeriesConfig.for_risk_management(RiskProfile.CONSERVATIVE)
# - EWMA decay: 0.88 (slower reaction)
# - Confidence level: 99%
# - Forecast horizon: 10

# Aggressive risk management
aggressive_config = TimeSeriesConfig.for_risk_management(RiskProfile.AGGRESSIVE)
# - EWMA decay: 0.99 (faster reaction)
# - Confidence level: 90%
# - Forecast horizon: 3

# Moderate (default)
moderate_config = TimeSeriesConfig.for_risk_management(RiskProfile.MODERATE)
# - EWMA decay: 0.94
# - Confidence level: 95%
# - Forecast horizon: 5
```

## ARIMA Forecasting

### Automatic Model Selection

```python
from axiom.models.base.factory import ModelFactory, ModelType

# Auto-select optimal (p,d,q) using AIC
arima_model = ModelFactory.create(ModelType.ARIMA)
result = arima_model.calculate(
    data=price_data,
    forecast_horizon=5,
    auto_select=True,  # Automatic order selection
    max_p=5,
    max_d=2,
    max_q=5
)

print(f"Selected order: {result.value['order']}")  # e.g., (2,1,1)
print(f"AIC: {result.value['aic']:.2f}")
print(f"Forecast: {result.value['forecast']}")
```

### Manual ARIMA Configuration

```python
from axiom.config.model_config import TimeSeriesConfig

# Custom ARIMA configuration
arima_config = TimeSeriesConfig(
    arima_auto_select=False,
    arima_ic="bic",  # Use BIC instead of AIC
    arima_max_p=3,
    arima_max_d=1,
    arima_max_q=3,
    arima_seasonal=True,
    arima_m=12,  # Monthly seasonality
    forecast_horizon=12
)

model = ModelFactory.create(ModelType.ARIMA, config=arima_config)
```

## GARCH Volatility Forecasting

### Standard GARCH(1,1)

```python
from axiom.models.base.factory import ModelFactory, ModelType

# GARCH(1,1) - industry standard
garch_model = ModelFactory.create(ModelType.GARCH)
result = garch_model.calculate(
    data=returns_data,
    forecast_horizon=5
)

print(f"Volatility forecast: {result.value['volatility_forecast']}")
print(f"Conditional variance: {result.value['conditional_variance']}")
print(f"Persistence: {result.value['persistence']:.4f}")
```

### GARCH with Student-t Distribution

```python
from axiom.config.model_config import TimeSeriesConfig

# GARCH with fat-tailed distribution
garch_config = TimeSeriesConfig(
    garch_order=(1, 1),
    garch_distribution="t",  # Student-t for fat tails
    garch_rescale=True
)

model = ModelFactory.create(ModelType.GARCH, config=garch_config)
result = model.calculate(data=returns_data)

# Better for financial data with extreme moves
```

### Volatility Targeting

```python
# GARCH with volatility targeting
garch_config = TimeSeriesConfig(
    garch_order=(1, 1),
    garch_vol_target=0.15  # Target 15% annualized volatility
)

model = ModelFactory.create(ModelType.GARCH, config=garch_config)
```

## EWMA Trend Following

### Basic EWMA

```python
from axiom.models.base.factory import ModelFactory, ModelType

# Simple EWMA
ewma_model = ModelFactory.create(ModelType.EWMA)
result = ewma_model.calculate(
    data=price_data,
    decay_factor=0.94  # RiskMetrics standard
)

print(f"EWMA values: {result.value['ewma']}")
print(f"Current trend: {result.value['trend']}")  # up/down/neutral
```

### Dual EWMA Crossover Strategy

```python
from axiom.config.model_config import TimeSeriesConfig

# Fast and slow EWMA for crossover signals
ewma_config = TimeSeriesConfig(
    ewma_decay_factor=0.94,
    ewma_fast_span=12,  # Fast EWMA
    ewma_slow_span=26,  # Slow EWMA
    ewma_min_periods=30
)

model = ModelFactory.create(ModelType.EWMA, config=ewma_config)
result = model.calculate(data=price_data)

# Trading signals
fast_ewma = result.value['fast_ewma']
slow_ewma = result.value['slow_ewma']
signal = "BUY" if fast_ewma[-1] > slow_ewma[-1] else "SELL"
```

## Configuration

### Using TimeSeriesConfig Class

```python
from axiom.config.model_config import TimeSeriesConfig

# Create custom configuration
ts_config = TimeSeriesConfig(
    # ARIMA
    arima_auto_select=True,
    arima_ic="aic",
    arima_max_p=5,
    arima_max_d=2,
    arima_max_q=5,
    arima_seasonal=False,
    
    # GARCH
    garch_order=(1, 1),
    garch_distribution="normal",
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

# Use with factory
from axiom.models.base.factory import ModelFactory, ModelType
model = ModelFactory.create(ModelType.ARIMA, config=ts_config)
```

### Environment Variables

```bash
# Add to .env file
TS_EWMA_LAMBDA=0.94
TS_FORECAST_HORIZON=5
TS_MIN_OBS=100

# Load automatically
from axiom.config.model_config import TimeSeriesConfig
config = TimeSeriesConfig.from_env()
```

## Advanced Features

### Seasonal ARIMA

```python
# Seasonal ARIMA for monthly data
seasonal_config = TimeSeriesConfig(
    arima_seasonal=True,
    arima_m=12,  # 12-month seasonality
    forecast_horizon=12
)

model = ModelFactory.create(ModelType.ARIMA, config=seasonal_config)
```

### GARCH with Different Distributions

```python
# Normal distribution (default)
garch_normal = TimeSeriesConfig(garch_distribution="normal")

# Student-t distribution (fat tails)
garch_t = TimeSeriesConfig(garch_distribution="t")

# GED distribution (Generalized Error Distribution)
garch_ged = TimeSeriesConfig(garch_distribution="ged")
```

### Multi-Step Forecasting

```python
# Long-horizon forecasting
long_term_config = TimeSeriesConfig(
    forecast_horizon=20,
    confidence_level=0.95
)

model = ModelFactory.create(ModelType.ARIMA, config=long_term_config)
result = model.calculate(data=price_data)

# Get confidence bands
lower_bound = result.value['confidence_intervals']['lower']
upper_bound = result.value['confidence_intervals']['upper']
```

## Performance Benchmarks

| Model | Data Size | Execution Time | Status |
|-------|-----------|----------------|--------|
| ARIMA auto-select | 250 observations | <50ms | ✅ |
| ARIMA manual | 250 observations | <20ms | ✅ |
| GARCH(1,1) | 1000 observations | <100ms | ✅ |
| EWMA | 1000 observations | <10ms | ✅ |

## Use Cases

### 1. VaR Calculation with GARCH

```python
# Forecast volatility for VaR
garch_model = ModelFactory.create(ModelType.GARCH)
garch_result = garch_model.calculate(data=returns_data)

# Use forecasted volatility in VaR
forecasted_vol = garch_result.value['volatility_forecast'][0]
var_95 = portfolio_value * forecasted_vol * 1.645  # 95% confidence
```

### 2. Mean Reversion Trading with ARIMA

```python
# Detect mean reversion opportunities
arima_model = ModelFactory.create(ModelType.ARIMA)
forecast = arima_model.calculate(data=price_data, forecast_horizon=1)

current_price = price_data[-1]
predicted_price = forecast.value['forecast'][0]

if current_price < predicted_price * 0.98:
    signal = "BUY"  # Price below forecast - mean reversion opportunity
elif current_price > predicted_price * 1.02:
    signal = "SELL"  # Price above forecast
```

### 3. Momentum Trading with EWMA

```python
# EWMA crossover strategy
ewma_config = TimeSeriesConfig.for_intraday_trading()
ewma_model = ModelFactory.create(ModelType.EWMA, config=ewma_config)
result = ewma_model.calculate(data=price_data)

# Generate trading signals
fast = result.value['fast_ewma']
slow = result.value['slow_ewma']

if fast[-1] > slow[-1] and fast[-2] <= slow[-2]:
    signal = "BUY"  # Golden cross
elif fast[-1] < slow[-1] and fast[-2] >= slow[-2]:
    signal = "SELL"  # Death cross
```

## Model Validation

### Residual Analysis

```python
# Check model fit quality
arima_result = arima_model.calculate(data=price_data)

# Access residuals
residuals = arima_result.value['residuals']

# Check for autocorrelation (should be close to zero)
from scipy.stats import acorr_ljungbox
lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10)
print(f"Ljung-Box p-value: {lb_pvalue[0]:.4f}")  # Should be > 0.05
```

### Out-of-Sample Testing

```python
# Split data for validation
train_data = price_data[:-20]
test_data = price_data[-20:]

# Fit on training data
model = ModelFactory.create(ModelType.ARIMA)
model.calculate(data=train_data)

# Forecast and compare
forecast = model.calculate(data=train_data, forecast_horizon=20)
predicted = forecast.value['forecast']

# Calculate forecast error
mse = np.mean((test_data - predicted) ** 2)
mae = np.mean(np.abs(test_data - predicted))
```

## References

- Box & Jenkins (1970). "Time Series Analysis: Forecasting and Control"
- Bollerslev (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- J.P. Morgan (1996). "RiskMetrics Technical Document" (EWMA λ=0.94)
- Engle (1982). "Autoregressive Conditional Heteroscedasticity"

---

**Last Updated**: 2025-10-23  
**Version**: 2.0.0 (Factory Pattern + Configuration System)