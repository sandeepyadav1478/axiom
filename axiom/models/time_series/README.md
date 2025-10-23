# Time Series Models for Algorithmic Trading

Production-ready implementations of ARIMA, GARCH, and EWMA models for price forecasting, volatility estimation, and trend analysis in algorithmic trading.

## Overview

This module provides three industry-standard time series models:

1. **ARIMA** - Autoregressive Integrated Moving Average for price/trend forecasting
2. **GARCH** - Generalized Autoregressive Conditional Heteroskedasticity for volatility forecasting
3. **EWMA** - Exponentially Weighted Moving Average for trend analysis and real-time signals

All models integrate seamlessly with the existing VaR and Portfolio Optimization models for comprehensive risk management.

## Quick Start

```python
from axiom.models.time_series import ARIMAModel, GARCHModel, EWMAModel
import yfinance as yf

# Fetch data
data = yf.Ticker("AAPL").history(period="1y")
prices = data['Close'].values
returns = (prices[1:] / prices[:-1]) - 1

# ARIMA for price forecasting
arima = ARIMAModel()  # Auto-selects optimal (p,d,q)
arima.fit(prices)
forecast = arima.forecast(horizon=5)
print(f"5-day price forecast: {forecast.forecast}")

# GARCH for volatility forecasting
garch = GARCHModel(order=(1, 1))
garch.fit(returns, use_returns=True)
vol_forecast = garch.forecast(horizon=10)
print(f"10-day volatility forecast: {vol_forecast.volatility}")

# EWMA for trend analysis
ewma = EWMAModel(decay_factor=0.94)  # RiskMetrics
ewma.fit(returns, use_returns=True, calculate_volatility=True)
trend = ewma.detect_trend()
print(f"Current trend: {trend['trend']}")
```

## Models

### ARIMA - Price Forecasting

**Features:**
- Auto-ARIMA with automatic (p,d,q) selection
- Information criteria-based model selection (AIC, BIC, HQIC)
- Confidence intervals for forecasts
- Stationarity testing and differencing
- Seasonal ARIMA support

**Use Cases:**
- Price trend forecasting
- Mean reversion identification
- Trading signal generation

**Example:**
```python
from axiom.models.time_series import ARIMAModel

# Automatic parameter selection
model = ARIMAModel()
model.fit(price_data)

# Or specify parameters
model = ARIMAModel(order=(2, 1, 2))
model.fit(price_data)

# Forecast with confidence intervals
forecast = model.forecast(horizon=5, confidence_level=0.95)
print(f"Forecast: {forecast.forecast}")
print(f"95% CI: {forecast.confidence_intervals}")
```

### GARCH - Volatility Forecasting

**Features:**
- Standard GARCH(p,q) implementation
- Volatility clustering detection
- Persistence and half-life calculation
- Multiple distribution support (Normal, Student-t, GED)
- Integration with VaR models

**Use Cases:**
- Volatility forecasting
- VaR calculations
- Options pricing
- Risk-adjusted position sizing

**Example:**
```python
from axiom.models.time_series import GARCHModel

# Fit GARCH(1,1) - industry standard
garch = GARCHModel(order=(1, 1))
garch.fit(returns, use_returns=True)

# Check for volatility clustering
clustering = garch.detect_volatility_clustering()
print(f"Clustering detected: {clustering['volatility_clustering_detected']}")
print(f"Half-life: {clustering['half_life']:.1f} periods")

# Forecast volatility
vol_forecast = garch.forecast(horizon=10)
annualized_vol = vol_forecast.get_annualized_volatility()
print(f"Annualized volatility: {annualized_vol * 100:.2f}%")
```

### EWMA - Trend Analysis

**Features:**
- RiskMetrics™ standard (λ = 0.94 for daily data)
- Real-time updating for live trading
- Trend detection and strength calculation
- Trading signal generation (dual EWMA crossover)
- Volatility estimation

**Use Cases:**
- Trend following strategies
- Real-time trading signals
- Fast volatility estimation
- Alternative to GARCH for simple cases

**Example:**
```python
from axiom.models.time_series import EWMAModel

# RiskMetrics configuration
ewma = EWMAModel(decay_factor=0.94)
ewma.fit(returns, use_returns=True, calculate_volatility=True)

# Detect trend
trend_info = ewma.detect_trend(lookback=30)
print(f"Trend: {trend_info['trend']}")
print(f"Strength: {trend_info['strength']:.3f}")

# Generate trading signals
signals = ewma.get_trading_signals(prices, fast_span=12, slow_span=26)
print(f"Position: {signals['current_position']}")

# Real-time update
new_mean, new_vol = ewma.update(new_return, update_volatility=True)
```

## Integration with Risk Models

### VaR Integration

Combine GARCH volatility forecasts with VaR calculations:

```python
from axiom.models.time_series import GARCHModel
from axiom.models.risk.var_models import VaRCalculator, VaRMethod

# Fit GARCH
garch = GARCHModel(order=(1, 1))
garch.fit(returns, use_returns=True)

# Forecast volatility
vol_forecast = garch.forecast(horizon=1)
next_day_vol = vol_forecast.volatility[0]

# Calculate VaR using GARCH volatility
portfolio_value = 1000000
var_amount = portfolio_value * 1.65 * next_day_vol  # 95% VaR

# Or use VaR calculator
var_calc = VaRCalculator()
var_result = var_calc.calculate_var(
    portfolio_value,
    returns,
    method=VaRMethod.HISTORICAL
)
```

### Portfolio Optimization Integration

Use time series models for dynamic portfolio optimization:

```python
from axiom.models.time_series import GARCHModel, EWMAModel
from axiom.models.portfolio.optimization import PortfolioOptimizer

# Forecast volatility for each asset
volatility_forecasts = {}
for symbol, returns in returns_dict.items():
    garch = GARCHModel(order=(1, 1))
    garch.fit(returns, use_returns=True)
    vol_forecast = garch.forecast(horizon=1)
    volatility_forecasts[symbol] = vol_forecast.volatility[0]

# Optimize portfolio with forecasted volatility
optimizer = PortfolioOptimizer()
result = optimizer.optimize(returns_df, method=OptimizationMethod.MAX_SHARPE)
```

## Configuration

Pre-configured settings for different trading styles:

```python
from axiom.models.time_series import TimeSeriesConfig

# Intraday trading
config = TimeSeriesConfig.for_intraday_trading()

# Swing trading (multi-day holds)
config = TimeSeriesConfig.for_swing_trading()

# Position trading (weeks to months)
config = TimeSeriesConfig.for_position_trading()

# Risk management
config = TimeSeriesConfig.for_risk_management(RiskProfile.CONSERVATIVE)
```

## Utility Functions

Helper functions for data preprocessing and analysis:

```python
from axiom.models.time_series import (
    prepare_returns,
    check_stationarity,
    calculate_acf,
    calculate_pacf,
    detect_seasonality,
    split_train_test,
    calculate_forecast_accuracy
)

# Prepare returns
returns = prepare_returns(prices, log_returns=True)

# Check stationarity
stationarity = check_stationarity(prices)
if not stationarity['is_stationary']:
    print("Consider differencing the series")

# Detect seasonality
seasonality = detect_seasonality(data, max_period=365)
if seasonality['has_seasonality']:
    print(f"Seasonal period: {seasonality['primary_period']} days")

# Split data
train, test = split_train_test(data, test_size=0.2)

# Evaluate forecast accuracy
accuracy = calculate_forecast_accuracy(actual, forecast)
print(f"RMSE: {accuracy['rmse']:.4f}")
print(f"MAPE: {accuracy['mape']:.2f}%")
```

## Running the Demo

```bash
# Run comprehensive demo with real market data
python demos/demo_time_series_models.py
```

The demo includes:
- ARIMA price forecasting for AAPL
- GARCH volatility forecasting
- EWMA trend analysis and trading signals
- Integration with VaR and Portfolio Optimization
- Forecast accuracy comparison

## Testing

```bash
# Run all tests
pytest tests/test_time_series_models.py -v

# Run specific test class
pytest tests/test_time_series_models.py::TestARIMAModel -v

# Run with coverage
pytest tests/test_time_series_models.py --cov=axiom.models.time_series
```

## Performance Considerations

- **ARIMA**: Computational cost increases with model order. Use auto-ARIMA for optimal selection.
- **GARCH**: Maximum likelihood estimation can be slow. Start with GARCH(1,1).
- **EWMA**: Very fast, suitable for real-time applications.

## Best Practices

1. **Data Quality**: Ensure clean, high-quality data without gaps
2. **Stationarity**: Check stationarity before ARIMA fitting
3. **Model Selection**: Use information criteria (AIC, BIC) for model comparison
4. **Validation**: Always validate forecasts on out-of-sample data
5. **Volatility Clustering**: Use GARCH when volatility clustering is detected
6. **Real-time**: Use EWMA for low-latency trading systems

## References

- Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity
- RiskMetrics Technical Document (1996). J.P. Morgan/Reuters

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or contributions, please open an issue on GitHub.