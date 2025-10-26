# External Libraries Integration

This module provides production-grade adapters and wrappers for external quantitative finance libraries, enhancing Axiom's capabilities while reducing custom code maintenance.

## 📚 Available Libraries

### 1. QuantLib - Fixed Income & Derivatives
**Installation**: `pip install QuantLib-Python`

**Features**:
- Industry-standard bond pricing (all bond types)
- 30+ day count conventions
- Comprehensive calendar support
- Yield curve construction
- Schedule generation

**Example**:
```python
from axiom.integrations.external_libs import QuantLibBondPricer, BondSpecification
from datetime import date

# Create bond pricer
pricer = QuantLibBondPricer()

# Define bond
bond = BondSpecification(
    face_value=1000,
    coupon_rate=0.05,  # 5% coupon
    issue_date=date(2020, 1, 1),
    maturity_date=date(2030, 1, 1)
)

# Price bond
result = pricer.price_bond(
    bond,
    settlement_date=date(2024, 1, 1),
    yield_rate=0.04
)

print(f"Clean Price: ${result.clean_price:.2f}")
print(f"YTM: {result.ytm:.4%}")
print(f"Duration: {result.duration:.2f}")
```

### 2. PyPortfolioOpt - Portfolio Optimization
**Installation**: `pip install PyPortfolioOpt`

**Features**:
- Mean-variance optimization
- Hierarchical Risk Parity (HRP)
- Black-Litterman model
- Efficient frontier
- Discrete allocation

**Example**:
```python
from axiom.integrations.external_libs import PyPortfolioOptAdapter, OptimizationObjective
import pandas as pd

# Create optimizer
optimizer = PyPortfolioOptAdapter()

# Load price data
prices = pd.DataFrame({
    'AAPL': [...],
    'GOOGL': [...],
    'MSFT': [...]
})

# Optimize for maximum Sharpe ratio
result = optimizer.optimize_portfolio(
    prices,
    objective=OptimizationObjective.MAX_SHARPE
)

print(f"Expected Return: {result.expected_return:.2%}")
print(f"Volatility: {result.volatility:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Weights: {result.weights}")
```

### 3. TA-Lib - Technical Analysis (150+ Indicators)
**Installation**: 
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Linux
# Download and compile from source
pip install TA-Lib
```

**Features**:
- 150+ technical indicators
- Extremely fast (C implementation)
- Industry-standard calculations
- Pattern recognition

**Example**:
```python
from axiom.integrations.external_libs import TALibIndicators
import pandas as pd

# Create indicators calculator
indicators = TALibIndicators()

# Calculate RSI
rsi = indicators.rsi(df['close'], timeperiod=14)

# Calculate MACD
macd, signal, hist = indicators.macd(df['close'])

# Calculate Bollinger Bands
upper, middle, lower = indicators.bollinger_bands(df['close'])

# Add multiple indicators to DataFrame
df = indicators.calculate_multiple_indicators(
    df,
    indicators=['rsi', 'macd', 'bbands', 'atr', 'obv']
)
```

### 4. pandas-ta - Technical Analysis in Pandas (130+ Indicators)
**Installation**: `pip install pandas-ta`

**Features**:
- 130+ indicators (pure Python)
- Pandas DataFrame native
- Custom indicator creation
- Strategy development

**Example**:
```python
from axiom.integrations.external_libs import PandasTAIntegration
import pandas as pd

# Create integration
pta = PandasTAIntegration()

# Add individual indicators
df = pta.add_rsi(df, length=14)
df = pta.add_macd(df)
df = pta.add_bbands(df)

# Or add all common indicators at once
df = pta.add_all_ta_indicators(df, strategy='common')

# Create custom strategy
custom_indicators = [
    {'name': 'rsi', 'length': 14},
    {'name': 'macd', 'fast': 12, 'slow': 26, 'signal': 9},
    {'name': 'bbands', 'length': 20, 'std': 2},
    {'name': 'atr', 'length': 14}
]
df = pta.create_custom_strategy(df, custom_indicators)
```

### 5. arch - ARCH/GARCH Volatility Models
**Installation**: `pip install arch`

**Features**:
- GARCH, EGARCH, TGARCH variants
- Multiple distributions
- Volatility forecasting
- Model diagnostics

**Example**:
```python
from axiom.integrations.external_libs import ArchGARCH, VolatilityModel
import pandas as pd

# Create GARCH model
garch = ArchGARCH()

# Fit GARCH(1,1) model
result = garch.fit_garch(
    returns,
    p=1, q=1,
    model_type=VolatilityModel.GARCH,
    dist=Distribution.STUDENTS_T
)

print(f"AIC: {result.aic:.2f}")
print(f"BIC: {result.bic:.2f}")

# Forecast volatility
forecast = garch.forecast_volatility(result, horizon=10)
print(f"10-day variance forecast: {forecast.variance}")

# Calculate VaR
var = garch.calculate_var(result, confidence_level=0.95)
```

## 🔧 Configuration

### Global Configuration
```python
from axiom.integrations.external_libs import LibraryConfig, set_config

# Create custom configuration
config = LibraryConfig(
    use_quantlib=True,
    use_pypfopt=True,
    use_talib=True,
    prefer_external=True,  # Prefer external over custom
    fallback_to_custom=True,  # Fallback if unavailable
    log_library_usage=True
)

# Set global configuration
set_config(config)
```

### Check Library Availability
```python
from axiom.integrations.external_libs import get_library_availability

# Check which libraries are available
availability = get_library_availability()

print(f"QuantLib: {availability['QuantLib']}")
print(f"PyPortfolioOpt: {availability['PyPortfolioOpt']}")
print(f"TA-Lib: {availability['TA-Lib']}")
print(f"pandas-ta: {availability['pandas-ta']}")
print(f"arch: {availability['arch']}")
```

## 📊 Complete Example: Portfolio Analysis

```python
from axiom.integrations.external_libs import (
    PyPortfolioOptAdapter,
    TALibIndicators,
    PandasTAIntegration,
    ArchGARCH,
    get_library_availability
)
import pandas as pd
import numpy as np

# Check availability
libs = get_library_availability()
print("Available libraries:", {k: v for k, v in libs.items() if v})

# Load data
prices = pd.read_csv('prices.csv', index_col='date', parse_dates=True)

# 1. Add technical indicators
if libs['pandas-ta']:
    pta = PandasTAIntegration()
    prices = pta.add_all_ta_indicators(prices, strategy='common')

# 2. Calculate returns
returns = prices['close'].pct_change().dropna()

# 3. Optimize portfolio
if libs['PyPortfolioOpt']:
    optimizer = PyPortfolioOptAdapter()
    opt_result = optimizer.optimize_portfolio(
        prices[['AAPL', 'GOOGL', 'MSFT']],
        objective=OptimizationObjective.MAX_SHARPE
    )
    print(f"\nOptimal Weights: {opt_result.weights}")
    print(f"Expected Return: {opt_result.expected_return:.2%}")
    print(f"Sharpe Ratio: {opt_result.sharpe_ratio:.2f}")

# 4. Model volatility
if libs['arch']:
    garch = ArchGARCH()
    vol_result = garch.fit_garch(returns, p=1, q=1)
    print(f"\nGARCH AIC: {vol_result.aic:.2f}")
    
    # Forecast
    forecast = garch.forecast_volatility(vol_result, horizon=5)
    print(f"5-day volatility forecast: {np.sqrt(forecast.variance[0]):.4f}")
```

## 🎯 Best Practices

### 1. Always Check Availability
```python
from axiom.integrations.external_libs import check_quantlib_availability

if check_quantlib_availability():
    # Use QuantLib
    pricer = QuantLibBondPricer()
else:
    # Use custom implementation
    pricer = CustomBondPricer()
```

### 2. Handle Errors Gracefully
```python
try:
    result = garch.fit_garch(returns, p=1, q=1)
except Exception as e:
    logger.error(f"GARCH fitting failed: {e}")
    # Fallback to simpler model
    result = estimate_rolling_volatility(returns)
```

### 3. Use Configuration for Flexibility
```python
from axiom.integrations.external_libs import get_config

config = get_config()
if config.use_quantlib and QUANTLIB_AVAILABLE:
    # Use QuantLib
    pass
elif config.fallback_to_custom:
    # Use custom implementation
    pass
```

### 4. Leverage Library-Specific Features
```python
# QuantLib's comprehensive calendars
pricer = QuantLibBondPricer()
# Automatically handles holidays, business days, etc.

# PyPortfolioOpt's discrete allocation
optimizer = PyPortfolioOptAdapter()
allocation = optimizer.discrete_allocation(
    weights,
    latest_prices,
    total_portfolio_value=100000
)
# Returns integer shares to buy
```

## 🔍 Comparison: External vs Custom

### Bond Pricing
- **QuantLib**: 30+ day count conventions, all bond types, production-tested
- **Custom**: 4 day count conventions, basic bond types

### Portfolio Optimization
- **PyPortfolioOpt**: Efficient frontier, HRP, Black-Litterman, discrete allocation
- **Custom**: Basic mean-variance, limited constraints

### Technical Indicators
- **TA-Lib**: 150+ indicators, C implementation (fast), industry-standard
- **pandas-ta**: 130+ indicators, pure Python (easy install)
- **Custom**: 10-20 indicators, Python implementation

### Volatility Models
- **arch**: GARCH variants, multiple distributions, comprehensive diagnostics
- **Custom**: Basic GARCH, normal distribution only

## 📈 Performance Considerations

### TA-Lib vs pandas-ta
- **TA-Lib**: 10-100x faster (C library), but harder to install
- **pandas-ta**: Slower but pure Python (easier installation)

### When to Use Which
- **Production systems**: Use TA-Lib for speed
- **Research/Prototyping**: Use pandas-ta for ease of use
- **Both available**: Use TA-Lib for heavy computations, pandas-ta for custom indicators

## 🚀 Installation Guide

### Quick Install (All Libraries)
```bash
pip install -r requirements.txt
```

### Individual Installation

**QuantLib**:
```bash
pip install QuantLib-Python
```

**PyPortfolioOpt**:
```bash
pip install PyPortfolioOpt
```

**TA-Lib** (requires system dependencies):
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Ubuntu/Debian
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

**pandas-ta** (no dependencies):
```bash
pip install pandas-ta
```

**arch**:
```bash
pip install arch
```

## 🧪 Testing

Run integration tests:
```bash
pytest tests/test_external_libs.py -v
```

Run performance comparison:
```bash
python tests/test_performance_comparison.py
```

## 📝 Contributing

When adding new library integrations:

1. Create wrapper in `axiom/integrations/external_libs/`
2. Follow the adapter pattern
3. Add availability checking
4. Update `__init__.py` exports
5. Add documentation and examples
6. Add tests

## 🔗 References

- [QuantLib Documentation](https://www.quantlib.org/docs.shtml)
- [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)
- [TA-Lib Documentation](https://mrjbq7.github.io/ta-lib/)
- [pandas-ta Documentation](https://github.com/twopirllc/pandas-ta)
- [arch Documentation](https://arch.readthedocs.io/)

## 📄 License

See LICENSE file in the project root.