# Portfolio Optimization Module

Production-ready portfolio optimization implementation for quantitative traders and institutional investors.

## Overview

This module provides comprehensive portfolio optimization and asset allocation tools based on Modern Portfolio Theory (MPT) and advanced quantitative finance techniques.

## Features

### 1. Portfolio Optimization (`optimization.py`)

#### Markowitz Mean-Variance Optimization
- **Maximum Sharpe Ratio**: Optimal risk-adjusted returns
- **Minimum Volatility**: Lowest risk portfolio
- **Efficient Return**: Target return with minimum risk
- **Risk Parity**: Equal risk contribution from each asset
- **CVaR Minimization**: Downside risk optimization

#### Efficient Frontier
- Generate complete efficient frontier with customizable points
- Identify maximum Sharpe ratio portfolio
- Find minimum volatility portfolio
- Export to DataFrame for visualization

#### Performance Metrics
- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
- **Risk Measures**: Volatility, VaR, CVaR, Maximum Drawdown
- **Benchmark-Relative**: Alpha, Beta, Information Ratio, Treynor Ratio
- **Comprehensive Analysis**: All metrics calculated in one pass

### 2. Asset Allocation (`allocation.py`)

#### Strategic Allocation Strategies
- **Equal Weight**: Simple 1/N allocation
- **Market Cap Weighted**: Proportional to market capitalization
- **Risk Parity**: Equal risk contribution
- **Minimum Variance**: Lowest portfolio volatility
- **Maximum Sharpe**: Best risk-adjusted returns
- **Hierarchical Risk Parity (HRP)**: Cluster-based diversification
- **Black-Litterman**: Market equilibrium + investor views
- **VaR-Constrained**: Risk budget aware allocation

#### Asset Class Framework
- Define asset classes with strategic weights
- Set min/max bounds per asset class
- Aggregate security-level to asset class weights
- Strategic vs tactical allocation tracking

#### Rebalancing
- Automatic rebalancing trigger detection
- Transaction cost modeling
- Turnover analysis
- Net benefit calculation

## Usage Examples

### Basic Portfolio Optimization

```python
from axiom.models.portfolio import PortfolioOptimizer, OptimizationMethod
import pandas as pd

# Load historical returns
returns = pd.read_csv('returns.csv', index_col=0)

# Create optimizer
optimizer = PortfolioOptimizer(risk_free_rate=0.03, periods_per_year=252)

# Optimize for maximum Sharpe ratio
result = optimizer.optimize(
    returns,
    method=OptimizationMethod.MAX_SHARPE
)

print(f"Expected Return: {result.metrics.expected_return*100:.2f}%")
print(f"Volatility: {result.metrics.volatility*100:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
print(f"Weights: {result.get_weights_dict()}")
```

### Efficient Frontier

```python
# Generate efficient frontier
frontier = optimizer.calculate_efficient_frontier(
    returns,
    n_points=100
)

# Get optimal portfolios
max_sharpe = frontier.get_max_sharpe_portfolio()
min_vol = frontier.get_min_volatility_portfolio()

# Export for plotting
df = frontier.to_dataframe()
```

### Asset Allocation

```python
from axiom.models.portfolio import AssetAllocator, AllocationStrategy, AssetClass

# Define asset classes
asset_classes = [
    AssetClass(
        name="Equities",
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        strategic_weight=0.6,
        min_weight=0.4,
        max_weight=0.8
    ),
    AssetClass(
        name="Bonds",
        symbols=['AGG', 'TLT'],
        strategic_weight=0.4,
        min_weight=0.2,
        max_weight=0.6
    )
]

# Create allocator
allocator = AssetAllocator(
    asset_classes=asset_classes,
    risk_free_rate=0.03
)

# Risk parity allocation
result = allocator.allocate(
    returns,
    strategy=AllocationStrategy.RISK_PARITY
)

print(f"Asset Weights: {result.weights}")
print(f"Asset Class Weights: {result.asset_class_weights}")
```

### Black-Litterman Model

```python
# Market equilibrium
market_caps = {
    'AAPL': 3000e9,
    'MSFT': 2800e9,
    'GOOGL': 1800e9
}

# Investor views
views = {
    'AAPL': 0.15,  # Expect 15% return
    'MSFT': 0.12   # Expect 12% return
}

view_confidences = {
    'AAPL': 0.8,  # 80% confident
    'MSFT': 0.7   # 70% confident
}

# Black-Litterman allocation
result = allocator.allocate(
    returns,
    strategy=AllocationStrategy.BLACK_LITTERMAN,
    market_caps=market_caps,
    views=views,
    view_confidences=view_confidences
)
```

### VaR Integration

```python
from axiom.models.risk.var_models import VaRCalculator, VaRMethod

# Optimize portfolio
opt_result = optimizer.optimize(returns, method=OptimizationMethod.MAX_SHARPE)

# Calculate portfolio returns
weights = np.array([opt_result.weights[i] for i in range(len(opt_result.weights))])
portfolio_returns = returns.values @ weights

# Calculate VaR
calculator = VaRCalculator()
var_result = calculator.calculate_var(
    portfolio_value=1_000_000,
    returns=portfolio_returns,
    method=VaRMethod.HISTORICAL,
    confidence_level=0.95
)

print(f"VaR (95%, 1d): ${var_result.var_amount:,.2f}")
print(f"CVaR: ${var_result.expected_shortfall:,.2f}")
```

### VaR-Constrained Allocation

```python
# Allocate with VaR budget constraint
var_budget = 0.05  # 5% daily VaR limit

result = allocator.allocate(
    returns,
    strategy=AllocationStrategy.VAR_CONSTRAINED,
    var_budget=var_budget
)
```

## Key Classes

### PortfolioOptimizer
Main optimization engine supporting multiple objectives and constraints.

**Methods:**
- `optimize()`: Optimize portfolio weights
- `calculate_efficient_frontier()`: Generate efficient frontier
- `calculate_metrics()`: Compute performance metrics

### AssetAllocator
Multi-strategy asset allocation engine.

**Methods:**
- `allocate()`: Calculate optimal allocation
- `rebalance()`: Compute rebalancing trades

### OptimizationResult
Container for optimization results with metrics and weights.

**Attributes:**
- `weights`: Optimal portfolio weights
- `metrics`: Performance metrics
- `success`: Optimization status
- `method`: Optimization method used

### PortfolioMetrics
Comprehensive portfolio performance metrics.

**Attributes:**
- `expected_return`: Annualized return
- `volatility`: Annualized volatility
- `sharpe_ratio`: Risk-adjusted return
- `sortino_ratio`: Downside risk-adjusted return
- `max_drawdown`: Maximum peak-to-trough decline
- `var_95`: 95% Value at Risk
- `cvar_95`: 95% Conditional VaR
- `alpha`, `beta`: CAPM metrics (if benchmark provided)

## Performance Considerations

### Optimization Speed
- Parametric VaR: Fastest (analytical)
- Historical Simulation: Fast (empirical)
- Monte Carlo: Slower (simulation-based)

### Memory Usage
- Efficient frontier: O(n_points × n_assets)
- Historical returns: O(periods × assets)
- Covariance matrix: O(n_assets²)

### Best Practices
1. Use at least 252 days (1 year) of historical data
2. Annualize returns with appropriate periods_per_year
3. Consider transaction costs in rebalancing
4. Validate optimization success flag
5. Use VaR constraints for risk-controlled strategies

## Integration with VaR Models

The portfolio optimization module seamlessly integrates with the VaR models:

```python
# Portfolio optimization -> VaR calculation
opt_result = optimizer.optimize(returns, method=OptimizationMethod.MAX_SHARPE)
portfolio_returns = returns.values @ opt_result.weights
var_result = calculator.calculate_var(1_000_000, portfolio_returns)

# VaR-constrained optimization
var_constrained = allocator.allocate(
    returns,
    strategy=AllocationStrategy.VAR_CONSTRAINED,
    var_budget=0.05
)
```

## Mathematical Foundations

### Mean-Variance Optimization
Portfolio return: μₚ = wᵀμ
Portfolio variance: σₚ² = wᵀΣw
Sharpe ratio: (μₚ - rₑ) / σₚ

### Risk Parity
Equal risk contribution: wᵢ × (Σw)ᵢ / σₚ = constant ∀i

### Black-Litterman
Posterior returns: E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹Π + PᵀΩ⁻¹Q]

### Hierarchical Risk Parity
1. Compute distance matrix from correlation
2. Hierarchical clustering (single linkage)
3. Quasi-diagonalization
4. Recursive bisection with inverse variance weighting

## Testing

Comprehensive test suite in `tests/test_portfolio_optimization.py`:
- 50+ test cases covering all major functionality
- Edge cases and error handling
- Integration tests with VaR models
- Performance benchmarking

Run tests:
```bash
pytest tests/test_portfolio_optimization.py -v
```

## Demos

Complete demonstration in `demos/demo_portfolio_optimization.py`:
- Basic optimization examples
- Efficient frontier visualization
- All allocation strategies
- VaR integration
- Rebalancing workflows

Run demo:
```bash
python demos/demo_portfolio_optimization.py
```

## Dependencies

- numpy: Numerical computing
- pandas: Data manipulation
- scipy: Optimization and statistics
- Standard library: dataclasses, datetime, enum

## References

1. Markowitz, H. (1952). "Portfolio Selection". Journal of Finance.
2. Black, F., & Litterman, R. (1992). "Global Portfolio Optimization". Financial Analysts Journal.
3. Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample". Journal of Portfolio Management.
4. Sharpe, W. (1994). "The Sharpe Ratio". Journal of Portfolio Management.

## License

Part of the Axiom quantitative finance framework.

## Support

For questions or issues, please refer to the main Axiom documentation or contact the development team.