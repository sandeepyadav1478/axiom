# Portfolio Optimization - Complete Reference

**Modern Portfolio Theory implementation for institutional investors**

## Overview

Comprehensive portfolio optimization with 6 methods and 8 allocation strategies.

### Optimization Methods

| Method | Objective | Performance | Use Case |
|--------|-----------|-------------|----------|
| **Max Sharpe** | Maximize risk-adjusted return | <50ms | General optimization |
| **Min Volatility** | Minimize portfolio risk | <30ms | Conservative portfolios |
| **Efficient Return** | Target specific return | <60ms | Return constraints |
| **Risk Parity** | Equal risk contribution | <80ms | Diversification |
| **Max Return** | Maximize expected return | <40ms | Aggressive strategy |
| **Min CVaR** | Minimize tail risk | <100ms | Risk-averse |

### Allocation Strategies

1. **Equal Weight** - Simple 1/N allocation
2. **Market Cap Weighted** - Proportional to market cap
3. **Risk Parity** - Equal risk contribution
4. **Minimum Variance** - Lowest volatility
5. **Maximum Sharpe** - Best risk-adjusted
6. **Hierarchical Risk Parity (HRP)** - Cluster-based
7. **Black-Litterman** - Market equilibrium + views
8. **VaR-Constrained** - Risk budget allocation

## Mathematical Framework

### Mean-Variance Optimization

**Portfolio Return:**
```
μₚ = wᵀμ
```

**Portfolio Variance:**
```
σₚ² = wᵀΣw
```

**Sharpe Ratio:**
```
SR = (μₚ - rₓ) / σₚ
```

**Optimization Problem:**
```
max  (wᵀμ - rₓ) / √(wᵀΣw)
s.t. wᵀ1 = 1
     w ≥ 0 (long-only)
```

### Risk Parity

**Equal Risk Contribution:**
```
wᵢ × (Σw)ᵢ / σₚ = constant ∀i

Minimize: Σᵢ Σⱼ (RCᵢ - RCⱼ)²
where RCᵢ = risk contribution of asset i
```

### Black-Litterman

**Posterior Returns:**
```
E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹Π + PᵀΩ⁻¹Q]

where:
Π = implied equilibrium returns
P = view matrix
Q = view returns
Ω = view uncertainty
τ = scalar uncertainty parameter
```

### Hierarchical Risk Parity

1. Compute distance matrix from correlation
2. Hierarchical clustering (single linkage)
3. Quasi-diagonalization
4. Recursive bisection with inverse variance

## Quick Start

```python
from axiom.models.portfolio import (
    PortfolioOptimizer,
    OptimizationMethod,
    AssetAllocator,
    AllocationStrategy
)
import pandas as pd

# Load returns
returns = pd.read_csv('returns.csv', index_col=0)

# Create optimizer
optimizer = PortfolioOptimizer(risk_free_rate=0.03)

# Optimize for maximum Sharpe ratio
result = optimizer.optimize(
    returns,
    method=OptimizationMethod.MAX_SHARPE
)

print(f"Expected Return: {result.metrics.expected_return*100:.2f}%")
print(f"Volatility: {result.metrics.volatility*100:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
print(f"Weights: {result.get_weights_dict()}")

# Generate efficient frontier
frontier = optimizer.calculate_efficient_frontier(
    returns,
    n_points=100
)

max_sharpe_portfolio = frontier.get_max_sharpe_portfolio()
min_vol_portfolio = frontier.get_min_volatility_portfolio()
```

## Asset Allocation

```python
from axiom.models.portfolio import AssetAllocator, AssetClass

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

# Black-Litterman with views
market_caps = {'AAPL': 3000e9, 'MSFT': 2800e9, 'GOOGL': 1800e9}
views = {'AAPL': 0.15, 'MSFT': 0.12}  # Expected returns
view_confidences = {'AAPL': 0.8, 'MSFT': 0.7}

bl_result = allocator.allocate(
    returns,
    strategy=AllocationStrategy.BLACK_LITTERMAN,
    market_caps=market_caps,
    views=views,
    view_confidences=view_confidences
)
```

## Performance Metrics

```python
# Comprehensive metrics
metrics = result.metrics

print(f"Expected Return: {metrics.expected_return*100:.2f}%")
print(f"Volatility: {metrics.volatility*100:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
print(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
print(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
print(f"VaR (95%): {metrics.var_95*100:.2f}%")
print(f"CVaR (95%): {metrics.cvar_95*100:.2f}%")

# Benchmark-relative metrics
if metrics.beta:
    print(f"Beta: {metrics.beta:.3f}")
    print(f"Alpha: {metrics.alpha*100:.2f}%")
    print(f"Information Ratio: {metrics.information_ratio:.3f}")
    print(f"Treynor Ratio: {metrics.treynor_ratio:.3f}")
```

## Configuration

```python
PORTFOLIO_CONFIG = {
    # Optimization
    "default_risk_free_rate": 0.03,
    "periods_per_year": 252,  # Daily returns
    "optimization_method": "max_sharpe",
    
    # Constraints
    "long_only": True,
    "fully_invested": True,
    "min_weight": 0.0,
    "max_weight": 1.0,
    "sector_limits": {},
    
    # Efficient Frontier
    "frontier_points": 100,
    "target_return_range": "auto",
    
    # Risk Parity
    "risk_parity_max_iter": 1000,
    "risk_parity_tolerance": 1e-6,
    
    # Black-Litterman
    "bl_tau": 0.05,  # Uncertainty in equilibrium
    "bl_confidence_method": "idzorek",
    
    # Rebalancing
    "rebalance_threshold": 0.05,  # 5% drift
    "transaction_costs": 0.001,  # 10 bps
    
    # Performance
    "cache_covariance": True,
    "use_shrinkage": False
}
```

## Integration with VaR

```python
from axiom.models.risk import VaRCalculator, VaRMethod

# Optimize portfolio
opt_result = optimizer.optimize(returns, method=OptimizationMethod.MAX_SHARPE)

# Calculate portfolio returns
portfolio_returns = returns.values @ opt_result.weights

# Calculate VaR
var_calc = VaRCalculator()
var_result = var_calc.calculate_var(
    portfolio_value=1_000_000,
    returns=portfolio_returns,
    method=VaRMethod.HISTORICAL,
    confidence_level=0.95
)

print(f"Portfolio VaR: ${var_result.var_amount:,.2f}")

# VaR-constrained optimization
var_constrained = allocator.allocate(
    returns,
    strategy=AllocationStrategy.VAR_CONSTRAINED,
    var_budget=0.05  # 5% daily VaR limit
)
```

## Rebalancing

```python
# Check rebalancing needs
current_weights = np.array([0.35, 0.25, 0.20, 0.15, 0.05])
target_weights = result.weights

rebalance_needed = allocator.check_rebalancing_needed(
    current_weights,
    target_weights,
    threshold=0.05  # 5% drift threshold
)

if rebalance_needed:
    trades = allocator.generate_rebalancing_trades(
        current_weights,
        target_weights,
        portfolio_value=1_000_000,
        transaction_costs=0.001
    )
    
    print("Rebalancing Trades:")
    for asset, trade in trades.items():
        print(f"{asset}: {trade['shares']} shares (${trade['value']:,.2f})")
```

## Performance Benchmarks

| Operation | Assets | Execution Time | Status |
|-----------|--------|----------------|--------|
| Max Sharpe | 10 | <50ms | ✅ |
| Efficient Frontier | 10 × 100 points | <2s | ✅ |
| Risk Parity | 20 | <80ms | ✅ |
| Black-Litterman | 15 | <100ms | ✅ |
| HRP | 30 | <150ms | ✅ |

## References

- Markowitz, H. (1952). "Portfolio Selection"
- Black, F. & Litterman, R. (1992). "Global Portfolio Optimization"
- Lopez de Prado, M. (2016). "Building Diversified Portfolios"
- Sharpe, W. (1994). "The Sharpe Ratio"

---

**Last Updated**: 2025-10-23  
**Version**: 1.0.0