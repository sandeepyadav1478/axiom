"""
Portfolio Optimization Module

Comprehensive portfolio management toolkit:
- Markowitz Mean-Variance Optimization
- Efficient Frontier Generation
- Portfolio Performance Metrics (Sharpe, Sortino, Calmar)
- Asset Allocation Strategies (Risk Parity, Black-Litterman, HRP)
- VaR-Integrated Risk Management
- Portfolio Rebalancing

Production-ready for quantitative traders and portfolio managers.
"""

from .optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    ConstraintType,
    PortfolioMetrics,
    OptimizationResult,
    EfficientFrontier,
    markowitz_optimization,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown
)

from .allocation import (
    AssetAllocator,
    AllocationStrategy,
    AssetClass,
    AllocationResult,
    equal_weight_allocation,
    risk_parity_allocation
)

__all__ = [
    # Optimization
    "PortfolioOptimizer",
    "OptimizationMethod",
    "ConstraintType",
    "PortfolioMetrics",
    "OptimizationResult",
    "EfficientFrontier",
    "markowitz_optimization",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    
    # Allocation
    "AssetAllocator",
    "AllocationStrategy",
    "AssetClass",
    "AllocationResult",
    "equal_weight_allocation",
    "risk_parity_allocation",
]

__version__ = "1.0.0"