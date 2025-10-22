"""
Portfolio Optimization Demo

Comprehensive demonstration of portfolio optimization capabilities:
1. Markowitz Mean-Variance Optimization
2. Efficient Frontier Generation
3. Portfolio Performance Metrics
4. Asset Allocation Strategies
5. Integration with VaR Models
6. Rebalancing Strategies

Production-ready examples for quantitative traders and portfolio managers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from axiom.models.portfolio.optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    markowitz_optimization,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown
)

from axiom.models.portfolio.allocation import (
    AssetAllocator,
    AllocationStrategy,
    AssetClass,
    equal_weight_allocation,
    risk_parity_allocation
)

from axiom.models.risk.var_models import (
    VaRCalculator,
    VaRMethod,
    calculate_portfolio_var
)


def generate_sample_data(n_periods=252, n_assets=5, seed=42):
    """
    Generate realistic sample return data for demonstration.
    
    Args:
        n_periods: Number of time periods (default: 252 trading days)
        n_assets: Number of assets
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with returns and asset metadata
    """
    np.random.seed(seed)
    
    # Define realistic assets
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM'][:n_assets]
    
    # Expected annual returns (realistic for tech/finance stocks)
    annual_returns = np.array([0.15, 0.12, 0.18, 0.14, 0.10])[:n_assets]
    daily_returns = annual_returns / 252
    
    # Annual volatilities
    annual_vols = np.array([0.25, 0.22, 0.28, 0.30, 0.18])[:n_assets]
    daily_vols = annual_vols / np.sqrt(252)
    
    # Correlation matrix (realistic correlations)
    if n_assets == 5:
        correlation = np.array([
            [1.00, 0.70, 0.65, 0.60, 0.30],
            [0.70, 1.00, 0.68, 0.62, 0.32],
            [0.65, 0.68, 1.00, 0.64, 0.28],
            [0.60, 0.62, 0.64, 1.00, 0.25],
            [0.30, 0.32, 0.28, 0.25, 1.00]
        ])
    else:
        correlation = np.eye(n_assets)
    
    # Generate correlated returns
    cov_matrix = np.outer(daily_vols, daily_vols) * correlation
    returns = np.random.multivariate_normal(daily_returns, cov_matrix, n_periods)
    
    # Create DataFrame
    df = pd.DataFrame(returns, columns=assets)
    
    # Add market caps (in billions)
    market_caps = {
        'AAPL': 3000, 'MSFT': 2800, 'GOOGL': 1800,
        'AMZN': 1600, 'JPM': 500
    }
    
    return df, {k: v for k, v in market_caps.items() if k in assets}


def demo_basic_optimization():
    """Demonstrate basic portfolio optimization."""
    print("=" * 80)
    print("1. BASIC PORTFOLIO OPTIMIZATION")
    print("=" * 80)
    
    # Generate sample data
    returns, market_caps = generate_sample_data()
    
    print(f"\nPortfolio: {list(returns.columns)}")
    print(f"Data period: {len(returns)} trading days (~1 year)")
    
    # Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.03, periods_per_year=252)
    
    # 1. Maximum Sharpe Ratio
    print("\n" + "-" * 40)
    print("Maximum Sharpe Ratio Portfolio")
    print("-" * 40)
    
    max_sharpe = optimizer.optimize(
        returns,
        method=OptimizationMethod.MAX_SHARPE
    )
    
    print(f"Status: {'✓ Success' if max_sharpe.success else '✗ Failed'}")
    print(f"Expected Return: {max_sharpe.metrics.expected_return*100:.2f}%")
    print(f"Volatility: {max_sharpe.metrics.volatility*100:.2f}%")
    print(f"Sharpe Ratio: {max_sharpe.metrics.sharpe_ratio:.3f}")
    print(f"\nOptimal Weights:")
    for asset, weight in max_sharpe.get_weights_dict().items():
        if weight > 0.01:  # Show weights > 1%
            print(f"  {asset}: {weight*100:.2f}%")
    
    # 2. Minimum Volatility
    print("\n" + "-" * 40)
    print("Minimum Volatility Portfolio")
    print("-" * 40)
    
    min_vol = optimizer.optimize(
        returns,
        method=OptimizationMethod.MIN_VOLATILITY
    )
    
    print(f"Expected Return: {min_vol.metrics.expected_return*100:.2f}%")
    print(f"Volatility: {min_vol.metrics.volatility*100:.2f}%")
    print(f"Sharpe Ratio: {min_vol.metrics.sharpe_ratio:.3f}")
    
    # 3. Risk Parity
    print("\n" + "-" * 40)
    print("Risk Parity Portfolio")
    print("-" * 40)
    
    risk_parity = optimizer.optimize(
        returns,
        method=OptimizationMethod.RISK_PARITY
    )
    
    print(f"Expected Return: {risk_parity.metrics.expected_return*100:.2f}%")
    print(f"Volatility: {risk_parity.metrics.volatility*100:.2f}%")
    print(f"Sharpe Ratio: {risk_parity.metrics.sharpe_ratio:.3f}")
    
    return returns, optimizer


def demo_efficient_frontier(returns, optimizer):
    """Demonstrate efficient frontier generation."""
    print("\n" + "=" * 80)
    print("2. EFFICIENT FRONTIER")
    print("=" * 80)
    
    print("\nGenerating efficient frontier with 100 portfolios...")
    
    frontier = optimizer.calculate_efficient_frontier(
        returns,
        n_points=100
    )
    
    print(f"Generated {len(frontier.returns)} efficient portfolios")
    print(f"Return range: {frontier.returns.min()*100:.2f}% to {frontier.returns.max()*100:.2f}%")
    print(f"Risk range: {frontier.risks.min()*100:.2f}% to {frontier.risks.max()*100:.2f}%")
    
    # Get special portfolios
    max_sharpe = frontier.get_max_sharpe_portfolio()
    min_vol = frontier.get_min_volatility_portfolio()
    
    print("\n" + "-" * 40)
    print("Maximum Sharpe Portfolio (from frontier)")
    print("-" * 40)
    print(f"Expected Return: {max_sharpe.metrics.expected_return*100:.2f}%")
    print(f"Volatility: {max_sharpe.metrics.volatility*100:.2f}%")
    print(f"Sharpe Ratio: {max_sharpe.metrics.sharpe_ratio:.3f}")
    
    print("\n" + "-" * 40)
    print("Minimum Volatility Portfolio (from frontier)")
    print("-" * 40)
    print(f"Expected Return: {min_vol.metrics.expected_return*100:.2f}%")
    print(f"Volatility: {min_vol.metrics.volatility*100:.2f}%")
    print(f"Sharpe Ratio: {min_vol.metrics.sharpe_ratio:.3f}")
    
    return frontier


def demo_performance_metrics(returns):
    """Demonstrate portfolio performance metrics calculation."""
    print("\n" + "=" * 80)
    print("3. PORTFOLIO PERFORMANCE METRICS")
    print("=" * 80)
    
    # Equal weight portfolio
    weights = np.ones(len(returns.columns)) / len(returns.columns)
    portfolio_returns = returns.values @ weights
    
    print("\nEqual-Weighted Portfolio Metrics:")
    print("-" * 40)
    
    # Calculate metrics
    sharpe = calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.03)
    sortino = calculate_sortino_ratio(portfolio_returns, risk_free_rate=0.03)
    max_dd = calculate_max_drawdown(portfolio_returns)
    
    annual_return = np.mean(portfolio_returns) * 252
    annual_vol = np.std(portfolio_returns) * np.sqrt(252)
    
    print(f"Annual Return: {annual_return*100:.2f}%")
    print(f"Annual Volatility: {annual_vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Sortino Ratio: {sortino:.3f}")
    print(f"Maximum Drawdown: {max_dd*100:.2f}%")
    
    # Calculate VaR
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
    
    print(f"95% VaR (daily): {var_95*100:.2f}%")
    print(f"95% CVaR (daily): {cvar_95*100:.2f}%")
    
    # Calmar ratio
    calmar = -annual_return / max_dd if max_dd < 0 else 0
    print(f"Calmar Ratio: {calmar:.3f}")


def demo_asset_allocation(returns, market_caps):
    """Demonstrate various asset allocation strategies."""
    print("\n" + "=" * 80)
    print("4. ASSET ALLOCATION STRATEGIES")
    print("=" * 80)
    
    # Define asset classes
    asset_classes = [
        AssetClass(
            name="Technology",
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            strategic_weight=0.7,
            min_weight=0.5,
            max_weight=0.8
        ),
        AssetClass(
            name="Financials",
            symbols=['JPM'],
            strategic_weight=0.3,
            min_weight=0.2,
            max_weight=0.5
        )
    ]
    
    allocator = AssetAllocator(
        asset_classes=asset_classes,
        risk_free_rate=0.03,
        rebalancing_threshold=0.05
    )
    
    strategies = [
        AllocationStrategy.EQUAL_WEIGHT,
        AllocationStrategy.RISK_PARITY,
        AllocationStrategy.MAX_SHARPE,
        AllocationStrategy.MIN_VARIANCE,
        AllocationStrategy.MARKET_CAP,
        AllocationStrategy.HIERARCHICAL_RISK_PARITY
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'-' * 40}")
        print(f"{strategy.value.upper().replace('_', ' ')}")
        print(f"{'-' * 40}")
        
        try:
            if strategy == AllocationStrategy.MARKET_CAP:
                result = allocator.allocate(
                    returns,
                    strategy=strategy,
                    market_caps=market_caps
                )
            else:
                result = allocator.allocate(returns, strategy=strategy)
            
            results[strategy.value] = result
            
            print(f"Expected Return: {result.metrics.expected_return*100:.2f}%")
            print(f"Volatility: {result.metrics.volatility*100:.2f}%")
            print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
            
            print("\nTop 3 Holdings:")
            top_holdings = result.get_top_holdings(3)
            for asset, weight in top_holdings.items():
                print(f"  {asset}: {weight*100:.2f}%")
            
            if result.asset_class_weights:
                print("\nAsset Class Allocation:")
                for ac, weight in result.asset_class_weights.items():
                    print(f"  {ac}: {weight*100:.2f}%")
        
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return results, allocator


def demo_black_litterman(returns, market_caps):
    """Demonstrate Black-Litterman model."""
    print("\n" + "=" * 80)
    print("5. BLACK-LITTERMAN MODEL")
    print("=" * 80)
    
    allocator = AssetAllocator(risk_free_rate=0.03)
    
    # Define views
    views = {
        'AAPL': 0.20,  # Expect 20% return for AAPL
        'MSFT': 0.15   # Expect 15% return for MSFT
    }
    
    view_confidences = {
        'AAPL': 0.7,  # 70% confident in AAPL view
        'MSFT': 0.8   # 80% confident in MSFT view
    }
    
    print("\nInvestor Views:")
    for asset, expected_return in views.items():
        confidence = view_confidences[asset]
        print(f"  {asset}: {expected_return*100:.0f}% return (confidence: {confidence*100:.0f}%)")
    
    # Market equilibrium (market cap weighted)
    print("\n" + "-" * 40)
    print("Market Equilibrium Portfolio")
    print("-" * 40)
    
    market_result = allocator.allocate(
        returns,
        strategy=AllocationStrategy.MARKET_CAP,
        market_caps=market_caps
    )
    
    print("Weights:")
    for asset, weight in market_result.weights.items():
        print(f"  {asset}: {weight*100:.2f}%")
    
    # Black-Litterman with views
    print("\n" + "-" * 40)
    print("Black-Litterman Portfolio (with views)")
    print("-" * 40)
    
    bl_result = allocator.allocate(
        returns,
        strategy=AllocationStrategy.BLACK_LITTERMAN,
        market_caps=market_caps,
        views=views,
        view_confidences=view_confidences
    )
    
    print("Weights:")
    for asset, weight in bl_result.weights.items():
        print(f"  {asset}: {weight*100:.2f}%")
    
    print("\nWeight Changes vs Market Portfolio:")
    for asset in returns.columns:
        market_weight = market_result.weights[asset]
        bl_weight = bl_result.weights[asset]
        change = (bl_weight - market_weight) * 100
        print(f"  {asset}: {change:+.2f}%")


def demo_var_integration(returns):
    """Demonstrate VaR model integration."""
    print("\n" + "=" * 80)
    print("6. VaR MODEL INTEGRATION")
    print("=" * 80)
    
    # Optimize portfolio
    optimizer = PortfolioOptimizer(risk_free_rate=0.03)
    result = optimizer.optimize(
        returns,
        method=OptimizationMethod.MAX_SHARPE
    )
    
    # Calculate portfolio returns
    weights_array = np.array([result.weights[i] for i in range(len(result.weights))])
    portfolio_returns = returns.values @ weights_array
    
    portfolio_value = 1_000_000  # $1M portfolio
    
    print(f"\nPortfolio Value: ${portfolio_value:,.0f}")
    print(f"Portfolio Strategy: Maximum Sharpe Ratio")
    
    # Calculate VaR using all three methods
    calculator = VaRCalculator(default_confidence=0.95)
    
    var_results = calculator.calculate_all_methods(
        portfolio_value=portfolio_value,
        returns=portfolio_returns,
        confidence_level=0.95,
        time_horizon_days=1,
        num_simulations=10000
    )
    
    print("\n" + "-" * 40)
    print("Value at Risk (95% confidence, 1-day)")
    print("-" * 40)
    
    for method, var_result in var_results.items():
        print(f"\n{method.upper()}:")
        print(f"  VaR: ${var_result.var_amount:,.2f} ({var_result.var_percentage*100:.2f}%)")
        print(f"  Expected Shortfall: ${var_result.expected_shortfall:,.2f}")
    
    # VaR summary
    summary = calculator.get_var_summary(var_results)
    
    print("\n" + "-" * 40)
    print("VaR Summary Statistics")
    print("-" * 40)
    print(f"VaR Range: ${summary['var_range']['min']:,.2f} - ${summary['var_range']['max']:,.2f}")
    print(f"Mean VaR: ${summary['var_range']['mean']:,.2f}")
    print(f"Median VaR: ${summary['var_range']['median']:,.2f}")
    
    # 10-day VaR (regulatory)
    print("\n" + "-" * 40)
    print("Regulatory VaR (99% confidence, 10-day)")
    print("-" * 40)
    
    from axiom.models.risk.var_models import regulatory_var
    
    reg_var = regulatory_var(portfolio_value, portfolio_returns)
    print(f"VaR: ${reg_var.var_amount:,.2f} ({reg_var.var_percentage*100:.2f}%)")
    print(f"Expected Shortfall: ${reg_var.expected_shortfall:,.2f}")
    
    # VaR-constrained allocation
    print("\n" + "-" * 40)
    print("VaR-Constrained Allocation")
    print("-" * 40)
    
    allocator = AssetAllocator()
    var_budget = 0.05  # 5% daily VaR limit
    
    print(f"VaR Budget: {var_budget*100:.2f}%")
    
    var_constrained = allocator.allocate(
        returns,
        strategy=AllocationStrategy.VAR_CONSTRAINED,
        var_budget=var_budget
    )
    
    print("\nOptimal Weights:")
    for asset, weight in var_constrained.weights.items():
        if weight > 0.01:
            print(f"  {asset}: {weight*100:.2f}%")
    
    # Verify VaR constraint
    var_port_returns = returns.values @ np.array([
        var_constrained.weights[asset] for asset in returns.columns
    ])
    
    var_check = calculator.calculate_var(
        portfolio_value=1.0,
        returns=var_port_returns,
        method=VaRMethod.HISTORICAL
    )
    
    print(f"\nActual VaR: {var_check.var_percentage*100:.2f}%")
    print(f"Constraint: {'✓ Satisfied' if var_check.var_percentage <= var_budget else '✗ Violated'}")


def demo_rebalancing():
    """Demonstrate portfolio rebalancing."""
    print("\n" + "=" * 80)
    print("7. PORTFOLIO REBALANCING")
    print("=" * 80)
    
    returns, market_caps = generate_sample_data()
    allocator = AssetAllocator(rebalancing_threshold=0.05)
    
    # Current portfolio (drifted from target)
    current_weights = {
        'AAPL': 0.35,  # Drifted up
        'MSFT': 0.15,  # Drifted down
        'GOOGL': 0.25,
        'AMZN': 0.15,  # Drifted down
        'JPM': 0.10
    }
    
    # Target allocation (risk parity)
    target_result = allocator.allocate(
        returns,
        strategy=AllocationStrategy.RISK_PARITY
    )
    
    print("\nCurrent Portfolio:")
    for asset, weight in current_weights.items():
        print(f"  {asset}: {weight*100:.2f}%")
    
    print("\nTarget Portfolio:")
    for asset, weight in target_result.weights.items():
        print(f"  {asset}: {weight*100:.2f}%")
    
    # Check rebalancing
    result_with_rebal = allocator.allocate(
        returns,
        strategy=AllocationStrategy.RISK_PARITY,
        current_weights=current_weights
    )
    
    print(f"\nRebalancing Required: {'Yes ✓' if result_with_rebal.rebalancing_required else 'No'}")
    print(f"Tracking Error: {result_with_rebal.tracking_error*100:.2f}%")
    
    # Calculate rebalancing trades
    rebalance_info = allocator.rebalance(
        current_weights,
        target_result.weights,
        transaction_cost=0.001  # 0.1% transaction cost
    )
    
    print("\n" + "-" * 40)
    print("Rebalancing Analysis")
    print("-" * 40)
    print(f"Total Turnover: {rebalance_info['turnover']*100:.2f}%")
    print(f"Transaction Cost: {rebalance_info['transaction_cost']*100:.4f}%")
    print(f"Net Benefit: {rebalance_info['net_benefit']*100:.4f}%")
    
    print("\nRequired Trades:")
    for asset, trade in sorted(rebalance_info['trades'].items(), key=lambda x: abs(x[1]), reverse=True):
        action = "BUY" if trade > 0 else "SELL"
        print(f"  {action} {asset}: {abs(trade)*100:.2f}%")


def demo_comparison_table(results):
    """Create comparison table of different strategies."""
    print("\n" + "=" * 80)
    print("8. STRATEGY COMPARISON")
    print("=" * 80)
    
    print("\n{:<25} {:>12} {:>12} {:>12}".format(
        "Strategy", "Return", "Volatility", "Sharpe"
    ))
    print("-" * 65)
    
    for strategy_name, result in results.items():
        if result.metrics:
            print("{:<25} {:>11.2f}% {:>11.2f}% {:>12.3f}".format(
                strategy_name.replace('_', ' ').title(),
                result.metrics.expected_return * 100,
                result.metrics.volatility * 100,
                result.metrics.sharpe_ratio
            ))


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("PORTFOLIO OPTIMIZATION COMPREHENSIVE DEMO")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # 1. Basic Optimization
        returns, optimizer = demo_basic_optimization()
        
        # 2. Efficient Frontier
        frontier = demo_efficient_frontier(returns, optimizer)
        
        # 3. Performance Metrics
        demo_performance_metrics(returns)
        
        # 4. Asset Allocation
        _, market_caps = generate_sample_data()
        allocation_results, allocator = demo_asset_allocation(returns, market_caps)
        
        # 5. Black-Litterman
        demo_black_litterman(returns, market_caps)
        
        # 6. VaR Integration
        demo_var_integration(returns)
        
        # 7. Rebalancing
        demo_rebalancing()
        
        # 8. Comparison
        demo_comparison_table(allocation_results)
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nAll portfolio optimization features demonstrated:")
        print("✓ Markowitz optimization (max Sharpe, min vol, risk parity)")
        print("✓ Efficient frontier generation")
        print("✓ Performance metrics (Sharpe, Sortino, Calmar, drawdown)")
        print("✓ Multiple allocation strategies")
        print("✓ Black-Litterman model with investor views")
        print("✓ VaR model integration")
        print("✓ Portfolio rebalancing")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()