"""
Quick verification script for Portfolio Optimization Module
"""

import numpy as np
import pandas as pd
import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from axiom.models.portfolio import (
            PortfolioOptimizer,
            OptimizationMethod,
            AssetAllocator,
            AllocationStrategy,
            markowitz_optimization
        )
        from axiom.models.risk.var_models import VaRCalculator, VaRMethod
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_optimization():
    """Test basic portfolio optimization."""
    print("\nTesting basic optimization...")
    try:
        from axiom.models.portfolio import PortfolioOptimizer, OptimizationMethod
        
        # Generate sample data
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 3) * 0.01 + 0.0002,
            columns=['Asset1', 'Asset2', 'Asset3']
        )
        
        # Optimize
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        result = optimizer.optimize(returns, method=OptimizationMethod.MAX_SHARPE)
        
        assert result.success, "Optimization failed"
        assert len(result.weights) == 3, "Wrong number of weights"
        assert abs(sum(result.weights) - 1.0) < 0.01, "Weights don't sum to 1"
        assert result.metrics.sharpe_ratio is not None, "Missing Sharpe ratio"
        
        print(f"✓ Max Sharpe optimization successful")
        print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
        print(f"  Return: {result.metrics.expected_return*100:.2f}%")
        print(f"  Volatility: {result.metrics.volatility*100:.2f}%")
        return True
    except Exception as e:
        print(f"✗ Optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_efficient_frontier():
    """Test efficient frontier generation."""
    print("\nTesting efficient frontier...")
    try:
        from axiom.models.portfolio import PortfolioOptimizer
        
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 3) * 0.01 + 0.0002,
            columns=['Asset1', 'Asset2', 'Asset3']
        )
        
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        frontier = optimizer.calculate_efficient_frontier(returns, n_points=20)
        
        assert len(frontier.returns) > 0, "No frontier points generated"
        assert len(frontier.returns) == len(frontier.risks), "Mismatched frontier data"
        
        print(f"✓ Efficient frontier generated with {len(frontier.returns)} points")
        return True
    except Exception as e:
        print(f"✗ Efficient frontier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_asset_allocation():
    """Test asset allocation strategies."""
    print("\nTesting asset allocation...")
    try:
        from axiom.models.portfolio import AssetAllocator, AllocationStrategy
        
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 4) * 0.01 + 0.0002,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        )
        
        allocator = AssetAllocator(risk_free_rate=0.02)
        
        # Test equal weight
        result = allocator.allocate(returns, strategy=AllocationStrategy.EQUAL_WEIGHT)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01, "Weights don't sum to 1"
        
        print(f"✓ Equal weight allocation successful")
        
        # Test risk parity
        result = allocator.allocate(returns, strategy=AllocationStrategy.RISK_PARITY)
        print(f"✓ Risk parity allocation successful")
        
        return True
    except Exception as e:
        print(f"✗ Asset allocation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_var_integration():
    """Test VaR model integration."""
    print("\nTesting VaR integration...")
    try:
        from axiom.models.portfolio import PortfolioOptimizer, OptimizationMethod
        from axiom.models.risk.var_models import VaRCalculator, VaRMethod
        
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 3) * 0.01 + 0.0002,
            columns=['Asset1', 'Asset2', 'Asset3']
        )
        
        # Optimize portfolio
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        result = optimizer.optimize(returns, method=OptimizationMethod.MAX_SHARPE)
        
        # Calculate portfolio returns
        weights = np.array([result.weights[i] for i in range(len(result.weights))])
        portfolio_returns = returns.values @ weights
        
        # Calculate VaR
        calculator = VaRCalculator()
        var_result = calculator.calculate_var(
            portfolio_value=1000000,
            returns=portfolio_returns,
            method=VaRMethod.HISTORICAL
        )
        
        assert var_result.var_amount > 0, "Invalid VaR amount"
        assert var_result.expected_shortfall is not None, "Missing CVaR"
        
        print(f"✓ VaR integration successful")
        print(f"  Portfolio Value: $1,000,000")
        print(f"  VaR (95%, 1d): ${var_result.var_amount:,.2f}")
        print(f"  CVaR: ${var_result.expected_shortfall:,.2f}")
        
        return True
    except Exception as e:
        print(f"✗ VaR integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_metrics():
    """Test performance metrics calculation."""
    print("\nTesting performance metrics...")
    try:
        from axiom.models.portfolio import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
            calculate_max_drawdown
        )
        
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0002
        
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        max_dd = calculate_max_drawdown(returns)
        
        assert not np.isnan(sharpe), "Invalid Sharpe ratio"
        assert not np.isnan(sortino), "Invalid Sortino ratio"
        assert max_dd <= 0, "Invalid max drawdown"
        
        print(f"✓ Performance metrics calculated")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Sortino Ratio: {sortino:.3f}")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
        
        return True
    except Exception as e:
        print(f"✗ Performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION MODULE VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Basic Optimization", test_basic_optimization),
        ("Efficient Frontier", test_efficient_frontier),
        ("Asset Allocation", test_asset_allocation),
        ("VaR Integration", test_var_integration),
        ("Performance Metrics", test_performance_metrics),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All verification tests passed!")
        print("\nPortfolio Optimization Module is ready for production use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())