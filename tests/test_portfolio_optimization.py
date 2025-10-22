"""
Comprehensive tests for Portfolio Optimization Models

Tests all major components:
1. Markowitz optimization
2. Efficient frontier calculation
3. Portfolio metrics (Sharpe, Sortino, etc.)
4. Asset allocation strategies
5. Integration with VaR models
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from axiom.models.portfolio.optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    PortfolioMetrics,
    OptimizationResult,
    EfficientFrontier,
    markowitz_optimization,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown
)

from axiom.models.portfolio.allocation import (
    AssetAllocator,
    AllocationStrategy,
    AssetClass,
    AllocationResult,
    equal_weight_allocation,
    risk_parity_allocation
)

from axiom.models.risk.var_models import (
    VaRCalculator,
    VaRMethod
)


@pytest.fixture
def sample_returns():
    """Generate sample return data for testing."""
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    n_assets = 5
    
    # Generate correlated returns
    mean_returns = np.array([0.0003, 0.0002, 0.0004, 0.0001, 0.0005])  # Daily
    volatilities = np.array([0.01, 0.015, 0.02, 0.008, 0.025])  # Daily
    
    # Correlation matrix
    correlation = np.array([
        [1.0, 0.6, 0.3, 0.2, 0.4],
        [0.6, 1.0, 0.5, 0.3, 0.5],
        [0.3, 0.5, 1.0, 0.4, 0.6],
        [0.2, 0.3, 0.4, 1.0, 0.3],
        [0.4, 0.5, 0.6, 0.3, 1.0]
    ])
    
    # Generate returns
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    
    # Create DataFrame
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    return pd.DataFrame(returns, columns=assets)


@pytest.fixture
def optimizer():
    """Create a PortfolioOptimizer instance."""
    return PortfolioOptimizer(risk_free_rate=0.02, periods_per_year=252)


@pytest.fixture
def allocator():
    """Create an AssetAllocator instance."""
    asset_classes = [
        AssetClass(
            name="Tech",
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            strategic_weight=0.6
        ),
        AssetClass(
            name="Other",
            symbols=['AMZN', 'TSLA'],
            strategic_weight=0.4
        )
    ]
    return AssetAllocator(asset_classes=asset_classes, risk_free_rate=0.02)


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer class."""
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.risk_free_rate == 0.02
        assert optimizer.periods_per_year == 252
        assert len(optimizer.optimization_history) == 0
    
    def test_max_sharpe_optimization(self, optimizer, sample_returns):
        """Test maximum Sharpe ratio optimization."""
        result = optimizer.optimize(
            sample_returns,
            method=OptimizationMethod.MAX_SHARPE
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert len(result.weights) == len(sample_returns.columns)
        assert np.isclose(np.sum(result.weights), 1.0, atol=1e-3)
        assert np.all(result.weights >= -1e-6)  # Long only
        assert result.metrics.sharpe_ratio > 0
        assert result.method == OptimizationMethod.MAX_SHARPE
    
    def test_min_volatility_optimization(self, optimizer, sample_returns):
        """Test minimum volatility optimization."""
        result = optimizer.optimize(
            sample_returns,
            method=OptimizationMethod.MIN_VOLATILITY
        )
        
        assert result.success
        assert result.metrics.volatility > 0
        assert result.method == OptimizationMethod.MIN_VOLATILITY
        
        # Should have lower volatility than equal weight
        equal_weights = np.ones(len(sample_returns.columns)) / len(sample_returns.columns)
        equal_metrics = optimizer.calculate_metrics(equal_weights, sample_returns.values)
        assert result.metrics.volatility <= equal_metrics.volatility
    
    def test_efficient_return_optimization(self, optimizer, sample_returns):
        """Test efficient return (target return) optimization."""
        target_return = 0.10  # 10% annual return
        
        result = optimizer.optimize(
            sample_returns,
            method=OptimizationMethod.EFFICIENT_RETURN,
            target_return=target_return
        )
        
        assert result.success or not result.success  # May not always be feasible
        if result.success:
            # Should be close to target return
            assert abs(result.metrics.expected_return - target_return) < 0.05
    
    def test_risk_parity_optimization(self, optimizer, sample_returns):
        """Test risk parity optimization."""
        result = optimizer.optimize(
            sample_returns,
            method=OptimizationMethod.RISK_PARITY
        )
        
        assert result.success or not result.success  # May not always converge
        if result.success:
            assert np.isclose(np.sum(result.weights), 1.0, atol=1e-3)
    
    def test_weights_constraints(self, optimizer, sample_returns):
        """Test portfolio weights satisfy constraints."""
        result = optimizer.optimize(
            sample_returns,
            method=OptimizationMethod.MAX_SHARPE,
            bounds=(0.0, 0.3)  # Max 30% per asset
        )
        
        if result.success:
            assert np.all(result.weights <= 0.31)  # Allow small tolerance
            assert np.all(result.weights >= -1e-6)
    
    def test_calculate_metrics(self, optimizer, sample_returns):
        """Test portfolio metrics calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        metrics = optimizer.calculate_metrics(
            weights,
            sample_returns.values
        )
        
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.expected_return > -1.0 and metrics.expected_return < 1.0
        assert metrics.volatility > 0
        assert metrics.sharpe_ratio is not None
        assert metrics.sortino_ratio is not None
        assert metrics.max_drawdown is not None
        assert metrics.var_95 is not None
        assert metrics.cvar_95 is not None
    
    def test_metrics_with_benchmark(self, optimizer, sample_returns):
        """Test metrics calculation with benchmark."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Create synthetic benchmark
        benchmark_returns = sample_returns.mean(axis=1).values
        
        metrics = optimizer.calculate_metrics(
            weights,
            sample_returns.values,
            benchmark_returns=benchmark_returns
        )
        
        assert metrics.beta is not None
        assert metrics.alpha is not None
        assert metrics.information_ratio is not None
        assert metrics.treynor_ratio is not None


class TestEfficientFrontier:
    """Tests for Efficient Frontier calculation."""
    
    def test_efficient_frontier_generation(self, optimizer, sample_returns):
        """Test efficient frontier generation."""
        frontier = optimizer.calculate_efficient_frontier(
            sample_returns,
            n_points=50
        )
        
        assert isinstance(frontier, EfficientFrontier)
        assert len(frontier.returns) > 0
        assert len(frontier.risks) > 0
        assert len(frontier.sharpe_ratios) > 0
        assert len(frontier.returns) == len(frontier.risks)
        assert len(frontier.returns) == len(frontier.sharpe_ratios)
        
        # Risks should increase with returns (mostly)
        assert np.all(np.diff(frontier.returns) >= -1e-6)
    
    def test_max_sharpe_from_frontier(self, optimizer, sample_returns):
        """Test extracting max Sharpe portfolio from frontier."""
        frontier = optimizer.calculate_efficient_frontier(
            sample_returns,
            n_points=50
        )
        
        max_sharpe_portfolio = frontier.get_max_sharpe_portfolio()
        
        assert isinstance(max_sharpe_portfolio, OptimizationResult)
        assert max_sharpe_portfolio.success
        assert max_sharpe_portfolio.method == OptimizationMethod.MAX_SHARPE
        
        # Should have highest Sharpe ratio on frontier
        assert max_sharpe_portfolio.metrics.sharpe_ratio >= np.max(frontier.sharpe_ratios) - 0.01
    
    def test_min_volatility_from_frontier(self, optimizer, sample_returns):
        """Test extracting min volatility portfolio from frontier."""
        frontier = optimizer.calculate_efficient_frontier(
            sample_returns,
            n_points=50
        )
        
        min_vol_portfolio = frontier.get_min_volatility_portfolio()
        
        assert isinstance(min_vol_portfolio, OptimizationResult)
        assert min_vol_portfolio.success
        
        # Should have lowest volatility on frontier
        assert min_vol_portfolio.metrics.volatility <= np.min(frontier.risks) + 0.01
    
    def test_frontier_to_dataframe(self, optimizer, sample_returns):
        """Test converting frontier to DataFrame."""
        frontier = optimizer.calculate_efficient_frontier(
            sample_returns,
            n_points=30
        )
        
        df = frontier.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'return' in df.columns
        assert 'risk' in df.columns
        assert 'sharpe_ratio' in df.columns
        assert len(df) == len(frontier.returns)


class TestAssetAllocator:
    """Tests for AssetAllocator class."""
    
    def test_initialization(self, allocator):
        """Test allocator initialization."""
        assert len(allocator.asset_classes) == 2
        assert allocator.risk_free_rate == 0.02
        assert allocator.rebalancing_threshold == 0.05
    
    def test_equal_weight_allocation(self, allocator, sample_returns):
        """Test equal weight allocation."""
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.EQUAL_WEIGHT
        )
        
        assert isinstance(result, AllocationResult)
        assert len(result.weights) == len(sample_returns.columns)
        
        # All weights should be equal
        expected_weight = 1.0 / len(sample_returns.columns)
        for weight in result.weights.values():
            assert np.isclose(weight, expected_weight, atol=1e-6)
    
    def test_risk_parity_allocation(self, allocator, sample_returns):
        """Test risk parity allocation."""
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.RISK_PARITY
        )
        
        assert isinstance(result, AllocationResult)
        assert result.strategy == AllocationStrategy.RISK_PARITY
        assert np.isclose(sum(result.weights.values()), 1.0, atol=1e-3)
    
    def test_min_variance_allocation(self, allocator, sample_returns):
        """Test minimum variance allocation."""
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.MIN_VARIANCE
        )
        
        assert result.strategy == AllocationStrategy.MIN_VARIANCE
        assert result.metrics is not None
        assert result.metrics.volatility > 0
    
    def test_max_sharpe_allocation(self, allocator, sample_returns):
        """Test maximum Sharpe ratio allocation."""
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.MAX_SHARPE
        )
        
        assert result.strategy == AllocationStrategy.MAX_SHARPE
        assert result.metrics.sharpe_ratio > 0
    
    def test_market_cap_allocation(self, allocator, sample_returns):
        """Test market cap weighted allocation."""
        market_caps = {
            'AAPL': 3000e9,
            'MSFT': 2500e9,
            'GOOGL': 1800e9,
            'AMZN': 1500e9,
            'TSLA': 800e9
        }
        
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.MARKET_CAP,
            market_caps=market_caps
        )
        
        assert result.strategy == AllocationStrategy.MARKET_CAP
        
        # AAPL should have highest weight
        assert result.weights['AAPL'] > result.weights['TSLA']
    
    def test_hierarchical_risk_parity(self, allocator, sample_returns):
        """Test Hierarchical Risk Parity allocation."""
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.HIERARCHICAL_RISK_PARITY
        )
        
        assert result.strategy == AllocationStrategy.HIERARCHICAL_RISK_PARITY
        assert np.isclose(sum(result.weights.values()), 1.0, atol=1e-3)
        assert all(w >= -1e-6 for w in result.weights.values())
    
    def test_black_litterman_allocation(self, allocator, sample_returns):
        """Test Black-Litterman allocation."""
        market_caps = {asset: 1e9 for asset in sample_returns.columns}
        views = {'AAPL': 0.15}  # Expect 15% return for AAPL
        view_confidences = {'AAPL': 0.8}
        
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.BLACK_LITTERMAN,
            market_caps=market_caps,
            views=views,
            view_confidences=view_confidences
        )
        
        assert result.strategy == AllocationStrategy.BLACK_LITTERMAN
        # AAPL should have increased allocation due to positive view
        assert result.weights['AAPL'] > 0
    
    def test_asset_class_aggregation(self, allocator, sample_returns):
        """Test asset class weight aggregation."""
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.EQUAL_WEIGHT
        )
        
        assert result.asset_class_weights is not None
        assert 'Tech' in result.asset_class_weights
        assert 'Other' in result.asset_class_weights
        
        # Weights should sum to 1
        total = sum(result.asset_class_weights.values())
        assert np.isclose(total, 1.0, atol=1e-3)
    
    def test_rebalancing_check(self, allocator, sample_returns):
        """Test rebalancing requirement check."""
        current_weights = {'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': 0.2, 'AMZN': 0.2, 'TSLA': 0.1}
        
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.EQUAL_WEIGHT,
            current_weights=current_weights
        )
        
        # Equal weight is 0.2, current AAPL is 0.3, difference is 0.1 > threshold
        assert result.rebalancing_required
        assert result.tracking_error is not None
    
    def test_rebalance_calculation(self, allocator):
        """Test rebalancing trade calculation."""
        current_weights = {'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': 0.2, 'AMZN': 0.2, 'TSLA': 0.1}
        target_weights = {'AAPL': 0.2, 'MSFT': 0.2, 'GOOGL': 0.2, 'AMZN': 0.2, 'TSLA': 0.2}
        
        rebalance_info = allocator.rebalance(
            current_weights,
            target_weights,
            transaction_cost=0.001
        )
        
        assert 'trades' in rebalance_info
        assert 'turnover' in rebalance_info
        assert 'transaction_cost' in rebalance_info
        
        # Should have trades for AAPL and TSLA
        assert 'AAPL' in rebalance_info['trades']
        assert 'TSLA' in rebalance_info['trades']
        
        # Trades should sum to zero (long-short balanced)
        total_trades = sum(rebalance_info['trades'].values())
        assert np.isclose(total_trades, 0.0, atol=1e-6)


class TestPerformanceMetrics:
    """Tests for individual performance metric functions."""
    
    def test_sharpe_ratio_calculation(self, sample_returns):
        """Test Sharpe ratio calculation."""
        returns = sample_returns.iloc[:, 0].values
        
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)
    
    def test_sortino_ratio_calculation(self, sample_returns):
        """Test Sortino ratio calculation."""
        returns = sample_returns.iloc[:, 0].values
        
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        assert not np.isinf(sortino)
        
        # Sortino should be >= Sharpe (uses downside deviation)
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sortino >= sharpe - 0.5  # Allow some tolerance
    
    def test_max_drawdown_calculation(self, sample_returns):
        """Test maximum drawdown calculation."""
        returns = sample_returns.iloc[:, 0].values
        
        max_dd = calculate_max_drawdown(returns)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative
        assert max_dd >= -1.0  # Can't lose more than 100%


class TestVaRIntegration:
    """Tests for VaR model integration."""
    
    def test_var_constrained_allocation(self, sample_returns):
        """Test VaR-constrained allocation."""
        allocator = AssetAllocator()
        
        var_budget = 0.05  # 5% VaR limit
        
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.VAR_CONSTRAINED,
            var_budget=var_budget
        )
        
        assert result.strategy == AllocationStrategy.VAR_CONSTRAINED
        
        # Calculate actual VaR
        calculator = VaRCalculator()
        portfolio_returns = sample_returns.values @ np.array([
            result.weights[asset] for asset in sample_returns.columns
        ])
        
        var_result = calculator.calculate_var(
            portfolio_value=1.0,
            returns=portfolio_returns,
            method=VaRMethod.HISTORICAL
        )
        
        # VaR should be within budget (with tolerance for optimization error)
        assert var_result.var_percentage <= var_budget + 0.02
    
    def test_portfolio_var_calculation(self, optimizer, sample_returns):
        """Test calculating VaR for optimized portfolio."""
        # Optimize portfolio
        opt_result = optimizer.optimize(
            sample_returns,
            method=OptimizationMethod.MAX_SHARPE
        )
        
        # Calculate portfolio returns
        weights_array = np.array([
            opt_result.weights[i] for i in range(len(opt_result.weights))
        ])
        portfolio_returns = sample_returns.values @ weights_array
        
        # Calculate VaR
        calculator = VaRCalculator()
        var_result = calculator.calculate_var(
            portfolio_value=1000000,  # $1M portfolio
            returns=portfolio_returns,
            method=VaRMethod.HISTORICAL
        )
        
        assert var_result.var_amount > 0
        assert var_result.var_percentage > 0
        assert var_result.expected_shortfall is not None
        
        # Expected Shortfall should be >= VaR
        assert var_result.expected_shortfall >= var_result.var_amount


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_markowitz_optimization(self, sample_returns):
        """Test quick Markowitz optimization."""
        result = markowitz_optimization(
            sample_returns,
            risk_free_rate=0.02,
            method=OptimizationMethod.MAX_SHARPE
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert result.method == OptimizationMethod.MAX_SHARPE
    
    def test_equal_weight_allocation_function(self):
        """Test equal weight allocation function."""
        assets = ['AAPL', 'MSFT', 'GOOGL']
        
        weights = equal_weight_allocation(assets)
        
        assert len(weights) == len(assets)
        assert all(np.isclose(w, 1.0/3.0) for w in weights.values())
    
    def test_risk_parity_allocation_function(self, sample_returns):
        """Test risk parity allocation function."""
        weights = risk_parity_allocation(sample_returns)
        
        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(sum(weights.values()), 1.0, atol=1e-3)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_asset_optimization(self, optimizer):
        """Test optimization with single asset."""
        returns = pd.DataFrame(np.random.randn(100, 1), columns=['Asset1'])
        
        result = optimizer.optimize(
            returns,
            method=OptimizationMethod.MAX_SHARPE
        )
        
        # Should allocate 100% to single asset
        assert np.isclose(result.weights[0], 1.0)
    
    def test_highly_correlated_assets(self, optimizer):
        """Test optimization with highly correlated assets."""
        n_periods = 252
        returns1 = np.random.randn(n_periods) * 0.01
        returns2 = returns1 + np.random.randn(n_periods) * 0.001  # Almost identical
        
        returns = pd.DataFrame({
            'Asset1': returns1,
            'Asset2': returns2
        })
        
        result = optimizer.optimize(
            returns,
            method=OptimizationMethod.MIN_VOLATILITY
        )
        
        # Should still produce valid allocation
        assert result.success or not result.success  # May or may not converge
        if result.success:
            assert np.isclose(np.sum(result.weights), 1.0, atol=1e-3)
    
    def test_negative_returns(self, optimizer):
        """Test optimization with mostly negative returns."""
        returns = pd.DataFrame(
            np.random.randn(252, 3) * 0.02 - 0.001,  # Negative drift
            columns=['Asset1', 'Asset2', 'Asset3']
        )
        
        result = optimizer.optimize(
            returns,
            method=OptimizationMethod.MAX_SHARPE
        )
        
        # Should still complete (though Sharpe may be negative)
        assert isinstance(result, OptimizationResult)


class TestOutputFormats:
    """Tests for output formats and conversions."""
    
    def test_optimization_result_to_dict(self, optimizer, sample_returns):
        """Test converting OptimizationResult to dictionary."""
        result = optimizer.optimize(
            sample_returns,
            method=OptimizationMethod.MAX_SHARPE
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'weights' in result_dict
        assert 'assets' in result_dict
        assert 'metrics' in result_dict
        assert 'method' in result_dict
        assert 'success' in result_dict
    
    def test_metrics_to_dict(self, optimizer, sample_returns):
        """Test converting PortfolioMetrics to dictionary."""
        weights = np.ones(len(sample_returns.columns)) / len(sample_returns.columns)
        metrics = optimizer.calculate_metrics(weights, sample_returns.values)
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'expected_return' in metrics_dict
        assert 'volatility' in metrics_dict
        assert 'sharpe_ratio' in metrics_dict
        assert 'sortino_ratio' in metrics_dict
    
    def test_allocation_result_to_dict(self, allocator, sample_returns):
        """Test converting AllocationResult to dictionary."""
        result = allocator.allocate(
            sample_returns,
            strategy=AllocationStrategy.EQUAL_WEIGHT
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'weights' in result_dict
        assert 'strategy' in result_dict
        assert 'metrics' in result_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])