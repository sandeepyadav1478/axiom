"""
Test Suite for Market Making Platform

Tests:
- RL spread optimization
- DRL auto-hedging
- Inventory management
- Market condition adaptation
"""

import pytest
import numpy as np
from axiom.derivatives.market_making.rl_spread_optimizer import RLSpreadOptimizer, MarketState
from axiom.derivatives.market_making.auto_hedger import DRLAutoHedger, PortfolioState


class TestRLSpreadOptimizer:
    """Test RL-based spread optimization"""
    
    @pytest.fixture
    def optimizer(self):
        return RLSpreadOptimizer(use_gpu=False)
    
    def test_normal_market_spreads(self, optimizer):
        """Test spread optimization in normal market"""
        state = MarketState(
            mid_price=100.0,
            bid_ask_spread=0.10,
            bid_size=1000,
            ask_size=1000,
            recent_volume=10000,
            volatility=0.20,
            inventory=0,
            max_inventory=1000,
            time_to_close=3.0,
            regime='normal'
        )
        
        spreads = optimizer.get_optimal_spreads(state)
        
        # Should provide reasonable spreads
        assert 0 < spreads.bid_offset < 0.05
        assert 0 < spreads.ask_offset < 0.05
        assert spreads.confidence > 0.5
        assert spreads.expected_pnl > 0
    
    def test_high_inventory_spreads(self, optimizer):
        """Test that spreads widen with high inventory"""
        normal_state = MarketState(
            mid_price=100.0, bid_ask_spread=0.10, bid_size=1000, ask_size=1000,
            recent_volume=10000, volatility=0.20, inventory=0, max_inventory=1000,
            time_to_close=3.0, regime='normal'
        )
        
        high_inventory_state = MarketState(
            mid_price=100.0, bid_ask_spread=0.10, bid_size=1000, ask_size=1000,
            recent_volume=10000, volatility=0.20, inventory=800, max_inventory=1000,
            time_to_close=3.0, regime='normal'
        )
        
        normal_spreads = optimizer.get_optimal_spreads(normal_state)
        high_inv_spreads = optimizer.get_optimal_spreads(high_inventory_state)
        
        # High inventory should result in wider spreads
        assert (high_inv_spreads.bid_offset + high_inv_spreads.ask_offset) > \
               (normal_spreads.bid_offset + normal_spreads.ask_offset)
    
    def test_high_volatility_spreads(self, optimizer):
        """Test that spreads widen in high volatility"""
        low_vol_state = MarketState(
            mid_price=100.0, bid_ask_spread=0.10, bid_size=1000, ask_size=1000,
            recent_volume=10000, volatility=0.15, inventory=0, max_inventory=1000,
            time_to_close=3.0, regime='normal'
        )
        
        high_vol_state = MarketState(
            mid_price=100.0, bid_ask_spread=0.10, bid_size=1000, ask_size=1000,
            recent_volume=10000, volatility=0.40, inventory=0, max_inventory=1000,
            time_to_close=3.0, regime='high_vol'
        )
        
        low_vol_spreads = optimizer.get_optimal_spreads(low_vol_state)
        high_vol_spreads = optimizer.get_optimal_spreads(high_vol_state)
        
        # Higher vol should mean wider spreads
        assert (high_vol_spreads.bid_offset + high_vol_spreads.ask_offset) > \
               (low_vol_spreads.bid_offset + low_vol_spreads.ask_offset)


class TestDRLAutoHedger:
    """Test DRL-based auto-hedging system"""
    
    @pytest.fixture
    def hedger(self):
        return DRLAutoHedger(use_gpu=False, target_delta=0.0)
    
    def test_hedge_recommendation(self, hedger):
        """Test hedge recommendation for delta exposure"""
        portfolio = PortfolioState(
            total_delta=500.0,  # Long 500 delta
            total_gamma=10.0,
            total_vega=2000.0,
            total_theta=-200.0,
            spot_price=100.0,
            volatility=0.25,
            positions=[],
            hedge_position=0.0,
            pnl=5000.0,
            time_to_close=3.0
        )
        
        hedge = hedger.get_optimal_hedge(portfolio)
        
        # Should recommend selling stock to reduce delta
        assert hedge.hedge_delta < 0  # Sell to offset long delta
        assert abs(hedge.expected_delta_after) < abs(portfolio.total_delta)
        assert hedge.confidence > 0.5
    
    def test_no_hedge_when_balanced(self, hedger):
        """Test that minimal hedging when delta is small"""
        portfolio = PortfolioState(
            total_delta=5.0,  # Very small delta
            total_gamma=10.0,
            total_vega=2000.0,
            total_theta=-200.0,
            spot_price=100.0,
            volatility=0.25,
            positions=[],
            hedge_position=0.0,
            pnl=5000.0,
            time_to_close=3.0
        )
        
        hedge = hedger.get_optimal_hedge(portfolio)
        
        # Hedge should be small
        assert abs(hedge.hedge_delta) < 100


class TestPerformance:
    """Performance benchmarks for market making"""
    
    @pytest.mark.benchmark
    def test_spread_optimizer_performance(self, benchmark):
        """Benchmark spread optimization speed"""
        optimizer = RLSpreadOptimizer(use_gpu=False)
        
        state = MarketState(
            mid_price=100.0, bid_ask_spread=0.10, bid_size=1000, ask_size=1000,
            recent_volume=10000, volatility=0.20, inventory=0, max_inventory=1000,
            time_to_close=3.0, regime='normal'
        )
        
        result = benchmark(optimizer.get_optimal_spreads, state)
        
        # Target: <1ms
        mean_ms = benchmark.stats['mean'] * 1000
        assert mean_ms < 1.0, f"Spread optimization {mean_ms:.2f}ms exceeds 1ms target"
    
    @pytest.mark.benchmark
    def test_hedger_performance(self, benchmark):
        """Benchmark auto-hedger speed"""
        hedger = DRLAutoHedger(use_gpu=False)
        
        portfolio = PortfolioState(
            total_delta=500.0, total_gamma=10.0, total_vega=2000.0, total_theta=-200.0,
            spot_price=100.0, volatility=0.25, positions=[], hedge_position=0.0,
            pnl=5000.0, time_to_close=3.0
        )
        
        result = benchmark(hedger.get_optimal_hedge, portfolio)
        
        # Target: <1ms
        mean_ms = benchmark.stats['mean'] * 1000
        assert mean_ms < 1.0, f"Auto-hedging {mean_ms:.2f}ms exceeds 1ms target"


# Run with: pytest tests/derivatives/test_market_making.py -v --benchmark-only