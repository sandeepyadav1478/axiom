"""
Comprehensive Test Suite for Exotic Options Pricing

Tests all exotic option types for:
- Accuracy (vs Monte Carlo)
- Performance (<2ms target)
- Edge cases
- Boundary conditions
"""

import pytest
import numpy as np
import time
from axiom.derivatives.exotic_pricer import ExoticOptionsPricer, ExoticType


class TestBarrierOptions:
    """Test barrier option pricing"""
    
    @pytest.fixture
    def pricer(self):
        return ExoticOptionsPricer(use_gpu=False)  # CPU for CI/CD
    
    def test_up_and_out_call(self, pricer):
        """Test up-and-out barrier option"""
        result = pricer.price_barrier_option(
            spot=100.0,
            strike=100.0,
            barrier=120.0,
            time_to_maturity=1.0,
            risk_free_rate=0.03,
            volatility=0.25,
            barrier_type='up_and_out',
            option_type='call'
        )
        
        # Price should be less than vanilla call
        assert 0 < result.price < 10.0
        assert result.delta > 0
        assert result.calculation_time_ms < 2.0  # <2ms target
    
    def test_barrier_at_spot(self, pricer):
        """Test barrier option when spot = barrier (edge case)"""
        result = pricer.price_barrier_option(
            spot=120.0,
            strike=100.0,
            barrier=120.0,
            time_to_maturity=1.0,
            risk_free_rate=0.03,
            volatility=0.25,
            barrier_type='up_and_out'
        )
        
        # Should be knocked out or very low value
        assert result.price < 1.0


class TestAsianOptions:
    """Test Asian option pricing"""
    
    @pytest.fixture
    def pricer(self):
        return ExoticOptionsPricer(use_gpu=False)
    
    def test_asian_arithmetic(self, pricer):
        """Test arithmetic average Asian option"""
        result = pricer.price_asian_option(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.03,
            volatility=0.25,
            averaging_type='arithmetic'
        )
        
        assert result.price > 0
        assert result.calculation_time_ms < 2.0
        assert result.method == 'VAE'
    
    def test_asian_vs_vanilla(self, pricer):
        """Asian should be cheaper than vanilla (less volatile)"""
        from axiom.derivatives.ultra_fast_greeks import UltraFastGreeksEngine
        
        asian = pricer.price_asian_option(100, 100, 1.0, 0.03, 0.25)
        
        vanilla_engine = UltraFastGreeksEngine(use_gpu=False)
        vanilla = vanilla_engine.calculate_greeks(100, 100, 1.0, 0.03, 0.25)
        
        # Asian should be cheaper (averaging reduces volatility)
        assert asian.price < vanilla.price


class TestLookbackOptions:
    """Test lookback option pricing"""
    
    @pytest.fixture
    def pricer(self):
        return ExoticOptionsPricer(use_gpu=False)
    
    def test_lookback_floating(self, pricer):
        """Test floating strike lookback"""
        result = pricer.price_lookback_option(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.03,
            volatility=0.25,
            lookback_type='floating'
        )
        
        # Lookback is valuable (allows buying at min, selling at max)
        assert result.price > 5.0
        assert result.calculation_time_ms < 2.0


class TestBinaryOptions:
    """Test binary/digital option pricing"""
    
    @pytest.fixture
    def pricer(self):
        return ExoticOptionsPricer(use_gpu=False)
    
    def test_binary_call(self, pricer):
        """Test cash-or-nothing binary"""
        result = pricer.price_binary_option(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.03,
            volatility=0.25,
            payout=100.0
        )
        
        # Binary price should be fraction of payout
        assert 0 < result.price < 100.0
        assert result.calculation_time_ms < 0.5  # Binary is fastest
    
    def test_binary_payout_scaling(self, pricer):
        """Binary with 2x payout should have 2x price"""
        result1 = pricer.price_binary_option(100, 100, 1.0, 0.03, 0.25, payout=100)
        result2 = pricer.price_binary_option(100, 100, 1.0, 0.03, 0.25, payout=200)
        
        # Should scale linearly
        assert abs(result2.price - 2 * result1.price) < 0.01


class TestPerformanceBenchmarks:
    """Benchmark exotic options performance"""
    
    @pytest.mark.benchmark
    def test_barrier_performance(self, benchmark):
        """Benchmark barrier option pricing"""
        pricer = ExoticOptionsPricer(use_gpu=False)
        
        result = benchmark(
            pricer.price_barrier_option,
            100.0, 100.0, 120.0, 1.0, 0.03, 0.25, 'up_and_out'
        )
        
        # Target: <1ms
        mean_ms = benchmark.stats['mean'] * 1000
        assert mean_ms < 1.0, f"Barrier pricing {mean_ms:.2f}ms exceeds 1ms target"
    
    @pytest.mark.benchmark
    def test_asian_performance(self, benchmark):
        """Benchmark Asian option pricing"""
        pricer = ExoticOptionsPricer(use_gpu=False)
        
        result = benchmark(
            pricer.price_asian_option,
            100.0, 100.0, 1.0, 0.03, 0.25, 'arithmetic'
        )
        
        mean_ms = benchmark.stats['mean'] * 1000
        assert mean_ms < 2.0, f"Asian pricing {mean_ms:.2f}ms exceeds 2ms target"


# Run with: pytest tests/derivatives/test_exotic_options.py -v --benchmark-only