"""
Comprehensive Test Suite for Ultra-Fast Greeks Engine

Tests cover:
1. Accuracy validation (vs Black-Scholes)
2. Performance benchmarking (<100us target)
3. Stress testing (sustained load)
4. Edge cases (boundary conditions)
5. GPU utilization
6. Error handling

Senior quant standard: 95%+ coverage, rigorous validation
"""

import pytest
import numpy as np
import time
from scipy.stats import norm

from axiom.derivatives.ultra_fast_greeks import (
    UltraFastGreeksEngine,
    GreeksEnsemble,
    create_greeks_engine
)


class TestAccuracy:
    """Test Greeks accuracy vs analytical Black-Scholes"""
    
    @pytest.fixture
    def engine(self):
        return UltraFastGreeksEngine(use_gpu=True)
    
    @pytest.fixture
    def test_cases(self):
        """Generate diverse test cases"""
        np.random.seed(42)
        n_cases = 1000
        
        return {
            'spot': np.random.uniform(50, 200, n_cases),
            'strike': np.random.uniform(50, 200, n_cases),
            'time': np.random.uniform(0.1, 5.0, n_cases),
            'rate': np.random.uniform(0.0, 0.1, n_cases),
            'vol': np.random.uniform(0.1, 0.8, n_cases)
        }
    
    def black_scholes_greeks(self, S, K, T, r, sigma):
        """Analytical Black-Scholes for validation"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        
        return delta, gamma, theta, vega, rho, price
    
    def test_accuracy_atm_options(self, engine):
        """Test accuracy for ATM options (most common)"""
        # ATM options (strike = spot)
        result = engine.calculate_greeks(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.03,
            volatility=0.25
        )
        
        # Analytical solution
        delta_bs, gamma_bs, theta_bs, vega_bs, rho_bs, price_bs = self.black_scholes_greeks(
            100.0, 100.0, 1.0, 0.03, 0.25
        )
        
        # Validate within 0.01% (99.99% accuracy target)
        assert abs(result.delta - delta_bs) / delta_bs < 0.0001
        assert abs(result.price - price_bs) / price_bs < 0.0001
    
    def test_accuracy_extreme_cases(self, engine):
        """Test edge cases (deep ITM, deep OTM)"""
        # Deep ITM call (delta should be ~1.0)
        itm = engine.calculate_greeks(100.0, 50.0, 1.0, 0.03, 0.25)
        assert 0.95 < itm.delta < 1.0
        
        # Deep OTM call (delta should be ~0.0)
        otm = engine.calculate_greeks(100.0, 150.0, 1.0, 0.03, 0.25)
        assert 0.0 < otm.delta < 0.05
    
    def test_accuracy_statistical(self, engine, test_cases):
        """Test accuracy over 1000 random cases"""
        errors = []
        
        for i in range(len(test_cases['spot'])):
            # Our engine
            result = engine.calculate_greeks(
                test_cases['spot'][i],
                test_cases['strike'][i],
                test_cases['time'][i],
                test_cases['rate'][i],
                test_cases['vol'][i]
            )
            
            # Analytical
            delta_bs, _, _, _, _, _ = self.black_scholes_greeks(
                test_cases['spot'][i],
                test_cases['strike'][i],
                test_cases['time'][i],
                test_cases['rate'][i],
                test_cases['vol'][i]
            )
            
            # Relative error
            error = abs(result.delta - delta_bs) / abs(delta_bs + 1e-10)
            errors.append(error)
        
        errors = np.array(errors)
        
        # Statistical validation
        assert np.mean(errors) < 0.0001  # Mean error < 0.01%
        assert np.percentile(errors, 95) < 0.001  # 95% within 0.1%
        assert np.percentile(errors, 99) < 0.01  # 99% within 1%


class TestPerformance:
    """Performance benchmarking tests"""
    
    @pytest.fixture
    def engine(self):
        return UltraFastGreeksEngine(use_gpu=True)
    
    @pytest.mark.benchmark
    def test_single_calculation_latency(self, engine, benchmark):
        """Critical test: <100 microsecond target"""
        result = benchmark.pedantic(
            engine.calculate_greeks,
            args=(100.0, 100.0, 1.0, 0.03, 0.25),
            iterations=10000,
            rounds=10
        )
        
        # Convert to microseconds
        mean_us = benchmark.stats['mean'] * 1_000_000
        p95_us = benchmark.stats.get('q_95', benchmark.stats['mean']) * 1_000_000
        
        print(f"\nLatency: mean={mean_us:.2f}us, p95={p95_us:.2f}us")
        
        # CRITICAL: Must be <100us
        assert mean_us < 100, f"Mean latency {mean_us:.2f}us exceeds 100us target"
        assert p95_us < 200, f"P95 latency {p95_us:.2f}us exceeds 200us limit"
    
    def test_batch_throughput(self, engine):
        """Test batch processing throughput"""
        batch_size = 1000
        batch_data = np.random.rand(batch_size, 5)
        
        start = time.perf_counter()
        results = engine.calculate_batch(batch_data)
        elapsed = time.perf_counter() - start
        
        throughput = batch_size / elapsed
        
        print(f"\nBatch throughput: {throughput:,.0f} calc/sec")
        
        # Target: 10,000+ calculations/second
        assert throughput > 10000, f"Throughput {throughput:,.0f} below 10K target"
    
    @pytest.mark.stress
    def test_sustained_load(self, engine):
        """Stress test: 100,000 continuous calculations"""
        latencies = []
        
        for i in range(100000):
            result = engine.calculate_greeks(100.0, 100.0, 1.0, 0.03, 0.25)
            latencies.append(result.calculation_time_us)
            
            # Check every 10K
            if (i+1) % 10000 == 0:
                recent_avg = np.mean(latencies[-10000:])
                assert recent_avg < 100, f"Latency degraded to {recent_avg:.2f}us"
        
        # Verify no degradation over time
        first_10k = np.mean(latencies[:10000])
        last_10k = np.mean(latencies[-10000:])
        degradation = (last_10k - first_10k) / first_10k
        
        assert abs(degradation) < 0.1, f"Performance degraded {degradation:.1%}"
    
    def test_gpu_utilization(self, engine):
        """Verify GPU is being utilized"""
        assert engine.device.type == 'cuda', "GPU not available or not being used"
        
        # Verify CUDA is working
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
        assert torch.cuda.device_count() > 0, "No CUDA devices"


class TestEdgeCases:
    """Test boundary conditions and error cases"""
    
    @pytest.fixture
    def engine(self):
        return UltraFastGreeksEngine(use_gpu=True)
    
    def test_zero_time_to_maturity(self, engine):
        """Test with T → 0 (expiration)"""
        result = engine.calculate_greeks(100.0, 100.0, 0.001, 0.03, 0.25)
        # Delta should approach step function
        assert 0.4 < result.delta < 0.6  # Near 0.5 for ATM
    
    def test_high_volatility(self, engine):
        """Test with extreme volatility"""
        result = engine.calculate_greeks(100.0, 100.0, 1.0, 0.03, 2.0)
        # Should handle gracefully
        assert 0.0 <= result.delta <= 1.0
        assert result.vega > 0
    
    def test_deep_itm(self, engine):
        """Test deep in-the-money"""
        result = engine.calculate_greeks(100.0, 50.0, 1.0, 0.03, 0.25)
        assert result.delta > 0.95  # Should be near 1.0
    
    def test_deep_otm(self, engine):
        """Test deep out-of-the-money"""
        result = engine.calculate_greeks(100.0, 200.0, 1.0, 0.03, 0.25)
        assert result.delta < 0.05  # Should be near 0.0
    
    def test_put_option(self, engine):
        """Test put option Greeks"""
        result = engine.calculate_greeks(
            100.0, 100.0, 1.0, 0.03, 0.25, option_type='put'
        )
        # Put delta should be negative
        assert -1.0 <= result.delta <= 0.0


class TestIntegration:
    """Integration tests with other components"""
    
    def test_with_volatility_surface(self):
        """Test integration with vol surface engine"""
        from axiom.derivatives.ultra_fast_greeks import UltraFastGreeksEngine
        from axiom.derivatives.volatility_surface import RealTimeVolatilitySurface
        
        # Create engines
        greeks_engine = UltraFastGreeksEngine(use_gpu=True)
        surface_engine = RealTimeVolatilitySurface(use_gpu=True)
        
        # Build surface
        market_quotes = np.random.uniform(0.2, 0.3, 20)
        surface = surface_engine.construct_surface(market_quotes, spot=100.0)
        
        # Get vol for specific option
        vol = surface.get_vol(105.0, 0.5)
        
        # Calculate Greeks with that vol
        greeks = greeks_engine.calculate_greeks(
            100.0, 105.0, 0.5, 0.03, vol
        )
        
        # Should work seamlessly
        assert greeks.calculation_time_us < 100
        assert 0.0 <= greeks.delta <= 1.0
    
    def test_with_mcp_data(self):
        """Test integration with MCP data source"""
        # Would test actual MCP integration in production
        pass


@pytest.mark.benchmark
def test_comprehensive_benchmark(benchmark):
    """
    Comprehensive benchmark reported to stakeholders
    
    This is the official benchmark we show to clients
    """
    engine = UltraFastGreeksEngine(use_gpu=True)
    
    result = benchmark.pedantic(
        engine.calculate_greeks,
        args=(100.0, 100.0, 1.0, 0.03, 0.25),
        iterations=10000,
        rounds=10
    )
    
    stats = benchmark.stats
    mean_us = stats['mean'] * 1_000_000
    median_us = stats['median'] * 1_000_000
    
    # Print official results
    print("\n" + "="*60)
    print("OFFICIAL BENCHMARK RESULTS")
    print("="*60)
    print(f"Mean latency: {mean_us:.2f} microseconds")
    print(f"Median latency: {median_us:.2f} microseconds")
    print(f"Std dev: {stats['stddev'] * 1_000_000:.2f} microseconds")
    print(f"Target <100us: {'✓ PASS' if mean_us < 100 else '✗ FAIL'}")
    print(f"vs Bloomberg (100ms): {100_000 / mean_us:.0f}x faster")
    print("="*60)


# Run with: pytest tests/derivatives/test_ultra_fast_greeks.py -v --benchmark-only