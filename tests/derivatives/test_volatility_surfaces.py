"""
Test Suite for Volatility Surface Engine

Tests:
- GAN-based surface construction
- SABR calibration
- Arbitrage-free constraints
- Real-time updates
- Interpolation accuracy
"""

import pytest
import numpy as np
from axiom.derivatives.volatility_surface import RealTimeVolatilitySurface, SABRCalibrator


class TestVolatilitySurface:
    """Test volatility surface construction"""
    
    @pytest.fixture
    def surface_engine(self):
        return RealTimeVolatilitySurface(use_gpu=False)
    
    def test_surface_construction(self, surface_engine):
        """Test GAN-based surface construction"""
        market_quotes = np.array([0.20, 0.22, 0.24, 0.26, 0.28] * 4)
        
        surface = surface_engine.construct_surface(
            market_quotes=market_quotes,
            spot=100.0
        )
        
        # Validate surface
        assert surface.strikes.shape[0] > 0
        assert surface.maturities.shape[0] > 0
        assert surface.surface.shape == (len(surface.strikes), len(surface.maturities))
        assert surface.arbitrage_free == True
        assert surface.construction_time_ms < 10.0  # <10ms even on CPU
    
    def test_surface_interpolation(self, surface_engine):
        """Test volatility lookup and interpolation"""
        market_quotes = np.random.uniform(0.2, 0.3, 20)
        surface = surface_engine.construct_surface(market_quotes, spot=100.0)
        
        # Test interpolation at various points
        test_points = [
            (95.0, 0.25),
            (100.0, 0.5),
            (105.0, 1.0),
            (110.0, 2.0)
        ]
        
        for strike, maturity in test_points:
            vol = surface.get_vol(strike, maturity)
            assert 0.05 < vol < 1.0  # Reasonable vol range
    
    def test_arbitrage_free_constraint(self, surface_engine):
        """Test that arbitrage-free constraints are enforced"""
        market_quotes = np.random.uniform(0.2, 0.3, 20)
        surface = surface_engine.construct_surface(market_quotes, spot=100.0)
        
        # Check calendar arbitrage: total variance increases with time
        for i in range(len(surface.strikes)):
            variances = surface.surface[i, :] ** 2 * surface.maturities
            
            # Each variance should be >= previous
            for j in range(1, len(variances)):
                assert variances[j] >= variances[j-1] * 0.99  # Allow small numerical errors
    
    def test_real_time_update(self, surface_engine):
        """Test real-time surface updates"""
        market_quotes = np.random.uniform(0.2, 0.3, 20)
        surface = surface_engine.construct_surface(market_quotes, spot=100.0)
        
        # Market moves, update surface
        new_quotes = market_quotes * 1.1  # 10% vol increase
        updated_surface = surface_engine.update_surface_realtime(surface, new_quotes)
        
        # Should be updated
        assert updated_surface.construction_time_ms < 5.0
        
        # Vols should be higher
        old_vol = surface.get_vol(100, 1.0)
        new_vol = updated_surface.get_vol(100, 1.0)
        assert new_vol > old_vol


class TestSABRCalibration:
    """Test SABR model calibration"""
    
    def test_sabr_calibration(self):
        """Test SABR parameter calibration"""
        sabr = SABRCalibrator()
        
        # Calibrate to market data
        strikes = np.array([90, 95, 100, 105, 110])
        market_vols = np.array([0.28, 0.25, 0.23, 0.25, 0.28])  # Vol smile
        
        params = sabr.calibrate(
            forward=100.0,
            strikes=strikes,
            market_vols=market_vols,
            maturity=1.0
        )
        
        # Should return reasonable parameters
        assert 0 < params['alpha'] < 1.0
        assert 0 <= params['beta'] <= 1.0
        assert -1 < params['rho'] < 1
        assert params['nu'] > 0
    
    def test_sabr_volatility_calculation(self):
        """Test SABR vol calculation"""
        sabr = SABRCalibrator()
        
        params = {
            'alpha': 0.2,
            'beta': 0.5,
            'rho': -0.3,
            'nu': 0.4
        }
        
        vol = sabr.get_volatility(
            forward=100.0,
            strike=105.0,
            maturity=1.0,
            params=params
        )
        
        assert 0.1 < vol < 0.5  # Reasonable vol


class TestPerformance:
    """Performance benchmarks for volatility surfaces"""
    
    @pytest.mark.benchmark
    def test_construction_performance(self, benchmark):
        """Benchmark surface construction speed"""
        surface_engine = RealTimeVolatilitySurface(use_gpu=False)
        market_quotes = np.random.uniform(0.2, 0.3, 20)
        
        result = benchmark(
            surface_engine.construct_surface,
            market_quotes=market_quotes,
            spot=100.0
        )
        
        # Target: <1ms on GPU, <10ms on CPU
        mean_ms = benchmark.stats['mean'] * 1000
        assert mean_ms < 10.0, f"Surface construction {mean_ms:.2f}ms too slow"
    
    @pytest.mark.benchmark
    def test_interpolation_performance(self, benchmark):
        """Benchmark surface interpolation speed"""
        surface_engine = RealTimeVolatilitySurface(use_gpu=False)
        market_quotes = np.random.uniform(0.2, 0.3, 20)
        surface = surface_engine.construct_surface(market_quotes, spot=100.0)
        
        result = benchmark(
            surface.get_vol,
            strike=105.0,
            maturity=0.5
        )
        
        # Target: <100 microseconds
        mean_us = benchmark.stats['mean'] * 1_000_000
        assert mean_us < 1000  # <1ms is acceptable on CPU


# Run with: pytest tests/derivatives/test_volatility_surfaces.py -v --benchmark-only