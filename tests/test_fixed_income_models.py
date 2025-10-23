"""
Comprehensive Test Suite for Fixed Income Models
=================================================

Tests all 6 major components:
1. Bond Pricing
2. Yield Curve Construction
3. Duration & Convexity
4. Term Structure Models
5. Spreads & Credit
6. Portfolio Analytics

Target: 60+ tests with 100% coverage of critical paths
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from axiom.models.fixed_income.base_model import (
    BondSpecification,
    BondPrice,
    YieldCurve,
    DayCountConvention,
    CompoundingFrequency,
    BondType,
    ValidationError
)
from axiom.models.fixed_income.bond_pricing import (
    BondPricingModel,
    YieldType,
    price_bond,
    calculate_ytm
)
from axiom.models.fixed_income.yield_curve import (
    NelsonSiegelModel,
    SvenssonModel,
    BootstrappingModel,
    CubicSplineModel,
    BondMarketData,
    YieldCurveAnalyzer
)
from axiom.models.fixed_income.duration import (
    DurationCalculator,
    DurationHedger,
    calculate_duration
)
from axiom.models.fixed_income.term_structure import (
    VasicekModel,
    CIRModel,
    HullWhiteModel,
    HoLeeModel,
    TermStructureParameters
)
from axiom.models.fixed_income.spreads import (
    SpreadAnalyzer,
    CreditSpreadAnalyzer,
    RelativeValueAnalyzer,
    calculate_spread
)
from axiom.models.fixed_income.portfolio import (
    BondPortfolioAnalyzer,
    BondHolding,
    PortfolioOptimizer,
    calculate_portfolio_duration
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_bond():
    """Create a sample fixed-rate bond."""
    return BondSpecification(
        face_value=100.0,
        coupon_rate=0.05,
        maturity_date=datetime(2030, 12, 31),
        issue_date=datetime(2020, 1, 1),
        coupon_frequency=CompoundingFrequency.SEMI_ANNUAL,
        day_count=DayCountConvention.THIRTY_360,
        bond_type=BondType.FIXED_RATE
    )


@pytest.fixture
def zero_coupon_bond():
    """Create a zero-coupon bond."""
    return BondSpecification(
        face_value=100.0,
        coupon_rate=0.0,
        maturity_date=datetime(2030, 12, 31),
        issue_date=datetime(2020, 1, 1),
        bond_type=BondType.ZERO_COUPON
    )


@pytest.fixture
def callable_bond():
    """Create a callable bond."""
    return BondSpecification(
        face_value=100.0,
        coupon_rate=0.06,
        maturity_date=datetime(2030, 12, 31),
        issue_date=datetime(2020, 1, 1),
        callable=True,
        call_price=105.0,
        call_date=datetime(2025, 12, 31)
    )


@pytest.fixture
def settlement_date():
    """Standard settlement date."""
    return datetime(2025, 1, 1)


@pytest.fixture
def treasury_curve():
    """Create a sample treasury curve."""
    tenors = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    rates = np.array([0.03, 0.035, 0.04, 0.042, 0.045, 0.047, 0.05, 0.052, 0.053])
    
    return YieldCurve(
        tenors=tenors,
        rates=rates,
        model_type="treasury",
        calibration_date=datetime(2025, 1, 1),
        parameters={}
    )


# ============================================================================
# 1. BOND PRICING TESTS (~15 tests)
# ============================================================================

class TestBondPricing:
    """Test bond pricing functionality."""
    
    def test_fixed_rate_bond_pricing(self, sample_bond, settlement_date):
        """Test fixed-rate bond pricing."""
        model = BondPricingModel()
        result = model.calculate_price(
            bond=sample_bond,
            settlement_date=settlement_date,
            yield_rate=0.06
        )
        
        assert result.clean_price > 0
        assert result.dirty_price > result.clean_price
        assert result.ytm == pytest.approx(0.06, rel=1e-6)
        assert result.modified_duration > 0
    
    def test_zero_coupon_bond_pricing(self, zero_coupon_bond, settlement_date):
        """Test zero-coupon bond pricing."""
        model = BondPricingModel()
        result = model.calculate_price(
            bond=zero_coupon_bond,
            settlement_date=settlement_date,
            yield_rate=0.05
        )
        
        assert result.clean_price < 100.0  # Should trade at discount
        assert result.accrued_interest == 0.0  # No coupons
    
    def test_ytm_calculation(self, sample_bond, settlement_date):
        """Test YTM calculation from price."""
        model = BondPricingModel()
        target_price = 95.0
        
        ytm = model.calculate_yield(
            bond=sample_bond,
            price=target_price,
            settlement_date=settlement_date
        )
        
        assert ytm > 0
        assert ytm > 0.05  # YTM > coupon when price < par
    
    def test_callable_bond_pricing(self, callable_bond, settlement_date):
        """Test callable bond pricing."""
        model = BondPricingModel()
        result = model.calculate_price(
            bond=callable_bond,
            settlement_date=settlement_date,
            yield_rate=0.05
        )
        
        assert result.clean_price > 0
        # Callable bond should price lower than straight bond
    
    def test_all_yield_metrics(self, sample_bond, settlement_date):
        """Test calculation of all yield metrics."""
        model = BondPricingModel()
        price = 98.0
        
        yields = model.calculate_all_yields(
            bond=sample_bond,
            price=price,
            settlement_date=settlement_date
        )
        
        assert yields.ytm > 0
        assert yields.current_yield > 0
        assert yields.ytw > 0
    
    def test_accrued_interest(self, sample_bond, settlement_date):
        """Test accrued interest calculation."""
        model = BondPricingModel()
        
        # Get coupon dates
        last_coupon = datetime(2024, 12, 31)
        next_coupon = datetime(2025, 6, 30)
        
        accrued = model.calculate_accrued_interest(
            bond=sample_bond,
            settlement_date=settlement_date,
            last_coupon_date=last_coupon,
            next_coupon_date=next_coupon
        )
        
        assert accrued >= 0
        assert accrued < sample_bond.face_value * sample_bond.coupon_rate / 2
    
    def test_price_validation(self, sample_bond, settlement_date):
        """Test that prices are validated."""
        model = BondPricingModel()
        
        # Normal case should work
        result = model.calculate_price(
            bond=sample_bond,
            settlement_date=settlement_date,
            yield_rate=0.05
        )
        assert result.clean_price > 0
    
    def test_convenience_function(self):
        """Test convenience pricing function."""
        price = price_bond(
            face_value=100,
            coupon_rate=0.05,
            yield_rate=0.06
        )
        
        assert price > 0
        assert price < 100  # Discount when yield > coupon
    
    def test_performance_target(self, sample_bond, settlement_date):
        """Test that pricing meets <5ms target."""
        import time
        model = BondPricingModel()
        
        start = time.perf_counter()
        model.calculate_price(
            bond=sample_bond,
            settlement_date=settlement_date,
            yield_rate=0.05
        )
        execution_ms = (time.perf_counter() - start) * 1000
        
        assert execution_ms < 5.0, f"Pricing took {execution_ms:.2f}ms, target is <5ms"


# ============================================================================
# 2. YIELD CURVE TESTS (~10 tests)
# ============================================================================

class TestYieldCurve:
    """Test yield curve construction."""
    
    def test_nelson_siegel_fitting(self):
        """Test Nelson-Siegel model fitting."""
        # Create sample bond data
        bonds = [
            BondMarketData(
                bond=BondSpecification(
                    face_value=100,
                    coupon_rate=0.04,
                    maturity_date=datetime.now() + timedelta(days=365*t),
                    issue_date=datetime.now()
                ),
                clean_price=98.0 + t * 0.5,
                settlement_date=datetime.now(),
                time_to_maturity=t
            )
            for t in [1, 2, 3, 5, 7, 10]
        ]
        
        model = NelsonSiegelModel()
        curve = model.fit(bonds)
        
        assert len(curve.tenors) > 0
        assert len(curve.rates) > 0
        assert "beta0" in curve.parameters
        assert "beta1" in curve.parameters
    
    def test_svensson_fitting(self):
        """Test Svensson model fitting."""
        bonds = [
            BondMarketData(
                bond=BondSpecification(
                    face_value=100,
                    coupon_rate=0.05,
                    maturity_date=datetime.now() + timedelta(days=365*t),
                    issue_date=datetime.now()
                ),
                clean_price=99.0,
                settlement_date=datetime.now(),
                time_to_maturity=t
            )
            for t in [1, 2, 3, 5, 7, 10, 15, 20]
        ]
        
        model = SvenssonModel()
        curve = model.fit(bonds)
        
        assert "beta3" in curve.parameters  # Svensson extension
        assert "lambda2" in curve.parameters
    
    def test_cubic_spline(self):
        """Test cubic spline interpolation."""
        tenors = np.array([1, 2, 3, 5, 7, 10])
        rates = np.array([0.03, 0.035, 0.04, 0.045, 0.048, 0.05])
        
        model = CubicSplineModel()
        curve = model.fit(tenors, rates)
        
        assert len(curve.tenors) > len(tenors)  # Interpolated points
        assert curve.model_type == "cubic_spline"
    
    def test_curve_interpolation(self, treasury_curve):
        """Test yield curve interpolation."""
        # Test at known point
        rate_5y = treasury_curve.get_rate(5.0)
        assert rate_5y == pytest.approx(0.045, rel=1e-6)
        
        # Test interpolation
        rate_4y = treasury_curve.get_rate(4.0)
        assert 0.042 < rate_4y < 0.045
    
    def test_forward_rate_calculation(self, treasury_curve):
        """Test forward rate calculation."""
        analyzer = YieldCurveAnalyzer()
        forward_rates = analyzer.calculate_forward_rates(treasury_curve)
        
        assert len(forward_rates) == len(treasury_curve.tenors) - 1
        assert all(r > 0 for r in forward_rates)
    
    def test_par_yield_calculation(self, treasury_curve):
        """Test par yield calculation."""
        analyzer = YieldCurveAnalyzer()
        par_yields = analyzer.calculate_par_yields(treasury_curve)
        
        assert len(par_yields) == len(treasury_curve.tenors)
        assert all(y > 0 for y in par_yields)
    
    def test_curve_shifting(self, treasury_curve):
        """Test parallel curve shift."""
        analyzer = YieldCurveAnalyzer()
        shifted = analyzer.shift_curve(treasury_curve, shift_bps=50, parallel=True)
        
        # All rates should increase by 50 bps
        diff = shifted.rates - treasury_curve.rates
        assert all(np.abs(diff - 0.005) < 1e-6)
    
    def test_discount_factor(self, treasury_curve):
        """Test discount factor calculation."""
        df = treasury_curve.get_discount_factor(5.0)
        
        assert 0 < df < 1
        assert df == pytest.approx(np.exp(-treasury_curve.get_rate(5.0) * 5), rel=1e-6)


# ============================================================================
# 3. DURATION & CONVEXITY TESTS (~10 tests)
# ============================================================================

class TestDuration:
    """Test duration and convexity calculations."""
    
    def test_macaulay_duration(self, sample_bond, settlement_date):
        """Test Macaulay duration calculation."""
        calc = DurationCalculator()
        duration = calc.calculate_macaulay_duration(
            bond=sample_bond,
            settlement_date=settlement_date,
            yield_rate=0.05
        )
        
        assert duration > 0
        assert duration < 10  # Should be less than maturity for coupon bond
    
    def test_modified_duration(self):
        """Test modified duration calculation."""
        calc = DurationCalculator()
        macaulay = 7.5
        modified = calc.calculate_modified_duration(
            macaulay_duration=macaulay,
            yield_rate=0.05,
            frequency=2
        )
        
        assert modified < macaulay
        assert modified == pytest.approx(macaulay / 1.025, rel=1e-6)
    
    def test_convexity(self, sample_bond, settlement_date):
        """Test convexity calculation."""
        calc = DurationCalculator()
        convexity = calc.calculate_convexity(
            bond=sample_bond,
            settlement_date=settlement_date,
            yield_rate=0.05
        )
        
        assert convexity > 0
    
    def test_dv01(self):
        """Test DV01 calculation."""
        calc = DurationCalculator()
        dv01 = calc.calculate_dv01(
            modified_duration=7.0,
            price=98.5
        )
        
        assert dv01 > 0
        assert dv01 == pytest.approx(7.0 * 98.5 / 10000, rel=1e-6)
    
    def test_all_duration_metrics(self, sample_bond, settlement_date):
        """Test calculation of all duration metrics."""
        calc = DurationCalculator()
        metrics = calc.calculate_all_metrics(
            bond=sample_bond,
            price=98.0,
            yield_rate=0.05,
            settlement_date=settlement_date
        )
        
        assert metrics.macaulay_duration > 0
        assert metrics.modified_duration > 0
        assert metrics.convexity > 0
        assert metrics.dv01 > 0
    
    def test_effective_duration(self, sample_bond, settlement_date):
        """Test effective duration calculation."""
        calc = DurationCalculator()
        eff_dur = calc.calculate_effective_duration(
            bond=sample_bond,
            base_price=98.0,
            settlement_date=settlement_date,
            base_yield=0.05,
            shock_bps=10
        )
        
        assert eff_dur > 0
    
    def test_hedge_ratio(self):
        """Test hedge ratio calculation."""
        hedger = DurationHedger()
        ratio = hedger.calculate_hedge_ratio(
            target_duration=7.0,
            hedge_duration=5.0,
            target_value=1000000
        )
        
        assert ratio < 0  # Hedge should be opposite direction
    
    def test_duration_convenience_function(self):
        """Test convenience duration function."""
        macaulay, modified = calculate_duration(
            coupon_rate=0.05,
            years_to_maturity=10,
            yield_rate=0.06
        )
        
        assert macaulay > 0
        assert modified < macaulay
    
    def test_performance_target(self, sample_bond, settlement_date):
        """Test that duration calculation meets <8ms target."""
        import time
        calc = DurationCalculator()
        
        start = time.perf_counter()
        calc.calculate_all_metrics(
            bond=sample_bond,
            price=98.0,
            yield_rate=0.05,
            settlement_date=settlement_date
        )
        execution_ms = (time.perf_counter() - start) * 1000
        
        assert execution_ms < 8.0, f"Duration calc took {execution_ms:.2f}ms, target is <8ms"


# ============================================================================
# 4. TERM STRUCTURE TESTS (~10 tests)
# ============================================================================

class TestTermStructure:
    """Test term structure models."""
    
    def test_vasicek_pricing(self):
        """Test Vasicek zero-coupon bond pricing."""
        model = VasicekModel()
        params = TermStructureParameters(
            initial_rate=0.05,
            mean_reversion_speed=0.1,
            long_term_mean=0.06,
            volatility=0.01
        )
        
        price = model.price_zero_coupon_bond(
            current_rate=0.05,
            time_to_maturity=5.0,
            params=params
        )
        
        assert 0 < price < 1  # Discount bond
    
    def test_vasicek_simulation(self):
        """Test Vasicek rate path simulation."""
        model = VasicekModel()
        params = TermStructureParameters(
            initial_rate=0.05,
            mean_reversion_speed=0.1,
            long_term_mean=0.06,
            volatility=0.01
        )
        
        paths = model.simulate_paths(
            params=params,
            n_paths=100,
            n_steps=50,
            time_horizon=5.0,
            seed=42
        )
        
        assert paths.shape == (100, 51)
        assert all(paths[:, 0] == 0.05)  # Initial rate
    
    def test_cir_pricing(self):
        """Test CIR zero-coupon bond pricing."""
        model = CIRModel()
        params = TermStructureParameters(
            initial_rate=0.05,
            mean_reversion_speed=0.15,
            long_term_mean=0.06,
            volatility=0.02
        )
        
        price = model.price_zero_coupon_bond(
            current_rate=0.05,
            time_to_maturity=10.0,
            params=params
        )
        
        assert 0 < price < 1
    
    def test_cir_simulation(self):
        """Test CIR rate path simulation."""
        model = CIRModel()
        params = TermStructureParameters(
            initial_rate=0.05,
            mean_reversion_speed=0.15,
            long_term_mean=0.06,
            volatility=0.02
        )
        
        paths = model.simulate_paths(
            params=params,
            n_paths=100,
            n_steps=50,
            time_horizon=5.0,
            method="euler",
            seed=42
        )
        
        assert paths.shape == (100, 51)
        assert all(paths[:, 0] == 0.05)
        assert all(paths.flatten() >= 0)  # CIR ensures non-negative rates
    
    def test_hull_white_model(self):
        """Test Hull-White model."""
        model = HullWhiteModel()
        params = TermStructureParameters(
            initial_rate=0.05,
            mean_reversion_speed=0.1,
            long_term_mean=0.06,
            volatility=0.01
        )
        
        price = model.price_zero_coupon_bond(
            current_rate=0.05,
            time_to_maturity=3.0,
            params=params
        )
        
        assert 0 < price < 1
    
    def test_ho_lee_lattice(self):
        """Test Ho-Lee binomial lattice."""
        model = HoLeeModel()
        lattice = model.build_lattice(
            initial_rate=0.05,
            volatility=0.01,
            n_steps=10,
            dt=0.1
        )
        
        assert lattice.shape == (11, 11)
        assert lattice[0, 0] == 0.05
    
    def test_feller_condition(self):
        """Test Feller condition check for CIR."""
        params = TermStructureParameters(
            initial_rate=0.05,
            mean_reversion_speed=0.1,
            long_term_mean=0.06,
            volatility=0.05  # High volatility
        )
        
        # Should log warning but not fail
        params.validate("cir")


# ============================================================================
# 5. SPREADS & CREDIT TESTS (~8 tests)
# ============================================================================

class TestSpreads:
    """Test spread and credit analytics."""
    
    def test_g_spread(self):
        """Test G-spread calculation."""
        analyzer = SpreadAnalyzer()
        g_spread = analyzer.calculate_g_spread(
            corporate_ytm=0.065,
            treasury_ytm=0.045
        )
        
        assert g_spread == pytest.approx(200, rel=1e-6)  # 200 bps
    
    def test_i_spread(self, treasury_curve):
        """Test I-spread calculation."""
        analyzer = SpreadAnalyzer()
        i_spread = analyzer.calculate_i_spread(
            bond_ytm=0.07,
            swap_curve=treasury_curve,
            time_to_maturity=5.0
        )
        
        assert i_spread > 0  # Corporate spread > treasury
    
    def test_z_spread(self, sample_bond, treasury_curve, settlement_date):
        """Test Z-spread calculation."""
        analyzer = SpreadAnalyzer()
        z_spread = analyzer.calculate_z_spread(
            bond=sample_bond,
            bond_price=95.0,
            treasury_curve=treasury_curve,
            settlement_date=settlement_date
        )
        
        assert z_spread > 0  # Positive spread for corporate bond
    
    def test_all_spreads(self, sample_bond, treasury_curve, settlement_date):
        """Test calculation of all spread metrics."""
        analyzer = SpreadAnalyzer()
        spreads = analyzer.calculate_all_spreads(
            bond=sample_bond,
            bond_price=97.0,
            bond_ytm=0.055,
            settlement_date=settlement_date,
            treasury_curve=treasury_curve,
            treasury_ytm=0.045
        )
        
        assert spreads.g_spread is not None
        assert spreads.z_spread is not None
    
    def test_default_probability_extraction(self):
        """Test default probability from spread."""
        analyzer = CreditSpreadAnalyzer()
        pd = analyzer.extract_default_probability(
            credit_spread=0.02,  # 200 bps
            time_to_maturity=5.0,
            recovery_rate=0.40
        )
        
        assert 0 < pd < 1
    
    def test_hazard_rate(self):
        """Test hazard rate calculation."""
        analyzer = CreditSpreadAnalyzer()
        hazard = analyzer.calculate_hazard_rate(
            credit_spread=0.015,
            recovery_rate=0.40
        )
        
        assert hazard > 0
    
    def test_richness_cheapness(self):
        """Test relative value analysis."""
        analyzer = RelativeValueAnalyzer()
        result = analyzer.calculate_richness_cheapness(
            market_spread=180,  # bps
            model_spread=200  # bps
        )
        
        assert result["classification"] == "RICH"  # Trading tight to model
    
    def test_butterfly_spread(self):
        """Test butterfly spread calculation."""
        analyzer = RelativeValueAnalyzer()
        butterfly = analyzer.calculate_butterfly_spread(
            short_spread=100,
            mid_spread=150,
            long_spread=180
        )
        
        # 2*150 - (100 + 180) = 20
        assert butterfly == pytest.approx(20, rel=1e-6)


# ============================================================================
# 6. PORTFOLIO TESTS (~10 tests)
# ============================================================================

class TestPortfolio:
    """Test portfolio analytics."""
    
    def test_portfolio_metrics(self, sample_bond, settlement_date):
        """Test portfolio metrics calculation."""
        holdings = [
            BondHolding(
                bond=sample_bond,
                quantity=100,
                market_value=9800,
                book_value=10000,
                weight=0.5,
                rating="AAA",
                sector="Corporate"
            ),
            BondHolding(
                bond=sample_bond,
                quantity=100,
                market_value=9800,
                book_value=10000,
                weight=0.5,
                rating="AA",
                sector="Financial"
            )
        ]
        
        analyzer = BondPortfolioAnalyzer()
        metrics = analyzer.calculate_portfolio_metrics(
            holdings=holdings,
            settlement_date=settlement_date
        )
        
        assert metrics.total_market_value == pytest.approx(19600, rel=1e-6)
        assert metrics.n_holdings == 2
        assert metrics.portfolio_duration > 0
    
    def test_performance_attribution(self, sample_bond):
        """Test performance attribution."""
        holdings_start = [
            BondHolding(
                bond=sample_bond,
                quantity=100,
                market_value=9800,
                book_value=10000,
                weight=1.0
            )
        ]
        
        holdings_end = [
            BondHolding(
                bond=sample_bond,
                quantity=100,
                market_value=9900,
                book_value=10000,
                weight=1.0
            )
        ]
        
        analyzer = BondPortfolioAnalyzer()
        attribution = analyzer.calculate_performance_attribution(
            holdings_start=holdings_start,
            holdings_end=holdings_end,
            coupon_income=250,
            period_days=30
        )
        
        assert "total_return" in attribution
        assert "yield_return" in attribution
        assert "price_return" in attribution
    
    def test_scenario_analysis(self, sample_bond, settlement_date):
        """Test scenario analysis."""
        holdings = [
            BondHolding(
                bond=sample_bond,
                quantity=100,
                market_value=9800,
                book_value=10000,
                weight=1.0
            )
        ]
        
        scenarios = [
            {"name": "Rates +100bp", "parallel_shift_bps": 100},
            {"name": "Rates -50bp", "parallel_shift_bps": -50}
        ]
        
        analyzer = BondPortfolioAnalyzer()
        results = analyzer.run_scenario_analysis(
            holdings=holdings,
            settlement_date=settlement_date,
            scenarios=scenarios
        )
        
        assert len(results["scenarios"]) == 2
        assert results["base_value"] > 0
    
    def test_concentration_analysis(self, sample_bond):
        """Test concentration risk analysis."""
        holdings = [
            BondHolding(
                bond=sample_bond,
                quantity=100,
                market_value=8000,
                book_value=10000,
                weight=0.8,
                sector="Corporate"
            ),
            BondHolding(
                bond=sample_bond,
                quantity=25,
                market_value=2000,
                book_value=2500,
                weight=0.2,
                sector="Financial"
            )
        ]
        
        analyzer = BondPortfolioAnalyzer()
        analysis = analyzer.analyze_concentration_risk(holdings)
        
        assert "hhi" in analysis
        assert "sector_exposure" in analysis
        # Should have breach for 80% in single sector
        assert analysis["total_breaches"] > 0
    
    def test_portfolio_duration_calculation(self):
        """Test portfolio duration calculation."""
        durations = [5.0, 7.0, 10.0]
        weights = [0.3, 0.4, 0.3]
        
        port_duration = calculate_portfolio_duration(durations, weights)
        
        expected = 0.3*5 + 0.4*7 + 0.3*10
        assert port_duration == pytest.approx(expected, rel=1e-6)
    
    def test_performance_target(self, sample_bond, settlement_date):
        """Test that portfolio analytics meet <100ms target for 100 bonds."""
        import time
        
        # Create 100 bonds
        holdings = [
            BondHolding(
                bond=sample_bond,
                quantity=10,
                market_value=980,
                book_value=1000,
                weight=0.01,
                rating="AAA",
                sector=f"Sector{i%10}"
            )
            for i in range(100)
        ]
        
        analyzer = BondPortfolioAnalyzer()
        
        start = time.perf_counter()
        analyzer.calculate_portfolio_metrics(
            holdings=holdings,
            settlement_date=settlement_date
        )
        execution_ms = (time.perf_counter() - start) * 1000
        
        assert execution_ms < 100.0, f"Portfolio analytics took {execution_ms:.2f}ms, target is <100ms"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across multiple modules."""
    
    def test_full_bond_analysis_workflow(self, sample_bond, treasury_curve, settlement_date):
        """Test complete bond analysis workflow."""
        # 1. Price the bond
        pricer = BondPricingModel()
        price_result = pricer.calculate_price(
            bond=sample_bond,
            settlement_date=settlement_date,
            yield_rate=0.055
        )
        
        # 2. Calculate duration metrics
        duration_calc = DurationCalculator()
        duration_metrics = duration_calc.calculate_all_metrics(
            bond=sample_bond,
            price=price_result.clean_price,
            yield_rate=0.055,
            settlement_date=settlement_date
        )
        
        # 3. Calculate spreads
        spread_analyzer = SpreadAnalyzer()
        spreads = spread_analyzer.calculate_all_spreads(
            bond=sample_bond,
            bond_price=price_result.clean_price,
            bond_ytm=0.055,
            settlement_date=settlement_date,
            treasury_curve=treasury_curve,
            treasury_ytm=0.045
        )
        
        # Verify all components worked
        assert price_result.clean_price > 0
        assert duration_metrics.modified_duration > 0
        assert spreads.g_spread is not None
    
    def test_portfolio_with_curves(self, sample_bond, treasury_curve, settlement_date):
        """Test portfolio analysis with yield curves."""
        holdings = [
            BondHolding(
                bond=sample_bond,
                quantity=100,
                market_value=9800,
                book_value=10000,
                weight=1.0
            )
        ]
        
        analyzer = BondPortfolioAnalyzer()
        metrics = analyzer.calculate_portfolio_metrics(
            holdings=holdings,
            settlement_date=settlement_date,
            yield_curve=treasury_curve
        )
        
        assert metrics.total_market_value > 0
        assert metrics.portfolio_duration > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])