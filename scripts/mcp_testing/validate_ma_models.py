"""
Quick validation script for M&A quantitative models.
Tests basic functionality without requiring pytest.
"""

import sys
import traceback

def test_imports():
    """Test that all models can be imported."""
    print("Testing imports...")
    try:
        from axiom.models.ma import (
            SynergyValuationModel,
            DealFinancingModel,
            MergerArbitrageModel,
            LBOModel,
            ValuationIntegrationModel,
            DealScreeningModel,
            CostSynergy,
            RevenueSynergy,
            OperationalImprovements,
            DealMetrics
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_synergy_model():
    """Test synergy valuation model."""
    print("\nTesting Synergy Valuation Model...")
    try:
        from axiom.models.ma import SynergyValuationModel, CostSynergy, RevenueSynergy
        
        model = SynergyValuationModel()
        
        cost_synergies = [
            CostSynergy("Test cost", 10_000_000, 1, category="operating")
        ]
        revenue_synergies = [
            RevenueSynergy("Test revenue", 15_000_000, 2, category="cross_sell")
        ]
        
        result = model.calculate(
            cost_synergies=cost_synergies,
            revenue_synergies=revenue_synergies,
            run_monte_carlo=False,
            run_sensitivity=False
        )
        
        assert result.success
        assert result.value.total_synergies_npv > 0
        assert result.metadata.execution_time_ms < 100
        
        print(f"✓ Synergy model works - NPV: ${result.value.total_synergies_npv:,.0f}")
        print(f"  Execution time: {result.metadata.execution_time_ms:.1f}ms")
        return True
    except Exception as e:
        print(f"✗ Synergy model failed: {e}")
        traceback.print_exc()
        return False


def test_financing_model():
    """Test deal financing model."""
    print("\nTesting Deal Financing Model...")
    try:
        from axiom.models.ma import DealFinancingModel
        
        model = DealFinancingModel()
        
        result = model.calculate(
            purchase_price=500_000_000,
            target_ebitda=50_000_000,
            acquirer_market_cap=2_000_000_000,
            acquirer_shares_outstanding=100_000_000,
            acquirer_eps=5.00
        )
        
        assert result.success
        assert result.value.wacc > 0
        assert result.metadata.execution_time_ms < 100
        
        print(f"✓ Financing model works - WACC: {result.value.wacc:.2%}")
        print(f"  EPS Impact: ${result.value.eps_impact:.2f}")
        print(f"  Execution time: {result.metadata.execution_time_ms:.1f}ms")
        return True
    except Exception as e:
        print(f"✗ Financing model failed: {e}")
        traceback.print_exc()
        return False


def test_arbitrage_model():
    """Test merger arbitrage model."""
    print("\nTesting Merger Arbitrage Model...")
    try:
        from axiom.models.ma import MergerArbitrageModel
        
        model = MergerArbitrageModel()
        
        result = model.calculate(
            target_price=48.50,
            offer_price=52.00,
            days_to_close=90
        )
        
        assert result.success
        assert result.value.deal_spread > 0
        assert result.metadata.execution_time_ms < 100
        
        print(f"✓ Merger arb model works - Spread: {result.value.deal_spread:.2%}")
        print(f"  Annualized Return: {result.value.annualized_return:.2%}")
        print(f"  Execution time: {result.metadata.execution_time_ms:.1f}ms")
        return True
    except Exception as e:
        print(f"✗ Merger arb model failed: {e}")
        traceback.print_exc()
        return False


def test_lbo_model():
    """Test LBO model."""
    print("\nTesting LBO Model...")
    try:
        from axiom.models.ma import LBOModel
        
        model = LBOModel()
        
        result = model.calculate(
            entry_ebitda=100_000_000,
            entry_multiple=10.0,
            holding_period=5,
            leverage_multiple=5.0
        )
        
        assert result.success
        assert result.value.irr > 0
        assert result.value.cash_on_cash > 0
        assert result.metadata.execution_time_ms < 150
        
        print(f"✓ LBO model works - IRR: {result.value.irr:.1%}")
        print(f"  Cash-on-Cash: {result.value.cash_on_cash:.2f}x")
        print(f"  Execution time: {result.metadata.execution_time_ms:.1f}ms")
        return True
    except Exception as e:
        print(f"✗ LBO model failed: {e}")
        traceback.print_exc()
        return False


def test_valuation_model():
    """Test valuation integration model."""
    print("\nTesting Valuation Integration Model...")
    try:
        from axiom.models.ma import ValuationIntegrationModel
        
        model = ValuationIntegrationModel()
        
        result = model.calculate(
            target_fcf=[50_000_000, 55_000_000, 60_000_000],
            discount_rate=0.10
        )
        
        assert result.success
        assert result.value.dcf_value > 0
        assert result.metadata.execution_time_ms < 100
        
        print(f"✓ Valuation model works - DCF Value: ${result.value.dcf_value:,.0f}")
        print(f"  Recommended Value: ${result.value.recommended_value:,.0f}")
        print(f"  Execution time: {result.metadata.execution_time_ms:.1f}ms")
        return True
    except Exception as e:
        print(f"✗ Valuation model failed: {e}")
        traceback.print_exc()
        return False


def test_screening_model():
    """Test deal screening model."""
    print("\nTesting Deal Screening Model...")
    try:
        from axiom.models.ma import DealScreeningModel, DealMetrics
        
        model = DealScreeningModel()
        
        metrics = DealMetrics(
            ebitda_multiple=10.5,
            revenue_multiple=2.0,
            ebitda_margin=0.20,
            revenue_growth=0.15,
            market_share=0.12,
            customer_concentration=0.15,
            technology_score=8.0
        )
        
        result = model.calculate(
            deal_id="TEST-001",
            deal_metrics=metrics,
            strategic_factors={'market_adjacency': 8.0},
            financial_factors={'cash_flow_quality': 8.0},
            risk_factors={'regulatory_risk': 3.0},
            purchase_price=100_000_000
        )
        
        assert result.success
        assert 0 <= result.value.overall_score <= 100
        assert result.metadata.execution_time_ms < 100
        
        print(f"✓ Screening model works - Overall Score: {result.value.overall_score:.1f}/100")
        print(f"  Recommendation: {result.value.recommendation}")
        print(f"  Execution time: {result.metadata.execution_time_ms:.1f}ms")
        return True
    except Exception as e:
        print(f"✗ Screening model failed: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test M&A configuration."""
    print("\nTesting M&A Configuration...")
    try:
        from axiom.config.model_config import MandAConfig
        
        # Test default config
        config = MandAConfig()
        assert config.synergy_discount_rate == 0.12
        assert config.target_irr == 0.20
        
        # Test conservative config
        conservative = MandAConfig.for_conservative_approach()
        assert conservative.synergy_discount_rate > config.synergy_discount_rate
        
        # Test aggressive config
        aggressive = MandAConfig.for_aggressive_approach()
        assert aggressive.synergy_discount_rate < config.synergy_discount_rate
        
        print("✓ Configuration system works")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 80)
    print(" " * 20 + "M&A QUANTITATIVE MODELS VALIDATION")
    print("=" * 80)
    
    tests = [
        test_imports,
        test_configuration,
        test_synergy_model,
        test_financing_model,
        test_arbitrage_model,
        test_lbo_model,
        test_valuation_model,
        test_screening_model
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 80)
    print(f"VALIDATION SUMMARY: {sum(results)}/{len(results)} tests passed")
    print("=" * 80)
    
    if all(results):
        print("\n✅ ALL VALIDATION TESTS PASSED!")
        print("M&A quantitative models are ready for production use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())