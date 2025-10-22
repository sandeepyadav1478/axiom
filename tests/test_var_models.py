"""
Comprehensive tests for Value at Risk (VaR) models.
Tests all three VaR methodologies and portfolio calculations.
"""

import numpy as np
import pytest
from axiom.models.risk import (
    VaRCalculator,
    VaRMethod,
    ParametricVaR,
    HistoricalSimulationVaR,
    MonteCarloVaR,
    calculate_portfolio_var,
    calculate_marginal_var,
    quick_var,
    regulatory_var
)


class TestParametricVaR:
    """Test Parametric VaR calculations."""
    
    def test_basic_parametric_var(self):
        """Test basic Parametric VaR calculation."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        result = ParametricVaR.calculate(
            portfolio_value=portfolio_value,
            returns=returns,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        assert result.var_amount > 0
        assert 0 < result.var_percentage < 1
        assert result.method == VaRMethod.PARAMETRIC
        assert result.confidence_level == 0.95
        assert result.portfolio_value == portfolio_value
        assert result.expected_shortfall is not None
        assert result.expected_shortfall > result.var_amount
    
    def test_parametric_var_different_confidence_levels(self):
        """Test that VaR increases with confidence level."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        var_90 = ParametricVaR.calculate(portfolio_value, returns, 0.90, 1)
        var_95 = ParametricVaR.calculate(portfolio_value, returns, 0.95, 1)
        var_99 = ParametricVaR.calculate(portfolio_value, returns, 0.99, 1)
        
        # VaR should increase with confidence level
        assert var_90.var_amount < var_95.var_amount < var_99.var_amount
    
    def test_parametric_var_time_scaling(self):
        """Test that VaR scales with time horizon."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        var_1d = ParametricVaR.calculate(portfolio_value, returns, 0.95, 1)
        var_10d = ParametricVaR.calculate(portfolio_value, returns, 0.95, 10)
        
        # 10-day VaR should be larger than 1-day VaR
        assert var_10d.var_amount > var_1d.var_amount
        # Should roughly scale by sqrt(10)
        scaling_factor = var_10d.var_amount / var_1d.var_amount
        assert 2.5 < scaling_factor < 4.0  # sqrt(10) ≈ 3.16


class TestHistoricalSimulationVaR:
    """Test Historical Simulation VaR calculations."""
    
    def test_basic_historical_var(self):
        """Test basic Historical Simulation VaR."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        result = HistoricalSimulationVaR.calculate(
            portfolio_value=portfolio_value,
            returns=returns,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        assert result.var_amount > 0
        assert result.method == VaRMethod.HISTORICAL
        assert result.portfolio_value == portfolio_value
        assert result.expected_shortfall >= result.var_amount
    
    def test_historical_var_percentile_logic(self):
        """Test that Historical VaR correctly identifies percentile."""
        # Create known distribution
        returns = np.array([-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.10])
        portfolio_value = 1_000_000
        
        # 90% confidence = 10% VaR (worst 10% of outcomes)
        result = HistoricalSimulationVaR.calculate(portfolio_value, returns, 0.90, 1)
        
        # With 10 observations, 90% confidence should give us the worst observation
        expected_var_pct = abs(np.percentile(returns, 10))
        assert abs(result.var_percentage - expected_var_pct) < 0.001
    
    def test_historical_var_multi_day(self):
        """Test Historical VaR with multi-day horizon."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 500)
        
        var_1d = HistoricalSimulationVaR.calculate(portfolio_value, returns, 0.95, 1)
        var_5d = HistoricalSimulationVaR.calculate(portfolio_value, returns, 0.95, 5)
        
        # Multi-day VaR should be larger
        assert var_5d.var_amount > var_1d.var_amount


class TestMonteCarloVaR:
    """Test Monte Carlo VaR calculations."""
    
    def test_basic_monte_carlo_var(self):
        """Test basic Monte Carlo VaR calculation."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        result = MonteCarloVaR.calculate(
            portfolio_value=portfolio_value,
            returns=returns,
            confidence_level=0.95,
            time_horizon_days=1,
            num_simulations=1000,
            random_seed=42  # For reproducibility
        )
        
        assert result.method == VaRMethod.MONTE_CARLO
        assert result.portfolio_value == portfolio_value
        assert result.metadata["num_simulations"] == 1000
    
    def test_monte_carlo_reproducibility(self):
        """Test that Monte Carlo is reproducible with same seed."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        result1 = MonteCarloVaR.calculate(
            portfolio_value, returns, 0.95, 1, 1000, random_seed=42
        )
        result2 = MonteCarloVaR.calculate(
            portfolio_value, returns, 0.95, 1, 1000, random_seed=42
        )
        
        # Same seed should give identical results
        assert abs(result1.var_amount - result2.var_amount) < 0.01
    
    def test_monte_carlo_simulation_count(self):
        """Test that more simulations improve stability."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        # Run with different simulation counts
        var_100 = MonteCarloVaR.calculate(portfolio_value, returns, 0.95, 1, 100, 42)
        var_10000 = MonteCarloVaR.calculate(portfolio_value, returns, 0.95, 1, 10000, 42)
        
        # Results should be in similar range (not testing exact equality due to randomness)
        ratio = var_10000.var_amount / var_100.var_amount if var_100.var_amount != 0 else 0
        assert 0.5 < abs(ratio) < 2.0  # Within reasonable range


class TestVaRCalculator:
    """Test unified VaR calculator."""
    
    def test_calculator_initialization(self):
        """Test VaR calculator initialization."""
        calculator = VaRCalculator(default_confidence=0.95)
        assert calculator.default_confidence == 0.95
        assert len(calculator.calculation_history) == 0
    
    def test_calculate_all_methods(self):
        """Test calculating VaR with all methods."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        calculator = VaRCalculator()
        all_results = calculator.calculate_all_methods(
            portfolio_value=portfolio_value,
            returns=returns,
            confidence_level=0.95,
            time_horizon_days=1,
            num_simulations=1000
        )
        
        assert "parametric" in all_results
        assert "historical" in all_results
        assert "monte_carlo" in all_results
        
        # All should have VaR amounts
        for method, result in all_results.items():
            assert result.var_amount is not None
            assert result.confidence_level == 0.95
    
    def test_var_summary(self):
        """Test VaR summary statistics."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        calculator = VaRCalculator()
        all_results = calculator.calculate_all_methods(portfolio_value, returns, 0.95, 1)
        summary = calculator.get_var_summary(all_results)
        
        assert "var_range" in summary
        assert "method_comparison" in summary
        assert summary["portfolio_value"] == portfolio_value
        assert summary["confidence_level"] == 0.95


class TestPortfolioVaR:
    """Test portfolio-level VaR calculations."""
    
    def test_portfolio_var_calculation(self):
        """Test multi-asset portfolio VaR."""
        positions = {
            "AAPL": {"value": 400_000, "weight": 0.40},
            "MSFT": {"value": 350_000, "weight": 0.35},
            "GOOGL": {"value": 250_000, "weight": 0.25}
        }
        
        returns_data = {
            "AAPL": np.random.normal(0.0005, 0.018, 252),
            "MSFT": np.random.normal(0.0005, 0.016, 252),
            "GOOGL": np.random.normal(0.0007, 0.020, 252)
        }
        
        result = calculate_portfolio_var(
            positions=positions,
            returns_data=returns_data,
            method=VaRMethod.HISTORICAL,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        assert result.portfolio_value == 1_000_000
        assert result.var_amount > 0
        assert result.method == VaRMethod.HISTORICAL


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_quick_var(self):
        """Test quick_var convenience function."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        var_amount = quick_var(portfolio_value, returns, confidence_level=0.95)
        
        assert var_amount > 0
        assert var_amount < portfolio_value  # Sanity check
    
    def test_regulatory_var(self):
        """Test regulatory VaR (Basel III standard)."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        result = regulatory_var(portfolio_value, returns)
        
        assert result.confidence_level == 0.99  # Basel III standard
        assert result.time_horizon_days == 10  # Basel III standard
        assert result.var_amount > 0


class TestVaRValidation:
    """Test VaR model validation and edge cases."""
    
    def test_var_with_zero_volatility(self):
        """Test VaR with zero volatility returns."""
        portfolio_value = 1_000_000
        returns = np.zeros(252)  # Zero returns
        
        result = ParametricVaR.calculate(portfolio_value, returns, 0.95, 1)
        
        # With zero volatility, VaR should be very small (just mean)
        assert result.var_percentage < 0.01
    
    def test_var_with_extreme_losses(self):
        """Test VaR captures extreme losses."""
        portfolio_value = 1_000_000
        # Create distribution with more extreme losses
        returns = np.concatenate([
            np.random.normal(0.001, 0.01, 240),
            np.array([-0.10, -0.10, -0.12, -0.12, -0.15, -0.15, -0.18, -0.18, -0.20, -0.20, -0.22, -0.25])  # More extreme losses
        ])
        
        result = HistoricalSimulationVaR.calculate(portfolio_value, returns, 0.95, 1)
        
        # VaR should capture the extreme losses (5% tail = ~13 worst observations)
        assert result.var_percentage > 0.01  # Should be at least 1%
    
    def test_var_result_serialization(self):
        """Test VaR result can be serialized."""
        portfolio_value = 1_000_000
        returns = np.random.normal(0.001, 0.02, 252)
        
        result = ParametricVaR.calculate(portfolio_value, returns, 0.95, 1)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "var_amount" in result_dict
        assert "confidence_level" in result_dict
        assert "method" in result_dict
        assert result_dict["method"] == "parametric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])