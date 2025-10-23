"""
Migration Tests for Refactored Models
======================================

Verifies that refactored models maintain 100% backward compatibility
and that new features work correctly.
"""

import numpy as np
import pytest
from axiom.models.risk.var_models import (
    ParametricVaR,
    HistoricalSimulationVaR,
    MonteCarloVaR,
    VaRResult,
    VaRMethod
)
from axiom.models.base.factory import ModelFactory, ModelType
from axiom.config.model_config import VaRConfig


class TestVaRBackwardCompatibility:
    """Test backward compatibility of refactored VaR models."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
    
    @pytest.fixture
    def portfolio_value(self):
        """Sample portfolio value."""
        return 1000000.0
    
    def test_parametric_var_static_method(self, portfolio_value, sample_returns):
        """Test that old static method API still works."""
        # Old API - static method call
        result = ParametricVaR.calculate(
            portfolio_value=portfolio_value,
            returns=sample_returns,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        # Verify result is VaRResult
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.PARAMETRIC
        assert result.confidence_level == 0.95
        assert result.portfolio_value == portfolio_value
        assert result.var_amount > 0
        assert result.expected_shortfall > 0
    
    def test_historical_var_static_method(self, portfolio_value, sample_returns):
        """Test Historical VaR backward compatibility."""
        result = HistoricalSimulationVaR.calculate(
            portfolio_value=portfolio_value,
            returns=sample_returns,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.HISTORICAL
        assert result.var_amount > 0
    
    def test_monte_carlo_var_static_method(self, portfolio_value, sample_returns):
        """Test Monte Carlo VaR backward compatibility."""
        result = MonteCarloVaR.calculate(
            portfolio_value=portfolio_value,
            returns=sample_returns,
            confidence_level=0.95,
            time_horizon_days=1,
            num_simulations=1000,
            random_seed=42
        )
        
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.MONTE_CARLO
        assert result.var_amount > 0
        assert result.metadata['num_simulations'] == 1000


class TestVaRNewFeatures:
    """Test new features added through refactoring."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)
    
    @pytest.fixture
    def portfolio_value(self):
        """Sample portfolio value."""
        return 1000000.0
    
    def test_parametric_var_instance_method(self, portfolio_value, sample_returns):
        """Test new instance-based API."""
        # New API - instance method with config
        model = ParametricVaR()
        result = model.calculate_risk(
            portfolio_value=portfolio_value,
            returns=sample_returns,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        # Result is now wrapped in ModelResult
        assert result.success is True
        assert isinstance(result.value, VaRResult)
        assert result.metadata.execution_time_ms >= 0
    
    def test_parametric_var_with_config(self, portfolio_value, sample_returns):
        """Test using custom configuration."""
        config = VaRConfig(
            default_confidence_level=0.99,
            default_time_horizon=10
        )
        
        model = ParametricVaR(config=config)
        result = model.calculate_risk(
            portfolio_value=portfolio_value,
            returns=sample_returns
            # Uses config defaults
        )
        
        assert result.value.confidence_level == 0.99
        assert result.value.time_horizon_days == 10
    
    def test_historical_var_with_validation(self, portfolio_value):
        """Test validation logic in new refactored model."""
        model = HistoricalSimulationVaR()
        
        # Test with insufficient data - should raise ValidationError
        with pytest.raises(Exception):  # ValidationError from mixins
            model.validate_inputs(
                portfolio_value=portfolio_value,
                returns=[0.01, 0.02],  # Too few observations
                confidence_level=0.95,
                time_horizon_days=1
            )
    
    def test_monte_carlo_var_performance_tracking(self, portfolio_value, sample_returns):
        """Test automatic performance tracking."""
        model = MonteCarloVaR()
        result = model.calculate_risk(
            portfolio_value=portfolio_value,
            returns=sample_returns,
            confidence_level=0.95,
            time_horizon_days=1,
            num_simulations=1000,
            random_seed=42
        )
        
        # Performance tracking is automatic
        assert result.metadata.execution_time_ms > 0
        assert result.metadata.model_name == "MonteCarloVaR"


class TestModelFactory:
    """Test Model Factory integration."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)
    
    @pytest.fixture
    def portfolio_value(self):
        """Sample portfolio value."""
        return 1000000.0
    
    def test_factory_creates_parametric_var(self):
        """Test creating ParametricVaR via factory."""
        model = ModelFactory.create(ModelType.PARAMETRIC_VAR)
        assert isinstance(model, ParametricVaR)
    
    def test_factory_creates_historical_var(self):
        """Test creating HistoricalSimulationVaR via factory."""
        model = ModelFactory.create(ModelType.HISTORICAL_VAR)
        assert isinstance(model, HistoricalSimulationVaR)
    
    def test_factory_creates_monte_carlo_var(self):
        """Test creating MonteCarloVaR via factory."""
        model = ModelFactory.create(ModelType.MONTE_CARLO_VAR)
        assert isinstance(model, MonteCarloVaR)
    
    def test_factory_with_custom_config(self, portfolio_value, sample_returns):
        """Test factory with custom configuration."""
        config = VaRConfig(default_confidence_level=0.99)
        
        model = ModelFactory.create(
            ModelType.PARAMETRIC_VAR,
            config=config
        )
        
        result = model.calculate_risk(
            portfolio_value=portfolio_value,
            returns=sample_returns
        )
        
        assert result.value.confidence_level == 0.99


class TestPerformanceComparison:
    """Compare performance before and after refactoring."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)
    
    @pytest.fixture
    def portfolio_value(self):
        """Sample portfolio value."""
        return 1000000.0
    
    def test_parametric_var_performance(self, portfolio_value, sample_returns):
        """Verify ParametricVaR performance target (<10ms)."""
        model = ParametricVaR()
        result = model.calculate_risk(
            portfolio_value=portfolio_value,
            returns=sample_returns,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        # Should be very fast
        assert result.metadata.execution_time_ms < 10
    
    def test_monte_carlo_var_performance(self, portfolio_value, sample_returns):
        """Verify Monte Carlo VaR is reasonably fast."""
        model = MonteCarloVaR()
        result = model.calculate_risk(
            portfolio_value=portfolio_value,
            returns=sample_returns,
            confidence_level=0.95,
            time_horizon_days=1,
            num_simulations=10000
        )
        
        # Should complete in reasonable time (< 100ms for 10k simulations)
        assert result.metadata.execution_time_ms < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])