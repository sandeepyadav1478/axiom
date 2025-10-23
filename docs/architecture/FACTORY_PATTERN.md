# Factory Pattern Architecture

**Centralized Model Creation with Dependency Injection**

## Overview

The Factory Pattern provides a single point of entry for creating any financial model in Axiom, with automatic configuration injection and plugin support. This eliminates the need to import specific model classes and manage their dependencies manually.

**Location**: [`axiom/models/base/factory.py`](../../axiom/models/base/factory.py)

## Benefits

| Benefit | Description |
|---------|-------------|
| **Single Entry Point** | Create any model through `ModelFactory.create()` |
| **Configuration Injection** | Automatic config loading from profiles, env, or files |
| **Type Safety** | [`ModelType`](../../axiom/models/base/factory.py:24) enum prevents typos |
| **Plugin Support** | Register custom models without modifying core code |
| **Easy Testing** | Mock factory for unit tests |
| **Discoverability** | List all available models programmatically |

## Basic Usage

### Creating Models

```python
from axiom.models.base.factory import ModelFactory, ModelType

# Create VaR model with default configuration
var_model = ModelFactory.create(ModelType.HISTORICAL_VAR)

# Create ARIMA model
arima_model = ModelFactory.create(ModelType.ARIMA)

# Create portfolio optimizer
optimizer = ModelFactory.create(ModelType.MARKOWITZ_OPTIMIZER)

# Use the models
result = var_model.calculate_risk(
    portfolio_value=1_000_000,
    returns=historical_returns,
    confidence_level=0.95
)
```

### With Custom Configuration

```python
from axiom.config.model_config import VaRConfig, ModelConfig

# Method 1: Direct configuration object
custom_config = VaRConfig(
    default_confidence_level=0.99,
    default_simulations=50000,
    parallel_mc=True
)
model = ModelFactory.create(ModelType.MONTE_CARLO_VAR, config=custom_config)

# Method 2: Use configuration profile
basel_config = ModelConfig.for_basel_iii_compliance()
model = ModelFactory.create(ModelType.PARAMETRIC_VAR, config=basel_config.var)

# Method 3: Pass configuration dictionary
model = ModelFactory.create(
    ModelType.HISTORICAL_VAR,
    config={"default_confidence_level": 0.99, "min_observations": 500}
)

# Method 4: Override individual parameters
model = ModelFactory.create(
    ModelType.MONTE_CARLO_VAR,
    default_simulations=100000,  # Override specific parameter
    parallel_mc=True
)
```

## ModelType Enumeration

All available models are defined in [`ModelType`](../../axiom/models/base/factory.py:24):

```python
class ModelType(Enum):
    """Enumeration of all available financial model types."""
    
    # Options Models
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL_TREE = "binomial_tree"
    MONTE_CARLO_OPTIONS = "monte_carlo_options"
    GREEKS_CALCULATOR = "greeks_calculator"
    IMPLIED_VOLATILITY = "implied_volatility"
    
    # Credit Risk Models
    KMV_MERTON_PD = "kmv_merton_pd"
    ALTMAN_Z_SCORE = "altman_z_score"
    CREDIT_VAR = "credit_var"
    PORTFOLIO_CREDIT_RISK = "portfolio_credit_risk"
    
    # VaR Models
    PARAMETRIC_VAR = "parametric_var"
    HISTORICAL_VAR = "historical_var"
    MONTE_CARLO_VAR = "monte_carlo_var"
    
    # Portfolio Models
    MARKOWITZ_OPTIMIZER = "markowitz_optimizer"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    
    # Time Series Models
    ARIMA = "arima"
    GARCH = "garch"
    EWMA = "ewma"
```

### Listing Available Models

```python
# List all registered models
models = ModelFactory.list_models()
for model_type, description in models.items():
    print(f"{model_type}: {description}")

# Output:
# historical_var: Historical simulation VaR using empirical distribution
# monte_carlo_var: Monte Carlo VaR with simulated scenarios
# arima: ARIMA(p,d,q) for price forecasting and trend prediction
# ...

# Get information about specific model
info = ModelFactory.get_model_info(ModelType.HISTORICAL_VAR)
print(f"Model class: {info.model_class}")
print(f"Config key: {info.config_key}")
print(f"Description: {info.description}")
```

## Model Registration

### Built-in Model Registration

Models are automatically registered on module load:

```python
# In axiom/models/base/factory.py
def _init_builtin_models():
    """Initialize registry with Axiom's built-in models."""
    
    # VaR Models
    ModelFactory.register_model(
        ModelType.PARAMETRIC_VAR.value,
        ParametricVaR,
        config_key="var",
        description="Parametric VaR using variance-covariance method"
    )
    
    ModelFactory.register_model(
        ModelType.HISTORICAL_VAR.value,
        HistoricalSimulationVaR,
        config_key="var",
        description="Historical simulation VaR using empirical distribution"
    )
    
    # Time Series Models
    ModelFactory.register_model(
        ModelType.ARIMA.value,
        ARIMAModel,
        config_key="time_series",
        description="ARIMA(p,d,q) for price forecasting"
    )
```

### Custom Model Registration

Register your own models without modifying core code:

```python
from axiom.models.base.base_model import BaseRiskModel
from axiom.models.base.factory import ModelFactory

# Define custom model
class CustomVaRModel(BaseRiskModel):
    """My custom VaR implementation."""
    
    def calculate_risk(self, **kwargs):
        # Your custom logic
        pass
    
    def validate_inputs(self, **kwargs):
        # Your validation
        pass

# Register with factory
ModelFactory.register_model(
    "custom_var",
    CustomVaRModel,
    config_key="var",
    description="Custom VaR methodology"
)

# Now create like any other model
custom_var = ModelFactory.create("custom_var")
```

## Plugin Manager

For more advanced plugin scenarios, use [`PluginManager`](../../axiom/models/base/factory.py:268):

```python
from axiom.models.base.factory import PluginManager

# Register plugin
PluginManager.register_plugin(
    "advanced_var",
    AdvancedVaRModel,
    config_key="var",
    description="Advanced VaR with custom features",
    override=False  # Prevent accidental override
)

# List plugins
plugins = PluginManager.list_plugins()
for plugin_name, description in plugins.items():
    print(f"{plugin_name}: {description}")

# Unregister plugin
PluginManager.unregister_plugin("advanced_var")
```

## Configuration Injection

The factory automatically injects configuration based on the model type:

```python
# Factory determines which config section to use
# VaR models → var config section
# Time series models → time_series config section
# Portfolio models → portfolio config section

# This happens automatically:
model = ModelFactory.create(ModelType.HISTORICAL_VAR)
# ↓
# 1. Look up model registration
# 2. Get config_key = "var"
# 3. Load global_config.var
# 4. Pass to model constructor
# 5. Return instantiated model

# You can override at any step:
custom_config = VaRConfig(default_confidence_level=0.99)
model = ModelFactory.create(ModelType.HISTORICAL_VAR, config=custom_config)
```

## Real-World Examples

### Example 1: Risk Management System

```python
from axiom.models.base.factory import ModelFactory, ModelType
from axiom.config.model_config import ModelConfig

# Load Basel III compliant configuration
config = ModelConfig.for_basel_iii_compliance()

# Create VaR model
var_model = ModelFactory.create(
    ModelType.PARAMETRIC_VAR,
    config=config.var
)

# Calculate 10-day VaR at 99.9% confidence
result = var_model.calculate_risk(
    portfolio_value=100_000_000,
    returns=returns_data,
    confidence_level=0.999,
    time_horizon=10
)

print(f"10-day VaR (99.9%): ${result.value['var']:,.0f}")
```

### Example 2: Multi-Model Comparison

```python
# Compare different VaR methodologies
methods = [
    ModelType.PARAMETRIC_VAR,
    ModelType.HISTORICAL_VAR,
    ModelType.MONTE_CARLO_VAR
]

results = {}
for method in methods:
    model = ModelFactory.create(method)
    result = model.calculate_risk(
        portfolio_value=1_000_000,
        returns=returns_data,
        confidence_level=0.95
    )
    results[method.value] = {
        "var": result.value["var"],
        "time_ms": result.metadata.execution_time_ms
    }

# Print comparison
for method, data in results.items():
    print(f"{method:20s}: ${data['var']:>12,.0f} ({data['time_ms']:>6.2f}ms)")

# Output:
# parametric_var      : $   50,234.12 (  0.85ms)
# historical_var      : $   52,891.45 (  3.21ms)
# monte_carlo_var     : $   51,567.89 (  8.94ms)
```

### Example 3: Trading Strategy with Multiple Models

```python
class TradingStrategy:
    """Trading strategy using multiple models from factory."""
    
    def __init__(self):
        # Create models via factory
        self.ewma_model = ModelFactory.create(ModelType.EWMA)
        self.garch_model = ModelFactory.create(ModelType.GARCH)
        self.var_model = ModelFactory.create(ModelType.HISTORICAL_VAR)
    
    def generate_signal(self, price_data):
        """Generate trading signal using multiple models."""
        
        # Trend signal from EWMA
        ewma_result = self.ewma_model.calculate(data=price_data)
        trend = ewma_result.value["trend"]
        
        # Volatility forecast from GARCH
        returns = np.diff(price_data) / price_data[:-1]
        garch_result = self.garch_model.calculate(data=returns)
        vol_forecast = garch_result.value["volatility_forecast"][0]
        
        # Risk limit from VaR
        var_result = self.var_model.calculate_risk(
            portfolio_value=1_000_000,
            returns=returns,
            confidence_level=0.95
        )
        var_limit = var_result.value["var"]
        
        # Combine signals
        if trend == "up" and vol_forecast < 0.15:
            return "BUY", var_limit
        elif trend == "down" or vol_forecast > 0.25:
            return "SELL", var_limit
        else:
            return "HOLD", var_limit

# Use strategy
strategy = TradingStrategy()
signal, risk_limit = strategy.generate_signal(price_history)
```

### Example 4: Plugin Development

```python
# Custom VaR plugin for hedge fund
class HedgeFundVaR(BaseRiskModel):
    """Custom VaR for hedge fund with leverage."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.max_leverage = config.get("max_leverage", 2.0)
    
    def calculate_risk(self, portfolio_value, returns, confidence_level=0.95):
        # Standard VaR calculation
        var_amount = np.percentile(returns, (1 - confidence_level) * 100)
        var_amount = abs(var_amount) * portfolio_value
        
        # Adjust for leverage
        leveraged_var = var_amount * self.max_leverage
        
        metadata = self._create_metadata(execution_time_ms=1.0)
        return ModelResult(
            value={
                "var": leveraged_var,
                "unleveraged_var": var_amount,
                "leverage": self.max_leverage
            },
            metadata=metadata
        )
    
    def validate_inputs(self, **kwargs):
        if "returns" not in kwargs:
            raise ValueError("Returns required")
        return True

# Register plugin
from axiom.models.base.factory import PluginManager

PluginManager.register_plugin(
    "hedge_fund_var",
    HedgeFundVaR,
    config_key="var",
    description="VaR adjusted for hedge fund leverage"
)

# Use plugin
hf_var = ModelFactory.create(
    "hedge_fund_var",
    max_leverage=3.0  # 3x leverage
)

result = hf_var.calculate_risk(
    portfolio_value=10_000_000,
    returns=returns_data
)

print(f"Leveraged VaR: ${result.value['var']:,.0f}")
print(f"Unleveraged VaR: ${result.value['unleveraged_var']:,.0f}")
```

## Testing with Factory

### Mocking the Factory

```python
import pytest
from unittest.mock import Mock, patch
from axiom.models.base.factory import ModelFactory, ModelType

def test_strategy_with_mocked_models():
    """Test trading strategy with mocked models."""
    
    # Create mock models
    mock_ewma = Mock()
    mock_ewma.calculate.return_value = Mock(
        value={"trend": "up"},
        metadata=Mock(execution_time_ms=1.0)
    )
    
    mock_var = Mock()
    mock_var.calculate_risk.return_value = Mock(
        value={"var": 50000},
        metadata=Mock(execution_time_ms=2.0)
    )
    
    # Patch factory
    with patch.object(ModelFactory, 'create') as mock_create:
        def create_side_effect(model_type, **kwargs):
            if model_type == ModelType.EWMA:
                return mock_ewma
            elif model_type == ModelType.HISTORICAL_VAR:
                return mock_var
        
        mock_create.side_effect = create_side_effect
        
        # Test your code
        strategy = TradingStrategy()
        signal, risk_limit = strategy.generate_signal(price_data)
        
        assert signal == "BUY"
        assert risk_limit == 50000
```

### Factory in Integration Tests

```python
def test_var_model_integration():
    """Integration test using real factory."""
    
    # Use real factory - no mocking
    model = ModelFactory.create(ModelType.HISTORICAL_VAR)
    
    # Test with real data
    returns = np.random.normal(0, 0.01, 252)
    result = model.calculate_risk(
        portfolio_value=1_000_000,
        returns=returns,
        confidence_level=0.95
    )
    
    # Verify result structure
    assert result.success
    assert "var" in result.value
    assert result.metadata.execution_time_ms > 0
    assert result.value["var"] > 0
```

## Advanced Patterns

### Factory with Dependency Injection Container

```python
class ModelContainer:
    """Dependency injection container for models."""
    
    def __init__(self, config=None):
        self.config = config or ModelConfig.from_env()
        self._models = {}
    
    def get_model(self, model_type: ModelType):
        """Get or create model (singleton per type)."""
        if model_type not in self._models:
            self._models[model_type] = ModelFactory.create(
                model_type,
                config=getattr(self.config, self._get_config_key(model_type))
            )
        return self._models[model_type]
    
    def _get_config_key(self, model_type: ModelType) -> str:
        """Determine config key from model type."""
        if "VAR" in model_type.value:
            return "var"
        elif model_type.value in ["arima", "garch", "ewma"]:
            return "time_series"
        elif model_type.value in ["markowitz_optimizer", "black_litterman"]:
            return "portfolio"
        return "options"

# Usage
container = ModelContainer()

# Get models (created once, reused)
var_model = container.get_model(ModelType.HISTORICAL_VAR)
arima_model = container.get_model(ModelType.ARIMA)
```

### Factory with Builder Pattern

```python
class ModelBuilder:
    """Builder for creating configured models."""
    
    def __init__(self):
        self._model_type = None
        self._config = {}
    
    def set_model_type(self, model_type: ModelType):
        self._model_type = model_type
        return self
    
    def with_confidence(self, level: float):
        self._config["default_confidence_level"] = level
        return self
    
    def with_simulations(self, n: int):
        self._config["default_simulations"] = n
        return self
    
    def with_parallel(self, enabled: bool = True):
        self._config["parallel_mc"] = enabled
        return self
    
    def build(self):
        if self._model_type is None:
            raise ValueError("Model type not set")
        return ModelFactory.create(self._model_type, **self._config)

# Usage
model = (ModelBuilder()
    .set_model_type(ModelType.MONTE_CARLO_VAR)
    .with_confidence(0.99)
    .with_simulations(100000)
    .with_parallel(True)
    .build())
```

## Best Practices

### 1. Always Use ModelType Enum

```python
# ✓ Good - type safe
model = ModelFactory.create(ModelType.HISTORICAL_VAR)

# ✗ Bad - prone to typos
model = ModelFactory.create("historicl_var")  # Typo!
```

### 2. Inject Configuration, Don't Hard-Code

```python
# ✓ Good - configuration injection
config = VaRConfig.from_env()
model = ModelFactory.create(ModelType.HISTORICAL_VAR, config=config)

# ✗ Bad - hard-coded values
model = ModelFactory.create(ModelType.HISTORICAL_VAR)
model.config["confidence_level"] = 0.95  # Modifying after creation
```

### 3. Use Appropriate Configuration Profiles

```python
# ✓ Good - use profiles for common scenarios
basel_config = ModelConfig.for_basel_iii_compliance()
model = ModelFactory.create(ModelType.PARAMETRIC_VAR, config=basel_config.var)

# ✗ Bad - manually setting all Basel III parameters
model = ModelFactory.create(
    ModelType.PARAMETRIC_VAR,
    default_confidence_level=0.999,
    default_time_horizon=10,
    min_observations=252,
    # ... many more parameters
)
```

### 4. Register Plugins Once at Startup

```python
# ✓ Good - register during app initialization
def initialize_app():
    PluginManager.register_plugin("custom_var", CustomVaRModel, ...)
    # Other initialization
    
# ✗ Bad - registering every time you need the model
def calculate_var():
    PluginManager.register_plugin("custom_var", ...)  # Don't do this!
    model = ModelFactory.create("custom_var")
```

## See Also

- [`BASE_CLASSES.md`](BASE_CLASSES.md) - Base class hierarchy
- [`MIXINS.md`](MIXINS.md) - Reusable functionality mixins
- [`CONFIGURATION_SYSTEM.md`](CONFIGURATION_SYSTEM.md) - Configuration architecture

---

**Last Updated**: 2025-10-23  
**Version**: 2.0.0