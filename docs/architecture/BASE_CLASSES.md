# Base Classes Architecture

**DRY Principles Through Abstract Base Classes**

## Overview

Axiom's base class hierarchy provides a consistent interface across all financial models, eliminating code duplication and ensuring standardized behavior. All models inherit from [`BaseFinancialModel`](../../axiom/models/base/base_model.py:83) or its specialized subclasses.

## Class Hierarchy

```python
BaseFinancialModel (abstract)
├── BasePricingModel (abstract)      # Options, bonds, derivatives
├── BaseRiskModel (abstract)          # VaR, credit risk, stress testing
├── BasePortfolioModel (abstract)     # Optimization, allocation
└── BaseTimeSeriesModel (abstract)    # ARIMA, GARCH, EWMA
```

## BaseFinancialModel

**Location**: [`axiom/models/base/base_model.py:83`](../../axiom/models/base/base_model.py:83)

### Core Interface

All financial models must implement:

```python
class BaseFinancialModel(ABC):
    """Abstract base class for all financial models."""
    
    @abstractmethod
    def calculate(self, **kwargs) -> ModelResult:
        """Core calculation method - must be implemented."""
        pass
    
    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """Validate model inputs - must be implemented."""
        pass
```

### Built-in Functionality

Every model automatically inherits:

1. **Configuration Management**
   ```python
   model.update_config({"risk_free_rate": 0.04})
   current_config = model.get_config()
   ```

2. **Performance Tracking**
   ```python
   # Automatic execution time tracking
   result = model.calculate(...)
   print(f"Time: {result.metadata.execution_time_ms}ms")
   ```

3. **Logging**
   ```python
   # Structured logging built-in
   model._log_calculation(portfolio_value=1_000_000)
   ```

4. **Metadata Generation**
   ```python
   # Automatic metadata creation
   metadata = model._create_metadata(
       execution_time_ms=10.5,
       warnings=["Low sample size"]
   )
   ```

### ModelResult Container

All calculations return [`ModelResult`](../../axiom/models/base/base_model.py:51):

```python
@dataclass
class ModelResult(Generic[T]):
    value: T                      # Calculation result
    metadata: ModelMetadata       # Execution info
    success: bool = True          # Success flag
    error_message: Optional[str] = None
```

## BasePricingModel

**Location**: [`axiom/models/base/base_model.py:226`](../../axiom/models/base/base_model.py:226)

Specialized for pricing models (options, bonds, etc.):

```python
class BasePricingModel(BaseFinancialModel):
    """Abstract base for pricing models."""
    
    @abstractmethod
    def price(self, **kwargs) -> float:
        """Calculate price - must be implemented."""
        pass
    
    def validate_price(self, price: float) -> bool:
        """Validate calculated price."""
        if price < 0:
            raise ValidationError(f"Price cannot be negative: {price}")
        if not np.isfinite(price):
            raise ValidationError(f"Price must be finite: {price}")
        return True
```

**Usage Example**:

```python
class BlackScholesModel(BasePricingModel):
    def price(self, S, K, T, r, sigma):
        # Black-Scholes formula
        price = self._calculate_bs_price(S, K, T, r, sigma)
        self.validate_price(price)
        return price
    
    def calculate(self, **kwargs):
        price = self.price(**kwargs)
        metadata = self._create_metadata(execution_time_ms=0.5)
        return ModelResult(value=price, metadata=metadata)
```

## BaseRiskModel

**Location**: [`axiom/models/base/base_model.py:269`](../../axiom/models/base/base_model.py:269)

Specialized for risk models (VaR, credit risk, etc.):

```python
class BaseRiskModel(BaseFinancialModel):
    """Abstract base for risk models."""
    
    @abstractmethod
    def calculate_risk(self, **kwargs) -> ModelResult:
        """Calculate risk metric - must be implemented."""
        pass
    
    def validate_confidence_level(self, confidence_level: float) -> bool:
        """Validate confidence level parameter."""
        if not 0 < confidence_level < 1:
            raise ValidationError(
                f"Confidence level must be between 0 and 1, got {confidence_level}"
            )
        return True
```

**Usage Example**:

```python
class HistoricalVaR(BaseRiskModel):
    def calculate_risk(self, portfolio_value, returns, confidence_level=0.95):
        self.validate_confidence_level(confidence_level)
        
        var_amount = self._calculate_historical_var(returns, confidence_level)
        var_amount *= portfolio_value
        
        metadata = self._create_metadata(execution_time_ms=2.5)
        return ModelResult(
            value={"var": var_amount},
            metadata=metadata
        )
```

## BasePortfolioModel

**Location**: [`axiom/models/base/base_model.py:312`](../../axiom/models/base/base_model.py:312)

Specialized for portfolio models:

```python
class BasePortfolioModel(BaseFinancialModel):
    """Abstract base for portfolio models."""
    
    @abstractmethod
    def optimize(self, **kwargs) -> ModelResult:
        """Optimize portfolio - must be implemented."""
        pass
    
    def validate_weights(self, weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Validate portfolio weights."""
        if np.any(weights < -tolerance):
            raise ValidationError("Weights cannot be negative in long-only portfolio")
        
        weight_sum = np.sum(weights)
        if not np.isclose(weight_sum, 1.0, atol=tolerance):
            raise ValidationError(f"Weights must sum to 1.0, got {weight_sum:.6f}")
        
        return True
```

## Benefits of Base Classes

### 1. Code Reuse

**Without Base Classes** (code duplication):
```python
# VaR model
class VaRModel:
    def __init__(self):
        self.logger = get_logger("VaR")
        self.config = {}
    
    def track_time(self):
        # 20 lines of timing code
        pass
    
    def log_result(self):
        # 15 lines of logging code
        pass

# Portfolio model
class PortfolioModel:
    def __init__(self):
        self.logger = get_logger("Portfolio")
        self.config = {}
    
    def track_time(self):
        # 20 lines of SAME timing code
        pass
    
    def log_result(self):
        # 15 lines of SAME logging code
        pass
```

**With Base Classes** (DRY):
```python
class VaRModel(BaseRiskModel):
    def calculate_risk(self, **kwargs):
        # Only business logic - all infrastructure inherited
        pass

class PortfolioModel(BasePortfolioModel):
    def optimize(self, **kwargs):
        # Only business logic - all infrastructure inherited
        pass
```

### 2. Consistency

All models have the same interface:
```python
# Same pattern for all models
result = model.calculate(...)  # or calculate_risk(), optimize(), etc.
print(result.metadata.execution_time_ms)
print(result.value)
```

### 3. Maintainability

Fix a bug once, fixed everywhere:
```python
# Update logging in BaseFinancialModel
# → Automatically fixed in ALL 20+ models
```

### 4. Extensibility

Easy to add new models:
```python
class CustomVaRModel(BaseRiskModel):
    def calculate_risk(self, **kwargs):
        # Your custom logic
        pass
    
    def validate_inputs(self, **kwargs):
        # Your custom validation
        pass
    
    # Everything else inherited automatically:
    # - Logging
    # - Performance tracking
    # - Configuration management
    # - Metadata generation
```

## Creating a New Model

### Step 1: Choose Base Class

```python
from axiom.models.base.base_model import BaseRiskModel

class MyCustomVaR(BaseRiskModel):
    pass
```

### Step 2: Implement Required Methods

```python
def calculate_risk(self, **kwargs) -> ModelResult:
    # Your calculation logic
    start = time.perf_counter()
    
    var_value = self._my_var_calculation(**kwargs)
    
    execution_time = (time.perf_counter() - start) * 1000
    metadata = self._create_metadata(execution_time_ms=execution_time)
    
    return ModelResult(
        value={"var": var_value},
        metadata=metadata
    )

def validate_inputs(self, **kwargs) -> bool:
    # Your validation logic
    if "returns" not in kwargs:
        raise ValidationError("Returns data required")
    return True
```

### Step 3: Use Inherited Functionality

```python
def calculate_risk(self, **kwargs):
    # Use inherited methods
    self._log_calculation(**kwargs)
    self.validate_confidence_level(kwargs.get("confidence_level", 0.95))
    
    # Your business logic
    result = self._my_calculation(**kwargs)
    
    # Track performance automatically
    return result
```

## Best Practices

### 1. Always Call Parent Initialization

```python
def __init__(self, config=None):
    super().__init__(config=config)  # Initialize base class
    # Your initialization
```

### 2. Use Validation Methods

```python
def calculate_risk(self, confidence_level=0.95, **kwargs):
    # Use inherited validation
    self.validate_confidence_level(confidence_level)
    
    # Your logic
    pass
```

### 3. Create Proper Metadata

```python
def calculate(self, **kwargs):
    start = time.perf_counter()
    
    # Calculation
    result = self._compute(**kwargs)
    
    # Proper metadata
    execution_time = (time.perf_counter() - start) * 1000
    metadata = self._create_metadata(
        execution_time_ms=execution_time,
        warnings=self._get_warnings()
    )
    
    return ModelResult(value=result, metadata=metadata)
```

### 4. Leverage Configuration

```python
def calculate(self, **kwargs):
    # Use configuration
    confidence = kwargs.get("confidence_level", self.config.get("default_confidence", 0.95))
    
    # Your logic
    pass
```

## Testing Base Classes

```python
import pytest
from axiom.models.base.base_model import BaseRiskModel, ModelResult

def test_base_risk_model_interface():
    """Test that BaseRiskModel enforces interface."""
    
    class IncompleteModel(BaseRiskModel):
        pass
    
    # Should raise TypeError - abstract methods not implemented
    with pytest.raises(TypeError):
        IncompleteModel()

def test_model_result_container():
    """Test ModelResult functionality."""
    from axiom.models.base.base_model import ModelMetadata
    
    metadata = ModelMetadata(
        model_name="TestModel",
        execution_time_ms=10.5
    )
    
    result = ModelResult(
        value={"var": 50000},
        metadata=metadata
    )
    
    assert result.success
    assert result.value["var"] == 50000
    assert result.metadata.execution_time_ms == 10.5
```

## See Also

- [`MIXINS.md`](MIXINS.md) - Reusable functionality mixins
- [`FACTORY_PATTERN.md`](FACTORY_PATTERN.md) - Model creation patterns
- [`CONFIGURATION_SYSTEM.md`](CONFIGURATION_SYSTEM.md) - Configuration management

---

**Last Updated**: 2025-10-23  
**Version**: 2.0.0