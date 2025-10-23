"""
Abstract Base Classes for All Financial Models
==============================================

Defines the standard interface and core functionality that all Axiom financial
models must implement, ensuring consistency and enabling DRY principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypeVar, Generic
from datetime import datetime
import time
import numpy as np

from axiom.core.logging.axiom_logger import get_logger


# Type variable for model results
T = TypeVar('T')


class ValidationError(Exception):
    """Raised when model input validation fails."""
    pass


@dataclass
class ModelMetadata:
    """Metadata for model execution."""
    model_name: str
    model_version: str = "1.0.0"
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    configuration: Dict[str, Any] = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "configuration": self.configuration,
            "warnings": self.warnings
        }


@dataclass
class ModelResult(Generic[T]):
    """
    Standard result container for all financial models.
    
    Provides consistent interface for:
    - Calculation results
    - Metadata and diagnostics
    - Validation status
    - Performance metrics
    """
    value: T
    metadata: ModelMetadata
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "value": self.value if isinstance(self.value, (int, float, str, bool, type(None))) else str(self.value),
            "metadata": self.metadata.to_dict(),
            "success": self.success
        }
        if self.error_message:
            result["error_message"] = self.error_message
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        status = "✓" if self.success else "✗"
        return f"ModelResult({status} value={self.value}, time={self.metadata.execution_time_ms:.2f}ms)"


class BaseFinancialModel(ABC):
    """
    Abstract base class for all financial models.
    
    All Axiom financial models inherit from this class and implement:
    - calculate() method for core computation
    - validate_inputs() for parameter validation
    - Standard logging and performance tracking
    - Configuration management
    
    This ensures:
    - Consistent API across all models
    - Standardized error handling
    - Built-in performance monitoring
    - Unified logging
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize base financial model.
        
        Args:
            config: Model-specific configuration dictionary
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        self.config = config or {}
        self.enable_logging = enable_logging
        self.enable_performance_tracking = enable_performance_tracking
        self.logger = get_logger(f"axiom.models.{self.__class__.__name__}")
        
        if self.enable_logging:
            self.logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def calculate(self, **kwargs) -> ModelResult:
        """
        Core calculation method - must be implemented by all models.
        
        Args:
            **kwargs: Model-specific parameters
        
        Returns:
            ModelResult containing calculation output and metadata
            
        Raises:
            ValidationError: If inputs are invalid
            ValueError: If calculation fails
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate model inputs - must be implemented by all models.
        
        Args:
            **kwargs: Model-specific parameters to validate
            
        Returns:
            True if all inputs are valid
            
        Raises:
            ValidationError: If any input is invalid with detailed message
        """
        pass
    
    def _create_metadata(
        self,
        execution_time_ms: float,
        warnings: Optional[list] = None
    ) -> ModelMetadata:
        """
        Create metadata for model result.
        
        Args:
            execution_time_ms: Execution time in milliseconds
            warnings: Optional list of warnings
            
        Returns:
            ModelMetadata instance
        """
        return ModelMetadata(
            model_name=self.__class__.__name__,
            execution_time_ms=execution_time_ms,
            configuration=self.config.copy(),
            warnings=warnings or []
        )
    
    def _track_performance(self, operation: str, start_time: float) -> float:
        """
        Track and log performance metrics.
        
        Args:
            operation: Name of the operation
            start_time: Start time from time.perf_counter()
            
        Returns:
            Execution time in milliseconds
        """
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging and self.enable_performance_tracking:
            self.logger.debug(
                f"{operation} completed",
                execution_time_ms=round(execution_time_ms, 3)
            )
        
        return execution_time_ms
    
    def _log_calculation(self, **kwargs):
        """Log calculation parameters."""
        if self.enable_logging:
            self.logger.debug(
                f"{self.__class__.__name__} calculation started",
                **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
            )
    
    def update_config(self, config: Dict[str, Any]):
        """
        Update model configuration at runtime.
        
        Args:
            config: New configuration values to merge
        """
        self.config.update(config)
        if self.enable_logging:
            self.logger.info(f"Configuration updated: {list(config.keys())}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current model configuration."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(config={len(self.config)} params)"


class BasePricingModel(BaseFinancialModel):
    """
    Abstract base class for pricing models (options, bonds, etc.).
    
    Adds pricing-specific functionality:
    - Price validation
    - Greeks calculation (optional)
    - Sensitivity analysis
    """
    
    @abstractmethod
    def price(self, **kwargs) -> float:
        """
        Calculate price - must be implemented by pricing models.
        
        Args:
            **kwargs: Model-specific pricing parameters
            
        Returns:
            Price as float
        """
        pass
    
    def validate_price(self, price: float) -> bool:
        """
        Validate calculated price.
        
        Args:
            price: Calculated price
            
        Returns:
            True if price is valid
            
        Raises:
            ValidationError: If price is invalid
        """
        if price < 0:
            raise ValidationError(f"Price cannot be negative: {price}")
        if not np.isfinite(price):
            raise ValidationError(f"Price must be finite: {price}")
        return True


class BaseRiskModel(BaseFinancialModel):
    """
    Abstract base class for risk models (VaR, Credit Risk, etc.).
    
    Adds risk-specific functionality:
    - Risk metric calculation
    - Confidence intervals
    - Scenario analysis
    """
    
    @abstractmethod
    def calculate_risk(self, **kwargs) -> ModelResult:
        """
        Calculate risk metric - must be implemented by risk models.
        
        Args:
            **kwargs: Model-specific risk parameters
            
        Returns:
            ModelResult containing risk metric and metadata
        """
        pass
    
    def validate_confidence_level(self, confidence_level: float) -> bool:
        """
        Validate confidence level parameter.
        
        Args:
            confidence_level: Confidence level (0 to 1)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If confidence level is invalid
        """
        if not 0 < confidence_level < 1:
            raise ValidationError(
                f"Confidence level must be between 0 and 1, got {confidence_level}"
            )
        return True


class BasePortfolioModel(BaseFinancialModel):
    """
    Abstract base class for portfolio models (optimization, allocation, etc.).
    
    Adds portfolio-specific functionality:
    - Weight validation
    - Constraint checking
    - Performance metrics
    """
    
    @abstractmethod
    def optimize(self, **kwargs) -> ModelResult:
        """
        Optimize portfolio - must be implemented by portfolio models.
        
        Args:
            **kwargs: Model-specific optimization parameters
            
        Returns:
            ModelResult containing optimal weights and metadata
        """
        pass
    
    def validate_weights(self, weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Validate portfolio weights.
        
        Args:
            weights: Portfolio weights array
            tolerance: Tolerance for sum-to-one constraint
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If weights are invalid
        """
        if np.any(weights < -tolerance):
            raise ValidationError("Weights cannot be negative in long-only portfolio")
        
        weight_sum = np.sum(weights)
        if not np.isclose(weight_sum, 1.0, atol=tolerance):
            raise ValidationError(
                f"Weights must sum to 1.0, got {weight_sum:.6f}"
            )
        
        return True


__all__ = [
    "ValidationError",
    "ModelMetadata",
    "ModelResult",
    "BaseFinancialModel",
    "BasePricingModel",
    "BaseRiskModel",
    "BasePortfolioModel",
]