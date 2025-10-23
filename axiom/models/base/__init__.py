"""
Base Classes and Patterns for All Financial Models
===================================================

This module provides the foundational architecture for all Axiom financial models,
implementing DRY principles through:

- Abstract base classes with standardized interfaces
- Reusable mixins for common functionality
- Factory pattern for model creation
- Shared validation and calculation logic
- Plugin system for extensibility

All financial models inherit from these base classes to ensure consistency
and eliminate code duplication.
"""

from .base_model import (
    BaseFinancialModel,
    ModelResult,
    ModelMetadata,
    ValidationError
)
from .factory import ModelFactory, ModelType
from .mixins import (
    MonteCarloMixin,
    NumericalMethodsMixin,
    PerformanceMixin,
    ValidationMixin,
    LoggingMixin
)

__all__ = [
    # Base classes
    "BaseFinancialModel",
    "ModelResult",
    "ModelMetadata",
    "ValidationError",
    
    # Factory
    "ModelFactory",
    "ModelType",
    
    # Mixins
    "MonteCarloMixin",
    "NumericalMethodsMixin",
    "PerformanceMixin",
    "ValidationMixin",
    "LoggingMixin",
]