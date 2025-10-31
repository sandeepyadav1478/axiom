"""
Data Quality Framework - Institutional Grade

Comprehensive data quality assurance for Axiom platform.
Ensures data legitimacy and compliance with world-class standards.
"""

from axiom.data_quality.validation.rules_engine import (
    DataValidationEngine,
    ValidationRule,
    ValidationResult,
    ValidationSeverity,
    ValidationCategory,
    ValidationError,
    get_validation_engine
)

__all__ = [
    "DataValidationEngine",
    "ValidationRule",
    "ValidationResult",
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationError",
    "get_validation_engine"
]

__version__ = "1.0.0"