"""
Axiom Core Logging Module
Enterprise-grade logging system with debug mode control
"""

from .axiom_logger import (
    AxiomLogger,
    LogLevel,
    get_logger,
)

# Create global logger instances
default_logger = get_logger("axiom")
provider_logger = get_logger("axiom.providers")
workflow_logger = get_logger("axiom.workflows")
validation_logger = get_logger("axiom.validation")

__all__ = [
    "AxiomLogger",
    "LogLevel",
    "get_logger",
    "default_logger",
    "provider_logger",
    "workflow_logger",
    "validation_logger",
]