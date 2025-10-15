"""Comprehensive error handling and validation for Axiom Investment Banking Analytics."""

import functools
import logging
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Error severity levels for investment banking operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in investment banking analytics."""

    AI_PROVIDER = "ai_provider"
    DATA_SOURCE = "data_source"
    FINANCIAL_VALIDATION = "financial_validation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    PROCESSING = "processing"
    COMPLIANCE = "compliance"


class AxiomError(Exception):
    """Base exception class for Axiom investment banking analytics."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.PROCESSING,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.utcnow()

        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "original_error": str(self.original_error) if self.original_error else None,
        }


class AIProviderError(AxiomError):
    """Error related to AI provider operations."""

    def __init__(self, message: str, provider: str, model: str = "", **kwargs):
        context = kwargs.get("context", {})
        context.update({"provider": provider, "model": model})
        super().__init__(
            message,
            category=ErrorCategory.AI_PROVIDER,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class FinancialDataError(AxiomError):
    """Error related to financial data processing or validation."""

    def __init__(self, message: str, data_source: str = "", **kwargs):
        context = kwargs.get("context", {})
        context.update({"data_source": data_source})
        super().__init__(
            message,
            category=ErrorCategory.FINANCIAL_VALIDATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class ComplianceError(AxiomError):
    """Error related to investment banking compliance requirements."""

    def __init__(self, message: str, compliance_rule: str = "", **kwargs):
        context = kwargs.get("context", {})
        context.update({"compliance_rule": compliance_rule})
        super().__init__(
            message,
            category=ErrorCategory.COMPLIANCE,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            **kwargs,
        )


class ConfigurationError(AxiomError):
    """Error related to system configuration."""

    def __init__(self, message: str, config_section: str = "", **kwargs):
        context = kwargs.get("context", {})
        context.update({"config_section": config_section})
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class ErrorHandler:
    """Centralized error handling for investment banking operations."""

    def __init__(self, logger_name: str = "axiom"):
        self.logger = logging.getLogger(logger_name)
        self.error_history: list[dict[str, Any]] = []
        self.max_history = 1000

    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        reraise: bool = True,
    ) -> AxiomError | None:
        """Handle and log errors with context."""

        # Convert to AxiomError if needed
        if isinstance(error, AxiomError):
            axiom_error = error
        else:
            axiom_error = AxiomError(
                message=str(error), context=context, original_error=error
            )

        # Log error
        self._log_error(axiom_error)

        # Store in history
        self._store_error(axiom_error)

        # Reraise if requested
        if reraise:
            raise axiom_error

        return axiom_error

    def _log_error(self, error: AxiomError):
        """Log error with appropriate level."""

        error_dict = error.to_dict()

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM ERROR: {error.message}", extra=error_dict)
        else:
            self.logger.info(f"LOW ERROR: {error.message}", extra=error_dict)

    def _store_error(self, error: AxiomError):
        """Store error in history."""

        self.error_history.append(error.to_dict())

        # Maintain history size
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history :]

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of recent errors."""

        if not self.error_history:
            return {"total_errors": 0, "by_category": {}, "by_severity": {}}

        # Count by category
        by_category = {}
        by_severity = {}

        for error in self.error_history[-100:]:  # Last 100 errors
            category = error.get("category", "unknown")
            severity = error.get("severity", "unknown")

            by_category[category] = by_category.get(category, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(self.error_history[-100:]),
            "by_category": by_category,
            "by_severity": by_severity,
            "latest_error": self.error_history[-1] if self.error_history else None,
        }


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_errors(
    error_types: type[Exception] | tuple = Exception,
    category: ErrorCategory = ErrorCategory.PROCESSING,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    reraise: bool = True,
    fallback_return: Any = None,
):
    """Decorator for handling errors in investment banking functions."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args)[:200],  # Truncate for logging
                    "kwargs": str(kwargs)[:200],
                }

                axiom_error = AxiomError(
                    message=f"Error in {func.__name__}: {str(e)}",
                    category=category,
                    severity=severity,
                    context=context,
                    original_error=e,
                )

                global_error_handler.handle_error(axiom_error, reraise=reraise)

                if not reraise:
                    return fallback_return

        return wrapper

    return decorator


def validate_investment_banking_data(
    data: dict[str, Any], required_fields: list[str]
) -> bool:
    """Validate investment banking data has required fields."""

    missing_fields = [
        field for field in required_fields if field not in data or data[field] is None
    ]

    if missing_fields:
        raise FinancialDataError(
            f"Missing required financial data fields: {missing_fields}",
            context={
                "provided_fields": list(data.keys()),
                "missing_fields": missing_fields,
            },
        )

    return True


def validate_financial_metrics(metrics: dict[str, float]) -> bool:
    """Validate financial metrics are reasonable."""

    validation_rules = {
        "revenue": lambda x: x >= 0,
        "ebitda": lambda x: True,  # Can be negative
        "debt": lambda x: x >= 0,
        "cash": lambda x: x >= 0,
        "valuation": lambda x: x > 0,
        "confidence": lambda x: 0 <= x <= 1,
    }

    invalid_metrics = []

    for metric, value in metrics.items():
        if metric in validation_rules:
            if not validation_rules[metric](value):
                invalid_metrics.append(f"{metric}: {value}")

    if invalid_metrics:
        raise FinancialDataError(
            f"Invalid financial metrics: {invalid_metrics}",
            context={"metrics": metrics, "invalid": invalid_metrics},
        )

    return True


def setup_logging(log_level: str = "INFO", log_file: str | None = None):
    """Setup logging for investment banking operations."""

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    # Configure axiom logger
    axiom_logger = logging.getLogger("axiom")
    axiom_logger.setLevel(getattr(logging, log_level.upper()))


class HealthChecker:
    """Health checking for investment banking system components."""

    def __init__(self):
        self.checks = {}

    def register_check(
        self, name: str, check_func: Callable[[], bool], description: str = ""
    ):
        """Register a health check function."""
        self.checks[name] = {"function": check_func, "description": description}

    def run_health_checks(self) -> dict[str, Any]:
        """Run all registered health checks."""

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "checks": {},
        }

        failed_checks = []

        for check_name, check_info in self.checks.items():
            try:
                result = check_info["function"]()
                results["checks"][check_name] = {
                    "status": "pass" if result else "fail",
                    "description": check_info["description"],
                }

                if not result:
                    failed_checks.append(check_name)

            except Exception as e:
                results["checks"][check_name] = {
                    "status": "error",
                    "error": str(e),
                    "description": check_info["description"],
                }
                failed_checks.append(check_name)

        if failed_checks:
            results["overall_status"] = "unhealthy"
            results["failed_checks"] = failed_checks

        return results


# Global health checker
global_health_checker = HealthChecker()
