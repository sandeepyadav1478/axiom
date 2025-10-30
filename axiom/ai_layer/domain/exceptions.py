"""
Custom Exception Hierarchy - Production Quality

Proper exception handling with:
- Specific exception types (not generic Exception)
- Context preservation (what was happening)
- Error codes (for client communication)
- Retry hints (can retry or not)
- Logging integration (structured logging)

Senior engineer principle: Exceptions are part of the domain model.
"""

from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCode(Enum):
    """Standardized error codes for client communication"""
    # Validation errors (4xx equivalent)
    INVALID_INPUT = "E1001"
    INVALID_GREEKS = "E1002"
    INVALID_PRICE = "E1003"
    VALIDATION_FAILED = "E1004"
    
    # Model errors (5xx equivalent)
    MODEL_ERROR = "E2001"
    MODEL_NOT_LOADED = "E2002"
    MODEL_INFERENCE_FAILED = "E2003"
    GPU_ERROR = "E2004"
    
    # Agent errors
    AGENT_NOT_AVAILABLE = "E3001"
    AGENT_TIMEOUT = "E3002"
    AGENT_OVERLOADED = "E3003"
    
    # System errors
    DATABASE_ERROR = "E4001"
    NETWORK_ERROR = "E4002"
    RATE_LIMIT_EXCEEDED = "E4003"
    CIRCUIT_BREAKER_OPEN = "E4004"


class AxiomBaseException(Exception):
    """
    Base exception for all Axiom errors
    
    All custom exceptions inherit from this.
    Provides common functionality:
    - Error codes
    - Context preservation
    - Retry hints
    - Structured logging
    - Timestamp
    """
    
    error_code: ErrorCode = ErrorCode.MODEL_ERROR
    severity: ErrorSeverity = ErrorSeverity.ERROR
    is_retryable: bool = False
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize exception
        
        Args:
            message: Human-readable error message
            context: Additional context (dict of relevant data)
            cause: Original exception if this wraps another
        """
        self.message = message
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary for logging/API response
        
        Returns structured error data
        """
        return {
            'error_code': self.error_code.value,
            'message': self.message,
            'severity': self.severity.value,
            'is_retryable': self.is_retryable,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'cause': str(self.cause) if self.cause else None
        }
    
    def log_error(self, logger):
        """
        Log error with structured logging
        
        Integrates with structlog or standard logging
        """
        logger.error(
            self.message,
            error_code=self.error_code.value,
            severity=self.severity.value,
            context=self.context,
            exc_info=self.cause
        )


# Validation errors (user/data issues)
class ValidationError(AxiomBaseException):
    """Base for validation errors"""
    error_code = ErrorCode.VALIDATION_FAILED
    severity = ErrorSeverity.WARNING
    is_retryable = False  # Validation errors don't benefit from retry


class InvalidInputError(ValidationError):
    """Input data is invalid"""
    error_code = ErrorCode.INVALID_INPUT


class InvalidGreeksError(ValidationError):
    """Greeks values are invalid"""
    error_code = ErrorCode.INVALID_GREEKS


class CrossValidationError(ValidationError):
    """Cross-validation failed"""
    error_code = ErrorCode.VALIDATION_FAILED


# Model errors (AI/ML issues)
class ModelError(AxiomBaseException):
    """Base for model-related errors"""
    error_code = ErrorCode.MODEL_ERROR
    severity = ErrorSeverity.ERROR
    is_retryable = True  # Model errors might be transient


class ModelNotLoadedError(ModelError):
    """Model not loaded or unavailable"""
    error_code = ErrorCode.MODEL_NOT_LOADED
    is_retryable = False  # Need to fix before retry


class ModelInferenceError(ModelError):
    """Model inference failed"""
    error_code = ErrorCode.MODEL_INFERENCE_FAILED
    is_retryable = True  # Might work on retry


class GPUError(ModelError):
    """GPU-related error"""
    error_code = ErrorCode.GPU_ERROR
    is_retryable = True  # Might recover or fallback to CPU


# Agent errors
class AgentError(AxiomBaseException):
    """Base for agent-related errors"""
    severity = ErrorSeverity.ERROR
    is_retryable = True  # Agent errors often transient


class AgentNotAvailableError(AgentError):
    """Agent not available or not responding"""
    error_code = ErrorCode.AGENT_NOT_AVAILABLE


class AgentTimeoutError(AgentError):
    """Agent didn't respond in time"""
    error_code = ErrorCode.AGENT_TIMEOUT


class AgentOverloadedError(AgentError):
    """Agent is overloaded"""
    error_code = ErrorCode.AGENT_OVERLOADED


# System errors
class SystemError(AxiomBaseException):
    """Base for system-level errors"""
    severity = ErrorSeverity.CRITICAL
    is_retryable = True


class DatabaseError(SystemError):
    """Database error"""
    error_code = ErrorCode.DATABASE_ERROR


class NetworkError(SystemError):
    """Network communication error"""
    error_code = ErrorCode.NETWORK_ERROR


class RateLimitError(SystemError):
    """Rate limit exceeded"""
    error_code = ErrorCode.RATE_LIMIT_EXCEEDED
    severity = ErrorSeverity.WARNING
    is_retryable = False  # Need to wait


class CircuitBreakerError(SystemError):
    """Circuit breaker is open"""
    error_code = ErrorCode.CIRCUIT_BREAKER_OPEN
    severity = ErrorSeverity.WARNING
    is_retryable = False  # Need to wait for circuit to close


# Example usage
if __name__ == "__main__":
    import structlog
    
    logger = structlog.get_logger()
    
    print("="*60)
    print("CUSTOM EXCEPTION HIERARCHY - PRODUCTION QUALITY")
    print("="*60)
    
    # Test 1: Validation error
    print("\n→ Test 1: Validation Error")
    try:
        raise InvalidGreeksError(
            message="Delta out of range",
            context={'delta': 1.5, 'expected_range': '[0, 1]'}
        )
    except InvalidGreeksError as e:
        print(f"   Caught: {type(e).__name__}")
        print(f"   Code: {e.error_code.value}")
        print(f"   Retryable: {e.is_retryable}")
        print(f"   Context: {e.context}")
    
    # Test 2: Model error with cause
    print("\n→ Test 2: Model Error with Cause")
    try:
        try:
            # Simulate CUDA error
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as cuda_error:
            raise GPUError(
                message="GPU failed during inference",
                context={'model': 'greeks_v2', 'batch_size': 1000},
                cause=cuda_error
            )
    except GPUError as e:
        print(f"   Caught: {type(e).__name__}")
        print(f"   Retryable: {e.is_retryable}")
        print(f"   Original cause: {e.cause}")
    
    # Test 3: Structured error dict
    print("\n→ Test 3: Structured Error Data")
    try:
        raise AgentTimeoutError(
            message="Pricing agent didn't respond",
            context={'agent': 'pricing', 'timeout_ms': 5000}
        )
    except AgentTimeoutError as e:
        error_dict = e.to_dict()
        
        print(f"   Error dict:")
        for key, value in error_dict.items():
            print(f"     {key}: {value}")
    
    print("\n" + "="*60)
    print("✓ Proper exception hierarchy")
    print("✓ Error codes for client communication")
    print("✓ Retry hints for error recovery")
    print("✓ Context preservation for debugging")
    print("✓ Structured logging integration")
    print("\nPRODUCTION-GRADE ERROR HANDLING")