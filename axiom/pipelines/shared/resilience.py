"""
Enterprise Resilience Patterns for Pipelines
Circuit breakers, retry logic, and fault tolerance
"""
import asyncio
import time
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before trying half-open
    
    
class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by stopping requests to failing services
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        
    def _should_attempt(self) -> bool:
        """Check if request should be attempted"""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if (
                self.last_failure_time and 
                time.time() - self.last_failure_time >= self.config.timeout
            ):
                logger.info(f"Circuit breaker {self.name}: Attempting half-open state")
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
                self.success_count = 0
                return True
            return False
            
        # HALF_OPEN state
        return True
        
    def _record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                logger.info(f"Circuit breaker {self.name}: Closing circuit")
                self.state = CircuitState.CLOSED
                self.success_count = 0
                
    def _record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker {self.name}: Re-opening circuit")
            self.state = CircuitState.OPEN
            self.success_count = 0
            
        elif self.failure_count >= self.config.failure_threshold:
            logger.error(f"Circuit breaker {self.name}: Opening circuit after {self.failure_count} failures")
            self.state = CircuitState.OPEN
            
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if not self._should_attempt():
            raise CircuitBreakerOpenError(
                f"Circuit breaker {self.name} is OPEN. Service unavailable."
            )
            
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise
            
    def get_status(self) -> dict:
        """Get current circuit breaker status"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class RetryStrategy:
    """
    Configurable retry strategy with exponential backoff
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        import random
        
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay = delay * (0.5 + random.random())
            
        return delay
        
    async def execute(
        self,
        func: Callable,
        *args,
        retry_on_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                return result
                
            except retry_on_exceptions as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    delay = self.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_attempts} attempts failed. "
                        f"Last error: {str(e)}"
                    )
                    
        # All attempts failed
        raise last_exception


class RateLimiter:
    """
    Token bucket rate limiter
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        
    async def acquire(self, tokens: int = 1):
        """Acquire tokens (blocks if not available)"""
        while True:
            now = time.time()
            
            # Add tokens based on time elapsed
            time_passed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + time_passed * self.rate
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
                
            # Wait for tokens to become available
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            
    def get_status(self) -> dict:
        """Get current rate limiter status"""
        return {
            'rate': self.rate,
            'capacity': self.capacity,
            'available_tokens': self.tokens
        }


class BulkheadPattern:
    """
    Bulkhead pattern: Isolate resources to prevent cascading failures
    """
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.active_count = 0
        
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with concurrency limits"""
        async with self.semaphore:
            self.active_count += 1
            try:
                return await func(*args, **kwargs)
            finally:
                self.active_count -= 1
                
    def get_status(self) -> dict:
        """Get current bulkhead status"""
        return {
            'max_concurrent': self.max_concurrent,
            'active': self.active_count,
            'available': self.max_concurrent - self.active_count
        }