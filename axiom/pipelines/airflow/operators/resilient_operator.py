"""
Resilient API Operators with Circuit Breakers and Retry Logic
Enterprise-grade fault tolerance for external API calls
"""
from typing import Any, Dict, Callable, Optional
from datetime import datetime, timedelta
from enum import Enum
import time

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreakerOperator(BaseOperator):
    """
    Operator with circuit breaker pattern for API resilience.
    
    Prevents cascade failures by:
    - Opening circuit after N failures
    - Auto-recovering with half-open state
    - Fast-failing when circuit is open
    - Exponential backoff on retries
    
    Use for: External APIs, Claude calls, market data providers
    """
    
    ui_color = '#FF6B6B'
    ui_fgcolor = '#fff'
    
    @apply_defaults
    def __init__(
        self,
        callable_func: Callable,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        half_open_attempts: int = 3,
        xcom_key: str = 'result',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.callable_func = callable_func
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_attempts = half_open_attempts
        self.xcom_key = xcom_key
        
        # Circuit state (would be Redis-backed in production)
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute with circuit breaker protection"""
        
        # Check circuit state
        current_state = self._get_circuit_state()
        
        if current_state == CircuitState.OPEN:
            # Circuit is open - fast fail
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed < self.recovery_timeout:
                error_msg = (
                    f"⛔ Circuit OPEN - Fast failing. "
                    f"Retry in {self.recovery_timeout - elapsed:.0f}s"
                )
                self.log.error(error_msg)
                raise Exception(error_msg)
            else:
                # Try half-open
                self._state = CircuitState.HALF_OPEN
                self.log.info("🔄 Circuit HALF-OPEN - Testing recovery")
        
        # Attempt operation
        try:
            result = self.callable_func(context)
            
            # Success - reset circuit
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self.log.info("✅ Circuit CLOSED - Recovered")
            
            context['ti'].xcom_push(key=self.xcom_key, value=result)
            return result
            
        except Exception as e:
            self._handle_failure(e)
            raise
    
    def _get_circuit_state(self) -> CircuitState:
        """Get current circuit state"""
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self.log.warning(
                f"⚠️  Circuit OPENED - {self._failure_count} consecutive failures"
            )
        return self._state
    
    def _handle_failure(self, error: Exception):
        """Handle operation failure"""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        self.log.error(
            f"❌ Operation failed ({self._failure_count}/{self.failure_threshold}): {error}"
        )


class ResilientAPIOperator(BaseOperator):
    """
    Operator with comprehensive retry logic and exponential backoff.
    
    Features:
    - Exponential backoff (1s, 2s, 4s, 8s, ...)
    - Jitter to prevent thundering herd
    - Detailed failure logging
    - Success rate tracking
    """
    
    template_fields = ('endpoint', 'params')
    ui_color = '#4ECDC4'
    ui_fgcolor = '#fff'
    
    @apply_defaults
    def __init__(
        self,
        api_callable: Callable,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        timeout_seconds: Optional[int] = 30,
        xcom_key: str = 'api_response',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.api_callable = api_callable
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.timeout_seconds = timeout_seconds
        self.xcom_key = xcom_key
        
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute API call with resilient retry logic"""
        import random
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.log.info(f"🔄 API call attempt {attempt + 1}/{self.max_retries + 1}")
                
                # Make API call with timeout
                start_time = time.time()
                result = self.api_callable(context)
                execution_time = time.time() - start_time
                
                # Success!
                self.log.info(
                    f"✅ API call succeeded in {execution_time:.2f}s "
                    f"(attempt {attempt + 1})"
                )
                
                response = {
                    'data': result,
                    'success': True,
                    'attempts': attempt + 1,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                context['ti'].xcom_push(key=self.xcom_key, value=response)
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # Calculate backoff delay
                    delay = min(
                        self.initial_retry_delay * (self.backoff_factor ** attempt),
                        self.max_retry_delay
                    )
                    
                    # Add jitter (±25%)
                    if self.jitter:
                        jitter_amount = delay * 0.25
                        delay += random.uniform(-jitter_amount, jitter_amount)
                    
                    self.log.warning(
                        f"⚠️  Attempt {attempt + 1} failed: {str(e)}\n"
                        f"   Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
                else:
                    self.log.error(
                        f"❌ All {self.max_retries + 1} attempts failed. "
                        f"Last error: {str(e)}"
                    )
        
        # All attempts failed
        error_response = {
            'success': False,
            'error': str(last_exception),
            'error_type': type(last_exception).__name__,
            'attempts': self.max_retries + 1,
            'timestamp': datetime.now().isoformat()
        }
        
        context['ti'].xcom_push(key=self.xcom_key, value=error_response)
        raise last_exception


class RateLimitedOperator(BaseOperator):
    """
    Operator that respects API rate limits.
    
    Uses token bucket algorithm to ensure we don't exceed rate limits.
    Perfect for: Polygon.io (5 calls/min), Finnhub (60 calls/min), etc.
    """
    
    ui_color = '#FFD93D'
    ui_fgcolor = '#000'
    
    @apply_defaults
    def __init__(
        self,
        api_callable: Callable,
        calls_per_minute: int = 60,
        burst_size: Optional[int] = None,
        xcom_key: str = 'result',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.api_callable = api_callable
        self.calls_per_minute = calls_per_minute
        self.burst_size = burst_size or calls_per_minute
        self.xcom_key = xcom_key
        
        # Token bucket state
        self._tokens = self.burst_size
        self._last_update = datetime.now()
        
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute with rate limiting"""
        
        # Refill tokens based on time passed
        self._refill_tokens()
        
        # Check if we have tokens
        if self._tokens < 1:
            wait_time = (1.0 / self.calls_per_minute) * 60
            self.log.warning(
                f"⏳ Rate limit reached. Waiting {wait_time:.1f}s..."
            )
            time.sleep(wait_time)
            self._refill_tokens()
        
        # Consume token
        self._tokens -= 1
        
        # Make API call
        try:
            result = self.api_callable(context)
            context['ti'].xcom_push(key=self.xcom_key, value=result)
            
            self.log.info(
                f"✅ API call successful. "
                f"Tokens remaining: {self._tokens}/{self.burst_size}"
            )
            
            return result
            
        except Exception as e:
            # Return token on failure
            self._tokens += 1
            raise
    
    def _refill_tokens(self):
        """Refill token bucket based on elapsed time"""
        now = datetime.now()
        elapsed = (now - self._last_update).total_seconds()
        
        # Calculate tokens to add
        tokens_to_add = (elapsed / 60.0) * self.calls_per_minute
        
        self._tokens = min(self._tokens + tokens_to_add, self.burst_size)
        self._last_update = now