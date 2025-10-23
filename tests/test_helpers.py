"""
Test Helpers and Utilities
===========================

Provides retry logic and utilities for tests with external dependencies.
"""

import functools
import time
from typing import Callable, Type, Tuple, Optional


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    verbose: bool = True
):
    """
    Decorator to retry test functions that may fail due to external dependencies.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        verbose: Whether to print retry information
        
    Usage:
        @retry_on_exception(max_retries=3, delay=1.0)
        def test_external_api():
            # Test that calls external API
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if verbose:
                        print(f"  Attempt {attempt + 1}/{max_retries} failed: {str(e)[:50]}...")
                    
                    if attempt < max_retries - 1:
                        if verbose:
                            print(f"  Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        if verbose:
                            print(f"  All {max_retries} attempts failed.")
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator


# Specific retry decorators for common scenarios
retry_on_network = retry_on_exception(
    max_retries=3,
    delay=2.0,
    exceptions=(ConnectionError, TimeoutError, OSError),
    verbose=True
)

retry_on_api_error = retry_on_exception(
    max_retries=3,
    delay=1.0,
    exceptions=(ConnectionError, TimeoutError, ValueError),
    verbose=True
)


__all__ = [
    "retry_on_exception",
    "retry_on_network",
    "retry_on_api_error",
]