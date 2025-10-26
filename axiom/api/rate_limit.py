"""
Rate Limiting for Axiom API using SlowAPI.

Features:
- Per-user rate limits
- Per-endpoint rate limits
- Redis-backed rate limiting (optional)
- Custom limit strategies
- Role-based limits
"""

import os
from typing import Callable

from fastapi import Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from axiom.api.auth import get_optional_user, User


# Rate limit configuration
DEFAULT_RATE_LIMIT = os.getenv("DEFAULT_RATE_LIMIT", "100/minute")
PREMIUM_RATE_LIMIT = os.getenv("PREMIUM_RATE_LIMIT", "1000/minute")
ADMIN_RATE_LIMIT = os.getenv("ADMIN_RATE_LIMIT", "10000/minute")

# Redis configuration (if available)
REDIS_URL = os.getenv("REDIS_URL", None)


def get_identifier(request: Request) -> str:
    """
    Get identifier for rate limiting.
    
    Uses authenticated user ID if available, otherwise IP address.
    """
    # Try to get user from request state (set by auth middleware)
    user = getattr(request.state, "user", None)
    
    if user and isinstance(user, User):
        # Use username as identifier for authenticated users
        return f"user:{user.username}"
    
    # Fall back to IP address for unauthenticated requests
    return f"ip:{get_remote_address(request)}"


def get_rate_limit_for_user(request: Request) -> str:
    """
    Get rate limit based on user role.
    
    - Admin: 10,000/minute
    - Premium users: 1,000/minute
    - Standard users: 100/minute
    - Unauthenticated: 50/minute
    """
    user = getattr(request.state, "user", None)
    
    if user and isinstance(user, User):
        if "admin" in user.roles:
            return ADMIN_RATE_LIMIT
        elif "premium" in user.roles:
            return PREMIUM_RATE_LIMIT
        else:
            return DEFAULT_RATE_LIMIT
    
    # Unauthenticated users get lower limit
    return "50/minute"


# Create limiter instance
if REDIS_URL:
    # Use Redis for distributed rate limiting
    limiter = Limiter(
        key_func=get_identifier,
        storage_uri=REDIS_URL,
        default_limits=[DEFAULT_RATE_LIMIT],
        headers_enabled=True,
    )
else:
    # Use in-memory storage (not suitable for multi-process deployments)
    limiter = Limiter(
        key_func=get_identifier,
        default_limits=[DEFAULT_RATE_LIMIT],
        headers_enabled=True,
    )


def rate_limit_by_role(default_limit: str = "100/minute"):
    """
    Decorator factory for role-based rate limiting.
    
    Usage:
        @app.get("/endpoint")
        @rate_limit_by_role("100/minute")
        async def my_endpoint():
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Apply the dynamic rate limit
        return limiter.limit(get_rate_limit_for_user)(func)
    return decorator


# Custom rate limit exceeded handler
def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors.
    
    Returns a JSON response with rate limit information.
    """
    from fastapi.responses import JSONResponse
    
    # Extract rate limit info
    retry_after = exc.headers.get("Retry-After", "60")
    limit = exc.headers.get("X-RateLimit-Limit", "unknown")
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Too many requests. Limit: {limit} requests per minute.",
            "retry_after": retry_after,
            "hint": "Consider upgrading to a premium plan for higher limits.",
        },
        headers={
            "Retry-After": retry_after,
            "X-RateLimit-Limit": limit,
            "X-RateLimit-Remaining": "0",
        },
    )


# Standard rate limit decorators for common use cases
def standard_rate_limit(func: Callable) -> Callable:
    """Apply standard rate limit (100/minute)."""
    return limiter.limit("100/minute")(func)


def high_rate_limit(func: Callable) -> Callable:
    """Apply high rate limit (1000/minute) for compute-intensive operations."""
    return limiter.limit("1000/minute")(func)


def low_rate_limit(func: Callable) -> Callable:
    """Apply low rate limit (10/minute) for expensive operations."""
    return limiter.limit("10/minute")(func)


# Endpoint-specific rate limits
ENDPOINT_LIMITS = {
    # Authentication endpoints
    "/api/v1/auth/token": "10/minute",
    "/api/v1/auth/refresh": "20/minute",
    
    # Computationally expensive endpoints
    "/api/v1/portfolio/optimize": "10/minute",
    "/api/v1/risk/monte-carlo": "5/minute",
    "/api/v1/ma/lbo-model": "10/minute",
    
    # Bulk operations
    "/api/v1/options/batch": "50/minute",
    "/api/v1/portfolio/batch-optimize": "5/minute",
    
    # Real-time data endpoints
    "/api/v1/market-data/stream": "1000/minute",
    "/api/v1/portfolio/real-time": "500/minute",
}


def get_endpoint_limit(request: Request) -> str:
    """Get rate limit for specific endpoint."""
    path = request.url.path
    return ENDPOINT_LIMITS.get(path, DEFAULT_RATE_LIMIT)