"""
Dependency injection utilities for Axiom API.

Provides reusable dependencies for authentication, validation, and common tasks.
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Query, Header

from axiom.api.auth import get_current_user, get_optional_user, User


# Pagination dependencies
def get_pagination(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return"),
) -> Dict[str, int]:
    """
    Pagination parameters dependency.
    
    Usage:
        @app.get("/items")
        async def list_items(pagination: dict = Depends(get_pagination)):
            skip = pagination["skip"]
            limit = pagination["limit"]
    """
    return {"skip": skip, "limit": limit}


# API version dependency
def get_api_version(
    x_api_version: Optional[str] = Header(None, description="API version"),
) -> str:
    """
    Get API version from header or default to v1.
    
    Usage:
        @app.get("/endpoint")
        async def my_endpoint(version: str = Depends(get_api_version)):
            pass
    """
    return x_api_version or "v1"


# Request ID dependency
def get_request_id(
    x_request_id: Optional[str] = Header(None, description="Request ID"),
) -> Optional[str]:
    """Get request ID from header."""
    return x_request_id


# Common query parameters
class CommonQueryParams:
    """Common query parameters for list endpoints."""
    
    def __init__(
        self,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        sort_by: Optional[str] = Query(None, description="Field to sort by"),
        order: str = Query("asc", regex="^(asc|desc)$", description="Sort order"),
    ):
        self.skip = skip
        self.limit = limit
        self.sort_by = sort_by
        self.order = order


# User rate limit info
async def get_user_rate_limit_info(
    current_user: Optional[User] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """
    Get rate limit information for current user.
    
    Returns limit and remaining requests.
    """
    if current_user:
        if "admin" in current_user.roles:
            return {"limit": 10000, "remaining": 9999, "tier": "admin"}
        elif "premium" in current_user.roles:
            return {"limit": 1000, "remaining": 999, "tier": "premium"}
        else:
            return {"limit": 100, "remaining": 99, "tier": "standard"}
    else:
        return {"limit": 50, "remaining": 49, "tier": "anonymous"}


# Feature flags dependency
def check_feature_flag(feature: str):
    """
    Dependency to check if a feature is enabled.
    
    Usage:
        @app.get("/beta-feature")
        async def beta_endpoint(_: None = Depends(check_feature_flag("beta_features"))):
            pass
    """
    async def _check_feature(current_user: User = Depends(get_current_user)):
        # Mock feature flag check
        enabled_features = {
            "beta_features": True,
            "advanced_analytics": True,
            "ai_insights": False,
        }
        
        if not enabled_features.get(feature, False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Feature '{feature}' is not enabled",
            )
        
        return None
    
    return _check_feature


# Cache dependency
class CacheSettings:
    """Cache settings dependency."""
    
    def __init__(
        self,
        no_cache: bool = Query(False, description="Bypass cache"),
        cache_ttl: int = Query(300, ge=0, le=3600, description="Cache TTL in seconds"),
    ):
        self.no_cache = no_cache
        self.cache_ttl = cache_ttl


# Response format dependency
def get_response_format(
    format: str = Query("json", regex="^(json|xml|csv)$", description="Response format"),
) -> str:
    """Get desired response format."""
    return format