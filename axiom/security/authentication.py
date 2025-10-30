"""
Authentication and Authorization for Production API

Implements:
- API key authentication
- JWT tokens
- Role-based access control (RBAC)
- Rate limiting per user

Security for production deployment.
"""

from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import jwt
import hashlib


API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


class AuthenticationManager:
    """
    Manage authentication for production API
    
    Supports:
    - API keys for programmatic access
    - JWT tokens for web UI
    - RBAC for different user levels
    """
    
    def __init__(self, secret_key: str = "your-secret-key-here"):
        self.secret_key = secret_key
        
        # API keys (in production, store in database)
        self.api_keys = {
            hashlib.sha256("demo-key-123".encode()).hexdigest(): {
                'user': 'demo_user',
                'tier': 'standard',
                'rate_limit': 100
            }
        }
        
        # User roles
        self.roles = {
            'admin': ['*'],  # Full access
            'analyst': ['portfolio', 'options', 'credit', 'ma'],
            'viewer': ['portfolio', 'credit']
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """
        Validate API key
        
        Args:
            api_key: API key from request header
            
        Returns:
            User info if valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            return self.api_keys[key_hash]
        
        return None
    
    def create_jwt_token(
        self,
        user_id: str,
        role: str,
        expires_delta: timedelta = timedelta(hours=24)
    ) -> str:
        """
        Create JWT token for web UI authentication
        
        Args:
            user_id: User identifier
            role: User role
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            'sub': user_id,
            'role': role,
            'exp': expire,
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        return token
    
    def decode_jwt_token(self, token: str) -> Optional[Dict]:
        """
        Decode and validate JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def check_permission(self, user_role: str, resource: str) -> bool:
        """
        Check if user has permission for resource
        
        Args:
            user_role: User's role
            resource: Resource being accessed
            
        Returns:
            True if authorized, False otherwise
        """
        if user_role not in self.roles:
            return False
        
        allowed = self.roles[user_role]
        
        # Admin has full access
        if '*' in allowed:
            return True
        
        # Check if resource in allowed list
        return resource in allowed


# Rate limiting
class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self):
        self.requests = {}  # user_id -> [(timestamp, count)]
    
    def check_rate_limit(
        self,
        user_id: str,
        limit: int = 100,
        window_minutes: int = 1
    ) -> bool:
        """
        Check if user is within rate limit
        
        Args:
            user_id: User identifier
            limit: Max requests per window
            window_minutes: Time window in minutes
            
        Returns:
            True if within limit, False if exceeded
        """
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)
        
        # Clean old requests
        if user_id in self.requests:
            self.requests[user_id] = [
                req for req in self.requests[user_id]
                if req > window_start
            ]
        else:
            self.requests[user_id] = []
        
        # Check limit
        if len(self.requests[user_id]) >= limit:
            return False
        
        # Add current request
        self.requests[user_id].append(now)
        
        return True


if __name__ == "__main__":
    print("Authentication Manager - Production Security")
    
    auth = AuthenticationManager()
    
    # Create token
    token = auth.create_jwt_token('user123', 'analyst')
    print(f"JWT token created")
    
    # Validate
    payload = auth.decode_jwt_token(token)
    print(f"Token valid: {payload is not None}")
    
    # Check permission
    can_access = auth.check_permission('analyst', 'portfolio')
    print(f"Analyst can access portfolio: {can_access}")
    
    # Rate limiting
    limiter = RateLimiter()
    within_limit = limiter.check_rate_limit('user123', limit=100)
    print(f"Within rate limit: {within_limit}")
    
    print("\n✓ Production security ready")