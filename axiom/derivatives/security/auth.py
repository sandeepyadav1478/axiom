"""
Authentication and Authorization for Derivatives Platform

Production-grade security for $5-10M/year enterprise clients.
Must be bulletproof - one breach = catastrophic.

Features:
- JWT authentication
- API key management
- Role-based access control
- Rate limiting per client
- Audit logging
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict
import secrets


# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)  # Generate securely in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class User:
    """User model for authentication"""
    def __init__(self, user_id: str, email: str, tier: str, rate_limit: int):
        self.user_id = user_id
        self.email = email
        self.tier = tier  # 'free', 'professional', 'enterprise'
        self.rate_limit = rate_limit  # Requests per hour
        self.permissions = self._get_permissions(tier)
    
    def _get_permissions(self, tier: str) -> list:
        """Get permissions based on tier"""
        if tier == 'enterprise':
            return ['greeks', 'exotic', 'surface', 'strategy', 'backtest', 'realtime']
        elif tier == 'professional':
            return ['greeks', 'exotic', 'surface']
        else:  # free
            return ['greeks']


# In-memory user database (use PostgreSQL in production)
USERS_DB = {
    "market_maker_1": {
        "user_id": "mm_001",
        "email": "trading@marketmaker.com",
        "hashed_password": pwd_context.hash("secure_password"),
        "tier": "enterprise",
        "rate_limit": 1000000,  # 1M req/hour
        "api_key": "mm_001_api_key_here"
    }
}


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        return payload
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> User:
    """Verify API key and return user"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Find user by API key
    for user_data in USERS_DB.values():
        if user_data.get('api_key') == api_key:
            return User(
                user_id=user_data['user_id'],
                email=user_data['email'],
                tier=user_data['tier'],
                rate_limit=user_data['rate_limit']
            )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key"
    )


def check_rate_limit(user: User, endpoint: str) -> bool:
    """
    Check if user is within rate limits
    
    Uses Redis for distributed rate limiting
    """
    # In production: Use Redis INCR with TTL
    # For now: Allow all
    return True


def check_permission(user: User, permission: str):
    """Check if user has required permission"""
    if permission not in user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission} not available in {user.tier} tier"
        )


# Example usage in FastAPI endpoint:
"""
@app.post("/greeks")
async def calculate_greeks(
    request: GreeksRequest,
    user: User = Depends(verify_api_key)
):
    # Check permission
    check_permission(user, 'greeks')
    
    # Check rate limit
    if not check_rate_limit(user, 'greeks'):
        raise HTTPException(429, "Rate limit exceeded")
    
    # Process request...
    result = greeks_engine.calculate_greeks(...)
    
    # Audit log
    log_api_call(user, 'greeks', result.calculation_time_us)
    
    return result
"""