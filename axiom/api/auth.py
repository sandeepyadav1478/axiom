"""
Authentication and Authorization for Axiom API.

Features:
- JWT token-based authentication
- API key authentication
- Role-based access control (RBAC)
- Token refresh mechanism
- Password hashing with bcrypt
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel


# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "axiom-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)

# API Key scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Bearer token scheme
bearer_scheme = HTTPBearer(auto_error=False)


# Pydantic models
class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    user_id: Optional[str] = None
    roles: list[str] = []
    exp: Optional[datetime] = None


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    roles: list[str] = []
    disabled: bool = False
    api_key: Optional[str] = None


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


# Mock user database (replace with real database)
FAKE_USERS_DB = {
    "demo": {
        "username": "demo",
        "email": "demo@axiom.com",
        "full_name": "Demo User",
        "hashed_password": pwd_context.hash("demo123"),
        "roles": ["user"],
        "disabled": False,
        "api_key": "axiom-demo-key-12345",
    },
    "admin": {
        "username": "admin",
        "email": "admin@axiom.com",
        "full_name": "Admin User",
        "hashed_password": pwd_context.hash("admin123"),
        "roles": ["admin", "user"],
        "disabled": False,
        "api_key": "axiom-admin-key-67890",
    },
}

# API Key to User mapping
API_KEY_TO_USER = {
    "axiom-demo-key-12345": "demo",
    "axiom-admin-key-67890": "admin",
}


# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# User utilities
def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in FAKE_USERS_DB:
        user_dict = FAKE_USERS_DB[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def authenticate_api_key(api_key: str) -> Optional[UserInDB]:
    """Authenticate user with API key."""
    username = API_KEY_TO_USER.get(api_key)
    if not username:
        return None
    return get_user(username)


# Token utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_tokens(user: UserInDB) -> Token:
    """Create access and refresh tokens for user."""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    token_data = {
        "sub": user.username,
        "user_id": user.username,  # In production, use actual user ID
        "roles": user.roles,
    }
    
    access_token = create_access_token(token_data, expires_delta=access_token_expires)
    refresh_token = create_refresh_token(token_data)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        roles: list = payload.get("roles", [])
        exp_timestamp: float = payload.get("exp")
        
        if username is None:
            return None
        
        exp = datetime.fromtimestamp(exp_timestamp) if exp_timestamp else None
        
        return TokenData(
            username=username,
            user_id=user_id,
            roles=roles,
            exp=exp,
        )
    except JWTError:
        return None


# Dependencies
async def get_current_user_from_token(token: str = Depends(oauth2_scheme)) -> Optional[User]:
    """Get current user from JWT token."""
    if not token:
        return None
    
    token_data = decode_token(token)
    if not token_data:
        return None
    
    user = get_user(token_data.username)
    if user is None:
        return None
    
    return User(**user.dict(exclude={"hashed_password"}))


async def get_current_user_from_api_key(api_key: str = Security(api_key_header)) -> Optional[User]:
    """Get current user from API key."""
    if not api_key:
        return None
    
    user = authenticate_api_key(api_key)
    if user is None:
        return None
    
    return User(**user.dict(exclude={"hashed_password"}))


async def get_current_user_from_bearer(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> Optional[User]:
    """Get current user from Bearer token."""
    if not credentials:
        return None
    
    token_data = decode_token(credentials.credentials)
    if not token_data:
        return None
    
    user = get_user(token_data.username)
    if user is None:
        return None
    
    return User(**user.dict(exclude={"hashed_password"}))


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key),
    bearer_user: Optional[User] = Depends(get_current_user_from_bearer),
) -> User:
    """
    Get current authenticated user.
    
    Tries multiple authentication methods:
    1. OAuth2 token
    2. API Key
    3. Bearer token
    
    Raises HTTPException if no valid authentication found.
    """
    user = token_user or api_key_user or bearer_user
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (wrapper for clarity)."""
    return current_user


def require_role(required_role: str):
    """
    Dependency to require a specific role.
    
    Usage:
        @app.get("/admin")
        async def admin_only(user: User = Depends(require_role("admin"))):
            pass
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if required_role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}",
            )
        return current_user
    
    return role_checker


def require_any_role(*required_roles: str):
    """
    Dependency to require any of the specified roles.
    
    Usage:
        @app.get("/premium")
        async def premium_only(user: User = Depends(require_any_role("admin", "premium"))):
            pass
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(required_roles)}",
            )
        return current_user
    
    return role_checker


# Optional authentication (for public endpoints with optional auth)
async def get_optional_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key),
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Useful for endpoints that are public but offer enhanced features for authenticated users.
    """
    user = token_user or api_key_user
    
    if user and user.disabled:
        return None
    
    return user