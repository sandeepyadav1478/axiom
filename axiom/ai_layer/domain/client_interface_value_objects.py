"""
Client Interface Domain Value Objects

Immutable value objects for client interaction domain.
Following DDD principles - these capture user sessions, requests, and responses.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on invalid requests)
- Rich behavior (session management, response formatting)
- Type-safe (using Decimal for precision, Enum for types)

These represent client interactions as first-class domain concepts.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from enum import Enum


class RequestType(str, Enum):
    """Client request types"""
    DASHBOARD = "dashboard"
    QUESTION = "question"
    REPORT = "report"
    EXPLAIN = "explain"
    ANALYZE = "analyze"


class ResponseFormat(str, Enum):
    """Response format"""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    PDF = "pdf"


class SessionStatus(str, Enum):
    """Client session status"""
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ClientSession:
    """
    Client session
    
    Immutable session state
    """
    session_id: str
    client_id: str
    
    # Session state
    status: SessionStatus
    
    # Conversation history
    message_count: int
    
    # Metadata
    started_at: datetime
    last_activity: datetime
    
    # Client details
    client_tier: str = "standard"  # standard, premium, enterprise
    
    def __post_init__(self):
        """Validate session"""
        if self.message_count < 0:
            raise ValueError("Message count must be non-negative")
        
        if self.last_activity < self.started_at:
            raise ValueError("Last activity cannot be before session start")
    
    def is_active(self) -> bool:
        """Check if session is active"""
        return self.status == SessionStatus.ACTIVE
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return self.status == SessionStatus.EXPIRED
    
    def get_duration(self) -> Decimal:
        """Get session duration in seconds"""
        delta = self.last_activity - self.started_at
        return Decimal(str(delta.total_seconds()))
    
    def get_idle_time(self) -> Decimal:
        """Get time since last activity in seconds"""
        delta = datetime.utcnow() - self.last_activity
        return Decimal(str(delta.total_seconds()))


@dataclass(frozen=True)
class UserQuery:
    """
    User query
    
    Immutable user question or request
    """
    query_id: str
    session_id: str
    client_id: str
    
    # Query details
    query_text: str
    request_type: RequestType
    
    # Context
    requires_portfolio_data: bool = False
    requires_market_data: bool = False
    requires_analytics: bool = False
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate query"""
        if not self.query_text or len(self.query_text.strip()) == 0:
            raise ValueError("Query text cannot be empty")
        
        if len(self.query_text) > 10000:
            raise ValueError("Query text too long (max 10000 characters)")
    
    def get_query_length(self) -> int:
        """Get query length in characters"""
        return len(self.query_text)
    
    def is_complex_query(self) -> bool:
        """Check if query is complex (requires multiple data sources)"""
        return sum([
            self.requires_portfolio_data,
            self.requires_market_data,
            self.requires_analytics
        ]) >= 2


# Example usage
if __name__ == "__main__":
    from axiom.ai_layer.infrastructure.observability import Logger
    
    logger = Logger("client_interface_domain_test")
    
    logger.info("test_starting", test="CLIENT INTERFACE DOMAIN VALUE OBJECTS")
    
    # Create session
    logger.info("creating_client_session")
    
    session = ClientSession(
        session_id="SESS-001",
        client_id="CLIENT-001",
        status=SessionStatus.ACTIVE,
        message_count=5,
        started_at=datetime(2024, 10, 30, 10, 0),
        last_activity=datetime(2024, 10, 30, 10, 15),
        client_tier="enterprise"
    )
    
    logger.info(
        "session_created",
        session_id=session.session_id,
        active=session.is_active(),
        duration=float(session.get_duration())
    )
    
    # Create user query
    logger.info("creating_user_query")
    
    query = UserQuery(
        query_id="QUERY-001",
        session_id="SESS-001",
        client_id="CLIENT-001",
        query_text="What is my current P&L and what is driving it?",
        request_type=RequestType.ANALYZE,
        requires_portfolio_data=True,
        requires_analytics=True
    )
    
    logger.info(
        "query_created",
        query_length=query.get_query_length(),
        complex=query.is_complex_query()
    )
    
    logger.info(
        "test_complete",
        artifacts_created=[
            "Immutable client interface objects",
            "Self-validating",
            "Rich domain behavior",
            "Session management",
            "Proper logging (no print)"
        ]
    )