"""
Real-Time Streaming API for Axiom Analytics Platform.

Production-grade WebSocket and SSE infrastructure for:
- Live market data streaming
- Real-time AI analysis results  
- News alerts and notifications
- Graph updates from Neo4j
- Quality metrics monitoring

Features:
- WebSocket endpoints for bidirectional communication
- Server-Sent Events (SSE) for dashboard streaming
- Redis pub/sub for multi-client broadcasting
- Connection management with heartbeat
- Automatic reconnection logic
- Load balancing support
"""

from .streaming_service import StreamingService, app
from .connection_manager import ConnectionManager
from .redis_pubsub import RedisPubSubManager
from .event_types import EventType, StreamEvent

__version__ = "1.0.0"

__all__ = [
    "StreamingService",
    "ConnectionManager",
    "RedisPubSubManager",
    "EventType",
    "StreamEvent",
    "app"
]