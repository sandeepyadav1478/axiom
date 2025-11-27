"""
Event Types for Real-Time Streaming.

Defines all event types that can be streamed through the system.
"""

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events that can be streamed."""
    
    # Market Data Events
    PRICE_UPDATE = "price_update"
    TRADE_EXECUTION = "trade_execution"
    ORDER_BOOK_UPDATE = "order_book_update"
    
    # News & Alerts
    NEWS_ALERT = "news_alert"
    MARKET_ALERT = "market_alert"
    SYSTEM_ALERT = "system_alert"
    
    # AI Analysis Events
    CLAUDE_ANALYSIS = "claude_analysis"
    LANGGRAPH_UPDATE = "langgraph_update"
    RAG_QUERY_RESULT = "rag_query_result"
    
    # Graph Updates
    NEO4J_GRAPH_UPDATE = "neo4j_graph_update"
    RELATIONSHIP_ADDED = "relationship_added"
    NODE_UPDATED = "node_updated"
    
    # Quality Metrics
    QUALITY_SCORE = "quality_score"
    ANOMALY_DETECTED = "anomaly_detected"
    VALIDATION_RESULT = "validation_result"
    
    # M&A Workflow Events
    DEAL_ANALYSIS_START = "deal_analysis_start"
    DEAL_ANALYSIS_PROGRESS = "deal_analysis_progress"
    DEAL_ANALYSIS_COMPLETE = "deal_analysis_complete"
    
    # System Events
    HEARTBEAT = "heartbeat"
    CONNECTION_STATUS = "connection_status"
    ERROR = "error"


class StreamEvent(BaseModel):
    """Standard event structure for streaming."""
    
    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    source: Optional[str] = Field(None, description="Event source/origin")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.json()


class PriceUpdateEvent(StreamEvent):
    """Price update event."""
    event_type: EventType = EventType.PRICE_UPDATE
    
    @classmethod
    def create(cls, symbol: str, price: float, volume: int, **kwargs):
        return cls(
            data={
                "symbol": symbol,
                "price": price,
                "volume": volume,
                **kwargs
            }
        )


class NewsAlertEvent(StreamEvent):
    """News alert event."""
    event_type: EventType = EventType.NEWS_ALERT
    
    @classmethod
    def create(cls, title: str, summary: str, url: str, sentiment: str = "neutral", **kwargs):
        return cls(
            data={
                "title": title,
                "summary": summary,
                "url": url,
                "sentiment": sentiment,
                **kwargs
            }
        )


class ClaudeAnalysisEvent(StreamEvent):
    """Claude analysis result event."""
    event_type: EventType = EventType.CLAUDE_ANALYSIS
    
    @classmethod
    def create(cls, query: str, answer: str, confidence: float, reasoning: list, **kwargs):
        return cls(
            data={
                "query": query,
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning,
                **kwargs
            }
        )


class GraphUpdateEvent(StreamEvent):
    """Neo4j graph update event."""
    event_type: EventType = EventType.NEO4J_GRAPH_UPDATE
    
    @classmethod
    def create(cls, update_type: str, node_id: str, properties: dict, **kwargs):
        return cls(
            data={
                "update_type": update_type,
                "node_id": node_id,
                "properties": properties,
                **kwargs
            }
        )


class QualityMetricEvent(StreamEvent):
    """Quality metric event."""
    event_type: EventType = EventType.QUALITY_SCORE
    
    @classmethod
    def create(cls, metric_name: str, score: float, threshold: float, status: str, **kwargs):
        return cls(
            data={
                "metric_name": metric_name,
                "score": score,
                "threshold": threshold,
                "status": status,
                **kwargs
            }
        )


class DealAnalysisEvent(StreamEvent):
    """M&A deal analysis event."""
    event_type: EventType = EventType.DEAL_ANALYSIS_PROGRESS
    
    @classmethod
    def create(cls, deal_id: str, stage: str, progress: float, message: str, **kwargs):
        return cls(
            data={
                "deal_id": deal_id,
                "stage": stage,
                "progress": progress,
                "message": message,
                **kwargs
            }
        )


class HeartbeatEvent(StreamEvent):
    """Heartbeat/keepalive event."""
    event_type: EventType = EventType.HEARTBEAT
    
    @classmethod
    def create(cls, server_time: Optional[datetime] = None):
        return cls(
            data={
                "server_time": (server_time or datetime.now()).isoformat(),
                "status": "alive"
            }
        )


__all__ = [
    "EventType",
    "StreamEvent",
    "PriceUpdateEvent",
    "NewsAlertEvent",
    "ClaudeAnalysisEvent",
    "GraphUpdateEvent",
    "QualityMetricEvent",
    "DealAnalysisEvent",
    "HeartbeatEvent"
]