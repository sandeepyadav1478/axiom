"""
Production Real-Time Streaming API.

FastAPI service with WebSocket and SSE endpoints for live market intelligence.
"""

import asyncio
import logging
from typing import Optional, Set, List
from datetime import datetime
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import os

from .connection_manager import ConnectionManager
from .redis_pubsub import RedisPubSubManager
from .event_types import (
    EventType, StreamEvent, PriceUpdateEvent, NewsAlertEvent,
    ClaudeAnalysisEvent, GraphUpdateEvent, QualityMetricEvent,
    DealAnalysisEvent
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Axiom Streaming API",
    description="Production real-time streaming API for live market intelligence",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global managers
connection_manager = ConnectionManager(
    heartbeat_interval=30,
    connection_timeout=90
)

redis_manager: Optional[RedisPubSubManager] = None


# Request/Response models
class SubscriptionRequest(BaseModel):
    """Subscription request model."""
    event_types: list[str] = Field(..., description="Event types to subscribe to")


class PublishRequest(BaseModel):
    """Publish event request model."""
    event_type: str = Field(..., description="Type of event")
    data: dict = Field(..., description="Event data")
    source: Optional[str] = Field(None, description="Event source")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    connections: int
    redis_connected: bool
    uptime_seconds: float

class IntelligenceRequest(BaseModel):
    """Intelligence analysis request model."""
    symbols: List[str] = Field(..., description="Stock symbols to analyze")
    timeframe: str = Field(default='1d', description="Data timeframe: 1h, 1d, 1w, 1m")
    analysis_type: str = Field(default='market_overview', description="Analysis type")



# Startup/Shutdown
start_time = datetime.now()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global redis_manager
    
    logger.info("Starting Axiom Streaming API...")
    
    # Initialize Redis if configured
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    try:
        redis_manager = RedisPubSubManager(redis_url=redis_url)
        await redis_manager.connect()
        
        # Subscribe to all event types for broadcasting
        await redis_manager.subscribe_pattern(
            pattern="*",
            callback=handle_redis_event
        )
        
        logger.info("Redis pub/sub initialized")
        
    except Exception as e:
        logger.warning(f"Redis initialization failed (will work without Redis): {e}")
        redis_manager = None
    
    logger.info("Axiom Streaming API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Axiom Streaming API...")
    
    if redis_manager:
        await redis_manager.disconnect()
    
    logger.info("Axiom Streaming API shutdown complete")


# WebSocket Endpoints
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time bidirectional communication.
    
    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    await connection_manager.connect(client_id, websocket)
    
    try:
        # Send welcome message
        welcome = StreamEvent(
            event_type=EventType.CONNECTION_STATUS,
            data={
                "status": "connected",
                "client_id": client_id,
                "server_time": datetime.now().isoformat()
            },
            source="streaming_api"
        )
        await connection_manager.send_personal_message(client_id, welcome)
        
        # Listen for client messages
        while True:
            data = await websocket.receive_json()
            
            # Handle subscription requests
            if data.get("action") == "subscribe":
                event_types = set(data.get("event_types", []))
                connection_manager.subscribe(client_id, event_types)
                
                response = StreamEvent(
                    event_type=EventType.CONNECTION_STATUS,
                    data={
                        "status": "subscribed",
                        "event_types": list(event_types)
                    }
                )
                await connection_manager.send_personal_message(client_id, response)
            
            # Handle unsubscribe requests
            elif data.get("action") == "unsubscribe":
                event_types = set(data.get("event_types", []))
                connection_manager.unsubscribe(client_id, event_types)
                
                response = StreamEvent(
                    event_type=EventType.CONNECTION_STATUS,
                    data={
                        "status": "unsubscribed",
                        "event_types": list(event_types)
                    }
                )
                await connection_manager.send_personal_message(client_id, response)
            
            # Handle ping/pong
            elif data.get("action") == "ping":
                pong = StreamEvent(
                    event_type=EventType.HEARTBEAT,
                    data={"status": "pong"}
                )
                await connection_manager.send_personal_message(client_id, pong)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
        
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        connection_manager.disconnect(client_id)


# Server-Sent Events (SSE) Endpoint
@app.get("/sse/{client_id}")
async def sse_endpoint(request: Request, client_id: str):
    """
    Server-Sent Events endpoint for one-way streaming.
    
    Ideal for dashboard updates where client doesn't need to send data.
    
    Args:
        request: FastAPI request
        client_id: Unique client identifier
    """
    async def event_generator():
        """Generate SSE events."""
        try:
            # Send initial connection event
            yield f"event: connection\ndata: {json.dumps({'status': 'connected', 'client_id': client_id})}\n\n"
            
            # Create a queue for this client
            event_queue = asyncio.Queue()
            
            # Store queue for broadcasting
            sse_queues[client_id] = event_queue
            
            # Send events from queue
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    
                    # Format as SSE
                    event_data = json.dumps(event.to_dict())
                    yield f"event: {event.event_type.value}\ndata: {event_data}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"
                    
        except Exception as e:
            logger.error(f"SSE error for {client_id}: {e}")
        finally:
            # Cleanup
            if client_id in sse_queues:
                del sse_queues[client_id]
            logger.info(f"SSE client {client_id} disconnected")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# Store SSE queues
sse_queues = {}


# REST API Endpoints
@app.post("/publish")
async def publish_event(request: PublishRequest):
    """
    Publish an event to all subscribers.
    
    Args:
        request: Event to publish
    """
    try:
        # Create event
        event = StreamEvent(
            event_type=EventType(request.event_type),
            data=request.data,
            source=request.source or "api"
        )
        
        # Broadcast via WebSocket
        await connection_manager.broadcast_to_subscribers(
            event,
            request.event_type
        )
        
        # Broadcast via SSE
        for queue in sse_queues.values():
            await queue.put(event)
        
        # Publish to Redis for other instances
        if redis_manager and redis_manager.is_connected():
            await redis_manager.publish(event)
        
        return {
            "status": "published",
            "event_type": request.event_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {request.event_type}")
    except Exception as e:
        logger.error(f"Error publishing event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PriceUpdateRequest(BaseModel):
    """Price update request model."""
    symbol: str
    price: float
    volume: int
    source: Optional[str] = None

@app.post("/publish/price")
async def publish_price_update(request: PriceUpdateRequest):
    """
    Publish a price update event.
    """
    symbol = request.symbol
    price = request.price
    volume = request.volume
    source = request.source
    event = PriceUpdateEvent.create(
        symbol=symbol,
        price=price,
        volume=volume,
        source=source or "api"
    )
    
    # Broadcast
    await connection_manager.broadcast_to_subscribers(
        event,
        EventType.PRICE_UPDATE.value
    )
    
    # Broadcast to SSE clients
    for queue in sse_queues.values():
        await queue.put(event)
    
    # Publish to Redis
    if redis_manager and redis_manager.is_connected():
        await redis_manager.publish(event)
    
    return {"status": "published", "event_type": "price_update"}


class NewsAlertRequest(BaseModel):
    """News alert request model."""
    title: str
    summary: str
    url: str
    sentiment: str = "neutral"

@app.post("/publish/news")
async def publish_news_alert(request: NewsAlertRequest):
    """Publish a news alert event."""
    title = request.title
    summary = request.summary
    url = request.url
    sentiment = request.sentiment
    event = NewsAlertEvent.create(
        title=title,
        summary=summary,
        url=url,
        sentiment=sentiment
    )
    
    await connection_manager.broadcast_to_subscribers(
        event,
        EventType.NEWS_ALERT.value
    )
    
    for queue in sse_queues.values():
        await queue.put(event)
    
    if redis_manager and redis_manager.is_connected():
        await redis_manager.publish(event)
    
    return {"status": "published", "event_type": "news_alert"}


class ClaudeAnalysisRequest(BaseModel):
    """Claude analysis request model."""
    query: str
    answer: str
    confidence: float
    reasoning: List[str]

@app.post("/publish/analysis")
async def publish_claude_analysis(request: ClaudeAnalysisRequest):
    """Publish Claude analysis result."""
    query = request.query
    answer = request.answer
    confidence = request.confidence
    reasoning = request.reasoning
    event = ClaudeAnalysisEvent.create(
        query=query,
        answer=answer,
        confidence=confidence,
        reasoning=reasoning
    )
    
    await connection_manager.broadcast_to_subscribers(
        event,
        EventType.CLAUDE_ANALYSIS.value
    )
    
    for queue in sse_queues.values():
        await queue.put(event)
    
    if redis_manager and redis_manager.is_connected():
        await redis_manager.publish(event)
    
    return {"status": "published", "event_type": "claude_analysis"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of streaming service
    """
    redis_connected = False
    if redis_manager:
        redis_connected = await redis_manager.health_check()
    
    connection_manager.health_check()
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        connections=len(connection_manager.active_connections),
        redis_connected=redis_connected,
        uptime_seconds=uptime
    )


@app.get("/stats")
async def get_stats():
    """Get detailed statistics."""
    connection_stats = connection_manager.get_connection_stats()
    
    redis_stats = {}
    if redis_manager:
        redis_stats = await redis_manager.get_stats()
    
    return {
        "connections": connection_stats,
        "redis": redis_stats,
        "sse_clients": len(sse_queues),
        "uptime_seconds": (datetime.now() - start_time).total_seconds()
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Axiom Streaming API",
        "version": "1.0.0",
        "description": "Production real-time streaming for market intelligence",
        "endpoints": {
            "websocket": "ws://host/ws/{client_id}",
            "sse": "GET /sse/{client_id}",
            "publish": "POST /publish",
            "health": "GET /health",
            "stats": "GET /stats"
        },
        "features": [
            "WebSocket bidirectional streaming",
            "Server-Sent Events for dashboards",
            "Redis pub/sub for horizontal scaling",
            "Automatic heartbeat and reconnection",
            "Multi-event type subscriptions"
        ]
    }


# Redis event handler
async def handle_redis_event(event: StreamEvent):
    """
    Handle events received from Redis.
    
    Broadcasts to all connected WebSocket and SSE clients.
    """
    # Broadcast to WebSocket clients
    await connection_manager.broadcast_to_subscribers(
        event,
        event.event_type.value
    )
    
    # Broadcast to SSE clients
    for queue in sse_queues.values():
        try:
            await queue.put(event)
        except:
            pass


class StreamingService:
    """
    Convenience wrapper for the streaming service.
    
    Allows programmatic event publishing from other parts of the application.
    """
    
    def __init__(self):
        self.connection_manager = connection_manager
        self.redis_manager = redis_manager
    
    async def publish(self, event: StreamEvent):
        """Publish an event."""
        # Broadcast to WebSocket
        await self.connection_manager.broadcast_to_subscribers(
            event,
            event.event_type.value
        )
        
        # Broadcast to SSE
        for queue in sse_queues.values():
            await queue.put(event)
        
        # Publish to Redis
        if self.redis_manager and self.redis_manager.is_connected():
            await self.redis_manager.publish(event)
    
    async def publish_price_update(self, symbol: str, price: float, volume: int):
        """Publish price update."""
        event = PriceUpdateEvent.create(symbol=symbol, price=price, volume=volume)
        await self.publish(event)
    
    async def publish_news(self, title: str, summary: str, url: str):
        """Publish news alert."""
        event = NewsAlertEvent.create(title=title, summary=summary, url=url)
        await self.publish(event)
    
    async def publish_analysis(self, query: str, answer: str, confidence: float, reasoning: list):
        """Publish Claude analysis."""
        event = ClaudeAnalysisEvent.create(
            query=query,
            answer=answer,
            confidence=confidence,
            reasoning=reasoning
        )
        await self.publish(event)
    
    async def publish_graph_update(self, update_type: str, node_id: str, properties: dict):
        """Publish Neo4j graph update."""
        event = GraphUpdateEvent.create(
            update_type=update_type,
            node_id=node_id,
            properties=properties
        )
        await self.publish(event)
    
    async def publish_quality_metric(self, metric_name: str, score: float, threshold: float, status: str):
        """Publish quality metric."""
        event = QualityMetricEvent.create(
            metric_name=metric_name,
            score=score,
            threshold=threshold,
            status=status
        )
        await self.publish(event)
    
    async def publish_deal_analysis(self, deal_id: str, stage: str, progress: float, message: str):
        """Publish deal analysis progress."""
        event = DealAnalysisEvent.create(
            deal_id=deal_id,
            stage=stage,
            progress=progress,
            message=message
        )
        await self.publish(event)


# ================================================================
# LangGraph Intelligence Integration
# ================================================================

from typing import List

@app.websocket("/ws/intelligence/{client_id}")
async def intelligence_websocket(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for streaming LangGraph intelligence.
    
    Provides real-time market analysis with Claude-powered insights.
    Updates every 60 seconds with fresh intelligence.
    """
    await websocket.accept()
    
    try:
        # Import intelligence service
        import sys
        sys.path.insert(0, '/app')
        from axiom.ai_layer.services.langgraph_intelligence_service import IntelligenceSynthesisService
        
        # Initialize service
        intelligence = IntelligenceSynthesisService()
        
        # Send welcome
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "service": "LangGraph Intelligence",
            "update_interval": 60
        })
        
        # Default symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        
        # Continuous intelligence streaming
        while True:
            try:
                # Generate intelligence report
                logger.info(f"Generating intelligence for {client_id}...")
                report = await intelligence.analyze_market(symbols, timeframe='1d')
                
                # Send to client
                await websocket.send_json({
                    "type": "intelligence_report",
                    "timestamp": datetime.now().isoformat(),
                    "report": report
                })
                
                logger.info(f"✅ Sent intelligence to {client_id}")
                
                # Wait for next cycle
                await asyncio.sleep(60)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Intelligence generation error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
                await asyncio.sleep(60)
                
    except WebSocketDisconnect:
        logger.info(f"Intelligence client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Intelligence WebSocket error: {e}")
    finally:
        if 'intelligence' in locals():
            intelligence.close()


@app.post("/intelligence/analyze")
async def get_intelligence_report(request: IntelligenceRequest):
    """
    Get one-time intelligence report via REST API.
    
    Uses LangGraph multi-agent analysis on current data.
    """
    symbols = request.symbols
    timeframe = request.timeframe
    analysis_type = request.analysis_type
    try:
        from axiom.ai_layer.services.langgraph_intelligence_service import IntelligenceSynthesisService
        
        intelligence = IntelligenceSynthesisService()
        
        report = await intelligence.analyze_market(
            symbols=symbols,
            analysis_type=analysis_type,
            timeframe=timeframe
        )
        
        intelligence.close()
        
        return report
        
    except Exception as e:
        logger.error(f"Intelligence report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

__all__ = ["app", "StreamingService"]