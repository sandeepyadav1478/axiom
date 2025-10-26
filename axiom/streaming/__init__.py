"""
Real-Time Data Streaming Infrastructure

Enterprise-grade streaming infrastructure leveraging battle-tested external libraries:
- websockets: WebSocket client/server
- redis: Real-time caching and pub/sub
- aiohttp: Async HTTP/WebSocket
- python-binance: Binance WebSocket feeds
- alpaca-trade-api: Alpaca real-time data
- polygon-api-client: Polygon.io streams

Features:
- Multi-connection WebSocket management
- Automatic reconnection with exponential backoff
- Real-time portfolio tracking with live P&L
- Sub-millisecond Redis caching
- Event processing pipeline
- Real-time risk monitoring
"""

from axiom.streaming.config import StreamingConfig
from axiom.streaming.websocket_manager import WebSocketManager
from axiom.streaming.redis_cache import RealTimeCache
from axiom.streaming.portfolio_tracker import PortfolioTracker, Position
from axiom.streaming.event_processor import EventProcessor
from axiom.streaming.risk_monitor import RealTimeRiskMonitor
from axiom.streaming.market_data import MarketDataStreamer

# Convenience aliases
RiskMonitor = RealTimeRiskMonitor

__all__ = [
    'StreamingConfig',
    'WebSocketManager',
    'RealTimeCache',
    'PortfolioTracker',
    'Position',
    'EventProcessor',
    'RealTimeRiskMonitor',
    'RiskMonitor',
    'MarketDataStreamer',
]

__version__ = '1.0.0'