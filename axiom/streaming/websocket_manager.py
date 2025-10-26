"""
WebSocket Manager

Robust WebSocket connection management with automatic reconnection,
heartbeat handling, and connection pooling.

Uses the `websockets` library for production-grade WebSocket handling.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Callable, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    # Fallback types for when websockets is not installed
    ConnectionClosed = Exception
    WebSocketException = Exception

from axiom.streaming.config import StreamingConfig


logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class ConnectionStats:
    """Connection statistics."""
    connection_time: float = 0.0
    reconnect_count: int = 0
    messages_received: int = 0
    messages_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    last_message_time: float = 0.0
    last_ping_time: float = 0.0
    avg_latency_ms: float = 0.0


class WebSocketConnection:
    """
    Individual WebSocket connection with reconnection logic.
    
    Features:
    - Automatic reconnection with exponential backoff
    - Heartbeat/ping-pong handling
    - Message buffering during reconnection
    - Connection health monitoring
    """
    
    def __init__(
        self,
        name: str,
        url: str,
        on_message: Callable,
        config: StreamingConfig,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
    ):
        """
        Initialize WebSocket connection.
        
        Args:
            name: Connection identifier
            url: WebSocket URL
            on_message: Message handler callback
            config: Streaming configuration
            on_connect: Optional connect callback
            on_disconnect: Optional disconnect callback
        """
        self.name = name
        self.url = url
        self.on_message = on_message
        self.config = config
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        
        self.websocket: Optional[Any] = None
        self.state = ConnectionState.DISCONNECTED
        self.stats = ConnectionStats()
        
        self._reconnect_delay = config.reconnect_delay
        self._message_buffer: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
    
    async def connect(self):
        """Establish WebSocket connection with reconnection logic."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets library not installed. "
                "Install with: pip install websockets"
            )
        
        self._running = True
        attempt = 0
        
        while self._running and attempt < self.config.reconnect_attempts:
            try:
                logger.info(f"Connecting to {self.name} at {self.url}")
                self.state = ConnectionState.CONNECTING
                
                # Connect with timeout
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.url,
                        ping_interval=self.config.ping_interval,
                        ping_timeout=self.config.connection_timeout,
                    ),
                    timeout=self.config.connection_timeout
                )
                
                self.state = ConnectionState.CONNECTED
                self.stats.connection_time = time.time()
                self.stats.reconnect_count += 1
                
                logger.info(f"Connected to {self.name}")
                
                # Call connect callback
                if self.on_connect:
                    await self.on_connect(self)
                
                # Start message handling
                await self._handle_messages()
                
            except asyncio.TimeoutError:
                logger.warning(f"Connection timeout for {self.name}")
                attempt += 1
                await self._reconnect_delay_wait(attempt)
                
            except ConnectionClosed as e:
                logger.warning(f"Connection closed for {self.name}: {e}")
                self.state = ConnectionState.RECONNECTING
                
                if self.on_disconnect:
                    await self.on_disconnect(self)
                
                attempt += 1
                await self._reconnect_delay_wait(attempt)
                
            except WebSocketException as e:
                logger.error(f"WebSocket error for {self.name}: {e}")
                attempt += 1
                await self._reconnect_delay_wait(attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error for {self.name}: {e}")
                attempt += 1
                await self._reconnect_delay_wait(attempt)
        
        if attempt >= self.config.reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for {self.name}")
            self.state = ConnectionState.CLOSED
    
    async def _handle_messages(self):
        """Handle incoming messages."""
        try:
            async for message in self.websocket:
                self.stats.messages_received += 1
                self.stats.bytes_received += len(message)
                self.stats.last_message_time = time.time()
                
                # Log latency if enabled
                if self.config.log_latency:
                    start_time = time.time()
                    await self.on_message(self.name, message)
                    latency = (time.time() - start_time) * 1000
                    self.stats.avg_latency_ms = (
                        0.9 * self.stats.avg_latency_ms + 0.1 * latency
                    )
                else:
                    await self.on_message(self.name, message)
                    
        except ConnectionClosed:
            logger.info(f"Connection closed for {self.name}, reconnecting...")
            raise
    
    async def _reconnect_delay_wait(self, attempt: int):
        """Wait with exponential backoff before reconnecting."""
        delay = min(
            self._reconnect_delay * (2 ** attempt),
            self.config.max_reconnect_delay
        )
        logger.info(f"Waiting {delay:.2f}s before reconnecting {self.name}")
        await asyncio.sleep(delay)
    
    async def send(self, message: Any):
        """
        Send message through WebSocket.
        
        Args:
            message: Message to send (will be JSON serialized if dict)
        """
        if self.websocket and self.state == ConnectionState.CONNECTED:
            try:
                if isinstance(message, dict):
                    message = json.dumps(message)
                
                await self.websocket.send(message)
                self.stats.messages_sent += 1
                self.stats.bytes_sent += len(message)
                
            except ConnectionClosed:
                logger.warning(f"Cannot send message, connection closed for {self.name}")
                # Buffer message for retry
                try:
                    self._message_buffer.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning(f"Message buffer full for {self.name}, dropping message")
        else:
            # Buffer message if not connected
            try:
                self._message_buffer.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning(f"Message buffer full for {self.name}, dropping message")
    
    async def close(self):
        """Close WebSocket connection."""
        self._running = False
        self.state = ConnectionState.CLOSED
        
        if self.websocket:
            await self.websocket.close()
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'name': self.name,
            'state': self.state.value,
            'reconnect_count': self.stats.reconnect_count,
            'messages_received': self.stats.messages_received,
            'messages_sent': self.stats.messages_sent,
            'bytes_received': self.stats.bytes_received,
            'bytes_sent': self.stats.bytes_sent,
            'avg_latency_ms': self.stats.avg_latency_ms,
            'uptime_seconds': time.time() - self.stats.connection_time if self.stats.connection_time > 0 else 0,
        }


class WebSocketManager:
    """
    Manage multiple WebSocket connections for real-time data.
    
    Features:
    - Multi-connection management
    - Automatic reconnection with exponential backoff
    - Heartbeat/ping-pong handling
    - Message queuing and buffering
    - Connection pooling
    - Performance metrics
    
    Uses the `websockets` library (12M+ downloads/month) for
    production-grade WebSocket handling.
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """
        Initialize WebSocket manager.
        
        Args:
            config: Streaming configuration
        """
        self.config = config or StreamingConfig()
        self.connections: Dict[str, WebSocketConnection] = {}
        self._tasks: Set[asyncio.Task] = set()
        
        logger.info("WebSocket Manager initialized")
    
    async def add_connection(
        self,
        name: str,
        url: str,
        on_message: Callable,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
    ) -> WebSocketConnection:
        """
        Add and start a new WebSocket connection.
        
        Args:
            name: Connection identifier
            url: WebSocket URL
            on_message: Message handler callback
            on_connect: Optional connect callback
            on_disconnect: Optional disconnect callback
        
        Returns:
            WebSocketConnection instance
        """
        if name in self.connections:
            logger.warning(f"Connection {name} already exists, closing old one")
            await self.remove_connection(name)
        
        connection = WebSocketConnection(
            name=name,
            url=url,
            on_message=on_message,
            config=self.config,
            on_connect=on_connect,
            on_disconnect=on_disconnect,
        )
        
        self.connections[name] = connection
        
        # Start connection in background
        task = asyncio.create_task(connection.connect())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        
        return connection
    
    async def remove_connection(self, name: str):
        """
        Remove and close a connection.
        
        Args:
            name: Connection identifier
        """
        if name in self.connections:
            connection = self.connections[name]
            await connection.close()
            del self.connections[name]
            logger.info(f"Connection {name} removed")
    
    async def send(self, name: str, message: Any):
        """
        Send message through a specific connection.
        
        Args:
            name: Connection identifier
            message: Message to send
        """
        if name in self.connections:
            await self.connections[name].send(message)
        else:
            logger.warning(f"Connection {name} not found")
    
    async def broadcast(self, message: Any):
        """
        Broadcast message to all connections.
        
        Args:
            message: Message to send
        """
        for connection in self.connections.values():
            await connection.send(message)
    
    def get_connection(self, name: str) -> Optional[WebSocketConnection]:
        """
        Get connection by name.
        
        Args:
            name: Connection identifier
        
        Returns:
            WebSocketConnection or None
        """
        return self.connections.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all connections.
        
        Returns:
            Dictionary of connection stats
        """
        return {
            name: conn.get_stats()
            for name, conn in self.connections.items()
        }
    
    async def close_all(self):
        """Close all connections."""
        for connection in list(self.connections.values()):
            await connection.close()
        
        self.connections.clear()
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        logger.info("All connections closed")
    
    def __len__(self) -> int:
        """Get number of active connections."""
        return len(self.connections)