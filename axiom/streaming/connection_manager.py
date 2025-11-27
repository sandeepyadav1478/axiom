"""
WebSocket Connection Manager.

Handles connection lifecycle, heartbeat, and client management.
"""

import asyncio
import logging
from typing import Dict, Set, Optional
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
import json

from .event_types import StreamEvent, HeartbeatEvent

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections with production features:
    - Connection tracking
    - Automatic heartbeat
    - Reconnection support
    - Broadcast messaging
    - Connection health monitoring
    """
    
    def __init__(
        self,
        heartbeat_interval: int = 30,  # seconds
        connection_timeout: int = 90,  # seconds
        max_reconnect_attempts: int = 5
    ):
        """
        Initialize connection manager.
        
        Args:
            heartbeat_interval: Seconds between heartbeat messages
            connection_timeout: Seconds before connection considered dead
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> event_types
        
        self.heartbeat_interval = heartbeat_interval
        self.connection_timeout = connection_timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("ConnectionManager initialized")
    
    async def connect(
        self,
        client_id: str,
        websocket: WebSocket,
        metadata: Optional[Dict] = None
    ):
        """
        Accept and register a new WebSocket connection.
        
        Args:
            client_id: Unique client identifier
            websocket: WebSocket connection
            metadata: Optional client metadata
        """
        await websocket.accept()
        
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": datetime.now(),
            "last_seen": datetime.now(),
            "message_count": 0,
            "subscriptions": set(),
            "metadata": metadata or {}
        }
        
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        
        # Start heartbeat if this is the first connection
        if len(self.active_connections) == 1:
            await self._start_heartbeat()
    
    def disconnect(self, client_id: str):
        """
        Disconnect and clean up a client connection.
        
        Args:
            client_id: Client to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
        if client_id in self.connection_metadata:
            del self.connection_metadata[client_id]
            
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        
        logger.info(f"Client {client_id} disconnected. Remaining connections: {len(self.active_connections)}")
        
        # Stop heartbeat if no connections remain
        if len(self.active_connections) == 0:
            self._stop_heartbeat()
    
    async def send_personal_message(
        self,
        client_id: str,
        message: StreamEvent
    ):
        """
        Send message to a specific client.
        
        Args:
            client_id: Target client
            message: Event to send
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message.to_dict())
                
                # Update metadata
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id]["message_count"] += 1
                    self.connection_metadata[client_id]["last_seen"] = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(
        self,
        message: StreamEvent,
        exclude_clients: Optional[Set[str]] = None
    ):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Event to broadcast
            exclude_clients: Optional set of client IDs to exclude
        """
        exclude_clients = exclude_clients or set()
        
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id in exclude_clients:
                continue
            
            try:
                await websocket.send_json(message.to_dict())
                
                # Update metadata
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id]["message_count"] += 1
                    self.connection_metadata[client_id]["last_seen"] = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_to_subscribers(
        self,
        message: StreamEvent,
        event_type: str
    ):
        """
        Broadcast message only to clients subscribed to the event type.
        
        Args:
            message: Event to broadcast
            event_type: Event type for subscription filtering
        """
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            # Check if client is subscribed to this event type
            if client_id in self.subscriptions:
                if event_type not in self.subscriptions[client_id]:
                    continue
            
            try:
                await websocket.send_json(message.to_dict())
                
                # Update metadata
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id]["message_count"] += 1
                    self.connection_metadata[client_id]["last_seen"] = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def subscribe(self, client_id: str, event_types: Set[str]):
        """
        Subscribe client to specific event types.
        
        Args:
            client_id: Client to subscribe
            event_types: Set of event type strings
        """
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = set()
        
        self.subscriptions[client_id].update(event_types)
        
        # Update metadata
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]["subscriptions"] = self.subscriptions[client_id]
        
        logger.info(f"Client {client_id} subscribed to: {event_types}")
    
    def unsubscribe(self, client_id: str, event_types: Set[str]):
        """
        Unsubscribe client from specific event types.
        
        Args:
            client_id: Client to unsubscribe
            event_types: Set of event type strings
        """
        if client_id in self.subscriptions:
            self.subscriptions[client_id].difference_update(event_types)
            
            # Update metadata
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["subscriptions"] = self.subscriptions[client_id]
            
            logger.info(f"Client {client_id} unsubscribed from: {event_types}")
    
    async def _start_heartbeat(self):
        """Start heartbeat background task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("Heartbeat started")
    
    def _stop_heartbeat(self):
        """Stop heartbeat background task."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            logger.info("Heartbeat stopped")
    
    async def _heartbeat_loop(self):
        """Background loop sending periodic heartbeats."""
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                
                if len(self.active_connections) > 0:
                    heartbeat = HeartbeatEvent.create()
                    await self.broadcast(heartbeat)
                    logger.debug(f"Heartbeat sent to {len(self.active_connections)} clients")
                    
        except asyncio.CancelledError:
            logger.info("Heartbeat loop cancelled")
        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")
    
    def get_connection_stats(self) -> Dict:
        """
        Get statistics about current connections.
        
        Returns:
            Dictionary with connection statistics
        """
        total_messages = sum(
            meta["message_count"]
            for meta in self.connection_metadata.values()
        )
        
        return {
            "active_connections": len(self.active_connections),
            "total_messages_sent": total_messages,
            "clients": {
                client_id: {
                    "connected_at": meta["connected_at"].isoformat(),
                    "last_seen": meta["last_seen"].isoformat(),
                    "message_count": meta["message_count"],
                    "subscriptions": list(meta["subscriptions"]),
                    "uptime_seconds": (datetime.now() - meta["connected_at"]).total_seconds()
                }
                for client_id, meta in self.connection_metadata.items()
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if connection manager is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        # Check for stale connections
        now = datetime.now()
        stale_threshold = timedelta(seconds=self.connection_timeout)
        
        stale_clients = []
        for client_id, meta in self.connection_metadata.items():
            if now - meta["last_seen"] > stale_threshold:
                stale_clients.append(client_id)
        
        # Clean up stale connections
        for client_id in stale_clients:
            logger.warning(f"Removing stale connection: {client_id}")
            self.disconnect(client_id)
        
        return True


__all__ = ["ConnectionManager"]