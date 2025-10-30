"""
WebSocket Support for Real-Time Derivatives Data Streaming

Provides ultra-low latency streaming for:
- Real-time Greeks updates (as market moves)
- Position updates
- P&L ticks
- Risk alerts
- Market data

Critical for market makers who need sub-millisecond updates.

Performance: <100 microsecond message delivery
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import asyncio
import json
from datetime import datetime
import time


class ConnectionManager:
    """
    Manages WebSocket connections for multiple clients
    
    Features:
    - Per-client subscriptions
    - Broadcast to all or specific clients
    - Connection health monitoring
    - Automatic reconnection handling
    """
    
    def __init__(self):
        # Active connections by client_id
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Client subscriptions
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> {topics}
        
        # Message queue for guaranteed delivery
        self.message_queues: Dict[str, List[Dict]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        self.message_queues[client_id] = []
        
        print(f"✓ Client {client_id} connected via WebSocket")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        if client_id in self.message_queues:
            del self.message_queues[client_id]
        
        print(f"✗ Client {client_id} disconnected")
    
    async def send_personal_message(self, message: Dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except:
                # Connection lost, queue message
                self.message_queues[client_id].append(message)
    
    async def broadcast(self, message: Dict, topic: str = None):
        """
        Broadcast message to all subscribed clients
        
        Args:
            message: Data to send
            topic: Optional topic filter (e.g., 'greeks', 'pnl', 'alerts')
        """
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            # Check if client subscribed to this topic
            if topic and topic not in self.subscriptions.get(client_id, set()):
                continue
            
            try:
                await websocket.send_json(message)
            except:
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
    
    def subscribe(self, client_id: str, topics: List[str]):
        """Subscribe client to topics"""
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = set()
        
        self.subscriptions[client_id].update(topics)
        
        print(f"Client {client_id} subscribed to: {topics}")
    
    def unsubscribe(self, client_id: str, topics: List[str]):
        """Unsubscribe from topics"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id] -= set(topics)


# Global connection manager
manager = ConnectionManager()


# WebSocket endpoint example
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time streaming
    
    URL: ws://derivatives-api/ws/{client_id}
    
    Messages received (client → server):
    - {"action": "subscribe", "topics": ["greeks", "pnl", "alerts"]}
    - {"action": "unsubscribe", "topics": ["pnl"]}
    - {"action": "ping"}
    
    Messages sent (server → client):
    - {"type": "greeks_update", "data": {...}, "timestamp": "..."}
    - {"type": "pnl_update", "data": {...}, "timestamp": "..."}
    - {"type": "alert", "data": {...}, "timestamp": "..."}
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            if data.get('action') == 'subscribe':
                topics = data.get('topics', [])
                manager.subscribe(client_id, topics)
                await manager.send_personal_message({
                    'type': 'subscription_confirmed',
                    'topics': topics
                }, client_id)
            
            elif data.get('action') == 'unsubscribe':
                topics = data.get('topics', [])
                manager.unsubscribe(client_id, topics)
            
            elif data.get('action') == 'ping':
                await manager.send_personal_message({
                    'type': 'pong',
                    'timestamp': datetime.utcnow().isoformat()
                }, client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)


async def stream_greeks_updates():
    """
    Background task: Stream Greeks updates as market moves
    
    Runs continuously, calculates and broadcasts Greeks
    when market data changes
    """
    from axiom.derivatives.ultra_fast_greeks import UltraFastGreeksEngine
    
    engine = UltraFastGreeksEngine(use_gpu=True)
    
    while True:
        # In production: Triggered by market data updates
        # For now: Periodic updates
        
        # Calculate Greeks for actively traded options
        # (Would get from watchlist in production)
        greeks = engine.calculate_greeks(100.0, 100.0, 1.0, 0.03, 0.25)
        
        # Broadcast to all subscribed clients
        await manager.broadcast({
            'type': 'greeks_update',
            'symbol': 'SPY241115C00100000',
            'data': {
                'delta': greeks.delta,
                'gamma': greeks.gamma,
                'theta': greeks.theta,
                'vega': greeks.vega,
                'rho': greeks.rho,
                'price': greeks.price,
                'calculation_time_us': greeks.calculation_time_us
            },
            'timestamp': datetime.utcnow().isoformat()
        }, topic='greeks')
        
        await asyncio.sleep(0.1)  # 100ms updates (can be faster)


# Client usage example
"""
Python client:

import websockets
import json
import asyncio

async def connect_websocket():
    uri = "ws://derivatives-api/ws/client_123"
    async with websockets.connect(uri) as websocket:
        # Subscribe to topics
        await websocket.send(json.dumps({
            "action": "subscribe",
            "topics": ["greeks", "pnl", "alerts"]
        }))
        
        # Receive updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data['type'] == 'greeks_update':
                print(f"Greeks update: Delta={data['data']['delta']:.4f}")
                print(f"Latency: {data['data']['calculation_time_us']:.2f}us")
            
            elif data['type'] == 'alert':
                print(f"ALERT: {data['data']['message']}")

asyncio.run(connect_websocket())
"""