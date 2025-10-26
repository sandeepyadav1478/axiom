"""
WebSocket endpoints for real-time streaming.

Features:
- Real-time portfolio updates
- Live market data streaming  
- Risk alert notifications
- Model calculation streaming
"""

import asyncio
import json
from typing import Dict, List, Set
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.websockets import WebSocketState

from axiom.api.auth import get_optional_user, User


router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections for different streams."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "portfolio": set(),
            "market_data": set(),
            "risk_alerts": set(),
            "analytics": set(),
        }
        self.user_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, stream_type: str, user: str = None):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections[stream_type].add(websocket)
        
        if user:
            if user not in self.user_connections:
                self.user_connections[user] = set()
            self.user_connections[user].add(websocket)
        
        print(f"✅ WebSocket connected: {stream_type} (user: {user})")
    
    def disconnect(self, websocket: WebSocket, stream_type: str, user: str = None):
        """Disconnect a WebSocket client."""
        if websocket in self.active_connections[stream_type]:
            self.active_connections[stream_type].remove(websocket)
        
        if user and user in self.user_connections:
            if websocket in self.user_connections[user]:
                self.user_connections[user].remove(websocket)
        
        print(f"❌ WebSocket disconnected: {stream_type}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message: {e}")
    
    async def broadcast(self, message: dict, stream_type: str):
        """Broadcast message to all clients on a stream."""
        disconnected = []
        
        for connection in self.active_connections[stream_type]:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_json(message)
                else:
                    disconnected.append(connection)
            except Exception as e:
                print(f"Error broadcasting: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.active_connections[stream_type].discard(connection)
    
    async def send_to_user(self, message: dict, username: str):
        """Send message to all connections of a specific user."""
        if username in self.user_connections:
            for connection in self.user_connections[username]:
                await self.send_personal_message(message, connection)


manager = ConnectionManager()


@router.websocket("/ws/portfolio/{portfolio_id}")
async def portfolio_stream(
    websocket: WebSocket,
    portfolio_id: str,
):
    """
    Stream real-time portfolio updates.
    
    **Updates Include**:
    - Position changes
    - P&L updates
    - Risk metrics
    - Performance attribution
    """
    await manager.connect(websocket, "portfolio", portfolio_id)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "portfolio_id": portfolio_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Main loop
        while True:
            # Listen for client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                
                # Handle client commands
                if message.get("command") == "subscribe":
                    await websocket.send_json({
                        "type": "subscription",
                        "status": "subscribed",
                        "topics": message.get("topics", []),
                    })
                
            except asyncio.TimeoutError:
                # Send periodic updates
                update = {
                    "type": "portfolio_update",
                    "portfolio_id": portfolio_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "total_value": 1_000_000.0,
                        "daily_pnl": 5_432.10,
                        "total_return": 0.0854,
                        "positions": [
                            {
                                "symbol": "AAPL",
                                "quantity": 100,
                                "price": 175.50,
                                "value": 17_550.0,
                                "pnl": 234.50,
                            }
                        ],
                        "risk_metrics": {
                            "volatility": 0.18,
                            "var_95": -12_500.0,
                            "beta": 1.05,
                        }
                    }
                }
                await websocket.send_json(update)
                
                # Wait before next update
                await asyncio.sleep(5)
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "portfolio", portfolio_id)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, "portfolio", portfolio_id)


@router.websocket("/ws/market-data/{symbol}")
async def market_data_stream(
    websocket: WebSocket,
    symbol: str,
):
    """
    Stream real-time market data for a symbol.
    
    **Data Includes**:
    - Price updates
    - Bid/Ask spreads
    - Volume
    - Trades
    """
    await manager.connect(websocket, "market_data", symbol)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Stream market data
        while True:
            try:
                # Check for client messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                message = json.loads(data)
                
                if message.get("command") == "ping":
                    await websocket.send_json({"type": "pong"})
            
            except asyncio.TimeoutError:
                # Send market data update
                import random
                base_price = 100.0
                price = base_price + random.uniform(-1, 1)
                
                update = {
                    "type": "quote",
                    "symbol": symbol,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "price": round(price, 2),
                        "bid": round(price - 0.05, 2),
                        "ask": round(price + 0.05, 2),
                        "volume": random.randint(100, 10000),
                        "change": round(random.uniform(-2, 2), 2),
                        "change_percent": round(random.uniform(-2, 2), 2),
                    }
                }
                await websocket.send_json(update)
                
                await asyncio.sleep(1)  # 1Hz updates
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "market_data", symbol)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, "market_data", symbol)


@router.websocket("/ws/risk-alerts")
async def risk_alerts_stream(websocket: WebSocket):
    """
    Stream real-time risk alerts and notifications.
    
    **Alert Types**:
    - VaR breach
    - Limit violations
    - Concentration risk
    - Market events
    """
    await manager.connect(websocket, "risk_alerts")
    
    try:
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Monitor for risk events
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
                # Simulate occasional risk alert
                import random
                if random.random() < 0.1:  # 10% chance
                    alert = {
                        "type": "risk_alert",
                        "severity": random.choice(["low", "medium", "high"]),
                        "timestamp": datetime.utcnow().isoformat(),
                        "alert": {
                            "category": "var_breach",
                            "message": "Portfolio VaR exceeded 95% confidence threshold",
                            "current_var": -125_000,
                            "limit": -100_000,
                            "recommended_action": "Consider reducing position sizes",
                        }
                    }
                    await websocket.send_json(alert)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "risk_alerts")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, "risk_alerts")


@router.websocket("/ws/analytics")
async def analytics_stream(websocket: WebSocket):
    """
    Stream real-time analytics and calculations.
    
    **Updates Include**:
    - Model calculations
    - Optimization results
    - Performance metrics
    """
    await manager.connect(websocket, "analytics")
    
    try:
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                
                # Handle calculation requests
                if message.get("command") == "calculate":
                    calc_type = message.get("type")
                    params = message.get("params", {})
                    
                    # Simulate calculation
                    await asyncio.sleep(0.5)
                    
                    result = {
                        "type": "calculation_result",
                        "calc_type": calc_type,
                        "timestamp": datetime.utcnow().isoformat(),
                        "result": {
                            "value": 12345.67,
                            "status": "completed",
                        }
                    }
                    await websocket.send_json(result)
            
            except asyncio.TimeoutError:
                continue
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "analytics")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, "analytics")


# Broadcast utility function for background tasks
async def broadcast_market_update(symbol: str, data: dict):
    """Broadcast market update to all subscribers."""
    message = {
        "type": "market_update",
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }
    await manager.broadcast(message, "market_data")


async def broadcast_risk_alert(alert: dict):
    """Broadcast risk alert to all subscribers."""
    message = {
        "type": "risk_alert",
        "timestamp": datetime.utcnow().isoformat(),
        **alert,
    }
    await manager.broadcast(message, "risk_alerts")