"""
Real-Time Portfolio Tracker

Live portfolio tracking with real-time P&L calculation, position monitoring,
and alert triggers.

Performance Target: <5ms for portfolio update processing
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from axiom.streaming.config import StreamingConfig
from axiom.streaming.redis_cache import RealTimeCache
from axiom.streaming.market_data import MarketDataStreamer
from axiom.streaming.adapters.base_adapter import TradeData


logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """
    Portfolio position.
    
    Attributes:
        symbol: Instrument symbol
        quantity: Position size (positive for long, negative for short)
        avg_cost: Average cost basis
        current_price: Current market price
        last_update: Last update timestamp
    """
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    @property
    def side(self) -> PositionSide:
        """Get position side."""
        return PositionSide.LONG if self.quantity > 0 else PositionSide.SHORT
    
    @property
    def abs_quantity(self) -> float:
        """Get absolute quantity."""
        return abs(self.quantity)
    
    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Calculate cost basis."""
        return self.quantity * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return (self.current_price - self.avg_cost) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.avg_cost == 0:
            return 0.0
        return ((self.current_price - self.avg_cost) / self.avg_cost) * 100
    
    def update_price(self, price: float):
        """
        Update current price.
        
        Args:
            price: New price
        """
        self.current_price = price
        self.last_update = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'side': self.side.value,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'last_update': self.last_update,
        }


@dataclass
class Alert:
    """Portfolio alert configuration."""
    alert_type: str
    threshold: float
    callback: Callable
    triggered: bool = False


@dataclass
class PortfolioStats:
    """Portfolio statistics."""
    total_positions: int = 0
    total_value: float = 0.0
    total_cost: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    updates_received: int = 0
    last_update: float = field(default_factory=time.time)


class PortfolioTracker:
    """
    Real-time portfolio tracking with live P&L.
    
    Features:
    - Live position tracking
    - Real-time P&L calculation
    - Alert triggers (stop-loss, profit targets)
    - Performance attribution
    - Portfolio metrics
    - Redis integration for persistence
    
    Performance Target: <5ms for portfolio update processing
    
    Example:
        ```python
        tracker = PortfolioTracker(cache, data_stream)
        
        positions = [
            Position("AAPL", quantity=100, avg_cost=150.0),
            Position("GOOGL", quantity=50, avg_cost=140.0),
        ]
        
        await tracker.track_portfolio(positions)
        
        # Get real-time summary
        summary = tracker.get_portfolio_summary()
        print(f"Total P&L: ${summary['total_pnl']:,.2f}")
        ```
    """
    
    def __init__(
        self,
        cache: RealTimeCache,
        data_stream: MarketDataStreamer,
        config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize portfolio tracker.
        
        Args:
            cache: Redis cache for price data
            data_stream: Market data streamer
            config: Streaming configuration
        """
        self.cache = cache
        self.data_stream = data_stream
        self.config = config or StreamingConfig()
        
        self.positions: Dict[str, Position] = {}
        self.alerts: Dict[str, List[Alert]] = {}
        self.stats = PortfolioStats()
        
        self._tracking = False
        self._update_callbacks: List[Callable] = []
        
        logger.info("Portfolio tracker initialized")
    
    async def track_portfolio(self, positions: List[Position]):
        """
        Start real-time tracking for portfolio.
        
        Args:
            positions: List of positions to track
        """
        # Store positions
        self.positions = {p.symbol: p for p in positions}
        self.stats.total_positions = len(self.positions)
        
        # Get symbols
        symbols = list(self.positions.keys())
        
        # Subscribe to price updates
        await self.data_stream.subscribe_trades(symbols, self._on_price_update)
        
        self._tracking = True
        logger.info(f"Tracking portfolio with {len(symbols)} positions: {symbols}")
    
    async def add_position(self, position: Position):
        """
        Add position to portfolio.
        
        Args:
            position: Position to add
        """
        if position.symbol in self.positions:
            # Update existing position
            existing = self.positions[position.symbol]
            
            # Calculate new average cost
            total_qty = existing.quantity + position.quantity
            if total_qty != 0:
                new_avg_cost = (
                    (existing.quantity * existing.avg_cost + 
                     position.quantity * position.avg_cost) / total_qty
                )
                existing.quantity = total_qty
                existing.avg_cost = new_avg_cost
            else:
                # Position closed
                del self.positions[position.symbol]
                return
        else:
            # Add new position
            self.positions[position.symbol] = position
            
            # Subscribe to price updates
            await self.data_stream.subscribe_trades(
                [position.symbol],
                self._on_price_update
            )
        
        self.stats.total_positions = len(self.positions)
        logger.info(f"Added/updated position: {position.symbol}")
    
    async def remove_position(self, symbol: str):
        """
        Remove position from portfolio.
        
        Args:
            symbol: Symbol to remove
        """
        if symbol in self.positions:
            del self.positions[symbol]
            self.stats.total_positions = len(self.positions)
            
            # Unsubscribe from updates
            await self.data_stream.unsubscribe([symbol])
            
            logger.info(f"Removed position: {symbol}")
    
    async def _on_price_update(self, trade: TradeData):
        """
        Handle real-time price update.
        
        Args:
            trade: Trade data
        """
        start_time = time.time()
        
        symbol = trade.symbol
        price = trade.price
        
        if symbol in self.positions:
            # Update position price
            self.positions[symbol].update_price(price)
            
            # Store price in cache
            await self.cache.set_price(symbol, price, trade.timestamp)
            
            # Update statistics
            self.stats.updates_received += 1
            self.stats.last_update = time.time()
            
            # Check alerts
            await self._check_alerts(symbol)
            
            # Call update callbacks
            for callback in self._update_callbacks:
                try:
                    await callback(symbol, self.positions[symbol])
                except Exception as e:
                    logger.error(f"Error in update callback: {e}")
            
            # Log latency
            if self.config.log_latency:
                latency = (time.time() - start_time) * 1000
                if latency > 5:  # Warn if >5ms
                    logger.warning(f"Portfolio update took {latency:.2f}ms")
    
    def add_alert(
        self,
        symbol: str,
        alert_type: str,
        threshold: float,
        callback: Callable,
    ):
        """
        Add alert for position.
        
        Args:
            symbol: Symbol to monitor
            alert_type: Alert type ('stop_loss', 'take_profit', 'price_above', 'price_below')
            threshold: Threshold value
            callback: Alert callback function
        """
        if symbol not in self.alerts:
            self.alerts[symbol] = []
        
        alert = Alert(
            alert_type=alert_type,
            threshold=threshold,
            callback=callback,
        )
        
        self.alerts[symbol].append(alert)
        logger.info(f"Added {alert_type} alert for {symbol} at {threshold}")
    
    async def _check_alerts(self, symbol: str):
        """
        Check and trigger alerts for symbol.
        
        Args:
            symbol: Symbol to check
        """
        if symbol not in self.alerts or symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        for alert in self.alerts[symbol]:
            if alert.triggered:
                continue
            
            should_trigger = False
            
            if alert.alert_type == 'stop_loss':
                # Trigger if P&L drops below threshold
                if position.unrealized_pnl < alert.threshold:
                    should_trigger = True
                    
            elif alert.alert_type == 'take_profit':
                # Trigger if P&L exceeds threshold
                if position.unrealized_pnl > alert.threshold:
                    should_trigger = True
                    
            elif alert.alert_type == 'price_above':
                # Trigger if price goes above threshold
                if position.current_price > alert.threshold:
                    should_trigger = True
                    
            elif alert.alert_type == 'price_below':
                # Trigger if price goes below threshold
                if position.current_price < alert.threshold:
                    should_trigger = True
            
            if should_trigger:
                alert.triggered = True
                try:
                    await alert.callback(symbol, position, alert)
                    logger.info(f"Alert triggered: {alert.alert_type} for {symbol}")
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def register_update_callback(self, callback: Callable):
        """
        Register callback for position updates.
        
        Args:
            callback: Async callback(symbol, position)
        """
        self._update_callbacks.append(callback)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary.
        
        Returns:
            Portfolio summary dictionary
        """
        total_value = sum(p.market_value for p in self.positions.values())
        total_cost = sum(p.cost_basis for p in self.positions.values())
        total_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0.0
        
        return {
            'timestamp': time.time(),
            'total_positions': len(self.positions),
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'total_pnl_pct': pnl_pct,
            'positions': {
                symbol: position.to_dict()
                for symbol, position in self.positions.items()
            }
        }
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol.
        
        Args:
            symbol: Symbol to get
        
        Returns:
            Position or None
        """
        return self.positions.get(symbol)
    
    def get_top_performers(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top performing positions.
        
        Args:
            n: Number of positions to return
        
        Returns:
            List of position data sorted by P&L
        """
        sorted_positions = sorted(
            self.positions.values(),
            key=lambda p: p.unrealized_pnl,
            reverse=True
        )
        
        return [p.to_dict() for p in sorted_positions[:n]]
    
    def get_worst_performers(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get worst performing positions.
        
        Args:
            n: Number of positions to return
        
        Returns:
            List of position data sorted by P&L (ascending)
        """
        sorted_positions = sorted(
            self.positions.values(),
            key=lambda p: p.unrealized_pnl
        )
        
        return [p.to_dict() for p in sorted_positions[:n]]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracker statistics.
        
        Returns:
            Statistics dictionary
        """
        summary = self.get_portfolio_summary()
        
        return {
            'tracking': self._tracking,
            'total_positions': self.stats.total_positions,
            'updates_received': self.stats.updates_received,
            'alerts_configured': sum(len(alerts) for alerts in self.alerts.values()),
            'total_value': summary['total_value'],
            'total_pnl': summary['total_pnl'],
            'total_pnl_pct': summary['total_pnl_pct'],
        }
    
    async def stop_tracking(self):
        """Stop portfolio tracking."""
        self._tracking = False
        
        # Unsubscribe from all symbols
        symbols = list(self.positions.keys())
        if symbols:
            await self.data_stream.unsubscribe(symbols)
        
        logger.info("Portfolio tracking stopped")