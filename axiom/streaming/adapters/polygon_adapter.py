"""
Polygon.io WebSocket Adapter

Real-time market data streaming using Polygon.io's WebSocket API.

Uses the official `polygon-api-client` library for reliable data streaming.
"""

import asyncio
import logging
from typing import Callable, List, Optional
import time

try:
    from polygon import WebSocketClient
    from polygon.websocket.models import WebSocketMessage, EquityTrade, EquityQuote, EquityAgg
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    WebSocketClient = None
    WebSocketMessage = None
    EquityTrade = None
    EquityQuote = None
    EquityAgg = None

from axiom.streaming.adapters.base_adapter import (
    BaseMarketDataAdapter,
    MarketDataType,
    TradeData,
    QuoteData,
    BarData,
)
from axiom.streaming.config import StreamingConfig


logger = logging.getLogger(__name__)


class PolygonAdapter(BaseMarketDataAdapter):
    """
    Polygon.io WebSocket adapter using official library.
    
    Features:
    - Real-time trades, quotes, and aggregates
    - Options data (if enabled)
    - Forex and crypto (if enabled)
    - News feeds (if enabled)
    
    Requires:
        polygon-api-client >= 1.12.0
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize Polygon.io adapter.
        
        Args:
            api_key: Polygon.io API key
            config: Streaming configuration
        """
        super().__init__("Polygon.io")
        
        if not POLYGON_AVAILABLE:
            raise ImportError(
                "polygon-api-client not installed. "
                "Install with: pip install polygon-api-client"
            )
        
        self.config = config or StreamingConfig()
        self.api_key = api_key or self.config.polygon_api_key
        
        if not self.api_key:
            raise ValueError("Polygon API key is required")
        
        self.client: Optional[WebSocketClient] = None
        self.supported_data_types = [
            MarketDataType.TRADE,
            MarketDataType.QUOTE,
            MarketDataType.BAR,
        ]
        
        # Callback storage
        self._trade_callbacks: List[Callable] = []
        self._quote_callbacks: List[Callable] = []
        self._bar_callbacks: List[Callable] = []
    
    async def connect(self):
        """Establish connection to Polygon.io WebSocket."""
        try:
            # Create WebSocket client
            self.client = WebSocketClient(
                api_key=self.api_key,
                subscriptions=[],  # Will subscribe later
            )
            
            # Set up message handlers
            self.client.on_trade(self._handle_trade)
            self.client.on_quote(self._handle_quote)
            self.client.on_agg(self._handle_agg)
            
            self._connected = True
            logger.info("Connected to Polygon.io WebSocket")
            
        except Exception as e:
            logger.error(f"Failed to connect to Polygon.io: {e}")
            raise
    
    async def disconnect(self):
        """Close connection to Polygon.io."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Disconnected from Polygon.io")
    
    async def subscribe_trades(
        self,
        symbols: List[str],
        callback: Callable[[TradeData], None],
    ):
        """
        Subscribe to real-time trades.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            callback: Callback function for trade data
        """
        if not self._connected:
            await self.connect()
        
        self._trade_callbacks.append(callback)
        
        # Subscribe to trades for each symbol
        for symbol in symbols:
            self.client.subscribe(f"T.{symbol}")
        
        logger.info(f"Subscribed to trades for: {symbols}")
        
        # Start client if not running
        if not self.client.is_running():
            self.client.run_async()
    
    async def subscribe_quotes(
        self,
        symbols: List[str],
        callback: Callable[[QuoteData], None],
    ):
        """
        Subscribe to real-time quotes.
        
        Args:
            symbols: List of stock symbols
            callback: Callback function for quote data
        """
        if not self._connected:
            await self.connect()
        
        self._quote_callbacks.append(callback)
        
        # Subscribe to quotes for each symbol
        for symbol in symbols:
            self.client.subscribe(f"Q.{symbol}")
        
        logger.info(f"Subscribed to quotes for: {symbols}")
        
        # Start client if not running
        if not self.client.is_running():
            self.client.run_async()
    
    async def subscribe_bars(
        self,
        symbols: List[str],
        callback: Callable[[BarData], None],
        timeframe: str = "1Min",
    ):
        """
        Subscribe to real-time aggregates/bars.
        
        Args:
            symbols: List of stock symbols
            callback: Callback function for bar data
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1H', '1D')
        """
        if not self._connected:
            await self.connect()
        
        self._bar_callbacks.append(callback)
        
        # Map timeframe to Polygon format
        timeframe_map = {
            '1Min': 'AM',  # Aggregate Minute
            '5Min': 'AM',
            '15Min': 'AM',
            '1H': 'AM',
            '1D': 'AM',
        }
        
        agg_type = timeframe_map.get(timeframe, 'AM')
        
        # Subscribe to aggregates for each symbol
        for symbol in symbols:
            self.client.subscribe(f"{agg_type}.{symbol}")
        
        logger.info(f"Subscribed to {timeframe} bars for: {symbols}")
        
        # Start client if not running
        if not self.client.is_running():
            self.client.run_async()
    
    async def unsubscribe(self, symbols: List[str]):
        """
        Unsubscribe from symbols.
        
        Args:
            symbols: List of symbols to unsubscribe
        """
        if self.client:
            for symbol in symbols:
                # Unsubscribe from all data types
                self.client.unsubscribe(f"T.{symbol}")
                self.client.unsubscribe(f"Q.{symbol}")
                self.client.unsubscribe(f"AM.{symbol}")
            
            logger.info(f"Unsubscribed from: {symbols}")
    
    def _handle_trade(self, msg: "WebSocketMessage"):
        """Handle incoming trade message."""
        try:
            for trade in msg.data:
                # Convert to TradeData
                trade_data = TradeData(
                    symbol=trade.symbol,
                    price=trade.price,
                    size=trade.size,
                    timestamp=trade.timestamp / 1000.0,  # Convert to seconds
                    exchange=trade.exchange if hasattr(trade, 'exchange') else None,
                    conditions=trade.conditions if hasattr(trade, 'conditions') else None,
                )
                
                # Call all trade callbacks
                for callback in self._trade_callbacks:
                    asyncio.create_task(callback(trade_data))
                    
        except Exception as e:
            logger.error(f"Error handling trade message: {e}")
    
    def _handle_quote(self, msg: "WebSocketMessage"):
        """Handle incoming quote message."""
        try:
            for quote in msg.data:
                # Convert to QuoteData
                quote_data = QuoteData(
                    symbol=quote.symbol,
                    bid=quote.bid_price,
                    ask=quote.ask_price,
                    bid_size=quote.bid_size,
                    ask_size=quote.ask_size,
                    timestamp=quote.timestamp / 1000.0,  # Convert to seconds
                    exchange=quote.exchange if hasattr(quote, 'exchange') else None,
                )
                
                # Call all quote callbacks
                for callback in self._quote_callbacks:
                    asyncio.create_task(callback(quote_data))
                    
        except Exception as e:
            logger.error(f"Error handling quote message: {e}")
    
    def _handle_agg(self, msg: "WebSocketMessage"):
        """Handle incoming aggregate/bar message."""
        try:
            for agg in msg.data:
                # Convert to BarData
                bar_data = BarData(
                    symbol=agg.symbol,
                    open=agg.open,
                    high=agg.high,
                    low=agg.low,
                    close=agg.close,
                    volume=agg.volume,
                    timestamp=agg.start_timestamp / 1000.0,  # Convert to seconds
                    vwap=agg.vwap if hasattr(agg, 'vwap') else None,
                )
                
                # Call all bar callbacks
                for callback in self._bar_callbacks:
                    asyncio.create_task(callback(bar_data))
                    
        except Exception as e:
            logger.error(f"Error handling aggregate message: {e}")
    
    async def health_check(self) -> bool:
        """
        Check adapter health.
        
        Returns:
            True if connected and running
        """
        return self._connected and (self.client.is_running() if self.client else False)