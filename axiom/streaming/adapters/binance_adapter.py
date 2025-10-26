"""
Binance WebSocket Adapter

Real-time cryptocurrency market data streaming using Binance's WebSocket API.

Uses the official `python-binance` library for reliable crypto data streaming.
"""

import asyncio
import logging
from typing import Callable, List, Optional
import time
import json

try:
    from binance import AsyncClient, BinanceSocketManager
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    AsyncClient = None
    BinanceSocketManager = None

from axiom.streaming.adapters.base_adapter import (
    BaseMarketDataAdapter,
    MarketDataType,
    TradeData,
    QuoteData,
    BarData,
)
from axiom.streaming.config import StreamingConfig


logger = logging.getLogger(__name__)


class BinanceAdapter(BaseMarketDataAdapter):
    """
    Binance WebSocket adapter using official library.
    
    Features:
    - Real-time crypto trades
    - Order book depth
    - Kline/candlestick data
    - Aggregate trades
    - 24hr ticker statistics
    
    Requires:
        python-binance >= 1.0.19
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        config: Optional[StreamingConfig] = None,
        testnet: bool = False,
    ):
        """
        Initialize Binance adapter.
        
        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret
            config: Streaming configuration
            testnet: Use testnet instead of mainnet
        """
        super().__init__("Binance")
        
        if not BINANCE_AVAILABLE:
            raise ImportError(
                "python-binance not installed. "
                "Install with: pip install python-binance"
            )
        
        self.config = config or StreamingConfig()
        self.api_key = api_key or self.config.binance_api_key
        self.api_secret = api_secret or self.config.binance_secret
        self.testnet = testnet
        
        self.client: Optional[AsyncClient] = None
        self.socket_manager: Optional[BinanceSocketManager] = None
        self.supported_data_types = [
            MarketDataType.TRADE,
            MarketDataType.QUOTE,
            MarketDataType.BAR,
            MarketDataType.ORDER_BOOK,
        ]
        
        # Callback storage
        self._trade_callbacks: List[Callable] = []
        self._quote_callbacks: List[Callable] = []
        self._bar_callbacks: List[Callable] = []
        
        # Active socket connections
        self._sockets = []
    
    async def connect(self):
        """Establish connection to Binance WebSocket."""
        try:
            # Create async client
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
            )
            
            # Create socket manager
            self.socket_manager = BinanceSocketManager(self.client)
            
            self._connected = True
            logger.info("Connected to Binance WebSocket")
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise
    
    async def disconnect(self):
        """Close connection to Binance."""
        # Close all sockets
        for socket in self._sockets:
            try:
                await socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
        
        self._sockets.clear()
        
        # Close client
        if self.client:
            await self.client.close_connection()
            self._connected = False
            logger.info("Disconnected from Binance")
    
    async def subscribe_trades(
        self,
        symbols: List[str],
        callback: Callable[[TradeData], None],
    ):
        """
        Subscribe to real-time aggregate trades.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            callback: Callback function for trade data
        """
        if not self._connected:
            await self.connect()
        
        self._trade_callbacks.append(callback)
        
        # Subscribe to aggregate trades for each symbol
        for symbol in symbols:
            socket = self.socket_manager.aggtrade_socket(symbol.lower())
            asyncio.create_task(self._handle_trade_socket(socket, symbol))
        
        logger.info(f"Subscribed to trades for: {symbols}")
    
    async def subscribe_quotes(
        self,
        symbols: List[str],
        callback: Callable[[QuoteData], None],
    ):
        """
        Subscribe to real-time best bid/ask (book ticker).
        
        Args:
            symbols: List of trading pairs
            callback: Callback function for quote data
        """
        if not self._connected:
            await self.connect()
        
        self._quote_callbacks.append(callback)
        
        # Subscribe to book ticker for each symbol
        for symbol in symbols:
            socket = self.socket_manager.symbol_book_ticker_socket(symbol.lower())
            asyncio.create_task(self._handle_quote_socket(socket, symbol))
        
        logger.info(f"Subscribed to quotes for: {symbols}")
    
    async def subscribe_bars(
        self,
        symbols: List[str],
        callback: Callable[[BarData], None],
        timeframe: str = "1m",
    ):
        """
        Subscribe to real-time kline/candlestick data.
        
        Args:
            symbols: List of trading pairs
            callback: Callback function for bar data
            timeframe: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d')
        """
        if not self._connected:
            await self.connect()
        
        self._bar_callbacks.append(callback)
        
        # Map standard timeframe to Binance format
        timeframe_map = {
            '1Min': '1m',
            '5Min': '5m',
            '15Min': '15m',
            '1H': '1h',
            '4H': '4h',
            '1D': '1d',
        }
        
        binance_timeframe = timeframe_map.get(timeframe, timeframe.lower())
        
        # Subscribe to klines for each symbol
        for symbol in symbols:
            socket = self.socket_manager.kline_socket(symbol.lower(), interval=binance_timeframe)
            asyncio.create_task(self._handle_bar_socket(socket, symbol))
        
        logger.info(f"Subscribed to {timeframe} bars for: {symbols}")
    
    async def subscribe_order_book(
        self,
        symbols: List[str],
        callback: Callable,
        depth: int = 10,
    ):
        """
        Subscribe to real-time order book depth.
        
        Args:
            symbols: List of trading pairs
            callback: Callback function for order book data
            depth: Order book depth (5, 10, or 20)
        """
        if not self._connected:
            await self.connect()
        
        # Valid depths: 5, 10, 20
        valid_depths = [5, 10, 20]
        if depth not in valid_depths:
            depth = 10
        
        # Subscribe to partial book depth for each symbol
        for symbol in symbols:
            socket = self.socket_manager.depth_socket(symbol.lower(), depth=depth)
            asyncio.create_task(self._handle_orderbook_socket(socket, symbol, callback))
        
        logger.info(f"Subscribed to order book (depth={depth}) for: {symbols}")
    
    async def _handle_trade_socket(self, socket, symbol: str):
        """Handle trade socket messages."""
        self._sockets.append(socket)
        
        async with socket as stream:
            while True:
                try:
                    msg = await stream.recv()
                    
                    # Convert to TradeData
                    trade_data = TradeData(
                        symbol=symbol,
                        price=float(msg['p']),
                        size=int(float(msg['q'])),
                        timestamp=msg['T'] / 1000.0,  # Convert to seconds
                        exchange='Binance',
                    )
                    
                    # Call all trade callbacks
                    for callback in self._trade_callbacks:
                        await callback(trade_data)
                        
                except Exception as e:
                    logger.error(f"Error in trade socket for {symbol}: {e}")
                    break
    
    async def _handle_quote_socket(self, socket, symbol: str):
        """Handle quote socket messages."""
        self._sockets.append(socket)
        
        async with socket as stream:
            while True:
                try:
                    msg = await stream.recv()
                    
                    # Convert to QuoteData
                    quote_data = QuoteData(
                        symbol=symbol,
                        bid=float(msg['b']),
                        ask=float(msg['a']),
                        bid_size=int(float(msg['B'])),
                        ask_size=int(float(msg['A'])),
                        timestamp=time.time(),
                        exchange='Binance',
                    )
                    
                    # Call all quote callbacks
                    for callback in self._quote_callbacks:
                        await callback(quote_data)
                        
                except Exception as e:
                    logger.error(f"Error in quote socket for {symbol}: {e}")
                    break
    
    async def _handle_bar_socket(self, socket, symbol: str):
        """Handle kline/bar socket messages."""
        self._sockets.append(socket)
        
        async with socket as stream:
            while True:
                try:
                    msg = await stream.recv()
                    kline = msg['k']
                    
                    # Only process closed candles
                    if kline['x']:  # Candle closed
                        # Convert to BarData
                        bar_data = BarData(
                            symbol=symbol,
                            open=float(kline['o']),
                            high=float(kline['h']),
                            low=float(kline['l']),
                            close=float(kline['c']),
                            volume=int(float(kline['v'])),
                            timestamp=kline['t'] / 1000.0,  # Convert to seconds
                        )
                        
                        # Call all bar callbacks
                        for callback in self._bar_callbacks:
                            await callback(bar_data)
                        
                except Exception as e:
                    logger.error(f"Error in bar socket for {symbol}: {e}")
                    break
    
    async def _handle_orderbook_socket(self, socket, symbol: str, callback: Callable):
        """Handle order book socket messages."""
        self._sockets.append(socket)
        
        async with socket as stream:
            while True:
                try:
                    msg = await stream.recv()
                    
                    # Extract bids and asks
                    bids = [(float(p), float(q)) for p, q in msg['bids']]
                    asks = [(float(p), float(q)) for p, q in msg['asks']]
                    
                    # Call callback with order book data
                    await callback({
                        'symbol': symbol,
                        'bids': bids,
                        'asks': asks,
                        'timestamp': time.time(),
                    })
                    
                except Exception as e:
                    logger.error(f"Error in order book socket for {symbol}: {e}")
                    break
    
    async def health_check(self) -> bool:
        """
        Check adapter health.
        
        Returns:
            True if connected
        """
        if not self._connected:
            return False
        
        try:
            # Ping the server
            if self.client:
                await self.client.ping()
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
        
        return False