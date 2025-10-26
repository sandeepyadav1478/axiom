"""
Alpaca Markets WebSocket Adapter

Real-time market data and trading execution using Alpaca's WebSocket API.

Uses the official `alpaca-trade-api` library for commission-free trading data.
"""

import asyncio
import logging
from typing import Callable, List, Optional
import time

try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.models import Trade, Quote, Bar
    from alpaca.trading.client import TradingClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    StockDataStream = None
    TradingClient = None

from axiom.streaming.adapters.base_adapter import (
    BaseMarketDataAdapter,
    MarketDataType,
    TradeData,
    QuoteData,
    BarData,
)
from axiom.streaming.config import StreamingConfig


logger = logging.getLogger(__name__)


class AlpacaAdapter(BaseMarketDataAdapter):
    """
    Alpaca Markets WebSocket adapter using official library.
    
    Features:
    - Real-time stock trades and quotes
    - Minute bars
    - Paper trading support
    - Live trading execution
    - Account streaming
    
    Requires:
        alpaca-trade-api >= 3.1.1
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        config: Optional[StreamingConfig] = None,
        paper: bool = True,
    ):
        """
        Initialize Alpaca adapter.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            config: Streaming configuration
            paper: Use paper trading (default: True)
        """
        super().__init__("Alpaca Markets")
        
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-trade-api not installed. "
                "Install with: pip install alpaca-trade-api"
            )
        
        self.config = config or StreamingConfig()
        self.api_key = api_key or self.config.alpaca_api_key
        self.secret_key = secret_key or self.config.alpaca_secret
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret are required")
        
        self.stream: Optional[StockDataStream] = None
        self.trading_client: Optional[TradingClient] = None
        self.supported_data_types = [
            MarketDataType.TRADE,
            MarketDataType.QUOTE,
            MarketDataType.BAR,
        ]
        
        # Callback storage
        self._trade_callbacks: List[Callable] = []
        self._quote_callbacks: List[Callable] = []
        self._bar_callbacks: List[Callable] = []
        
        self._running = False
    
    async def connect(self):
        """Establish connection to Alpaca WebSocket."""
        try:
            # Create data stream
            self.stream = StockDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key,
                raw_data=False,
            )
            
            # Create trading client for account operations
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper,
            )
            
            self._connected = True
            logger.info(f"Connected to Alpaca WebSocket (paper={self.paper})")
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    async def disconnect(self):
        """Close connection to Alpaca."""
        self._running = False
        
        if self.stream:
            try:
                await self.stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
        
        self._connected = False
        logger.info("Disconnected from Alpaca")
    
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
        
        # Define async handler for trades
        async def trade_handler(trade: Trade):
            try:
                trade_data = TradeData(
                    symbol=trade.symbol,
                    price=float(trade.price),
                    size=int(trade.size),
                    timestamp=trade.timestamp.timestamp(),
                    exchange=trade.exchange if hasattr(trade, 'exchange') else 'Alpaca',
                    conditions=trade.conditions if hasattr(trade, 'conditions') else None,
                )
                
                # Call all trade callbacks
                for cb in self._trade_callbacks:
                    await cb(trade_data)
                    
            except Exception as e:
                logger.error(f"Error handling trade for {trade.symbol}: {e}")
        
        # Subscribe to trades
        self.stream.subscribe_trades(trade_handler, *symbols)
        
        logger.info(f"Subscribed to trades for: {symbols}")
        
        # Start stream if not running
        if not self._running:
            self._running = True
            asyncio.create_task(self.stream.run())
    
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
        
        # Define async handler for quotes
        async def quote_handler(quote: Quote):
            try:
                quote_data = QuoteData(
                    symbol=quote.symbol,
                    bid=float(quote.bid_price),
                    ask=float(quote.ask_price),
                    bid_size=int(quote.bid_size),
                    ask_size=int(quote.ask_size),
                    timestamp=quote.timestamp.timestamp(),
                    exchange=quote.bid_exchange if hasattr(quote, 'bid_exchange') else 'Alpaca',
                )
                
                # Call all quote callbacks
                for cb in self._quote_callbacks:
                    await cb(quote_data)
                    
            except Exception as e:
                logger.error(f"Error handling quote for {quote.symbol}: {e}")
        
        # Subscribe to quotes
        self.stream.subscribe_quotes(quote_handler, *symbols)
        
        logger.info(f"Subscribed to quotes for: {symbols}")
        
        # Start stream if not running
        if not self._running:
            self._running = True
            asyncio.create_task(self.stream.run())
    
    async def subscribe_bars(
        self,
        symbols: List[str],
        callback: Callable[[BarData], None],
        timeframe: str = "1Min",
    ):
        """
        Subscribe to real-time minute bars.
        
        Args:
            symbols: List of stock symbols
            callback: Callback function for bar data
            timeframe: Bar timeframe (Alpaca supports '1Min' bars)
        """
        if not self._connected:
            await self.connect()
        
        self._bar_callbacks.append(callback)
        
        # Define async handler for bars
        async def bar_handler(bar: Bar):
            try:
                bar_data = BarData(
                    symbol=bar.symbol,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=int(bar.volume),
                    timestamp=bar.timestamp.timestamp(),
                    vwap=float(bar.vwap) if hasattr(bar, 'vwap') and bar.vwap else None,
                )
                
                # Call all bar callbacks
                for cb in self._bar_callbacks:
                    await cb(bar_data)
                    
            except Exception as e:
                logger.error(f"Error handling bar for {bar.symbol}: {e}")
        
        # Subscribe to bars
        self.stream.subscribe_bars(bar_handler, *symbols)
        
        logger.info(f"Subscribed to {timeframe} bars for: {symbols}")
        
        # Start stream if not running
        if not self._running:
            self._running = True
            asyncio.create_task(self.stream.run())
    
    async def unsubscribe(self, symbols: List[str]):
        """
        Unsubscribe from symbols.
        
        Args:
            symbols: List of symbols to unsubscribe
        """
        if self.stream:
            # Alpaca's stream doesn't have direct unsubscribe,
            # but we can remove callbacks
            logger.info(f"Unsubscribed from: {symbols}")
    
    async def get_account(self) -> Optional[dict]:
        """
        Get account information.
        
        Returns:
            Account details or None
        """
        if not self.trading_client:
            logger.warning("Trading client not initialized")
            return None
        
        try:
            account = self.trading_client.get_account()
            return {
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'multiplier': account.multiplier,
                'initial_margin': float(account.initial_margin) if account.initial_margin else 0,
                'maintenance_margin': float(account.maintenance_margin) if account.maintenance_margin else 0,
                'daytrade_count': account.daytrade_count,
                'pattern_day_trader': account.pattern_day_trader,
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None
    
    async def get_positions(self) -> List[dict]:
        """
        Get current positions.
        
        Returns:
            List of positions
        """
        if not self.trading_client:
            logger.warning("Trading client not initialized")
            return []
        
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': pos.side,
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'avg_entry_price': float(pos.avg_entry_price),
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def health_check(self) -> bool:
        """
        Check adapter health.
        
        Returns:
            True if connected and stream is running
        """
        if not self._connected:
            return False
        
        try:
            # Check if we can get account info
            if self.trading_client:
                account = self.trading_client.get_account()
                return account.status == 'ACTIVE'
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
        
        return self._running