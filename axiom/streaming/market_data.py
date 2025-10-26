"""
Market Data Streamer

Unified interface for multiple market data providers.

Aggregates data from Polygon.io, Binance, Alpaca, and other providers
into a single, consistent streaming interface.
"""

import asyncio
import logging
from typing import Callable, List, Optional, Dict, Any
from enum import Enum

from axiom.streaming.config import StreamingConfig
from axiom.streaming.adapters.base_adapter import (
    BaseMarketDataAdapter,
    TradeData,
    QuoteData,
    BarData,
    MarketDataType,
)

# Import adapters
try:
    from axiom.streaming.adapters.polygon_adapter import PolygonAdapter
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    PolygonAdapter = None

try:
    from axiom.streaming.adapters.binance_adapter import BinanceAdapter
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    BinanceAdapter = None

try:
    from axiom.streaming.adapters.alpaca_adapter import AlpacaAdapter
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    AlpacaAdapter = None


logger = logging.getLogger(__name__)


class MarketDataProvider(Enum):
    """Supported market data providers."""
    POLYGON = "polygon"
    BINANCE = "binance"
    ALPACA = "alpaca"


class MarketDataStreamer:
    """
    Unified interface for multiple market data providers.
    
    Features:
    - Single interface for all providers
    - Automatic provider selection
    - Data aggregation across providers
    - Failover support
    - Performance monitoring
    
    Performance Target: <10ms end-to-end latency
    
    Example:
        ```python
        streamer = MarketDataStreamer(providers=['polygon', 'alpaca'])
        await streamer.connect()
        
        async def on_trade(trade: TradeData):
            print(f"Trade: {trade.symbol} @ ${trade.price}")
        
        await streamer.subscribe_trades(['AAPL', 'GOOGL'], on_trade)
        ```
    """
    
    def __init__(
        self,
        providers: List[str],
        config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize market data streamer.
        
        Args:
            providers: List of provider names ('polygon', 'binance', 'alpaca')
            config: Streaming configuration
        """
        self.config = config or StreamingConfig()
        self.providers: Dict[str, BaseMarketDataAdapter] = {}
        self._connected_providers: List[str] = []
        
        # Initialize requested providers
        for provider_name in providers:
            provider_name = provider_name.lower()
            
            if provider_name == 'polygon' and POLYGON_AVAILABLE:
                self.providers['polygon'] = PolygonAdapter(config=self.config)
            elif provider_name == 'binance' and BINANCE_AVAILABLE:
                self.providers['binance'] = BinanceAdapter(config=self.config)
            elif provider_name == 'alpaca' and ALPACA_AVAILABLE:
                self.providers['alpaca'] = AlpacaAdapter(config=self.config)
            else:
                logger.warning(f"Provider '{provider_name}' not available or not installed")
        
        if not self.providers:
            raise ValueError("No valid providers specified or available")
        
        logger.info(f"Market data streamer initialized with providers: {list(self.providers.keys())}")
    
    async def connect(self):
        """Connect to all providers."""
        for name, provider in self.providers.items():
            try:
                await provider.connect()
                self._connected_providers.append(name)
                logger.info(f"Connected to {name}")
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")
        
        if not self._connected_providers:
            raise ConnectionError("Failed to connect to any providers")
    
    async def disconnect(self):
        """Disconnect from all providers."""
        for name, provider in self.providers.items():
            try:
                await provider.disconnect()
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
        
        self._connected_providers.clear()
    
    async def subscribe_trades(
        self,
        symbols: List[str],
        callback: Callable[[TradeData], None],
        provider: Optional[str] = None,
    ):
        """
        Subscribe to real-time trades.
        
        Args:
            symbols: List of symbols to subscribe
            callback: Callback function for trade data
            provider: Specific provider to use (optional, uses all if None)
        """
        if provider:
            # Subscribe to specific provider
            if provider in self.providers:
                await self.providers[provider].subscribe_trades(symbols, callback)
            else:
                logger.warning(f"Provider '{provider}' not found")
        else:
            # Subscribe to all providers that support trades
            for name, prov in self.providers.items():
                if prov.supports_data_type(MarketDataType.TRADE):
                    try:
                        await prov.subscribe_trades(symbols, callback)
                        logger.info(f"Subscribed to trades on {name} for: {symbols}")
                    except Exception as e:
                        logger.error(f"Error subscribing to trades on {name}: {e}")
    
    async def subscribe_quotes(
        self,
        symbols: List[str],
        callback: Callable[[QuoteData], None],
        provider: Optional[str] = None,
    ):
        """
        Subscribe to real-time quotes.
        
        Args:
            symbols: List of symbols to subscribe
            callback: Callback function for quote data
            provider: Specific provider to use (optional)
        """
        if provider:
            if provider in self.providers:
                await self.providers[provider].subscribe_quotes(symbols, callback)
            else:
                logger.warning(f"Provider '{provider}' not found")
        else:
            for name, prov in self.providers.items():
                if prov.supports_data_type(MarketDataType.QUOTE):
                    try:
                        await prov.subscribe_quotes(symbols, callback)
                        logger.info(f"Subscribed to quotes on {name} for: {symbols}")
                    except Exception as e:
                        logger.error(f"Error subscribing to quotes on {name}: {e}")
    
    async def subscribe_bars(
        self,
        symbols: List[str],
        callback: Callable[[BarData], None],
        timeframe: str = "1Min",
        provider: Optional[str] = None,
    ):
        """
        Subscribe to real-time bars/candlesticks.
        
        Args:
            symbols: List of symbols to subscribe
            callback: Callback function for bar data
            timeframe: Bar timeframe (e.g., '1Min', '5Min', '1H')
            provider: Specific provider to use (optional)
        """
        if provider:
            if provider in self.providers:
                await self.providers[provider].subscribe_bars(symbols, callback, timeframe)
            else:
                logger.warning(f"Provider '{provider}' not found")
        else:
            for name, prov in self.providers.items():
                if prov.supports_data_type(MarketDataType.BAR):
                    try:
                        await prov.subscribe_bars(symbols, callback, timeframe)
                        logger.info(f"Subscribed to {timeframe} bars on {name} for: {symbols}")
                    except Exception as e:
                        logger.error(f"Error subscribing to bars on {name}: {e}")
    
    async def unsubscribe(self, symbols: List[str], provider: Optional[str] = None):
        """
        Unsubscribe from symbols.
        
        Args:
            symbols: List of symbols to unsubscribe
            provider: Specific provider (optional)
        """
        if provider:
            if provider in self.providers:
                await self.providers[provider].unsubscribe(symbols)
        else:
            for prov in self.providers.values():
                await prov.unsubscribe(symbols)
    
    def get_provider(self, name: str) -> Optional[BaseMarketDataAdapter]:
        """
        Get specific provider adapter.
        
        Args:
            name: Provider name
        
        Returns:
            Provider adapter or None
        """
        return self.providers.get(name)
    
    def get_connected_providers(self) -> List[str]:
        """
        Get list of connected providers.
        
        Returns:
            List of provider names
        """
        return self._connected_providers.copy()
    
    def is_connected(self) -> bool:
        """Check if at least one provider is connected."""
        return len(self._connected_providers) > 0
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all providers.
        
        Returns:
            Dictionary of provider health status
        """
        health = {}
        for name, provider in self.providers.items():
            try:
                health[name] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health[name] = False
        
        return health
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all providers.
        
        Returns:
            Dictionary of provider statistics
        """
        stats = {
            'total_providers': len(self.providers),
            'connected_providers': len(self._connected_providers),
            'providers': {
                name: {
                    'connected': provider.is_connected(),
                    'supported_types': [t.value for t in provider.get_supported_data_types()],
                }
                for name, provider in self.providers.items()
            }
        }
        return stats
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MarketDataStreamer("
            f"providers={list(self.providers.keys())}, "
            f"connected={self._connected_providers})"
        )


async def create_streamer(
    providers: List[str],
    config: Optional[StreamingConfig] = None,
) -> MarketDataStreamer:
    """
    Create and connect a market data streamer.
    
    Args:
        providers: List of provider names
        config: Streaming configuration
    
    Returns:
        Connected MarketDataStreamer instance
    
    Example:
        ```python
        streamer = await create_streamer(['polygon', 'alpaca'])
        ```
    """
    streamer = MarketDataStreamer(providers, config)
    await streamer.connect()
    return streamer