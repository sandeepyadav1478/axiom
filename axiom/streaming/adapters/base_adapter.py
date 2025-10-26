"""
Base Market Data Adapter

Abstract base class for all market data provider adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Any, Dict
import logging


logger = logging.getLogger(__name__)


class MarketDataType(Enum):
    """Types of market data."""
    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    ORDER_BOOK = "order_book"
    NEWS = "news"


@dataclass
class TradeData:
    """Real-time trade data."""
    symbol: str
    price: float
    size: int
    timestamp: float
    exchange: Optional[str] = None
    conditions: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'size': self.size,
            'timestamp': self.timestamp,
            'exchange': self.exchange,
            'conditions': self.conditions,
        }


@dataclass
class QuoteData:
    """Real-time quote data."""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: float
    exchange: Optional[str] = None
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'timestamp': self.timestamp,
            'exchange': self.exchange,
            'spread': self.spread,
            'mid_price': self.mid_price,
        }


@dataclass
class BarData:
    """Real-time bar/candlestick data."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: float
    vwap: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timestamp': self.timestamp,
            'vwap': self.vwap,
        }


class BaseMarketDataAdapter(ABC):
    """
    Abstract base class for market data adapters.
    
    All provider-specific adapters (Polygon, Binance, Alpaca) inherit from this.
    
    Attributes:
        provider_name: Name of the data provider
        supported_data_types: List of supported data types
    """
    
    def __init__(self, provider_name: str):
        """
        Initialize adapter.
        
        Args:
            provider_name: Name of the provider
        """
        self.provider_name = provider_name
        self.supported_data_types: List[MarketDataType] = []
        self._callbacks: Dict[str, List[Callable]] = {}
        self._connected = False
        
        logger.info(f"Initialized {provider_name} adapter")
    
    @abstractmethod
    async def connect(self):
        """Establish connection to data provider."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to data provider."""
        pass
    
    @abstractmethod
    async def subscribe_trades(
        self,
        symbols: List[str],
        callback: Callable[[TradeData], None],
    ):
        """
        Subscribe to real-time trades.
        
        Args:
            symbols: List of symbols to subscribe
            callback: Callback function for trade data
        """
        pass
    
    @abstractmethod
    async def subscribe_quotes(
        self,
        symbols: List[str],
        callback: Callable[[QuoteData], None],
    ):
        """
        Subscribe to real-time quotes.
        
        Args:
            symbols: List of symbols to subscribe
            callback: Callback function for quote data
        """
        pass
    
    @abstractmethod
    async def subscribe_bars(
        self,
        symbols: List[str],
        callback: Callable[[BarData], None],
        timeframe: str = "1Min",
    ):
        """
        Subscribe to real-time bars.
        
        Args:
            symbols: List of symbols to subscribe
            callback: Callback function for bar data
            timeframe: Bar timeframe (e.g., '1Min', '5Min', '1H')
        """
        pass
    
    async def unsubscribe(self, symbols: List[str]):
        """
        Unsubscribe from symbols.
        
        Args:
            symbols: List of symbols to unsubscribe
        """
        logger.info(f"Unsubscribing from {symbols}")
    
    def register_callback(self, data_type: str, callback: Callable):
        """
        Register callback for data type.
        
        Args:
            data_type: Type of data (trades, quotes, bars)
            callback: Callback function
        """
        if data_type not in self._callbacks:
            self._callbacks[data_type] = []
        
        self._callbacks[data_type].append(callback)
    
    async def _invoke_callbacks(self, data_type: str, data: Any):
        """
        Invoke all callbacks for data type.
        
        Args:
            data_type: Type of data
            data: Data to pass to callbacks
        """
        if data_type in self._callbacks:
            for callback in self._callbacks[data_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {data_type}: {e}")
    
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._connected
    
    def get_supported_data_types(self) -> List[MarketDataType]:
        """Get list of supported data types."""
        return self.supported_data_types
    
    def supports_data_type(self, data_type: MarketDataType) -> bool:
        """
        Check if data type is supported.
        
        Args:
            data_type: Data type to check
        
        Returns:
            True if supported, False otherwise
        """
        return data_type in self.supported_data_types
    
    async def health_check(self) -> bool:
        """
        Check adapter health.
        
        Returns:
            True if healthy, False otherwise
        """
        return self._connected
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(provider={self.provider_name}, connected={self._connected})"