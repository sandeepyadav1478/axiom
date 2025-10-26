"""
Market Data Adapters

Provider-specific adapters for real-time market data streaming.

Supported providers:
- Polygon.io: Professional market data
- Binance: Cryptocurrency data
- Alpaca Markets: Commission-free trading and data
"""

from axiom.streaming.adapters.base_adapter import (
    BaseMarketDataAdapter,
    MarketDataType,
    TradeData,
    QuoteData,
    BarData,
)

__all__ = [
    'BaseMarketDataAdapter',
    'MarketDataType',
    'TradeData',
    'QuoteData',
    'BarData',
]