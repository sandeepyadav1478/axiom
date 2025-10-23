"""
Base Model for Market Microstructure Analysis
==============================================

Provides base classes and data structures for high-frequency trading analytics
and market microstructure analysis.

Performance Target: <1ms per 1000 ticks processing
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np
import pandas as pd
import time

from axiom.models.base.base_model import BaseFinancialModel, ModelResult, ModelMetadata
from axiom.models.base.mixins import ValidationMixin, PerformanceMixin
from axiom.core.logging.axiom_logger import get_logger


@dataclass
class TickData:
    """
    High-frequency tick data structure.
    
    Optimized for fast processing of market microstructure events.
    All arrays must have the same length.
    """
    timestamp: pd.DatetimeIndex
    price: np.ndarray  # Trade prices
    volume: np.ndarray  # Trade volumes
    bid: np.ndarray  # Best bid price
    ask: np.ndarray  # Best ask price
    bid_size: np.ndarray  # Best bid size
    ask_size: np.ndarray  # Best ask size
    trade_direction: Optional[np.ndarray] = None  # 1=buy-initiated, -1=sell-initiated, 0=unknown
    
    def __post_init__(self):
        """Validate tick data consistency."""
        lengths = [
            len(self.timestamp),
            len(self.price),
            len(self.volume),
            len(self.bid),
            len(self.ask),
            len(self.bid_size),
            len(self.ask_size)
        ]
        
        if len(set(lengths)) != 1:
            raise ValueError(f"All tick data arrays must have same length. Got: {lengths}")
        
        if self.trade_direction is not None and len(self.trade_direction) != lengths[0]:
            raise ValueError("trade_direction length must match other arrays")
    
    @property
    def n_ticks(self) -> int:
        """Number of ticks in dataset."""
        return len(self.timestamp)
    
    @property
    def midpoint(self) -> np.ndarray:
        """Calculate midpoint prices."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> np.ndarray:
        """Calculate bid-ask spreads."""
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> np.ndarray:
        """Calculate spreads in basis points."""
        return (self.spread / self.midpoint) * 10000
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame({
            'timestamp': self.timestamp,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
        })
        
        if self.trade_direction is not None:
            df['trade_direction'] = self.trade_direction
        
        return df


@dataclass
class OrderBookSnapshot:
    """
    Order book snapshot at a specific time.
    
    Contains multi-level order book data for liquidity analysis.
    """
    timestamp: datetime
    bids: np.ndarray  # Price levels (descending)
    bid_sizes: np.ndarray  # Size at each bid level
    asks: np.ndarray  # Price levels (ascending)
    ask_sizes: np.ndarray  # Size at each ask level
    
    @property
    def best_bid(self) -> float:
        """Best bid price."""
        return self.bids[0] if len(self.bids) > 0 else 0.0
    
    @property
    def best_ask(self) -> float:
        """Best ask price."""
        return self.asks[0] if len(self.asks) > 0 else 0.0
    
    @property
    def midpoint(self) -> float:
        """Midpoint price."""
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.best_ask - self.best_bid
    
    @property
    def total_bid_depth(self) -> float:
        """Total size on bid side."""
        return np.sum(self.bid_sizes)
    
    @property
    def total_ask_depth(self) -> float:
        """Total size on ask side."""
        return np.sum(self.ask_sizes)
    
    def depth_imbalance(self) -> float:
        """
        Calculate order book depth imbalance.
        
        Returns:
            Imbalance ratio: (bid_depth - ask_depth) / (bid_depth + ask_depth)
            Range: [-1, 1] where positive = more bids, negative = more asks
        """
        total_bid = self.total_bid_depth
        total_ask = self.total_ask_depth
        
        if total_bid + total_ask == 0:
            return 0.0
        
        return (total_bid - total_ask) / (total_bid + total_ask)


@dataclass
class TradeData:
    """
    Individual trade information.
    
    Used for trade-by-trade analysis and classification.
    """
    timestamp: datetime
    price: float
    volume: float
    trade_direction: Optional[int] = None  # 1=buy, -1=sell, 0=unknown
    trade_id: Optional[str] = None
    
    def is_buy(self) -> bool:
        """Check if trade is buy-initiated."""
        return self.trade_direction == 1
    
    def is_sell(self) -> bool:
        """Check if trade is sell-initiated."""
        return self.trade_direction == -1


@dataclass
class MicrostructureMetrics:
    """
    Comprehensive market microstructure metrics.
    
    Contains results from all microstructure analysis components.
    """
    # Order Flow
    order_flow_imbalance: Optional[float] = None
    signed_volume: Optional[float] = None
    vpin: Optional[float] = None  # Volume-Synchronized Probability of Informed Trading
    flow_toxicity: Optional[float] = None
    
    # Execution Benchmarks
    vwap: Optional[float] = None
    twap: Optional[float] = None
    participation_rate: Optional[float] = None
    
    # Liquidity Measures
    quoted_spread: Optional[float] = None
    effective_spread: Optional[float] = None
    realized_spread: Optional[float] = None
    amihud_illiquidity: Optional[float] = None
    roll_spread: Optional[float] = None
    
    # Market Impact
    kyle_lambda: Optional[float] = None  # Kyle's lambda (price impact coefficient)
    market_impact_bps: Optional[float] = None
    temporary_impact: Optional[float] = None
    permanent_impact: Optional[float] = None
    
    # Spread Components
    order_processing_cost: Optional[float] = None
    adverse_selection_cost: Optional[float] = None
    inventory_cost: Optional[float] = None
    
    # Price Discovery
    information_share: Optional[float] = None
    price_discovery_contribution: Optional[float] = None
    quote_to_trade_ratio: Optional[float] = None
    
    # Order Book
    depth_imbalance: Optional[float] = None
    book_pressure: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BaseMarketMicrostructureModel(BaseFinancialModel, ValidationMixin, PerformanceMixin):
    """
    Abstract base class for market microstructure models.
    
    Provides common functionality for:
    - High-frequency tick data processing
    - Order flow analysis
    - Liquidity measurement
    - Market impact estimation
    
    All microstructure models inherit from this class and implement
    specific calculation methods.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize market microstructure model.
        
        Args:
            config: Model-specific configuration
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(config, enable_logging, enable_performance_tracking)
        self.logger = get_logger(f"axiom.models.microstructure.{self.__class__.__name__}")
    
    @abstractmethod
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate microstructure metrics from tick data.
        
        Args:
            tick_data: High-frequency tick data
            **kwargs: Model-specific parameters
            
        Returns:
            MicrostructureMetrics with calculated values
        """
        pass
    
    def process_tick_data(
        self,
        ticks: TickData,
        batch_size: int = 1000
    ) -> List[MicrostructureMetrics]:
        """
        Process tick data in batches for optimal performance.
        
        Args:
            ticks: Tick data to process
            batch_size: Number of ticks per batch
            
        Returns:
            List of metrics for each batch
        """
        start_time = time.perf_counter()
        
        n_ticks = ticks.n_ticks
        n_batches = (n_ticks + batch_size - 1) // batch_size
        
        results = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_ticks)
            
            # Create batch
            batch_ticks = TickData(
                timestamp=ticks.timestamp[start_idx:end_idx],
                price=ticks.price[start_idx:end_idx],
                volume=ticks.volume[start_idx:end_idx],
                bid=ticks.bid[start_idx:end_idx],
                ask=ticks.ask[start_idx:end_idx],
                bid_size=ticks.bid_size[start_idx:end_idx],
                ask_size=ticks.ask_size[start_idx:end_idx],
                trade_direction=ticks.trade_direction[start_idx:end_idx] if ticks.trade_direction is not None else None
            )
            
            # Calculate metrics for batch
            metrics = self.calculate_metrics(batch_ticks)
            results.append(metrics)
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            ticks_per_ms = n_ticks / execution_time if execution_time > 0 else 0
            self.logger.info(
                f"Processed {n_ticks} ticks in {n_batches} batches",
                execution_time_ms=round(execution_time, 3),
                ticks_per_ms=round(ticks_per_ms, 2)
            )
        
        return results
    
    def validate_inputs(self, tick_data: TickData, **kwargs) -> bool:
        """
        Validate tick data inputs.
        
        Args:
            tick_data: Tick data to validate
            **kwargs: Additional parameters to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if tick_data.n_ticks == 0:
            raise ValueError("Tick data is empty")
        
        # Validate prices are positive
        if np.any(tick_data.price <= 0):
            raise ValueError("Prices must be positive")
        
        if np.any(tick_data.bid <= 0) or np.any(tick_data.ask <= 0):
            raise ValueError("Bid/ask prices must be positive")
        
        # Validate bid < ask
        if np.any(tick_data.bid >= tick_data.ask):
            raise ValueError("Bid must be less than ask")
        
        # Validate volumes are non-negative
        if np.any(tick_data.volume < 0):
            raise ValueError("Volumes cannot be negative")
        
        if np.any(tick_data.bid_size < 0) or np.any(tick_data.ask_size < 0):
            raise ValueError("Order sizes cannot be negative")
        
        return True
    
    def calculate(self, **kwargs) -> ModelResult:
        """
        Standard calculate interface for base model compatibility.
        
        This delegates to calculate_metrics for microstructure-specific logic.
        """
        start_time = time.perf_counter()
        
        # Extract tick data from kwargs
        tick_data = kwargs.pop('tick_data', None)  # Remove from kwargs to avoid duplication
        if tick_data is None:
            raise ValueError("tick_data is required")
        
        # Validate inputs
        self.validate_inputs(tick_data=tick_data, **kwargs)
        
        # Calculate metrics
        metrics = self.calculate_metrics(tick_data, **kwargs)
        
        # Create result
        execution_time_ms = self._track_performance("microstructure_calculation", start_time)
        metadata = self._create_metadata(execution_time_ms)
        
        return ModelResult(
            value=metrics,
            metadata=metadata,
            success=True
        )


__all__ = [
    "TickData",
    "OrderBookSnapshot",
    "TradeData",
    "MicrostructureMetrics",
    "BaseMarketMicrostructureModel",
]