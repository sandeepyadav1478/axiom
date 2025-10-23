"""
Order Flow Analysis for High-Frequency Trading
===============================================

Implements institutional-grade order flow analysis including:
- Order Flow Imbalance (OFI)
- Trade Classification (Lee-Ready, Tick Test, Quote Rule)
- Volume Profile Analysis
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Flow Toxicity Indicators

Performance Target: <5ms for real-time OFI calculation

Mathematical References:
- Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World"
- Lee, C. M., & Ready, M. J. (1991). "Inferring Trade Direction from Intraday Data"
- Hasbrouck, J. (2007). "Empirical Market Microstructure"
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time

from axiom.models.microstructure.base_model import (
    BaseMarketMicrostructureModel,
    TickData,
    MicrostructureMetrics,
    ModelResult
)


@dataclass
class OrderFlowMetrics:
    """Container for order flow analysis results."""
    # Order Flow Imbalance
    ofi: float  # Order Flow Imbalance
    buy_volume: float  # Total buy volume
    sell_volume: float  # Total sell volume
    net_volume: float  # Buy volume - Sell volume
    
    # Volume Profile
    volume_at_bid: float  # Volume at bid
    volume_at_ask: float  # Volume at ask
    volume_between: float  # Volume between bid-ask
    
    # Flow Toxicity
    vpin: float  # Volume-Synchronized Probability of Informed Trading
    toxicity_score: float  # Flow toxicity measure
    
    # Trade Classification Stats
    buy_trades: int  # Number of buy trades
    sell_trades: int  # Number of sell trades
    unknown_trades: int  # Number of unclassified trades
    
    # Book Pressure
    order_imbalance: float  # Order book imbalance
    trade_imbalance: float  # Trade imbalance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ofi': self.ofi,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'net_volume': self.net_volume,
            'volume_at_bid': self.volume_at_bid,
            'volume_at_ask': self.volume_at_ask,
            'volume_between': self.volume_between,
            'vpin': self.vpin,
            'toxicity_score': self.toxicity_score,
            'buy_trades': self.buy_trades,
            'sell_trades': self.sell_trades,
            'unknown_trades': self.unknown_trades,
            'order_imbalance': self.order_imbalance,
            'trade_imbalance': self.trade_imbalance
        }


class OrderFlowAnalyzer(BaseMarketMicrostructureModel):
    """
    Order Flow Analysis Model.
    
    Analyzes order flow imbalance, trade direction, and volume profiles
    for high-frequency trading signals.
    
    Features:
    - Real-time OFI calculation (<5ms)
    - Multiple trade classification algorithms
    - VPIN calculation
    - Volume profile analysis
    - Flow toxicity indicators
    
    Usage:
        analyzer = OrderFlowAnalyzer(config={
            'ofi_window': 100,
            'vpin_buckets': 50,
            'classification_method': 'lee_ready'
        })
        
        metrics = analyzer.calculate_metrics(tick_data)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize Order Flow Analyzer.
        
        Args:
            config: Configuration dictionary with:
                - ofi_window: Window size for OFI calculation (default: 100)
                - vpin_buckets: Number of volume buckets for VPIN (default: 50)
                - classification_method: 'lee_ready', 'tick_test', 'quote_rule', 'bvc'
                - toxicity_threshold: Threshold for toxic flow (default: 0.7)
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        # Configuration
        self.ofi_window = self.config.get('ofi_window', 100)
        self.vpin_buckets = self.config.get('vpin_buckets', 50)
        self.classification_method = self.config.get('classification_method', 'lee_ready')
        self.toxicity_threshold = self.config.get('toxicity_threshold', 0.7)
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate comprehensive order flow metrics.
        
        Args:
            tick_data: High-frequency tick data
            **kwargs: Additional parameters
            
        Returns:
            MicrostructureMetrics with order flow analysis
        """
        start_time = time.perf_counter()
        
        # Validate inputs
        self.validate_inputs(tick_data)
        
        # Classify trades if not already classified
        if tick_data.trade_direction is None:
            tick_data.trade_direction = self.classify_trades(tick_data)
        
        # Calculate OFI
        ofi = self.calculate_ofi(tick_data)
        
        # Calculate volume statistics
        buy_volume, sell_volume = self._calculate_directional_volume(tick_data)
        net_volume = buy_volume - sell_volume
        
        # Calculate volume at price levels
        vol_at_bid, vol_at_ask, vol_between = self._calculate_volume_profile(tick_data)
        
        # Calculate VPIN
        vpin = self.calculate_vpin(tick_data)
        
        # Calculate flow toxicity
        toxicity = self._calculate_toxicity(tick_data, vpin)
        
        # Trade classification stats
        buy_trades = np.sum(tick_data.trade_direction == 1)
        sell_trades = np.sum(tick_data.trade_direction == -1)
        unknown_trades = np.sum(tick_data.trade_direction == 0)
        
        # Calculate order and trade imbalance
        order_imbalance = self._calculate_order_imbalance(tick_data)
        trade_imbalance = net_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.debug(
                f"Order flow analysis completed",
                execution_time_ms=round(execution_time, 3),
                ofi=round(ofi, 4),
                vpin=round(vpin, 4)
            )
        
        # Package results
        flow_metrics = OrderFlowMetrics(
            ofi=ofi,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            net_volume=net_volume,
            volume_at_bid=vol_at_bid,
            volume_at_ask=vol_at_ask,
            volume_between=vol_between,
            vpin=vpin,
            toxicity_score=toxicity,
            buy_trades=int(buy_trades),
            sell_trades=int(sell_trades),
            unknown_trades=int(unknown_trades),
            order_imbalance=order_imbalance,
            trade_imbalance=trade_imbalance
        )
        
        return MicrostructureMetrics(
            order_flow_imbalance=ofi,
            signed_volume=net_volume,
            vpin=vpin,
            flow_toxicity=toxicity
        )
    
    def classify_trades(self, tick_data: TickData) -> np.ndarray:
        """
        Classify trades as buy or sell initiated.
        
        Supports multiple classification algorithms:
        - Lee-Ready: Quote rule + tick test
        - Tick test: Compare to previous trade
        - Quote rule: Compare to bid-ask midpoint
        - BVC: Bulk volume classification
        
        Args:
            tick_data: Tick data to classify
            
        Returns:
            Array of trade directions (1=buy, -1=sell, 0=unknown)
        """
        method = self.classification_method.lower()
        
        if method == 'lee_ready':
            return self._lee_ready_algorithm(tick_data)
        elif method == 'tick_test':
            return self._tick_test(tick_data)
        elif method == 'quote_rule':
            return self._quote_rule(tick_data)
        elif method == 'bvc':
            return self._bulk_volume_classification(tick_data)
        else:
            raise ValueError(f"Unknown classification method: {method}")
    
    def _lee_ready_algorithm(self, tick_data: TickData) -> np.ndarray:
        """
        Lee-Ready algorithm for trade classification.
        
        Uses quote rule (compare to midpoint) and tick test (compare to previous price)
        as a tie-breaker.
        
        Reference:
        Lee, C. M., & Ready, M. J. (1991). "Inferring Trade Direction from Intraday Data"
        
        Args:
            tick_data: Tick data
            
        Returns:
            Trade directions (1=buy, -1=sell, 0=unknown)
        """
        n = tick_data.n_ticks
        directions = np.zeros(n, dtype=np.int8)
        midpoints = tick_data.midpoint
        
        for i in range(n):
            price = tick_data.price[i]
            mid = midpoints[i]
            
            # Quote rule: compare to midpoint
            if price > mid:
                directions[i] = 1  # Buy
            elif price < mid:
                directions[i] = -1  # Sell
            else:
                # Tick test: compare to previous price
                if i > 0:
                    prev_price = tick_data.price[i-1]
                    if price > prev_price:
                        directions[i] = 1  # Uptick -> Buy
                    elif price < prev_price:
                        directions[i] = -1  # Downtick -> Sell
                    else:
                        # Zero tick: use previous direction
                        if i > 1:
                            directions[i] = directions[i-1]
                        else:
                            directions[i] = 0  # Unknown
                else:
                    directions[i] = 0  # First trade, unknown
        
        return directions
    
    def _tick_test(self, tick_data: TickData) -> np.ndarray:
        """
        Tick test for trade classification.
        
        Classifies based on price movement relative to previous trade.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Trade directions (1=buy, -1=sell, 0=unknown)
        """
        n = tick_data.n_ticks
        directions = np.zeros(n, dtype=np.int8)
        
        for i in range(1, n):
            price_change = tick_data.price[i] - tick_data.price[i-1]
            
            if price_change > 0:
                directions[i] = 1  # Uptick -> Buy
            elif price_change < 0:
                directions[i] = -1  # Downtick -> Sell
            else:
                # Zero tick: use previous direction
                directions[i] = directions[i-1]
        
        return directions
    
    def _quote_rule(self, tick_data: TickData) -> np.ndarray:
        """
        Quote rule for trade classification.
        
        Classifies based on trade price relative to bid-ask midpoint.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Trade directions (1=buy, -1=sell, 0=unknown)
        """
        midpoints = tick_data.midpoint
        directions = np.zeros(tick_data.n_ticks, dtype=np.int8)
        
        # Vectorized comparison
        directions[tick_data.price > midpoints] = 1  # Above mid -> Buy
        directions[tick_data.price < midpoints] = -1  # Below mid -> Sell
        # At midpoint remains 0 (unknown)
        
        return directions
    
    def _bulk_volume_classification(self, tick_data: TickData) -> np.ndarray:
        """
        Bulk Volume Classification (BVC).
        
        Uses volume-weighted approach to classify trades in bulk.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Trade directions (1=buy, -1=sell, 0=unknown)
        """
        # Start with quote rule
        directions = self._quote_rule(tick_data)
        
        # Adjust based on volume clustering
        n = tick_data.n_ticks
        window = min(10, n // 10)  # Adaptive window
        
        for i in range(window, n):
            # Look at volume-weighted direction in window
            window_dirs = directions[i-window:i]
            window_vols = tick_data.volume[i-window:i]
            
            # Volume-weighted average direction
            if np.sum(window_vols) > 0:
                weighted_dir = np.sum(window_dirs * window_vols) / np.sum(window_vols)
                
                # If current trade is ambiguous, use weighted direction
                if directions[i] == 0:
                    directions[i] = 1 if weighted_dir > 0 else -1 if weighted_dir < 0 else 0
        
        return directions
    
    def calculate_ofi(self, tick_data: TickData) -> float:
        """
        Calculate Order Flow Imbalance (OFI).
        
        OFI measures the imbalance between buy and sell order flow:
        OFI = Σ(buy_volume - sell_volume) / total_volume
        
        Args:
            tick_data: Tick data with classified trades
            
        Returns:
            Order flow imbalance [-1, 1]
        """
        if tick_data.trade_direction is None:
            raise ValueError("Trades must be classified before calculating OFI")
        
        # Calculate signed volume
        signed_volume = tick_data.volume * tick_data.trade_direction
        
        # Use rolling window if specified
        if self.ofi_window < tick_data.n_ticks:
            # Use most recent window
            signed_volume = signed_volume[-self.ofi_window:]
            total_volume = np.sum(np.abs(tick_data.volume[-self.ofi_window:]))
        else:
            total_volume = np.sum(tick_data.volume)
        
        # Calculate OFI
        net_signed_volume = np.sum(signed_volume)
        
        if total_volume > 0:
            ofi = net_signed_volume / total_volume
        else:
            ofi = 0.0
        
        return float(ofi)
    
    def calculate_vpin(self, tick_data: TickData) -> float:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading).
        
        VPIN estimates the probability of informed trading based on order flow imbalance
        across volume buckets.
        
        Reference:
        Easley, D., López de Prado, M., & O'Hara, M. (2012). 
        "Flow Toxicity and Liquidity in a High-Frequency World"
        
        Formula:
        VPIN = Σ|V_buy - V_sell| / (2 * Σ V_total)
        
        Args:
            tick_data: Tick data with classified trades
            
        Returns:
            VPIN score [0, 1]
        """
        if tick_data.trade_direction is None:
            raise ValueError("Trades must be classified before calculating VPIN")
        
        # Calculate total volume per bucket
        total_volume = np.sum(tick_data.volume)
        if total_volume == 0:
            return 0.0
        
        volume_per_bucket = total_volume / self.vpin_buckets
        
        # Allocate trades to volume buckets
        buy_volumes = []
        sell_volumes = []
        
        current_bucket_volume = 0
        current_buy_volume = 0
        current_sell_volume = 0
        
        for i in range(tick_data.n_ticks):
            volume = tick_data.volume[i]
            direction = tick_data.trade_direction[i]
            
            # Add to current bucket
            if direction == 1:
                current_buy_volume += volume
            elif direction == -1:
                current_sell_volume += volume
            
            current_bucket_volume += volume
            
            # Check if bucket is full
            if current_bucket_volume >= volume_per_bucket:
                buy_volumes.append(current_buy_volume)
                sell_volumes.append(current_sell_volume)
                
                # Reset bucket
                current_bucket_volume = 0
                current_buy_volume = 0
                current_sell_volume = 0
        
        # Add remaining partial bucket if significant
        if current_bucket_volume > volume_per_bucket * 0.1:
            buy_volumes.append(current_buy_volume)
            sell_volumes.append(current_sell_volume)
        
        if len(buy_volumes) == 0:
            return 0.0
        
        # Calculate VPIN
        buy_volumes = np.array(buy_volumes)
        sell_volumes = np.array(sell_volumes)
        
        # VPIN = average absolute order imbalance
        order_imbalances = np.abs(buy_volumes - sell_volumes)
        total_volumes = buy_volumes + sell_volumes
        
        # Avoid division by zero
        mask = total_volumes > 0
        if not np.any(mask):
            return 0.0
        
        vpin = np.sum(order_imbalances[mask]) / (2 * np.sum(total_volumes[mask]))
        
        return float(vpin)
    
    def _calculate_directional_volume(self, tick_data: TickData) -> Tuple[float, float]:
        """Calculate total buy and sell volumes."""
        if tick_data.trade_direction is None:
            return 0.0, 0.0
        
        buy_mask = tick_data.trade_direction == 1
        sell_mask = tick_data.trade_direction == -1
        
        buy_volume = np.sum(tick_data.volume[buy_mask])
        sell_volume = np.sum(tick_data.volume[sell_mask])
        
        return float(buy_volume), float(sell_volume)
    
    def _calculate_volume_profile(self, tick_data: TickData) -> Tuple[float, float, float]:
        """
        Calculate volume at different price levels.
        
        Returns:
            Tuple of (volume_at_bid, volume_at_ask, volume_between)
        """
        vol_at_bid = 0.0
        vol_at_ask = 0.0
        vol_between = 0.0
        
        for i in range(tick_data.n_ticks):
            price = tick_data.price[i]
            volume = tick_data.volume[i]
            bid = tick_data.bid[i]
            ask = tick_data.ask[i]
            
            if price <= bid:
                vol_at_bid += volume
            elif price >= ask:
                vol_at_ask += volume
            else:
                vol_between += volume
        
        return vol_at_bid, vol_at_ask, vol_between
    
    def _calculate_toxicity(self, tick_data: TickData, vpin: float) -> float:
        """
        Calculate flow toxicity score.
        
        Combines VPIN with other flow metrics to assess information asymmetry.
        
        Args:
            tick_data: Tick data
            vpin: VPIN score
            
        Returns:
            Toxicity score [0, 1]
        """
        # Base toxicity from VPIN
        toxicity = vpin
        
        # Adjust based on trade frequency
        # Rapid trading with high VPIN indicates toxic flow
        avg_time_between_trades = np.mean(np.diff(tick_data.timestamp.astype(np.int64))) / 1e9  # seconds
        if avg_time_between_trades < 1.0:  # Less than 1 second
            toxicity *= 1.2
        
        # Cap at 1.0
        toxicity = min(toxicity, 1.0)
        
        return float(toxicity)
    
    def _calculate_order_imbalance(self, tick_data: TickData) -> float:
        """
        Calculate order book imbalance.
        
        Returns:
            Order imbalance [-1, 1]
        """
        # Average bid-ask size imbalance
        bid_sizes = tick_data.bid_size
        ask_sizes = tick_data.ask_size
        
        total_bid = np.sum(bid_sizes)
        total_ask = np.sum(ask_sizes)
        
        if total_bid + total_ask > 0:
            imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        else:
            imbalance = 0.0
        
        return float(imbalance)


__all__ = [
    "OrderFlowAnalyzer",
    "OrderFlowMetrics",
]