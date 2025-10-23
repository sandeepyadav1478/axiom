"""
Liquidity Metrics for Market Microstructure Analysis
=====================================================

Institutional-grade liquidity measurement tools including:
- Spread-based measures (quoted, effective, realized, Roll, high-low)
- Price impact measures (Amihud, Pastor-Stambaugh, market impact)
- Volume-based metrics (turnover, depth, resilience)
- Order book metrics (depth ratio, slope, weighted depth)

Performance Target: <10ms for comprehensive liquidity analysis

Mathematical References:
- Amihud, Y. (2002). "Illiquidity and stock returns"
- Roll, R. (1984). "A simple implicit measure of the effective bid-ask spread"
- Pastor, L., & Stambaugh, R. F. (2003). "Liquidity risk and expected stock returns"
- Hasbrouck, J. (2009). "Trading costs and returns for U.S. equities"
"""

import numpy as np
import pandas as pd
from typing import List,  Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time

from axiom.models.microstructure.base_model import (
    BaseMarketMicrostructureModel,
    TickData,
    OrderBookSnapshot,
    MicrostructureMetrics,
    ModelResult
)


@dataclass
class LiquidityMetrics:
    """Container for comprehensive liquidity metrics."""
    # Spread-based measures
    quoted_spread: float  # Bid-ask spread
    quoted_spread_bps: float  # Spread in basis points
    effective_spread: float  # 2 * |trade_price - midpoint|
    effective_spread_bps: float  # Effective spread in bps
    realized_spread: float  # Temporary impact
    realized_spread_bps: float  # Realized spread in bps
    roll_spread: float  # Roll's implicit spread estimator
    high_low_spread: float  # High-low spread estimator
    
    # Price impact measures
    amihud_illiquidity: float  # Amihud ILLIQ ratio
    pastor_stambaugh_gamma: float  # Pastor-Stambaugh liquidity measure
    market_impact_coef: float  # Market impact coefficient
    price_impact_per_dollar: float  # Price impact per dollar traded
    
    # Temporary vs permanent impact
    temporary_impact: float  # Temporary price impact
    permanent_impact: float  # Permanent price impact
    impact_ratio: float  # Temporary/Permanent ratio
    
    # Volume-based metrics
    turnover_rate: float  # Trading volume / market cap
    trading_activity_index: float  # Relative trading activity
    depth: float  # Average order book depth
    resilience: float  # Speed of price recovery
    hui_heubel_ratio: float  # Hui-Heubel liquidity ratio
    
    # Order book metrics
    bid_ask_depth_ratio: float  # Bid depth / Ask depth
    order_book_slope: float  # Price impact of order book
    volume_weighted_depth: float  # Volume-weighted average depth
    cumulative_depth_5_levels: float  # Depth in top 5 levels
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'quoted_spread': self.quoted_spread,
            'quoted_spread_bps': self.quoted_spread_bps,
            'effective_spread': self.effective_spread,
            'effective_spread_bps': self.effective_spread_bps,
            'realized_spread': self.realized_spread,
            'realized_spread_bps': self.realized_spread_bps,
            'roll_spread': self.roll_spread,
            'high_low_spread': self.high_low_spread,
            'amihud_illiquidity': self.amihud_illiquidity,
            'pastor_stambaugh_gamma': self.pastor_stambaugh_gamma,
            'market_impact_coef': self.market_impact_coef,
            'price_impact_per_dollar': self.price_impact_per_dollar,
            'temporary_impact': self.temporary_impact,
            'permanent_impact': self.permanent_impact,
            'impact_ratio': self.impact_ratio,
            'turnover_rate': self.turnover_rate,
            'trading_activity_index': self.trading_activity_index,
            'depth': self.depth,
            'resilience': self.resilience,
            'hui_heubel_ratio': self.hui_heubel_ratio,
            'bid_ask_depth_ratio': self.bid_ask_depth_ratio,
            'order_book_slope': self.order_book_slope,
            'volume_weighted_depth': self.volume_weighted_depth,
            'cumulative_depth_5_levels': self.cumulative_depth_5_levels
        }


class LiquidityAnalyzer(BaseMarketMicrostructureModel):
    """
    Comprehensive Liquidity Analysis Model.
    
    Measures market liquidity through multiple dimensions:
    - Transaction costs (spreads)
    - Price impact
    - Trading volume
    - Order book depth
    - Market resilience
    
    Features:
    - Multi-dimensional liquidity assessment
    - Real-time liquidity monitoring
    - Comparative liquidity analysis
    - Liquidity-adjusted trading strategies
    
    Performance Target: <10ms for full analysis
    
    Usage:
        analyzer = LiquidityAnalyzer(config={
            'illiquidity_window': 20,
            'spread_estimator': 'roll'
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
        Initialize Liquidity Analyzer.
        
        Args:
            config: Configuration dictionary with:
                - illiquidity_window: Window for Amihud calculation (default: 20)
                - spread_estimator: 'roll', 'high_low', 'quoted' (default: 'roll')
                - market_cap: Market capitalization for turnover (optional)
                - depth_levels: Number of order book levels to analyze (default: 5)
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        self.illiquidity_window = self.config.get('illiquidity_window', 20)
        self.spread_estimator = self.config.get('spread_estimator', 'roll')
        self.market_cap = self.config.get('market_cap', None)
        self.depth_levels = self.config.get('depth_levels', 5)
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate comprehensive liquidity metrics.
        
        Args:
            tick_data: High-frequency tick data
            **kwargs: Additional parameters (order_book_snapshots, returns, etc.)
            
        Returns:
            MicrostructureMetrics with liquidity analysis
        """
        start_time = time.perf_counter()
        
        # Validate inputs
        self.validate_inputs(tick_data)
        
        # Calculate spread-based measures
        quoted_spread, quoted_spread_bps = self._calculate_quoted_spread(tick_data)
        effective_spread, effective_spread_bps = self._calculate_effective_spread(tick_data)
        realized_spread, realized_spread_bps = self._calculate_realized_spread(tick_data)
        roll_spread = self._calculate_roll_spread(tick_data)
        high_low_spread = self._calculate_high_low_spread(tick_data)
        
        # Calculate price impact measures
        amihud = self._calculate_amihud_illiquidity(tick_data)
        ps_gamma = self._calculate_pastor_stambaugh(tick_data)
        mi_coef = self._calculate_market_impact_coefficient(tick_data)
        price_impact_per_dollar = self._calculate_price_impact_per_dollar(tick_data)
        
        # Decompose impact
        temporary_impact, permanent_impact = self._decompose_impact(tick_data)
        impact_ratio = temporary_impact / permanent_impact if permanent_impact > 0 else 0.0
        
        # Calculate volume-based metrics
        turnover = self._calculate_turnover_rate(tick_data)
        activity_index = self._calculate_trading_activity_index(tick_data)
        depth = self._calculate_average_depth(tick_data)
        resilience = self._calculate_resilience(tick_data)
        hui_heubel = self._calculate_hui_heubel_ratio(tick_data)
        
        # Calculate order book metrics
        order_book_snapshots = kwargs.get('order_book_snapshots', None)
        if order_book_snapshots:
            depth_ratio = self._calculate_bid_ask_depth_ratio(order_book_snapshots)
            book_slope = self._calculate_order_book_slope(order_book_snapshots)
            vw_depth = self._calculate_volume_weighted_depth(order_book_snapshots)
            cum_depth = self._calculate_cumulative_depth(order_book_snapshots)
        else:
            # Use simplified metrics from tick data
            depth_ratio = float(np.mean(tick_data.bid_size / tick_data.ask_size))
            book_slope = 0.0
            vw_depth = depth
            cum_depth = depth * self.depth_levels
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.debug(
                f"Liquidity analysis completed",
                execution_time_ms=round(execution_time, 3),
                quoted_spread_bps=round(quoted_spread_bps, 2),
                amihud_illiquidity=round(amihud, 6)
            )
        
        # Package all metrics
        liquidity_metrics = LiquidityMetrics(
            quoted_spread=quoted_spread,
            quoted_spread_bps=quoted_spread_bps,
            effective_spread=effective_spread,
            effective_spread_bps=effective_spread_bps,
            realized_spread=realized_spread,
            realized_spread_bps=realized_spread_bps,
            roll_spread=roll_spread,
            high_low_spread=high_low_spread,
            amihud_illiquidity=amihud,
            pastor_stambaugh_gamma=ps_gamma,
            market_impact_coef=mi_coef,
            price_impact_per_dollar=price_impact_per_dollar,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            impact_ratio=impact_ratio,
            turnover_rate=turnover,
            trading_activity_index=activity_index,
            depth=depth,
            resilience=resilience,
            hui_heubel_ratio=hui_heubel,
            bid_ask_depth_ratio=depth_ratio,
            order_book_slope=book_slope,
            volume_weighted_depth=vw_depth,
            cumulative_depth_5_levels=cum_depth
        )
        
        return MicrostructureMetrics(
            quoted_spread=quoted_spread,
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            amihud_illiquidity=amihud,
            roll_spread=roll_spread
        )
    
    def _calculate_quoted_spread(self, tick_data: TickData) -> Tuple[float, float]:
        """
        Calculate quoted (bid-ask) spread.
        
        Quoted Spread = Ask - Bid
        
        Args:
            tick_data: Tick data
            
        Returns:
            Tuple of (spread, spread_bps)
        """
        spreads = tick_data.ask - tick_data.bid
        avg_spread = np.mean(spreads)
        
        midpoints = tick_data.midpoint
        avg_midpoint = np.mean(midpoints)
        
        spread_bps = (avg_spread / avg_midpoint) * 10000 if avg_midpoint > 0 else 0.0
        
        return float(avg_spread), float(spread_bps)
    
    def _calculate_effective_spread(self, tick_data: TickData) -> Tuple[float, float]:
        """
        Calculate effective spread.
        
        Effective Spread = 2 × |Trade Price - Midpoint|
        
        Args:
            tick_data: Tick data
            
        Returns:
            Tuple of (spread, spread_bps)
        """
        midpoints = tick_data.midpoint
        effective_spreads = 2 * np.abs(tick_data.price - midpoints)
        avg_effective = np.mean(effective_spreads)
        
        avg_midpoint = np.mean(midpoints)
        spread_bps = (avg_effective / avg_midpoint) * 10000 if avg_midpoint > 0 else 0.0
        
        return float(avg_effective), float(spread_bps)
    
    def _calculate_realized_spread(self, tick_data: TickData) -> Tuple[float, float]:
        """
        Calculate realized spread (temporary impact).
        
        Realized Spread = 2 × Direction × (Trade Price - Midpoint_t+τ)
        
        where Direction = +1 for buy, -1 for sell
        
        Args:
            tick_data: Tick data
            
        Returns:
            Tuple of (spread, spread_bps)
        """
        if tick_data.trade_direction is None or tick_data.n_ticks < 2:
            return 0.0, 0.0
        
        # Calculate realized spreads using future midpoint
        realized_spreads = []
        for i in range(tick_data.n_ticks - 1):
            direction = tick_data.trade_direction[i]
            trade_price = tick_data.price[i]
            future_midpoint = tick_data.midpoint[i + 1]
            
            realized_spread = 2 * direction * (trade_price - future_midpoint)
            realized_spreads.append(realized_spread)
        
        if len(realized_spreads) > 0:
            avg_realized = np.mean(realized_spreads)
            avg_midpoint = np.mean(tick_data.midpoint[:-1])
            spread_bps = (avg_realized / avg_midpoint) * 10000 if avg_midpoint > 0 else 0.0
        else:
            avg_realized = 0.0
            spread_bps = 0.0
        
        return float(avg_realized), float(spread_bps)
    
    def _calculate_roll_spread(self, tick_data: TickData) -> float:
        """
        Calculate Roll's implicit spread estimator.
        
        Roll Spread = 2 × √(-Cov(ΔP_t, ΔP_{t-1}))
        
        Reference:
        Roll, R. (1984). "A simple implicit measure of the effective bid-ask spread"
        
        Args:
            tick_data: Tick data
            
        Returns:
            Estimated spread
        """
        if tick_data.n_ticks < 3:
            return 0.0
        
        # Calculate price changes
        price_changes = np.diff(tick_data.price)
        
        # Calculate serial covariance
        if len(price_changes) > 1:
            cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
            
            # Roll spread is 2 × sqrt(-cov)
            if cov < 0:
                roll_spread = 2 * np.sqrt(-cov)
            else:
                roll_spread = 0.0  # Positive cov indicates no bid-ask bounce
        else:
            roll_spread = 0.0
        
        return float(roll_spread)
    
    def _calculate_high_low_spread(self, tick_data: TickData) -> float:
        """
        Calculate high-low spread estimator.
        
        Corwin-Schultz Estimator:
        Spread = 2(e^α - 1) / (1 + e^α)
        where α is based on high-low range
        
        Args:
            tick_data: Tick data
            
        Returns:
            Estimated spread
        """
        # Use tick-level high-low as proxy
        high = np.max(tick_data.ask)
        low = np.min(tick_data.bid)
        
        if low > 0:
            # Simplified version
            hl_range = np.log(high / low)
            alpha = hl_range / np.sqrt(2)
            
            # Corwin-Schultz formula
            spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        else:
            spread = 0.0
        
        return float(spread)
    
    def _calculate_amihud_illiquidity(self, tick_data: TickData) -> float:
        """
        Calculate Amihud illiquidity ratio (ILLIQ).
        
        ILLIQ = Average(|Return| / Dollar Volume)
        
        Reference:
        Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and time-series effects"
        
        Args:
            tick_data: Tick data
            
        Returns:
            Amihud illiquidity ratio (higher = less liquid)
        """
        if tick_data.n_ticks < 2:
            return 0.0
        
        # Calculate returns
        returns = np.diff(tick_data.price) / tick_data.price[:-1]
        abs_returns = np.abs(returns)
        
        # Calculate dollar volumes (price × volume)
        dollar_volumes = tick_data.price[1:] * tick_data.volume[1:]
        
        # Calculate ILLIQ for each period
        illiq_values = []
        for i in range(len(abs_returns)):
            if dollar_volumes[i] > 0:
                illiq = abs_returns[i] / dollar_volumes[i]
                illiq_values.append(illiq)
        
        if len(illiq_values) > 0:
            amihud = np.mean(illiq_values) * 1e6  # Scale by 10^6 for readability
        else:
            amihud = 0.0
        
        return float(amihud)
    
    def _calculate_pastor_stambaugh(self, tick_data: TickData) -> float:
        """
        Calculate Pastor-Stambaugh liquidity measure (gamma).
        
        This measures the return reversal associated with volume.
        
        Reference:
        Pastor, L., & Stambaugh, R. F. (2003). "Liquidity risk and expected stock returns"
        
        Args:
            tick_data: Tick data
            
        Returns:
            Pastor-Stambaugh gamma (liquidity measure)
        """
        if tick_data.n_ticks < 3:
            return 0.0
        
        # Calculate returns and signed volumes
        returns = np.diff(tick_data.price) / tick_data.price[:-1]
        volumes = tick_data.volume[1:]
        
        # Simple regression: r_t+1 = γ × (sign(r_t) × volume_t) + ε
        if len(returns) > 1:
            lagged_returns = returns[:-1]
            future_returns = returns[1:]
            lagged_volumes = volumes[:-1]
            
            # Signed volume (volume × sign of return)
            signed_volumes = lagged_volumes * np.sign(lagged_returns)
            
            # Simple linear regression
            if np.std(signed_volumes) > 0:
                gamma = np.cov(future_returns, signed_volumes)[0, 1] / np.var(signed_volumes)
            else:
                gamma = 0.0
        else:
            gamma = 0.0
        
        return float(gamma)
    
    def _calculate_market_impact_coefficient(self, tick_data: TickData) -> float:
        """
        Calculate market impact coefficient.
        
        MI = Cov(Price Change, Signed Volume) / Var(Signed Volume)
        
        Args:
            tick_data: Tick data
            
        Returns:
            Market impact coefficient
        """
        if tick_data.trade_direction is None or tick_data.n_ticks < 2:
            return 0.0
        
        # Price changes
        price_changes = np.diff(tick_data.price)
        
        # Signed volumes
        signed_volumes = tick_data.volume[1:] * tick_data.trade_direction[1:]
        
        # Calculate coefficient
        if np.std(signed_volumes) > 0:
            mi_coef = np.cov(price_changes, signed_volumes)[0, 1] / np.var(signed_volumes)
        else:
            mi_coef = 0.0
        
        return float(mi_coef)
    
    def _calculate_price_impact_per_dollar(self, tick_data: TickData) -> float:
        """
        Calculate price impact per dollar traded.
        
        Impact = Average(|ΔPrice| / DollarVolume)
        
        Args:
            tick_data: Tick data
            
        Returns:
            Price impact per dollar (in basis points)
        """
        if tick_data.n_ticks < 2:
            return 0.0
        
        price_changes = np.abs(np.diff(tick_data.price))
        dollar_volumes = tick_data.price[1:] * tick_data.volume[1:]
        
        impacts = []
        for i in range(len(price_changes)):
            if dollar_volumes[i] > 0:
                impact = (price_changes[i] / tick_data.price[i+1]) / dollar_volumes[i]
                impacts.append(impact * 10000)  # Convert to bps
        
        if len(impacts) > 0:
            avg_impact = np.mean(impacts)
        else:
            avg_impact = 0.0
        
        return float(avg_impact)
    
    def _decompose_impact(self, tick_data: TickData) -> Tuple[float, float]:
        """
        Decompose price impact into temporary and permanent components.
        
        Temporary: Price moves due to liquidity provision, reverts
        Permanent: Information in the trade, persists
        
        Args:
            tick_data: Tick data
            
        Returns:
            Tuple of (temporary_impact, permanent_impact)
        """
        if tick_data.trade_direction is None or tick_data.n_ticks < 3:
            return 0.0, 0.0
        
        # Temporary impact: immediate price move that reverts
        # Measured by effective spread
        effective_spread, _ = self._calculate_effective_spread(tick_data)
        temporary_impact = effective_spread / 2  # One-way impact
        
        # Permanent impact: lasting price change
        # Measured by price change over longer horizon
        short_term_changes = np.diff(tick_data.price[:tick_data.n_ticks//2])
        long_term_changes = np.diff(tick_data.price[tick_data.n_ticks//2:])
        
        if len(short_term_changes) > 0 and len(long_term_changes) > 0:
            permanent_impact = np.mean(np.abs(long_term_changes))
        else:
            permanent_impact = temporary_impact  # Fallback
        
        return float(temporary_impact), float(permanent_impact)
    
    def _calculate_turnover_rate(self, tick_data: TickData) -> float:
        """
        Calculate turnover rate.
        
        Turnover = Trading Volume / Market Cap
        
        Args:
            tick_data: Tick data
            
        Returns:
            Turnover rate
        """
        total_volume = np.sum(tick_data.volume)
        
        if self.market_cap is not None and self.market_cap > 0:
            turnover = total_volume / self.market_cap
        else:
            # Use relative volume as proxy
            avg_price = np.mean(tick_data.price)
            dollar_volume = total_volume * avg_price
            turnover = dollar_volume / 1e9  # Normalize to billions
        
        return float(turnover)
    
    def _calculate_trading_activity_index(self, tick_data: TickData) -> float:
        """
        Calculate trading activity index.
        
        TAI = Number of trades / Time period
        
        Args:
            tick_data: Tick data
            
        Returns:
            Trading activity index
        """
        n_trades = tick_data.n_ticks
        
        if tick_data.n_ticks > 1:
            time_span = (tick_data.timestamp[-1] - tick_data.timestamp[0]).total_seconds()
            if time_span > 0:
                activity = n_trades / (time_span / 3600)  # Trades per hour
            else:
                activity = 0.0
        else:
            activity = 0.0
        
        return float(activity)
    
    def _calculate_average_depth(self, tick_data: TickData) -> float:
        """
        Calculate average order book depth.
        
        Depth = Average(Bid Size + Ask Size)
        
        Args:
            tick_data: Tick data
            
        Returns:
            Average depth
        """
        total_depth = tick_data.bid_size + tick_data.ask_size
        avg_depth = np.mean(total_depth)
        
        return float(avg_depth)
    
    def _calculate_resilience(self, tick_data: TickData) -> float:
        """
        Calculate market resilience (speed of price recovery).
        
        Resilience = 1 / Half-life of price impact
        
        Args:
            tick_data: Tick data
            
        Returns:
            Resilience measure (higher = more resilient)
        """
        if tick_data.n_ticks < 3:
            return 0.0
        
        # Calculate autocorrelation of returns (negative = mean reversion)
        returns = np.diff(tick_data.price) / tick_data.price[:-1]
        
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            # Higher negative autocorr = faster mean reversion = higher resilience
            if autocorr < 0:
                resilience = -autocorr  # Convert to positive measure
            else:
                resilience = 0.0
        else:
            resilience = 0.0
        
        return float(resilience)
    
    def _calculate_hui_heubel_ratio(self, tick_data: TickData) -> float:
        """
        Calculate Hui-Heubel liquidity ratio.
        
        HH = (High - Low) / (Volume / Price)
        
        Lower ratio indicates higher liquidity.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Hui-Heubel ratio
        """
        if tick_data.n_ticks == 0:
            return 0.0
        
        high = np.max(tick_data.price)
        low = np.min(tick_data.price)
        total_volume = np.sum(tick_data.volume)
        avg_price = np.mean(tick_data.price)
        
        if total_volume > 0 and avg_price > 0:
            hui_heubel = (high - low) / (total_volume / avg_price)
        else:
            hui_heubel = 0.0
        
        return float(hui_heubel)
    
    def _calculate_bid_ask_depth_ratio(
        self,
        order_book_snapshots: List[OrderBookSnapshot]
    ) -> float:
        """Calculate average bid/ask depth ratio."""
        if not order_book_snapshots:
            return 1.0
        
        ratios = []
        for snapshot in order_book_snapshots:
            total_bid = snapshot.total_bid_depth
            total_ask = snapshot.total_ask_depth
            
            if total_ask > 0:
                ratio = total_bid / total_ask
                ratios.append(ratio)
        
        return float(np.mean(ratios)) if ratios else 1.0
    
    def _calculate_order_book_slope(
        self,
        order_book_snapshots: List[OrderBookSnapshot]
    ) -> float:
        """Calculate average order book slope (price impact per unit volume)."""
        if not order_book_snapshots:
            return 0.0
        
        slopes = []
        for snapshot in order_book_snapshots:
            # Calculate average price change per unit volume
            if len(snapshot.bids) > 1:
                bid_slope = np.mean(np.diff(snapshot.bids) / snapshot.bid_sizes[1:])
                slopes.append(abs(bid_slope))
            
            if len(snapshot.asks) > 1:
                ask_slope = np.mean(np.diff(snapshot.asks) / snapshot.ask_sizes[1:])
                slopes.append(abs(ask_slope))
        
        return float(np.mean(slopes)) if slopes else 0.0
    
    def _calculate_volume_weighted_depth(
        self,
        order_book_snapshots: List[OrderBookSnapshot]
    ) -> float:
        """Calculate volume-weighted average depth."""
        if not order_book_snapshots:
            return 0.0
        
        total_depth = 0.0
        total_weight = 0.0
        
        for snapshot in order_book_snapshots:
            depth = snapshot.total_bid_depth + snapshot.total_ask_depth
            weight = 1.0  # Could weight by spread or other factors
            
            total_depth += depth * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_depth / total_weight
        return 0.0
    
    def _calculate_cumulative_depth(
        self,
        order_book_snapshots: List[OrderBookSnapshot],
        levels: Optional[int] = None
    ) -> float:
        """Calculate cumulative depth in top N levels."""
        if not order_book_snapshots:
            return 0.0
        
        if levels is None:
            levels = self.depth_levels
        
        cumulative_depths = []
        for snapshot in order_book_snapshots:
            bid_depth = np.sum(snapshot.bid_sizes[:levels])
            ask_depth = np.sum(snapshot.ask_sizes[:levels])
            cumulative_depths.append(bid_depth + ask_depth)
        
        return float(np.mean(cumulative_depths)) if cumulative_depths else 0.0


__all__ = [
    "LiquidityAnalyzer",
    "LiquidityMetrics",
]