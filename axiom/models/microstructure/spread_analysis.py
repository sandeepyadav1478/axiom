"""
Bid-Ask Spread Dynamics Analysis
=================================

Advanced spread analysis for market microstructure including:
- Spread decomposition (Glosten-Harris, MRR models)
- Order processing, adverse selection, inventory costs
- Intraday spread patterns (U-shaped, event-driven)
- Cross-asset spread analysis
- Microstructure noise filtering

Performance Target: <8ms for spread decomposition

Mathematical References:
- Glosten, L. R., & Harris, L. E. (1988). "Estimating the components of the bid/ask spread"
- Madhavan, A., Richardson, M., & Roomans, M. (1997). "Why do security prices change?"
- Stoll, H. R. (1989). "Inferring the components of the bid-ask spread"
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, time
import time as time_module

from axiom.models.microstructure.base_model import (
    BaseMarketMicrostructureModel,
    TickData,
    MicrostructureMetrics,
    ModelResult
)


@dataclass
class SpreadComponents:
    """Decomposition of bid-ask spread into components."""
    # Total spread
    total_spread: float
    total_spread_bps: float
    
    # Components
    order_processing_cost: float  # Fixed cost of providing liquidity
    adverse_selection_cost: float  # Cost of trading with informed traders
    inventory_holding_cost: float  # Cost of holding risky inventory
    
    # Component percentages
    order_processing_pct: float
    adverse_selection_pct: float
    inventory_holding_pct: float
    
    # Model quality
    r_squared: float  # Model fit quality
    estimation_method: str  # Which model was used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_spread': self.total_spread,
            'total_spread_bps': self.total_spread_bps,
            'order_processing_cost': self.order_processing_cost,
            'adverse_selection_cost': self.adverse_selection_cost,
            'inventory_holding_cost': self.inventory_holding_cost,
            'order_processing_pct': self.order_processing_pct,
            'adverse_selection_pct': self.adverse_selection_pct,
            'inventory_holding_pct': self.inventory_holding_pct,
            'r_squared': self.r_squared,
            'estimation_method': self.estimation_method
        }


@dataclass
class IntradaySpreadPattern:
    """Intraday spread patterns and characteristics."""
    # Time-of-day patterns
    opening_spread: float  # Average spread at open
    closing_spread: float  # Average spread at close
    midday_spread: float  # Average spread during midday
    u_shape_coefficient: float  # Measure of U-shaped pattern
    
    # Volatility patterns
    spread_volatility: float  # Std dev of spreads
    spread_range: float  # Max - Min spread
    
    # Event patterns
    news_impact: float  # Spread expansion during news
    auction_impact: float  # Spread behavior near auctions
    
    # Statistical measures
    mean_spread: float
    median_spread: float
    spread_stability: float  # Inverse of coefficient of variation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'opening_spread': self.opening_spread,
            'closing_spread': self.closing_spread,
            'midday_spread': self.midday_spread,
            'u_shape_coefficient': self.u_shape_coefficient,
            'spread_volatility': self.spread_volatility,
            'spread_range': self.spread_range,
            'news_impact': self.news_impact,
            'auction_impact': self.auction_impact,
            'mean_spread': self.mean_spread,
            'median_spread': self.median_spread,
            'spread_stability': self.spread_stability
        }


class SpreadDecompositionModel(BaseMarketMicrostructureModel):
    """
    Spread Decomposition Model.
    
    Decomposes bid-ask spread into economic components:
    1. Order processing cost: Fixed cost of liquidity provision
    2. Adverse selection cost: Cost of trading with informed traders
    3. Inventory holding cost: Risk of holding inventory
    
    Implements:
    - Glosten-Harris model
    - Madhavan-Richardson-Roomans (MRR) model
    - Stoll's three-component model
    
    References:
    - Glosten, L. R., & Harris, L. E. (1988). "Estimating the components of the bid/ask spread"
    - Madhavan, A., Richardson, M., & Roomans, M. (1997)
    
    Usage:
        model = SpreadDecompositionModel(config={
            'method': 'glosten_harris'
        })
        
        components = model.decompose_spread(tick_data)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize Spread Decomposition Model.
        
        Args:
            config: Configuration dictionary with:
                - method: 'glosten_harris', 'mrr', 'stoll' (default: 'glosten_harris')
                - estimation_window: Window for estimation (default: 100)
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        self.method = self.config.get('method', 'glosten_harris')
        self.estimation_window = self.config.get('estimation_window', 100)
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate spread decomposition metrics.
        
        Args:
            tick_data: Tick data with trades
            **kwargs: Additional parameters
            
        Returns:
            MicrostructureMetrics with spread components
        """
        start_time = time_module.perf_counter()
        
        components = self.decompose_spread(tick_data)
        
        execution_time = (time_module.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.debug(
                f"Spread decomposition completed",
                execution_time_ms=round(execution_time, 3),
                method=self.method,
                adverse_selection_pct=round(components.adverse_selection_pct, 2)
            )
        
        return MicrostructureMetrics(
            order_processing_cost=components.order_processing_cost,
            adverse_selection_cost=components.adverse_selection_cost,
            inventory_cost=components.inventory_holding_cost
        )
    
    def decompose_spread(self, tick_data: TickData) -> SpreadComponents:
        """
        Decompose spread into components.
        
        Args:
            tick_data: Tick data
            
        Returns:
            SpreadComponents with decomposition
        """
        if self.method == 'glosten_harris':
            return self._glosten_harris_decomposition(tick_data)
        elif self.method == 'mrr':
            return self._mrr_decomposition(tick_data)
        elif self.method == 'stoll':
            return self._stoll_decomposition(tick_data)
        else:
            raise ValueError(f"Unknown decomposition method: {self.method}")
    
    def _glosten_harris_decomposition(self, tick_data: TickData) -> SpreadComponents:
        """
        Glosten-Harris spread decomposition.
        
        Model:
            ΔP_t = c + φΔQ_t + zQ_t + ε_t
        
        where:
            c = order processing cost
            φ = adverse selection component
            z = inventory component
            Q_t = trade direction indicator
        
        Args:
            tick_data: Tick data
            
        Returns:
            SpreadComponents
        """
        if tick_data.trade_direction is None or tick_data.n_ticks < 3:
            return self._default_components(tick_data)
        
        # Calculate price changes
        price_changes = np.diff(tick_data.price)
        
        # Trade directions
        Q_t = tick_data.trade_direction[1:]
        Q_t_minus_1 = tick_data.trade_direction[:-1]
        
        # Change in trade direction
        delta_Q = Q_t - Q_t_minus_1
        
        # Use most recent window
        if len(price_changes) > self.estimation_window:
            price_changes = price_changes[-self.estimation_window:]
            delta_Q = delta_Q[-self.estimation_window:]
            Q_t = Q_t[-self.estimation_window:]
        
        # Estimate via regression: ΔP = c + φΔQ + zQ
        # Simplified: use covariances
        
        # Order processing cost (constant)
        c = np.mean(price_changes)
        
        # Adverse selection (coefficient on ΔQ)
        if np.std(delta_Q) > 0:
            phi = np.cov(price_changes, delta_Q)[0, 1] / np.var(delta_Q)
        else:
            phi = 0.0
        
        # Inventory component (coefficient on Q)
        if np.std(Q_t) > 0:
            z = np.cov(price_changes, Q_t)[0, 1] / np.var(Q_t)
        else:
            z = 0.0
        
        # Calculate R-squared
        predicted = c + phi * delta_Q + z * Q_t
        ss_res = np.sum((price_changes - predicted) ** 2)
        ss_tot = np.sum((price_changes - np.mean(price_changes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Total spread
        avg_spread = np.mean(tick_data.spread)
        avg_spread_bps = np.mean(tick_data.spread_bps)
        
        # Convert to spread components
        # Order processing is 2c (round-trip)
        order_proc = abs(2 * c)
        # Adverse selection is 2φ
        adverse_sel = abs(2 * phi)
        # Inventory is abs(z)
        inventory = abs(z)
        
        # Calculate percentages
        total_components = order_proc + adverse_sel + inventory
        if total_components > 0:
            order_proc_pct = (order_proc / total_components) * 100
            adverse_sel_pct = (adverse_sel / total_components) * 100
            inventory_pct = (inventory / total_components) * 100
        else:
            order_proc_pct = adverse_sel_pct = inventory_pct = 0.0
        
        return SpreadComponents(
            total_spread=avg_spread,
            total_spread_bps=avg_spread_bps,
            order_processing_cost=order_proc,
            adverse_selection_cost=adverse_sel,
            inventory_holding_cost=inventory,
            order_processing_pct=order_proc_pct,
            adverse_selection_pct=adverse_sel_pct,
            inventory_holding_pct=inventory_pct,
            r_squared=max(0.0, r_squared),
            estimation_method='glosten_harris'
        )
    
    def _mrr_decomposition(self, tick_data: TickData) -> SpreadComponents:
        """
        Madhavan-Richardson-Roomans (MRR) spread decomposition.
        
        Focuses on temporary vs permanent price impacts.
        
        Args:
            tick_data: Tick data
            
        Returns:
            SpreadComponents
        """
        if tick_data.trade_direction is None or tick_data.n_ticks < 3:
            return self._default_components(tick_data)
        
        # MRR model focuses on revision in quotes
        # Simplified implementation using price changes
        
        price_changes = np.diff(tick_data.price)
        Q_t = tick_data.trade_direction[1:]
        
        # Use window
        if len(price_changes) > self.estimation_window:
            price_changes = price_changes[-self.estimation_window:]
            Q_t = Q_t[-self.estimation_window:]
        
        # Temporary component (mean-reverting)
        if len(price_changes) > 1:
            autocorr = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
            temporary = abs(autocorr)
        else:
            temporary = 0.0
        
        # Permanent component
        if np.std(Q_t) > 0:
            permanent = abs(np.cov(price_changes, Q_t)[0, 1] / np.var(Q_t))
        else:
            permanent = 0.0
        
        # Order processing as residual
        avg_spread = np.mean(tick_data.spread)
        order_proc = max(0, avg_spread - temporary - permanent)
        
        # Adverse selection is the permanent component
        adverse_sel = permanent
        
        # Inventory is the temporary component
        inventory = temporary
        
        # Calculate percentages
        total = order_proc + adverse_sel + inventory
        if total > 0:
            order_proc_pct = (order_proc / total) * 100
            adverse_sel_pct = (adverse_sel / total) * 100
            inventory_pct = (inventory / total) * 100
        else:
            order_proc_pct = adverse_sel_pct = inventory_pct = 0.0
        
        return SpreadComponents(
            total_spread=avg_spread,
            total_spread_bps=np.mean(tick_data.spread_bps),
            order_processing_cost=order_proc,
            adverse_selection_cost=adverse_sel,
            inventory_holding_cost=inventory,
            order_processing_pct=order_proc_pct,
            adverse_selection_pct=adverse_sel_pct,
            inventory_holding_pct=inventory_pct,
            r_squared=0.0,  # Not directly calculable in this simplified version
            estimation_method='mrr'
        )
    
    def _stoll_decomposition(self, tick_data: TickData) -> SpreadComponents:
        """
        Stoll's three-component spread decomposition.
        
        Args:
            tick_data: Tick data
            
        Returns:
            SpreadComponents
        """
        # Simplified Stoll model
        # Use empirical splits based on literature
        avg_spread = np.mean(tick_data.spread)
        avg_spread_bps = np.mean(tick_data.spread_bps)
        
        # Typical splits from empirical studies:
        # Order processing: ~40%
        # Adverse selection: ~40%
        # Inventory: ~20%
        
        order_proc = avg_spread * 0.40
        adverse_sel = avg_spread * 0.40
        inventory = avg_spread * 0.20
        
        return SpreadComponents(
            total_spread=avg_spread,
            total_spread_bps=avg_spread_bps,
            order_processing_cost=order_proc,
            adverse_selection_cost=adverse_sel,
            inventory_holding_cost=inventory,
            order_processing_pct=40.0,
            adverse_selection_pct=40.0,
            inventory_holding_pct=20.0,
            r_squared=0.0,
            estimation_method='stoll_empirical'
        )
    
    def _default_components(self, tick_data: TickData) -> SpreadComponents:
        """Return default spread components when estimation fails."""
        avg_spread = np.mean(tick_data.spread) if tick_data.n_ticks > 0 else 0.0
        avg_spread_bps = np.mean(tick_data.spread_bps) if tick_data.n_ticks > 0 else 0.0
        
        return SpreadComponents(
            total_spread=avg_spread,
            total_spread_bps=avg_spread_bps,
            order_processing_cost=0.0,
            adverse_selection_cost=0.0,
            inventory_holding_cost=0.0,
            order_processing_pct=0.0,
            adverse_selection_pct=0.0,
            inventory_holding_pct=0.0,
            r_squared=0.0,
            estimation_method='default'
        )


class IntradaySpreadAnalyzer(BaseMarketMicrostructureModel):
    """
    Intraday Spread Pattern Analyzer.
    
    Analyzes time-of-day patterns in bid-ask spreads:
    - U-shaped pattern (high at open/close, low at midday)
    - Opening/closing auction effects
    - News-driven spread expansion
    - Event-driven analysis
    
    Features:
    - Pattern detection
    - Anomaly identification
    - Statistical characterization
    - Cross-day comparison
    
    Usage:
        analyzer = IntradaySpreadAnalyzer()
        pattern = analyzer.analyze_patterns(tick_data)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """Initialize Intraday Spread Analyzer."""
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        self.opening_window_minutes = self.config.get('opening_window_minutes', 30)
        self.closing_window_minutes = self.config.get('closing_window_minutes', 30)
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate intraday spread metrics.
        
        Args:
            tick_data: Tick data covering trading day
            **kwargs: Additional parameters
            
        Returns:
            MicrostructureMetrics with spread patterns
        """
        pattern = self.analyze_patterns(tick_data)
        
        return MicrostructureMetrics(
            quoted_spread=pattern.mean_spread
        )
    
    def analyze_patterns(self, tick_data: TickData) -> IntradaySpreadPattern:
        """
        Analyze intraday spread patterns.
        
        Args:
            tick_data: Full day of tick data
            
        Returns:
            IntradaySpreadPattern with pattern characteristics
        """
        if tick_data.n_ticks == 0:
            return self._default_pattern()
        
        # Extract spreads and times
        spreads = tick_data.spread_bps
        timestamps = tick_data.timestamp
        
        # Calculate time-of-day buckets
        opening_spreads = []
        midday_spreads = []
        closing_spreads = []
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            minute = ts.minute
            time_minutes = hour * 60 + minute
            
            # Assume market hours: 9:30 AM - 4:00 PM (570 - 960 minutes)
            market_open = 570  # 9:30 AM
            market_close = 960  # 4:00 PM
            
            # Opening period
            if market_open <= time_minutes < market_open + self.opening_window_minutes:
                opening_spreads.append(spreads[i])
            # Closing period
            elif market_close - self.closing_window_minutes <= time_minutes <= market_close:
                closing_spreads.append(spreads[i])
            # Midday (11:00 AM - 2:00 PM)
            elif 660 <= time_minutes < 840:
                midday_spreads.append(spreads[i])
        
        # Calculate averages
        opening_spread = np.mean(opening_spreads) if opening_spreads else np.mean(spreads)
        midday_spread = np.mean(midday_spreads) if midday_spreads else np.mean(spreads)
        closing_spread = np.mean(closing_spreads) if closing_spreads else np.mean(spreads)
        
        # U-shape coefficient: (open + close) / (2 * midday)
        # Higher value = stronger U-shape
        if midday_spread > 0:
            u_shape = (opening_spread + closing_spread) / (2 * midday_spread)
        else:
            u_shape = 1.0
        
        # Volatility metrics
        spread_volatility = np.std(spreads)
        spread_range = np.max(spreads) - np.min(spreads)
        
        # Statistical measures
        mean_spread = np.mean(spreads)
        median_spread = np.median(spreads)
        
        # Stability (inverse coefficient of variation)
        if mean_spread > 0:
            spread_stability = 1.0 / (spread_volatility / mean_spread)
        else:
            spread_stability = 0.0
        
        # Event impacts (simplified - would need event data)
        news_impact = 0.0  # Placeholder
        auction_impact = 0.0  # Placeholder
        
        return IntradaySpreadPattern(
            opening_spread=opening_spread,
            closing_spread=closing_spread,
            midday_spread=midday_spread,
            u_shape_coefficient=u_shape,
            spread_volatility=spread_volatility,
            spread_range=spread_range,
            news_impact=news_impact,
            auction_impact=auction_impact,
            mean_spread=mean_spread,
            median_spread=median_spread,
            spread_stability=spread_stability
        )
    
    def _default_pattern(self) -> IntradaySpreadPattern:
        """Return default pattern when no data."""
        return IntradaySpreadPattern(
            opening_spread=0.0,
            closing_spread=0.0,
            midday_spread=0.0,
            u_shape_coefficient=1.0,
            spread_volatility=0.0,
            spread_range=0.0,
            news_impact=0.0,
            auction_impact=0.0,
            mean_spread=0.0,
            median_spread=0.0,
            spread_stability=0.0
        )
    
    def detect_u_shape(self, tick_data: TickData) -> bool:
        """
        Detect if spread exhibits U-shaped pattern.
        
        Args:
            tick_data: Tick data
            
        Returns:
            True if U-shaped pattern detected
        """
        pattern = self.analyze_patterns(tick_data)
        
        # U-shape if coefficient > 1.1 (10% higher at open/close than midday)
        return pattern.u_shape_coefficient > 1.1


class MicrostructureNoiseFilter(BaseMarketMicrostructureModel):
    """
    Microstructure Noise Filtering.
    
    Filters out bid-ask bounce and other microstructure noise:
    - Optimal sampling frequency
    - Noise-to-signal ratio
    - Realized variance adjustment
    
    Usage:
        filter = MicrostructureNoiseFilter()
        filtered_prices = filter.filter_noise(tick_data)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """Initialize Microstructure Noise Filter."""
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        self.filter_method = self.config.get('filter_method', 'roll')
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """Calculate noise metrics."""
        noise_ratio = self.calculate_noise_ratio(tick_data)
        
        return MicrostructureMetrics()
    
    def filter_noise(self, tick_data: TickData) -> np.ndarray:
        """
        Filter microstructure noise from prices.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Filtered prices
        """
        if self.filter_method == 'roll':
            return self._roll_filter(tick_data)
        elif self.filter_method == 'midpoint':
            return tick_data.midpoint
        else:
            return tick_data.price
    
    def _roll_filter(self, tick_data: TickData) -> np.ndarray:
        """Apply Roll's noise filter."""
        # Use midpoint to avoid bid-ask bounce
        return tick_data.midpoint
    
    def calculate_noise_ratio(self, tick_data: TickData) -> float:
        """
        Calculate noise-to-signal ratio.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Noise ratio (higher = more noise)
        """
        if tick_data.n_ticks < 2:
            return 0.0
        
        # Calculate realized variance at different frequencies
        returns_1tick = np.diff(tick_data.price) / tick_data.price[:-1]
        
        # Subsample to reduce noise
        if tick_data.n_ticks >= 10:
            prices_5tick = tick_data.price[::5]
            returns_5tick = np.diff(prices_5tick) / prices_5tick[:-1]
            
            var_1tick = np.var(returns_1tick)
            var_5tick = np.var(returns_5tick)
            
            # Noise ratio: (var_high_freq - var_low_freq) / var_low_freq
            if var_5tick > 0:
                noise_ratio = (var_1tick - var_5tick) / var_5tick
            else:
                noise_ratio = 0.0
        else:
            noise_ratio = 0.0
        
        return float(max(0.0, noise_ratio))


__all__ = [
    "SpreadDecompositionModel",
    "IntradaySpreadAnalyzer",
    "MicrostructureNoiseFilter",
    "SpreadComponents",
    "IntradaySpreadPattern",
]