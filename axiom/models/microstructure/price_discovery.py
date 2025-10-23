"""
Price Discovery and Market Quality Analysis
============================================

Institutional-grade price discovery analysis including:
- Hasbrouck Information Share (HIS)
- Component Share (CS)
- Information Leadership Share (ILS)
- Quote activity metrics
- Market quality indicators
- Price efficiency measures

Performance Target: <12ms for price discovery analysis

Mathematical References:
- Hasbrouck, J. (1995). "One security, many markets: Determining the contributions to price discovery"
- Gonzalo, J., & Granger, C. (1995). "Estimation of common long-memory components in cointegrated systems"
- Baillie, G. B., Booth, G. G., Tse, Y., & Zabotina, T. (2002). "Price discovery and common factor models"
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from scipy import stats
import time

from axiom.models.microstructure.base_model import (
    BaseMarketMicrostructureModel,
    TickData,
    MicrostructureMetrics,
    ModelResult
)


@dataclass
class PriceDiscoveryMetrics:
    """Container for price discovery analysis results."""
    # Information share metrics
    information_share: float  # Hasbrouck information share
    component_share: float  # Gonzalo-Granger component share
    information_leadership: float  # ILS measure
    price_discovery_contribution: float  # Overall contribution
    
    # Quote activity
    quote_to_trade_ratio: float  # Quotes per trade
    quote_update_frequency: float  # Updates per second
    effective_spread_ratio: float  # Effective/quoted spread
    price_improvement_probability: float  # % trades with improvement
    
    # Market quality
    price_efficiency: float  # Variance ratio or similar
    quote_stability: float  # Stability of quotes
    information_asymmetry: float  # Measure of asymmetry
    adverse_selection_component: float  # From spread decomposition
    
    # Autocorrelation measures
    return_autocorrelation: float  # First-order autocorr of returns
    quote_autocorrelation: float  # First-order autocorr of quote changes
    
    # Random walk deviation
    variance_ratio: float  # Variance ratio test statistic
    random_walk_deviation: float  # Deviation from random walk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'information_share': self.information_share,
            'component_share': self.component_share,
            'information_leadership': self.information_leadership,
            'price_discovery_contribution': self.price_discovery_contribution,
            'quote_to_trade_ratio': self.quote_to_trade_ratio,
            'quote_update_frequency': self.quote_update_frequency,
            'effective_spread_ratio': self.effective_spread_ratio,
            'price_improvement_probability': self.price_improvement_probability,
            'price_efficiency': self.price_efficiency,
            'quote_stability': self.quote_stability,
            'information_asymmetry': self.information_asymmetry,
            'adverse_selection_component': self.adverse_selection_component,
            'return_autocorrelation': self.return_autocorrelation,
            'quote_autocorrelation': self.quote_autocorrelation,
            'variance_ratio': self.variance_ratio,
            'random_walk_deviation': self.random_walk_deviation
        }


class InformationShareModel(BaseMarketMicrostructureModel):
    """
    Hasbrouck Information Share Model.
    
    Measures the contribution of a trading venue or market to price discovery.
    Information share represents the proportion of the variance in the common
    efficient price that is explained by innovations in a particular market.
    
    Reference:
    Hasbrouck, J. (1995). "One security, many markets: Determining the 
    contributions to price discovery"
    
    Model:
        For cointegrated price series, the information share of market i is:
        IS_i = (ψ_i)² × Var(ε_i) / Var(r_t)
    
    where:
        ψ_i = loading coefficient
        ε_i = innovation in market i
        r_t = common efficient price
    
    Usage:
        model = InformationShareModel()
        is_value = model.calculate_information_share(tick_data)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize Information Share Model.
        
        Args:
            config: Configuration dictionary
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        self.estimation_window = self.config.get('estimation_window', 100)
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate information share metrics.
        
        Args:
            tick_data: Tick data
            **kwargs: Additional parameters
            
        Returns:
            MicrostructureMetrics with information share
        """
        start_time = time.perf_counter()
        
        info_share = self.calculate_information_share(tick_data)
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.debug(
                f"Information share calculation completed",
                execution_time_ms=round(execution_time, 3),
                information_share=round(info_share, 4)
            )
        
        return MicrostructureMetrics(
            information_share=info_share,
            price_discovery_share=info_share
        )
    
    def calculate_information_share(self, tick_data: TickData) -> float:
        """
        Calculate Hasbrouck information share.
        
        Simplified version using price variance decomposition.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Information share [0, 1]
        """
        if tick_data.n_ticks < 2:
            return 0.0
        
        # Calculate price changes (innovations)
        price_changes = np.diff(tick_data.price)
        
        # Use most recent window
        if len(price_changes) > self.estimation_window:
            price_changes = price_changes[-self.estimation_window:]
        
        # Calculate variance of innovations
        innovation_var = np.var(price_changes)
        
        # For single market, information share is based on 
        # how much of the variance is "signal" vs "noise"
        
        # Estimate signal variance using autocorrelation
        if len(price_changes) > 1:
            autocorr = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
            
            # Positive autocorr suggests noise (bid-ask bounce)
            # Negative autocorr suggests mean reversion
            # Signal variance ≈ total variance × (1 - |autocorr|)
            signal_ratio = max(0, 1 - abs(autocorr))
        else:
            signal_ratio = 0.5
        
        # Information share is the signal component
        information_share = signal_ratio
        
        return float(min(max(information_share, 0.0), 1.0))
    
    def calculate_component_share(self, tick_data: TickData) -> float:
        """
        Calculate Gonzalo-Granger component share.
        
        Alternative measure of price discovery contribution.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Component share [0, 1]
        """
        if tick_data.n_ticks < 2:
            return 0.0
        
        # Simplified: use relative contribution to price variance
        price_changes = np.diff(tick_data.price)
        
        if len(price_changes) > self.estimation_window:
            price_changes = price_changes[-self.estimation_window:]
        
        # Component share based on permanent vs temporary price movements
        # Estimate using long-horizon vs short-horizon variance
        if len(price_changes) >= 10:
            short_var = np.var(price_changes[:len(price_changes)//2])
            long_var = np.var(price_changes)
            
            if long_var > 0:
                component_share = short_var / long_var
            else:
                component_share = 0.5
        else:
            component_share = 0.5
        
        return float(min(max(component_share, 0.0), 1.0))


class MarketQualityAnalyzer(BaseMarketMicrostructureModel):
    """
    Market Quality Analyzer.
    
    Assesses overall market quality through multiple dimensions:
    - Price efficiency (variance ratio tests)
    - Quote quality (stability, update frequency)
    - Information asymmetry
    - Transaction costs
    
    Features:
    - Comprehensive quality scoring
    - Time-series analysis
    - Cross-market comparison
    - Anomaly detection
    
    Usage:
        analyzer = MarketQualityAnalyzer()
        quality = analyzer.analyze_quality(tick_data)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """Initialize Market Quality Analyzer."""
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        self.variance_ratio_lags = self.config.get('variance_ratio_lags', [2, 5, 10])
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate market quality metrics.
        
        Args:
            tick_data: Tick data
            **kwargs: Additional parameters
            
        Returns:
            MicrostructureMetrics with quality indicators
        """
        start_time = time.perf_counter()
        
        # Calculate various quality metrics
        vr = self.calculate_variance_ratio(tick_data)
        efficiency = self.calculate_price_efficiency(tick_data)
        quote_stability = self.calculate_quote_stability(tick_data)
        asymmetry = self.calculate_information_asymmetry(tick_data)
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.debug(
                f"Market quality analysis completed",
                execution_time_ms=round(execution_time, 3),
                variance_ratio=round(vr, 4),
                price_efficiency=round(efficiency, 4)
            )
        
        return MicrostructureMetrics()
    
    def analyze_quality(self, tick_data: TickData) -> PriceDiscoveryMetrics:
        """
        Comprehensive market quality analysis.
        
        Args:
            tick_data: Tick data
            
        Returns:
            PriceDiscoveryMetrics with all quality indicators
        """
        # Information share metrics
        info_model = InformationShareModel()
        info_share = info_model.calculate_information_share(tick_data)
        component_share = info_model.calculate_component_share(tick_data)
        
        # Information leadership (simplified as average of shares)
        info_leadership = (info_share + component_share) / 2
        price_discovery = info_share
        
        # Quote activity metrics
        quote_to_trade = self._calculate_quote_to_trade_ratio(tick_data)
        quote_frequency = self._calculate_quote_update_frequency(tick_data)
        effective_ratio = self._calculate_effective_spread_ratio(tick_data)
        price_improvement = self._calculate_price_improvement_probability(tick_data)
        
        # Market quality indicators
        efficiency = self.calculate_price_efficiency(tick_data)
        quote_stability = self.calculate_quote_stability(tick_data)
        asymmetry = self.calculate_information_asymmetry(tick_data)
        adverse_selection = self._estimate_adverse_selection(tick_data)
        
        # Autocorrelation measures
        return_autocorr = self._calculate_return_autocorrelation(tick_data)
        quote_autocorr = self._calculate_quote_autocorrelation(tick_data)
        
        # Variance ratio
        vr = self.calculate_variance_ratio(tick_data)
        rw_deviation = abs(vr - 1.0)  # Deviation from random walk
        
        return PriceDiscoveryMetrics(
            information_share=info_share,
            component_share=component_share,
            information_leadership=info_leadership,
            price_discovery_contribution=price_discovery,
            quote_to_trade_ratio=quote_to_trade,
            quote_update_frequency=quote_frequency,
            effective_spread_ratio=effective_ratio,
            price_improvement_probability=price_improvement,
            price_efficiency=efficiency,
            quote_stability=quote_stability,
            information_asymmetry=asymmetry,
            adverse_selection_component=adverse_selection,
            return_autocorrelation=return_autocorr,
            quote_autocorrelation=quote_autocorr,
            variance_ratio=vr,
            random_walk_deviation=rw_deviation
        )
    
    def calculate_variance_ratio(self, tick_data: TickData, lag: int = 2) -> float:
        """
        Calculate variance ratio test statistic.
        
        Tests if returns follow a random walk. Under random walk:
        Var(r_t + r_{t+1} + ... + r_{t+q-1}) / q = Var(r_t)
        
        Variance ratio = [Var(q-period return) / q] / Var(1-period return)
        
        VR = 1 under random walk
        VR < 1 indicates mean reversion
        VR > 1 indicates momentum
        
        Args:
            tick_data: Tick data
            lag: Number of periods for variance ratio
            
        Returns:
            Variance ratio
        """
        if tick_data.n_ticks < lag + 1:
            return 1.0
        
        # Calculate returns
        returns = np.diff(tick_data.price) / tick_data.price[:-1]
        
        if len(returns) < lag + 1:
            return 1.0
        
        # 1-period variance
        var_1 = np.var(returns)
        
        # q-period returns (overlapping)
        q_returns = []
        for i in range(len(returns) - lag + 1):
            q_return = np.sum(returns[i:i+lag])
            q_returns.append(q_return)
        
        if len(q_returns) > 0 and var_1 > 0:
            var_q = np.var(q_returns)
            vr = (var_q / lag) / var_1
        else:
            vr = 1.0
        
        return float(vr)
    
    def calculate_price_efficiency(self, tick_data: TickData) -> float:
        """
        Calculate price efficiency score.
        
        Higher values indicate more efficient prices (closer to random walk).
        
        Args:
            tick_data: Tick data
            
        Returns:
            Efficiency score [0, 1]
        """
        # Use variance ratio as proxy
        vr = self.calculate_variance_ratio(tick_data)
        
        # Convert to efficiency score
        # Efficiency = 1 when VR = 1 (perfect random walk)
        # Decreases as VR deviates from 1
        efficiency = 1.0 / (1.0 + abs(vr - 1.0))
        
        return float(efficiency)
    
    def calculate_quote_stability(self, tick_data: TickData) -> float:
        """
        Calculate quote stability measure.
        
        Measures how stable bid/ask quotes are over time.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Stability score [0, 1]
        """
        if tick_data.n_ticks < 2:
            return 0.0
        
        # Calculate quote changes
        bid_changes = np.abs(np.diff(tick_data.bid))
        ask_changes = np.abs(np.diff(tick_data.ask))
        
        # Normalize by average prices
        avg_bid = np.mean(tick_data.bid)
        avg_ask = np.mean(tick_data.ask)
        
        if avg_bid > 0 and avg_ask > 0:
            bid_volatility = np.mean(bid_changes) / avg_bid
            ask_volatility = np.mean(ask_changes) / avg_ask
            avg_volatility = (bid_volatility + ask_volatility) / 2
            
            # Stability is inverse of volatility
            stability = 1.0 / (1.0 + avg_volatility * 100)
        else:
            stability = 0.0
        
        return float(stability)
    
    def calculate_information_asymmetry(self, tick_data: TickData) -> float:
        """
        Calculate information asymmetry measure.
        
        Higher values indicate greater information asymmetry.
        
        Args:
            tick_data: Tick data
            
        Returns:
            Asymmetry measure [0, 1]
        """
        if tick_data.trade_direction is None or tick_data.n_ticks < 2:
            return 0.0
        
        # Use order flow imbalance as proxy for information asymmetry
        signed_volume = tick_data.volume * tick_data.trade_direction
        total_volume = np.sum(tick_data.volume)
        
        if total_volume > 0:
            imbalance = abs(np.sum(signed_volume)) / total_volume
        else:
            imbalance = 0.0
        
        # High imbalance suggests asymmetric information
        return float(imbalance)
    
    def _calculate_quote_to_trade_ratio(self, tick_data: TickData) -> float:
        """Calculate quote-to-trade ratio."""
        # Simplified: assume each tick represents a quote update
        # In reality, would need separate quote and trade data
        n_trades = tick_data.n_ticks
        n_quotes = n_trades  # Simplified assumption
        
        if n_trades > 0:
            ratio = n_quotes / n_trades
        else:
            ratio = 0.0
        
        return float(ratio)
    
    def _calculate_quote_update_frequency(self, tick_data: TickData) -> float:
        """Calculate quote update frequency (updates per second)."""
        if tick_data.n_ticks < 2:
            return 0.0
        
        time_span = (tick_data.timestamp[-1] - tick_data.timestamp[0]).total_seconds()
        
        if time_span > 0:
            frequency = tick_data.n_ticks / time_span
        else:
            frequency = 0.0
        
        return float(frequency)
    
    def _calculate_effective_spread_ratio(self, tick_data: TickData) -> float:
        """Calculate ratio of effective to quoted spread."""
        quoted_spread = np.mean(tick_data.spread)
        
        # Effective spread = 2 * |price - midpoint|
        effective_spreads = 2 * np.abs(tick_data.price - tick_data.midpoint)
        avg_effective = np.mean(effective_spreads)
        
        if quoted_spread > 0:
            ratio = avg_effective / quoted_spread
        else:
            ratio = 1.0
        
        return float(ratio)
    
    def _calculate_price_improvement_probability(self, tick_data: TickData) -> float:
        """Calculate probability of price improvement."""
        # Price improvement: trades inside the spread
        midpoints = tick_data.midpoint
        
        # Count trades at midpoint (simplified measure of improvement)
        at_midpoint = np.abs(tick_data.price - midpoints) < (tick_data.spread * 0.1)
        
        if tick_data.n_ticks > 0:
            prob = np.sum(at_midpoint) / tick_data.n_ticks
        else:
            prob = 0.0
        
        return float(prob)
    
    def _estimate_adverse_selection(self, tick_data: TickData) -> float:
        """Estimate adverse selection component of spread."""
        if tick_data.trade_direction is None or tick_data.n_ticks < 2:
            return 0.0
        
        # Adverse selection ≈ permanent price impact
        # Measured by correlation between trade direction and future price changes
        
        price_changes = np.diff(tick_data.price)
        directions = tick_data.trade_direction[:-1]
        
        if len(price_changes) > 1 and np.std(directions) > 0:
            corr = np.corrcoef(price_changes, directions)[0, 1]
            adverse_selection = abs(corr)
        else:
            adverse_selection = 0.0
        
        return float(adverse_selection)
    
    def _calculate_return_autocorrelation(self, tick_data: TickData) -> float:
        """Calculate first-order autocorrelation of returns."""
        if tick_data.n_ticks < 3:
            return 0.0
        
        returns = np.diff(tick_data.price) / tick_data.price[:-1]
        
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            autocorr = 0.0
        
        return float(autocorr)
    
    def _calculate_quote_autocorrelation(self, tick_data: TickData) -> float:
        """Calculate first-order autocorrelation of quote changes."""
        if tick_data.n_ticks < 3:
            return 0.0
        
        mid_changes = np.diff(tick_data.midpoint)
        
        if len(mid_changes) > 1:
            autocorr = np.corrcoef(mid_changes[:-1], mid_changes[1:])[0, 1]
        else:
            autocorr = 0.0
        
        return float(autocorr)


__all__ = [
    "InformationShareModel",
    "MarketQualityAnalyzer",
    "PriceDiscoveryMetrics",
]