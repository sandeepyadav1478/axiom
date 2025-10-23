"""
Merger Arbitrage Models
=======================

Comprehensive merger arbitrage framework for M&A deal trading:
- Deal spread analysis and annualized returns
- Implied probability of deal closure
- Expected return calculation
- Optimal hedge ratio determination
- Position sizing using Kelly criterion
- Risk-adjusted position management
- Deal event analysis

Performance target: <20ms for spread analysis, <10ms for position sizing
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import time

from axiom.models.ma.base_model import BaseMandAModel, MergerArbPosition
from axiom.models.base.base_model import ModelResult
from axiom.config.model_config import MandAConfig


@dataclass
class DealEvent:
    """M&A deal event for tracking."""
    event_type: str  # announcement, hsr_filing, shareholder_vote, regulatory_approval, completion
    event_date: datetime
    impact: float  # Expected spread impact
    probability: float  # Probability of successful outcome


class MergerArbitrageModel(BaseMandAModel):
    """
    Comprehensive merger arbitrage analysis model.
    
    Features:
    - Deal spread calculation and analysis
    - Annualized return calculation
    - Market-implied probability of closure
    - Expected return with downside scenarios
    - Optimal hedge ratio (delta-neutral positioning)
    - Kelly criterion position sizing
    - Risk metrics (VaR, max drawdown scenarios)
    - Deal event probability modeling
    - Break-even analysis
    
    Performance: <20ms for spread analysis, <10ms for position sizing
    """
    
    def __init__(
        self,
        config: Optional[MandAConfig] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize merger arbitrage model.
        
        Args:
            config: M&A configuration
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(
            config=config.__dict__ if config and hasattr(config, '__dict__') else (config or {}),
            enable_logging=enable_logging,
            enable_performance_tracking=enable_performance_tracking
        )
        
        # Set default config values
        self.config.setdefault('min_deal_spread_bps', 100)
        self.config.setdefault('max_position_size_pct', 0.10)
        self.config.setdefault('hedge_ratio_method', 'optimal')
        self.config.setdefault('default_close_probability', 0.85)
        self.config.setdefault('kelly_fraction', 0.25)
        self.config.setdefault('break_fee_pct', 0.03)
    
    def calculate(
        self,
        target_price: float,
        offer_price: float,
        acquirer_price: Optional[float] = None,
        days_to_close: int = 90,
        deal_type: str = "cash",  # cash, stock, mixed
        stock_exchange_ratio: Optional[float] = None,
        probability_of_close: Optional[float] = None,
        downside_price: Optional[float] = None,
        break_fee: Optional[float] = None,
        portfolio_value: float = 1000000,
        target_volatility: float = 0.30,
        acquirer_volatility: Optional[float] = None
    ) -> ModelResult[MergerArbPosition]:
        """
        Calculate comprehensive merger arbitrage position analysis.
        
        Args:
            target_price: Current target trading price
            offer_price: Offer price (cash or implied value)
            acquirer_price: Acquirer stock price (for stock deals)
            days_to_close: Expected days until deal closes
            deal_type: Type of deal (cash, stock, mixed)
            stock_exchange_ratio: Share exchange ratio (for stock deals)
            probability_of_close: Estimated probability deal closes
            downside_price: Target price if deal breaks
            break_fee: Termination fee (% or absolute)
            portfolio_value: Total portfolio value for position sizing
            target_volatility: Target stock volatility
            acquirer_volatility: Acquirer stock volatility
            
        Returns:
            ModelResult containing MergerArbPosition
        """
        start_time = time.perf_counter()
        
        # Validate inputs
        self.validate_inputs(
            target_price=target_price,
            offer_price=offer_price
        )
        
        # Use config defaults
        probability_of_close = probability_of_close or self.config['default_close_probability']
        
        # Calculate deal spread
        deal_spread = self._calculate_deal_spread(
            target_price, offer_price, deal_type,
            acquirer_price, stock_exchange_ratio
        )
        
        # Calculate annualized return
        annualized_return = self._calculate_annualized_return(
            deal_spread, days_to_close
        )
        
        # Calculate implied probability
        implied_probability = self._calculate_implied_probability(
            target_price, offer_price, downside_price or target_price * 0.80,
            break_fee or self.config['break_fee_pct'] * offer_price
        )
        
        # Calculate expected return
        expected_return = self._calculate_expected_return(
            deal_spread, probability_of_close,
            target_price, downside_price or target_price * 0.80
        )
        
        # Calculate break-even probability
        break_even_prob = self._calculate_breakeven_probability(
            target_price, offer_price, downside_price or target_price * 0.80
        )
        
        # Determine optimal hedge ratio
        if deal_type == "stock" and acquirer_price:
            hedge_ratio = self._calculate_hedge_ratio(
                stock_exchange_ratio or 1.0,
                target_volatility,
                acquirer_volatility or target_volatility * 0.8,
                self.config['hedge_ratio_method']
            )
        else:
            hedge_ratio = 0.0  # No hedge for cash deals
        
        # Calculate position size using Kelly criterion
        kelly_optimal_size = self._calculate_kelly_position(
            probability_of_close,
            deal_spread,
            (target_price - (downside_price or target_price * 0.80)) / target_price
        )
        
        # Adjust for Kelly fraction safety margin
        kelly_fraction = self.config['kelly_fraction']
        position_size = min(
            kelly_optimal_size * kelly_fraction,
            portfolio_value * self.config['max_position_size_pct']
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            target_price, offer_price, downside_price or target_price * 0.80,
            probability_of_close, position_size, target_volatility, days_to_close
        )
        
        # Position values
        target_position = position_size
        acquirer_hedge = position_size * hedge_ratio if deal_type == "stock" else 0.0
        
        # Create result
        position = MergerArbPosition(
            target_position=target_position,
            acquirer_hedge=acquirer_hedge,
            deal_spread=deal_spread,
            annualized_return=annualized_return,
            implied_probability=implied_probability,
            expected_return=expected_return,
            position_size=position_size,
            kelly_optimal_size=kelly_optimal_size,
            risk_metrics=risk_metrics,
            hedge_ratio=hedge_ratio,
            break_even_prob=break_even_prob,
            days_to_close=days_to_close
        )
        
        # Track performance
        execution_time_ms = self._track_performance("merger_arbitrage", start_time)
        
        # Create metadata
        metadata = self._create_metadata(execution_time_ms)
        
        return ModelResult(
            value=position,
            metadata=metadata,
            success=True
        )
    
    def _calculate_deal_spread(
        self,
        target_price: float,
        offer_price: float,
        deal_type: str,
        acquirer_price: Optional[float],
        exchange_ratio: Optional[float]
    ) -> float:
        """
        Calculate deal spread.
        
        Args:
            target_price: Current target price
            offer_price: Offer price
            deal_type: Type of deal
            acquirer_price: Acquirer price (for stock deals)
            exchange_ratio: Share exchange ratio
            
        Returns:
            Deal spread as percentage
        """
        if deal_type == "stock" and acquirer_price and exchange_ratio:
            # Stock deal: implied value - current price
            implied_value = acquirer_price * exchange_ratio
            spread = (implied_value - target_price) / target_price
        else:
            # Cash deal or mixed
            spread = (offer_price - target_price) / target_price
        
        return spread
    
    def _calculate_annualized_return(
        self,
        spread: float,
        days_to_close: int
    ) -> float:
        """
        Calculate annualized return from spread.
        
        Args:
            spread: Deal spread
            days_to_close: Days until expected close
            
        Returns:
            Annualized return
        """
        if days_to_close <= 0:
            return 0.0
        
        # Annualize the spread
        annualized = spread * (365 / days_to_close)
        
        return annualized
    
    def _calculate_implied_probability(
        self,
        target_price: float,
        offer_price: float,
        downside_price: float,
        break_fee: float
    ) -> float:
        """
        Calculate market-implied probability of deal closing.
        
        Uses risk-neutral valuation:
        Current Price = P_close * Offer + (1 - P_close) * (Downside + Break Fee)
        
        Args:
            target_price: Current trading price
            offer_price: Offer price
            downside_price: Price if deal breaks
            break_fee: Termination fee
            
        Returns:
            Implied probability
        """
        # Adjust downside for break fee
        downside_value = downside_price + break_fee
        
        # Solve for implied probability
        if offer_price > downside_value:
            implied_prob = (target_price - downside_value) / (offer_price - downside_value)
        else:
            implied_prob = 0.5  # Default if no spread
        
        # Bound between 0 and 1
        return max(0.0, min(1.0, implied_prob))
    
    def _calculate_expected_return(
        self,
        spread: float,
        probability: float,
        target_price: float,
        downside_price: float
    ) -> float:
        """
        Calculate probability-weighted expected return.
        
        Args:
            spread: Deal spread
            probability: Probability of close
            target_price: Current target price
            downside_price: Downside if deal breaks
            
        Returns:
            Expected return
        """
        # Upside scenario (deal closes)
        upside_return = spread
        
        # Downside scenario (deal breaks)
        downside_return = (downside_price - target_price) / target_price
        
        # Probability-weighted expected return
        expected = probability * upside_return + (1 - probability) * downside_return
        
        return expected
    
    def _calculate_breakeven_probability(
        self,
        target_price: float,
        offer_price: float,
        downside_price: float
    ) -> float:
        """
        Calculate minimum probability needed for positive expected return.
        
        Args:
            target_price: Current target price
            offer_price: Offer price
            downside_price: Downside price
            
        Returns:
            Break-even probability
        """
        upside = offer_price - target_price
        downside = target_price - downside_price
        
        if upside + downside > 0:
            breakeven = downside / (upside + downside)
        else:
            breakeven = 0.5
        
        return max(0.0, min(1.0, breakeven))
    
    def _calculate_hedge_ratio(
        self,
        exchange_ratio: float,
        target_volatility: float,
        acquirer_volatility: float,
        method: str
    ) -> float:
        """
        Calculate optimal hedge ratio for stock deals.
        
        Args:
            exchange_ratio: Share exchange ratio
            target_volatility: Target stock volatility
            acquirer_volatility: Acquirer stock volatility
            method: Hedging method
            
        Returns:
            Optimal hedge ratio
        """
        if method == "static":
            # Simple exchange ratio hedge
            return exchange_ratio
        
        elif method == "optimal":
            # Variance-minimizing hedge ratio
            # Assumes correlation of 0.5 for M&A stocks
            correlation = 0.5
            optimal_ratio = exchange_ratio * (target_volatility / acquirer_volatility) * correlation
            return optimal_ratio
        
        else:  # "dynamic" or other
            # Dynamic hedge considers volatility ratio
            vol_ratio = target_volatility / acquirer_volatility if acquirer_volatility > 0 else 1.0
            return exchange_ratio * vol_ratio
    
    def _calculate_kelly_position(
        self,
        probability: float,
        expected_gain: float,
        expected_loss: float
    ) -> float:
        """
        Calculate Kelly criterion optimal position size.
        
        Kelly = (p * b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        
        Args:
            probability: Probability of winning
            expected_gain: Expected gain if win
            expected_loss: Expected loss if lose
            
        Returns:
            Kelly optimal position as fraction of portfolio
        """
        if expected_loss <= 0 or expected_gain <= 0:
            return 0.0
        
        win_prob = probability
        loss_prob = 1 - probability
        win_loss_ratio = expected_gain / expected_loss
        
        # Kelly formula
        kelly = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        
        # Bound to reasonable range
        return max(0.0, min(1.0, kelly))
    
    def _calculate_risk_metrics(
        self,
        target_price: float,
        offer_price: float,
        downside_price: float,
        probability: float,
        position_size: float,
        volatility: float,
        days_to_close: int
    ) -> Dict[str, float]:
        """
        Calculate risk metrics for the position.
        
        Args:
            target_price: Current target price
            offer_price: Offer price
            downside_price: Downside price
            probability: Close probability
            position_size: Position size
            volatility: Stock volatility
            days_to_close: Days to close
            
        Returns:
            Dictionary of risk metrics
        """
        # Maximum gain and loss
        max_gain = (offer_price - target_price) * position_size / target_price
        max_loss = (target_price - downside_price) * position_size / target_price
        
        # Value at Risk (95% confidence, deal breaks scenario)
        var_95 = max_loss * (1 - probability) * 1.65  # 1.65 = z-score for 95%
        
        # Expected shortfall (CVaR)
        cvar_95 = max_loss * (1 - probability) * 1.96  # Worst 5% scenarios
        
        # Sharpe ratio (annualized, simplified)
        expected_return_annual = self._calculate_annualized_return(
            (offer_price - target_price) / target_price,
            days_to_close
        )
        return_volatility = volatility * np.sqrt(252 / days_to_close) if days_to_close > 0 else volatility
        sharpe_ratio = (expected_return_annual * probability) / return_volatility if return_volatility > 0 else 0.0
        
        # Win/Loss ratio
        win_loss_ratio = max_gain / max_loss if max_loss > 0 else 999
        
        return {
            'max_gain': max_gain,
            'max_loss': max_loss,
            'value_at_risk_95': var_95,
            'expected_shortfall_95': cvar_95,
            'sharpe_ratio': sharpe_ratio,
            'win_loss_ratio': win_loss_ratio,
            'position_volatility': volatility * (position_size / target_price)
        }
    
    def calculate_value(
        self,
        target_price: float,
        offer_price: float,
        **kwargs
    ) -> float:
        """
        Calculate deal spread.
        
        Args:
            target_price: Current target price
            offer_price: Offer price
            **kwargs: Additional parameters
            
        Returns:
            Deal spread
        """
        result = self.calculate(target_price, offer_price, **kwargs)
        return result.value.deal_spread
    
    def analyze_deal_events(
        self,
        target_price: float,
        offer_price: float,
        events: List[DealEvent],
        current_date: datetime
    ) -> Dict[str, Any]:
        """
        Analyze impact of deal events on probability and returns.
        
        Args:
            target_price: Current target price
            offer_price: Offer price
            events: List of deal events
            current_date: Current date
            
        Returns:
            Event analysis results
        """
        # Sort events by date
        future_events = [e for e in events if e.event_date > current_date]
        future_events.sort(key=lambda x: x.event_date)
        
        # Calculate cumulative probability
        cumulative_prob = 1.0
        for event in future_events:
            cumulative_prob *= event.probability
        
        # Calculate time-weighted spread evolution
        spread_evolution = []
        remaining_days = (future_events[-1].event_date - current_date).days if future_events else 0
        
        for event in future_events:
            days_to_event = (event.event_date - current_date).days
            expected_spread = (offer_price - target_price) / target_price * cumulative_prob
            spread_evolution.append({
                'event': event.event_type,
                'date': event.event_date.isoformat(),
                'days_away': days_to_event,
                'expected_spread': expected_spread,
                'cumulative_probability': cumulative_prob
            })
        
        return {
            'final_probability': cumulative_prob,
            'event_count': len(future_events),
            'days_to_completion': remaining_days,
            'spread_evolution': spread_evolution
        }
    
    def calculate_collar_value(
        self,
        target_price: float,
        collar_floor: float,
        collar_cap: float,
        acquirer_price: float,
        exchange_ratio: float,
        acquirer_volatility: float,
        days_to_close: int
    ) -> Dict[str, float]:
        """
        Value collar provisions in stock deals.
        
        Collars protect both parties from excessive stock price moves.
        
        Args:
            target_price: Current target price
            collar_floor: Minimum exchange ratio price
            collar_cap: Maximum exchange ratio price
            acquirer_price: Current acquirer price
            exchange_ratio: Base exchange ratio
            acquirer_volatility: Acquirer stock volatility
            days_to_close: Days until close
            
        Returns:
            Collar valuation results
        """
        # Calculate probability of hitting collar bounds
        time_factor = np.sqrt(days_to_close / 365)
        z_score_floor = (np.log(collar_floor / acquirer_price)) / (acquirer_volatility * time_factor)
        z_score_cap = (np.log(collar_cap / acquirer_price)) / (acquirer_volatility * time_factor)
        
        from scipy.stats import norm
        prob_below_floor = norm.cdf(z_score_floor)
        prob_above_cap = 1 - norm.cdf(z_score_cap)
        prob_in_collar = 1 - prob_below_floor - prob_above_cap
        
        # Expected value under different scenarios
        expected_value = (
            prob_in_collar * (acquirer_price * exchange_ratio) +
            prob_below_floor * (collar_floor * exchange_ratio) +
            prob_above_cap * (collar_cap * exchange_ratio)
        )
        
        return {
            'prob_below_floor': prob_below_floor,
            'prob_above_cap': prob_above_cap,
            'prob_in_collar': prob_in_collar,
            'expected_value': expected_value,
            'collar_protection_value': expected_value - (acquirer_price * exchange_ratio)
        }
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate merger arbitrage inputs.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid
        """
        # Call parent validation
        super().validate_inputs(**kwargs)
        
        # Additional merger arb-specific validation
        if 'target_price' in kwargs and kwargs['target_price'] <= 0:
            raise ValueError("Target price must be positive")
        
        if 'offer_price' in kwargs and kwargs['offer_price'] <= 0:
            raise ValueError("Offer price must be positive")
        
        if 'days_to_close' in kwargs and kwargs['days_to_close'] <= 0:
            raise ValueError("Days to close must be positive")
        
        return True


__all__ = [
    "MergerArbitrageModel",
    "DealEvent",
]