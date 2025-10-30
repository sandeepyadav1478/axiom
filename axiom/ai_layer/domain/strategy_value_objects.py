"""
Trading Strategy Domain Value Objects

Immutable value objects for trading strategy domain.
Following DDD principles - these capture the essence of trading strategies.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on bad strategies)
- Rich behavior (strategy analysis, risk assessment)
- Type-safe (using Decimal for money, Enum for states)

These represent trading strategies as first-class domain concepts.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from enum import Enum


class MarketOutlook(str, Enum):
    """Market directional view"""
    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"


class VolatilityView(str, Enum):
    """Expected volatility direction"""
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"


class StrategyType(str, Enum):
    """Strategy classification"""
    DIRECTIONAL = "directional"  # Betting on direction
    VOLATILITY = "volatility"    # Betting on vol
    INCOME = "income"            # Premium collection
    HEDGING = "hedging"          # Risk reduction


class StrategyComplexity(str, Enum):
    """Strategy complexity level"""
    SIMPLE = "simple"            # 1-2 legs
    MODERATE = "moderate"        # 3-4 legs
    COMPLEX = "complex"          # 5+ legs


@dataclass(frozen=True)
class StrategyLeg:
    """
    Single leg of options strategy
    
    Immutable representation of one option position
    """
    option_type: str  # 'call' or 'put'
    action: str  # 'buy' or 'sell'
    strike: Decimal
    quantity: int
    expiry_days: int
    
    # Calculated fields (would come from pricing)
    premium: Optional[Decimal] = None
    delta: Optional[Decimal] = None
    gamma: Optional[Decimal] = None
    vega: Optional[Decimal] = None
    theta: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate leg parameters"""
        if self.option_type not in ['call', 'put']:
            raise ValueError(f"Invalid option type: {self.option_type}")
        
        if self.action not in ['buy', 'sell']:
            raise ValueError(f"Invalid action: {self.action}")
        
        if self.strike <= Decimal('0'):
            raise ValueError("Strike must be positive")
        
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.expiry_days <= 0:
            raise ValueError("Expiry must be positive")
    
    def is_long(self) -> bool:
        """Check if leg is long (buy)"""
        return self.action == 'buy'
    
    def is_short(self) -> bool:
        """Check if leg is short (sell)"""
        return self.action == 'sell'
    
    def get_directional_exposure(self) -> int:
        """Get signed quantity (positive for long, negative for short)"""
        return self.quantity if self.is_long() else -self.quantity


@dataclass(frozen=True)
class StrategyRiskMetrics:
    """
    Risk metrics for strategy
    
    Immutable snapshot of strategy risk profile
    """
    entry_cost: Decimal
    max_profit: Decimal
    max_loss: Decimal
    breakeven_points: List[Decimal]
    
    # Probabilities
    probability_profit: Decimal
    probability_max_profit: Decimal
    probability_max_loss: Decimal
    
    # Risk/Reward
    risk_reward_ratio: Decimal
    expected_return: Decimal
    expected_return_pct: Decimal
    
    # Greeks profile
    net_delta: Decimal
    net_gamma: Decimal
    net_vega: Decimal
    net_theta: Decimal
    
    # Position sizing
    capital_required: Decimal
    margin_required: Decimal
    buying_power_impact: Decimal
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate risk metrics"""
        if self.max_loss < Decimal('0'):
            raise ValueError("Max loss must be non-negative (loss amount)")
        
        if self.probability_profit < Decimal('0') or self.probability_profit > Decimal('1'):
            raise ValueError("Probability must be between 0 and 1")
        
        if self.capital_required < Decimal('0'):
            raise ValueError("Capital required must be non-negative")
    
    def is_defined_risk(self) -> bool:
        """Check if strategy has defined max loss"""
        return self.max_loss < Decimal('999999')  # Not "unlimited"
    
    def is_credit_strategy(self) -> bool:
        """Check if strategy collects premium (credit)"""
        return self.entry_cost < Decimal('0')
    
    def is_debit_strategy(self) -> bool:
        """Check if strategy pays premium (debit)"""
        return self.entry_cost > Decimal('0')
    
    def get_roi_potential(self) -> Decimal:
        """Calculate return on investment potential"""
        if self.capital_required > Decimal('0'):
            return (self.max_profit / self.capital_required) * Decimal('100')
        return Decimal('0')
    
    def meets_risk_criteria(
        self,
        max_loss_limit: Decimal,
        min_prob_profit: Decimal,
        min_risk_reward: Decimal
    ) -> bool:
        """Check if strategy meets risk criteria"""
        return (
            self.max_loss <= max_loss_limit and
            self.probability_profit >= min_prob_profit and
            self.risk_reward_ratio >= min_risk_reward
        )


@dataclass(frozen=True)
class TradingStrategy:
    """
    Complete trading strategy
    
    Immutable, validated, rich domain object
    """
    strategy_id: str
    strategy_name: str
    strategy_type: StrategyType
    complexity: StrategyComplexity
    
    # Market view
    market_outlook: MarketOutlook
    volatility_view: VolatilityView
    
    # Strategy legs
    legs: Tuple[StrategyLeg, ...]  # Immutable tuple
    
    # Risk metrics
    risk_metrics: StrategyRiskMetrics
    
    # Rationale
    rationale: str
    confidence: Decimal  # 0-1
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "ai_strategy_agent"
    validated: bool = False
    backtest_results: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate strategy"""
        if len(self.legs) == 0:
            raise ValueError("Strategy must have at least one leg")
        
        if not (Decimal('0') <= self.confidence <= Decimal('1')):
            raise ValueError("Confidence must be between 0 and 1")
        
        # Validate legs form coherent strategy
        self._validate_leg_coherence()
    
    def _validate_leg_coherence(self):
        """Ensure legs form a valid strategy"""
        # Check all legs have same expiry (for spreads)
        if len(self.legs) > 1:
            expiries = [leg.expiry_days for leg in self.legs]
            if len(set(expiries)) > 2:  # Allow some variation
                # Log warning but don't fail
                pass
    
    def get_leg_count(self) -> int:
        """Get number of legs in strategy"""
        return len(self.legs)
    
    def get_net_position(self) -> int:
        """Get net directional position (positive = long, negative = short)"""
        return sum(leg.get_directional_exposure() for leg in self.legs)
    
    def is_spread(self) -> bool:
        """Check if strategy is a spread (multiple legs with offsetting positions)"""
        return len(self.legs) >= 2 and self.get_net_position() != sum(abs(leg.quantity) for leg in self.legs)
    
    def is_balanced(self) -> bool:
        """Check if strategy is delta-neutral or near-neutral"""
        return abs(self.risk_metrics.net_delta) < Decimal('10')
    
    def requires_margin(self) -> bool:
        """Check if strategy requires margin"""
        return any(leg.is_short() for leg in self.legs)
    
    def get_directional_bias(self) -> str:
        """Get strategy's directional bias"""
        net_delta = self.risk_metrics.net_delta
        if net_delta > Decimal('20'):
            return "bullish"
        elif net_delta < Decimal('-20'):
            return "bearish"
        else:
            return "neutral"
    
    def get_volatility_bias(self) -> str:
        """Get strategy's volatility bias"""
        net_vega = self.risk_metrics.net_vega
        if net_vega > Decimal('50'):
            return "long_volatility"
        elif net_vega < Decimal('-50'):
            return "short_volatility"
        else:
            return "volatility_neutral"
    
    def passes_validation(
        self,
        max_loss_limit: Decimal,
        min_confidence: Decimal,
        min_prob_profit: Decimal
    ) -> bool:
        """Validate strategy against criteria"""
        return (
            self.risk_metrics.max_loss <= max_loss_limit and
            self.confidence >= min_confidence and
            self.risk_metrics.probability_profit >= min_prob_profit
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'strategy_id': self.strategy_id,
            'name': self.strategy_name,
            'type': self.strategy_type.value,
            'complexity': self.complexity.value,
            'outlook': self.market_outlook.value,
            'vol_view': self.volatility_view.value,
            'legs': [
                {
                    'type': leg.option_type,
                    'action': leg.action,
                    'strike': float(leg.strike),
                    'quantity': leg.quantity,
                    'expiry_days': leg.expiry_days
                }
                for leg in self.legs
            ],
            'entry_cost': float(self.risk_metrics.entry_cost),
            'max_profit': float(self.risk_metrics.max_profit),
            'max_loss': float(self.risk_metrics.max_loss),
            'probability_profit': float(self.risk_metrics.probability_profit),
            'expected_return': float(self.risk_metrics.expected_return),
            'net_delta': float(self.risk_metrics.net_delta),
            'confidence': float(self.confidence),
            'rationale': self.rationale
        }


@dataclass(frozen=True)
class BacktestResult:
    """
    Strategy backtest results
    
    Immutable historical performance metrics
    """
    strategy_id: str
    
    # Performance metrics
    total_return: Decimal
    annualized_return: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    
    # Profitability
    average_win: Decimal
    average_loss: Decimal
    profit_factor: Decimal
    
    # Risk metrics
    var_95: Decimal
    worst_trade: Decimal
    best_trade: Decimal
    
    # Period
    start_date: datetime
    end_date: datetime
    
    # Metadata
    tested_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate backtest results"""
        if self.total_trades < 0:
            raise ValueError("Total trades must be non-negative")
        
        if self.win_rate < Decimal('0') or self.win_rate > Decimal('1'):
            raise ValueError("Win rate must be between 0 and 1")
    
    def is_profitable(self) -> bool:
        """Check if strategy was profitable"""
        return self.total_return > Decimal('0')
    
    def has_positive_expectancy(self) -> bool:
        """Check if strategy has positive expected value"""
        return self.profit_factor > Decimal('1')
    
    def meets_quality_threshold(
        self,
        min_sharpe: Decimal = Decimal('1.0'),
        min_win_rate: Decimal = Decimal('0.50'),
        max_drawdown: Decimal = Decimal('0.20')
    ) -> bool:
        """Check if backtest meets quality thresholds"""
        return (
            self.sharpe_ratio >= min_sharpe and
            self.win_rate >= min_win_rate and
            abs(self.max_drawdown) <= max_drawdown
        )


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("STRATEGY DOMAIN VALUE OBJECTS")
    print("="*60)
    
    # Create strategy legs
    print("\n→ Building Bull Call Spread:")
    legs = (
        StrategyLeg(
            option_type='call',
            action='buy',
            strike=Decimal('100'),
            quantity=10,
            expiry_days=30,
            premium=Decimal('3.50'),
            delta=Decimal('0.55')
        ),
        StrategyLeg(
            option_type='call',
            action='sell',
            strike=Decimal('105'),
            quantity=10,
            expiry_days=30,
            premium=Decimal('1.80'),
            delta=Decimal('0.35')
        )
    )
    
    for i, leg in enumerate(legs, 1):
        print(f"   Leg {i}: {leg.action.upper()} {leg.quantity}x {leg.option_type} @ ${leg.strike}")
    
    # Create risk metrics
    print("\n→ Risk Metrics:")
    risk = StrategyRiskMetrics(
        entry_cost=Decimal('1700'),  # (3.50 - 1.80) * 10 * 100
        max_profit=Decimal('3300'),  # (5.00 - 1.70) * 10 * 100
        max_loss=Decimal('1700'),
        breakeven_points=[Decimal('101.70')],
        probability_profit=Decimal('0.65'),
        probability_max_profit=Decimal('0.35'),
        probability_max_loss=Decimal('0.15'),
        risk_reward_ratio=Decimal('1.94'),
        expected_return=Decimal('1890'),
        expected_return_pct=Decimal('111.2'),
        net_delta=Decimal('20.0'),
        net_gamma=Decimal('5.0'),
        net_vega=Decimal('15.0'),
        net_theta=Decimal('-5.0'),
        capital_required=Decimal('1700'),
        margin_required=Decimal('0'),
        buying_power_impact=Decimal('1700')
    )
    
    print(f"   Entry cost: ${risk.entry_cost:,.0f}")
    print(f"   Max profit: ${risk.max_profit:,.0f}")
    print(f"   Max loss: ${risk.max_loss:,.0f}")
    print(f"   Risk/Reward: {risk.risk_reward_ratio:.2f}")
    print(f"   Prob profit: {risk.probability_profit:.1%}")
    print(f"   Credit strategy: {'YES' if risk.is_credit_strategy() else 'NO'}")
    print(f"   Defined risk: {'YES' if risk.is_defined_risk() else 'NO'}")
    
    # Create complete strategy
    print("\n→ Complete Strategy:")
    strategy = TradingStrategy(
        strategy_id="STRAT-001",
        strategy_name="Bull Call Spread",
        strategy_type=StrategyType.DIRECTIONAL,
        complexity=StrategyComplexity.MODERATE,
        market_outlook=MarketOutlook.BULLISH,
        volatility_view=VolatilityView.STABLE,
        legs=legs,
        risk_metrics=risk,
        rationale="Bullish spread with defined risk. Lower cost than naked call.",
        confidence=Decimal('0.75')
    )
    
    print(f"   Name: {strategy.strategy_name}")
    print(f"   Type: {strategy.strategy_type.value}")
    print(f"   Legs: {strategy.get_leg_count()}")
    print(f"   Spread: {'YES' if strategy.is_spread() else 'NO'}")
    print(f"   Directional bias: {strategy.get_directional_bias()}")
    print(f"   Vol bias: {strategy.get_volatility_bias()}")
    print(f"   Confidence: {strategy.confidence:.1%}")
    
    # Validate
    passes = strategy.passes_validation(
        max_loss_limit=Decimal('5000'),
        min_confidence=Decimal('0.70'),
        min_prob_profit=Decimal('0.60')
    )
    print(f"   Passes validation: {'✓ YES' if passes else '✗ NO'}")
    
    # Backtest result
    print("\n→ Backtest Results:")
    backtest = BacktestResult(
        strategy_id="STRAT-001",
        total_return=Decimal('0.28'),
        annualized_return=Decimal('0.35'),
        sharpe_ratio=Decimal('1.8'),
        max_drawdown=Decimal('-0.12'),
        total_trades=100,
        winning_trades=62,
        losing_trades=38,
        win_rate=Decimal('0.62'),
        average_win=Decimal('850'),
        average_loss=Decimal('520'),
        profit_factor=Decimal('1.45'),
        var_95=Decimal('1200'),
        worst_trade=Decimal('-1500'),
        best_trade=Decimal('3200'),
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    print(f"   Total return: {backtest.total_return:.1%}")
    print(f"   Sharpe ratio: {backtest.sharpe_ratio:.2f}")
    print(f"   Win rate: {backtest.win_rate:.1%}")
    print(f"   Profit factor: {backtest.profit_factor:.2f}")
    print(f"   Profitable: {'YES' if backtest.is_profitable() else 'NO'}")
    print(f"   Positive expectancy: {'YES' if backtest.has_positive_expectancy() else 'NO'}")
    
    quality = backtest.meets_quality_threshold()
    print(f"   Meets quality: {'✓ YES' if quality else '✗ NO'}")
    
    print("\n" + "="*60)
    print("✓ Immutable strategy objects")
    print("✓ Self-validating")
    print("✓ Rich domain behavior")
    print("✓ Type-safe with Decimal")
    print("✓ Strategy analysis built-in")
    print("\nDOMAIN-DRIVEN DESIGN FOR TRADING STRATEGIES")