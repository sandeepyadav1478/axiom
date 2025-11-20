"""
Portfolio Hedging Domain Value Objects

Immutable value objects for portfolio hedging domain.
Following DDD principles - these capture the essence of hedging strategies.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on bad hedges)
- Rich behavior (hedge effectiveness analysis, cost optimization)
- Type-safe (using Decimal for precision, Enum for states)

These represent hedging operations as first-class domain concepts.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from enum import Enum


class HedgeType(str, Enum):
    """Type of hedge"""
    DELTA = "delta"  # Delta hedging
    GAMMA = "gamma"  # Gamma hedging
    VEGA = "vega"  # Vega hedging
    MULTI_GREEK = "multi_greek"  # Multiple Greeks


class HedgeUrgency(str, Enum):
    """Hedge urgency level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HedgeStrategy(str, Enum):
    """Hedging strategy"""
    STATIC = "static"  # Fixed hedge ratio
    DYNAMIC = "dynamic"  # Continuous rebalancing
    THRESHOLD = "threshold"  # Hedge when exceeds threshold
    TIME_BASED = "time_based"  # Periodic rebalancing
    DRL = "drl"  # Deep RL optimized


@dataclass(frozen=True)
class PortfolioGreeksSnapshot:
    """
    Greeks snapshot for hedging decisions
    
    Immutable portfolio Greeks at point in time
    """
    total_delta: Decimal
    total_gamma: Decimal
    total_vega: Decimal
    total_theta: Decimal
    total_rho: Decimal
    
    # Market data
    spot_price: Decimal
    volatility: Decimal
    risk_free_rate: Decimal
    
    # Position data
    position_count: int
    notional_exposure: Decimal
    current_hedge_position: Decimal  # Current underlying hedge (shares)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate snapshot"""
        if self.position_count < 0:
            raise ValueError("Position count cannot be negative")
        
        if self.spot_price <= Decimal('0'):
            raise ValueError("Spot price must be positive")
        
        if self.volatility <= Decimal('0'):
            raise ValueError("Volatility must be positive")
    
    def get_net_delta(self) -> Decimal:
        """Get net delta including current hedge"""
        return self.total_delta + self.current_hedge_position
    
    def is_delta_neutral(self, tolerance: Decimal = Decimal('10')) -> bool:
        """Check if portfolio is delta neutral within tolerance"""
        return abs(self.get_net_delta()) <= tolerance
    
    def get_delta_risk_value(self) -> Decimal:
        """Calculate dollar risk from delta exposure"""
        return abs(self.total_delta) * self.spot_price * self.volatility
    
    def get_gamma_risk_value(self) -> Decimal:
        """Calculate dollar risk from gamma exposure"""
        # Simplified gamma risk calculation
        return abs(self.total_gamma) * self.spot_price * self.spot_price * (self.volatility ** Decimal('2'))


@dataclass(frozen=True)
class HedgeRecommendation:
    """
    Hedge recommendation from DRL agent
    
    Immutable recommendation with rationale
    """
    hedge_id: str
    hedge_type: HedgeType
    
    # Hedge specification
    hedge_quantity: Decimal  # Shares to buy/sell (+ buy, - sell)
    hedge_instrument: str  # 'underlying', 'future', 'option'
    
    # Expected outcomes
    expected_delta_after: Decimal
    expected_gamma_after: Decimal
    expected_vega_after: Decimal
    
    # Cost analysis
    transaction_cost: Decimal
    slippage_cost: Decimal
    total_cost: Decimal
    cost_per_delta_hedged: Decimal
    
    # Quality metrics
    urgency: HedgeUrgency
    confidence: Decimal  # 0-1
    effectiveness_score: Decimal  # 0-1, how well it hedges
    
    # Analysis
    rationale: str
    risk_reduction_pct: Decimal
    cost_benefit_ratio: Decimal  # Risk reduced / Cost
    
    # Strategy
    strategy_used: HedgeStrategy
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate hedge recommendation"""
        if not (Decimal('0') <= self.confidence <= Decimal('1')):
            raise ValueError("Confidence must be between 0 and 1")
        
        if not (Decimal('0') <= self.effectiveness_score <= Decimal('1')):
            raise ValueError("Effectiveness score must be between 0 and 1")
        
        if self.total_cost < Decimal('0'):
            raise ValueError("Total cost must be non-negative")
    
    def is_worth_executing(
        self,
        min_effectiveness: Decimal = Decimal('0.70'),
        max_cost: Decimal = Decimal('10000'),
        min_cost_benefit: Decimal = Decimal('2.0')
    ) -> bool:
        """Check if hedge is worth executing"""
        return (
            self.effectiveness_score >= min_effectiveness and
            self.total_cost <= max_cost and
            self.cost_benefit_ratio >= min_cost_benefit
        )
    
    def is_critical(self) -> bool:
        """Check if hedge is critical (must execute immediately)"""
        return self.urgency == HedgeUrgency.CRITICAL
    
    def get_net_delta_reduction(self, current_delta: Decimal) -> Decimal:
        """Calculate delta reduction from hedge"""
        return abs(current_delta) - abs(self.expected_delta_after)


@dataclass(frozen=True)
class HedgeExecution:
    """
    Executed hedge record
    
    Immutable execution record for audit trail
    """
    hedge_id: str
    recommendation: HedgeRecommendation
    
    # Actual execution
    actual_quantity: Decimal
    actual_price: Decimal
    actual_cost: Decimal
    
    # Results
    actual_delta_after: Decimal
    actual_gamma_after: Decimal
    delta_error: Decimal  # Difference from expected
    
    # Execution details
    venue: str
    execution_time_ms: Decimal
    
    # Quality
    slippage_bps: Decimal
    execution_quality_score: Decimal  # 0-1
    
    # Status
    fully_executed: bool
    
    # Optional fields
    order_id: Optional[str] = None
    
    # Metadata
    executed_at: datetime = field(default_factory=datetime.utcnow)
    executed_by: str = "hedging_agent"
    
    def __post_init__(self):
        """Validate execution"""
        if self.actual_price <= Decimal('0'):
            raise ValueError("Price must be positive")
        
        if self.actual_cost < Decimal('0'):
            raise ValueError("Cost must be non-negative")
    
    def get_cost_variance(self) -> Decimal:
        """Calculate variance between expected and actual cost"""
        return abs(self.actual_cost - self.recommendation.total_cost)
    
    def get_cost_variance_pct(self) -> Decimal:
        """Calculate cost variance as percentage"""
        if self.recommendation.total_cost > Decimal('0'):
            return (self.get_cost_variance() / self.recommendation.total_cost) * Decimal('100')
        return Decimal('0')
    
    def was_effective(self, tolerance_pct: Decimal = Decimal('10')) -> bool:
        """Check if hedge was effective within tolerance"""
        error_pct = abs(self.delta_error / self.recommendation.expected_delta_after * Decimal('100')) if self.recommendation.expected_delta_after != Decimal('0') else Decimal('0')
        return error_pct <= tolerance_pct


@dataclass(frozen=True)
class HedgingPolicy:
    """
    Hedging policy configuration
    
    Immutable hedging rules and thresholds
    """
    # Target exposures
    target_delta: Decimal
    target_gamma: Optional[Decimal] = None
    target_vega: Optional[Decimal] = None
    
    # Thresholds for rebalancing
    delta_threshold: Decimal = Decimal('50')  # Hedge when delta exceeds this
    gamma_threshold: Optional[Decimal] = None
    
    # Cost constraints
    max_hedge_cost: Decimal = Decimal('10000')
    min_cost_benefit_ratio: Decimal = Decimal('2.0')  # Min risk reduced per $ spent
    
    # Rebalancing frequency
    max_time_between_hedges_hours: Decimal = Decimal('4')
    min_time_between_hedges_minutes: Decimal = Decimal('5')
    
    # Strategy
    strategy: HedgeStrategy = HedgeStrategy.DYNAMIC
    
    def __post_init__(self):
        """Validate policy"""
        if self.delta_threshold < Decimal('0'):
            raise ValueError("Delta threshold must be non-negative")
        
        if self.max_hedge_cost <= Decimal('0'):
            raise ValueError("Max hedge cost must be positive")
        
        if self.min_cost_benefit_ratio <= Decimal('0'):
            raise ValueError("Min cost-benefit ratio must be positive")
    
    def requires_hedge(
        self,
        current_delta: Decimal,
        time_since_last_hedge_minutes: Decimal
    ) -> bool:
        """Check if portfolio requires hedging"""
        # Check delta threshold
        if abs(current_delta) >= self.delta_threshold:
            return True
        
        # Check time-based
        if self.strategy == HedgeStrategy.TIME_BASED:
            if time_since_last_hedge_minutes >= self.max_time_between_hedges_hours * Decimal('60'):
                return True
        
        return False
    
    def can_hedge_now(self, time_since_last_hedge_minutes: Decimal) -> bool:
        """Check if enough time has passed to hedge again"""
        return time_since_last_hedge_minutes >= self.min_time_between_hedges_minutes


@dataclass(frozen=True)
class HedgingStatistics:
    """
    Aggregated hedging statistics
    
    Immutable performance metrics
    """
    total_hedges: int
    successful_hedges: int
    failed_hedges: int
    
    # Cost metrics
    total_hedge_cost: Decimal
    average_hedge_cost: Decimal
    total_slippage_cost: Decimal
    
    # Effectiveness
    average_delta_error: Decimal
    average_effectiveness_score: Decimal
    delta_neutral_time_pct: Decimal  # % of time delta neutral
    
    # Risk reduction
    total_risk_reduced: Decimal
    average_cost_benefit_ratio: Decimal
    
    # Time period
    start_time: datetime
    end_time: datetime
    
    def __post_init__(self):
        """Validate statistics"""
        if self.total_hedges < 0:
            raise ValueError("Total hedges must be non-negative")
    
    def get_success_rate(self) -> Decimal:
        """Calculate hedge success rate"""
        if self.total_hedges > 0:
            return Decimal(str(self.successful_hedges)) / Decimal(str(self.total_hedges))
        return Decimal('0')
    
    def get_average_cost_per_hedge(self) -> Decimal:
        """Calculate average cost per hedge"""
        if self.total_hedges > 0:
            return self.total_hedge_cost / Decimal(str(self.total_hedges))
        return Decimal('0')
    
    def is_cost_effective(
        self,
        min_cost_benefit: Decimal = Decimal('2.0'),
        max_avg_cost: Decimal = Decimal('500')
    ) -> bool:
        """Check if hedging is cost effective"""
        return (
            self.average_cost_benefit_ratio >= min_cost_benefit and
            self.average_hedge_cost <= max_avg_cost
        )


# Example usage
if __name__ == "__main__":
    from axiom.ai_layer.infrastructure.observability import Logger
    
    logger = Logger("hedging_domain_test")
    
    logger.info("test_starting", test="HEDGING DOMAIN VALUE OBJECTS")
    
    # Create portfolio Greeks snapshot
    logger.info("creating_greeks_snapshot")
    
    greeks = PortfolioGreeksSnapshot(
        total_delta=Decimal('250.0'),
        total_gamma=Decimal('15.0'),
        total_vega=Decimal('3500.0'),
        total_theta=Decimal('-150.0'),
        total_rho=Decimal('500.0'),
        spot_price=Decimal('100.0'),
        volatility=Decimal('0.25'),
        risk_free_rate=Decimal('0.03'),
        position_count=50,
        notional_exposure=Decimal('500000'),
        current_hedge_position=Decimal('0')
    )
    
    logger.info(
        "greeks_snapshot_created",
        total_delta=float(greeks.total_delta),
        net_delta=float(greeks.get_net_delta()),
        delta_neutral=greeks.is_delta_neutral()
    )
    
    # Create hedge recommendation
    logger.info("creating_hedge_recommendation")
    
    hedge = HedgeRecommendation(
        hedge_id="HEDGE-001",
        hedge_type=HedgeType.DELTA,
        hedge_quantity=Decimal('-250'),  # Sell 250 shares to neutralize
        hedge_instrument="underlying",
        expected_delta_after=Decimal('0'),
        expected_gamma_after=Decimal('15.0'),
        expected_vega_after=Decimal('3500.0'),
        transaction_cost=Decimal('75.0'),
        slippage_cost=Decimal('25.0'),
        total_cost=Decimal('100.0'),
        cost_per_delta_hedged=Decimal('0.40'),
        urgency=HedgeUrgency.MEDIUM,
        confidence=Decimal('0.92'),
        effectiveness_score=Decimal('0.95'),
        rationale="Delta exceeds threshold, high confidence neutralization",
        risk_reduction_pct=Decimal('95.0'),
        cost_benefit_ratio=Decimal('237.5'),  # Risk reduced / Cost
        strategy_used=HedgeStrategy.DRL
    )
    
    logger.info(
        "hedge_recommendation_created",
        hedge_quantity=float(hedge.hedge_quantity),
        total_cost=float(hedge.total_cost),
        effectiveness=float(hedge.effectiveness_score),
        worth_executing=hedge.is_worth_executing()
    )
    
    # Create hedging policy
    logger.info("creating_hedging_policy")
    
    policy = HedgingPolicy(
        target_delta=Decimal('0'),
        delta_threshold=Decimal('50'),
        max_hedge_cost=Decimal('5000'),
        min_cost_benefit_ratio=Decimal('2.0'),
        max_time_between_hedges_hours=Decimal('4'),
        min_time_between_hedges_minutes=Decimal('5'),
        strategy=HedgeStrategy.DYNAMIC
    )
    
    requires = policy.requires_hedge(greeks.total_delta, Decimal('10'))
    logger.info(
        "policy_evaluated",
        requires_hedge=requires,
        delta_threshold=float(policy.delta_threshold)
    )
    
    # Create execution record
    logger.info("creating_execution_record")
    
    execution = HedgeExecution(
        hedge_id="HEDGE-001",
        recommendation=hedge,
        actual_quantity=Decimal('-250'),
        actual_price=Decimal('100.0'),
        actual_cost=Decimal('105.0'),
        actual_delta_after=Decimal('2.0'),
        actual_gamma_after=Decimal('15.0'),
        delta_error=Decimal('2.0'),
        venue="NASDAQ",
        execution_time_ms=Decimal('8.5'),
        order_id="ORD-12345",
        slippage_bps=Decimal('2.0'),
        execution_quality_score=Decimal('0.90'),
        fully_executed=True
    )
    
    logger.info(
        "execution_recorded",
        actual_cost=float(execution.actual_cost),
        cost_variance=float(execution.get_cost_variance()),
        was_effective=execution.was_effective()
    )
    
    # Create statistics
    logger.info("creating_statistics")
    
    stats = HedgingStatistics(
        total_hedges=1000,
        successful_hedges=972,
        failed_hedges=28,
        total_hedge_cost=Decimal('125000'),
        average_hedge_cost=Decimal('125'),
        total_slippage_cost=Decimal('15000'),
        average_delta_error=Decimal('3.5'),
        average_effectiveness_score=Decimal('0.88'),
        delta_neutral_time_pct=Decimal('92.5'),
        total_risk_reduced=Decimal('5000000'),
        average_cost_benefit_ratio=Decimal('40.0'),
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 12, 31)
    )
    
    logger.info(
        "statistics_created",
        success_rate=float(stats.get_success_rate()),
        cost_effective=stats.is_cost_effective(),
        avg_cost=float(stats.get_average_cost_per_hedge())
    )
    
    logger.info(
        "test_complete",
        artifacts_created=[
            "Immutable hedging objects",
            "Self-validating",
            "Rich domain behavior",
            "Type-safe with Decimal",
            "Complete hedge lifecycle",
            "Cost-benefit analysis built-in",
            "Proper logging (no print)"
        ]
    )