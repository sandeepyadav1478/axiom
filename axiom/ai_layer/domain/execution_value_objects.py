"""
Order Execution Domain Value Objects

Immutable value objects for order execution domain.
Following DDD principles - these capture the essence of order execution.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on bad orders)
- Rich behavior (order lifecycle, execution quality analysis)
- Type-safe (using Decimal for prices, Enum for states)

These represent orders and executions as first-class domain concepts.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from enum import Enum


class OrderSide(str, Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order lifecycle status"""
    PENDING = "pending"
    ROUTED = "routed"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Order time in force"""
    DAY = "day"
    GTC = "gtc"  # Good til cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class Venue(str, Enum):
    """Execution venues"""
    CBOE = "CBOE"
    ISE = "ISE"
    PHLX = "PHLX"
    AMEX = "AMEX"
    BATS = "BATS"
    BOX = "BOX"
    MIAX = "MIAX"
    NASDAQ = "NASDAQ"
    NYSE_ARCA = "NYSE_ARCA"
    PEARL = "PEARL"


class Urgency(str, Enum):
    """Order urgency level"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Order:
    """
    Immutable order representation
    
    Complete order specification with validation
    """
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    
    # Pricing
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    
    # Execution parameters
    time_in_force: TimeInForce = TimeInForce.DAY
    urgency: Urgency = Urgency.NORMAL
    
    # Routing preferences
    preferred_venues: Tuple[Venue, ...] = field(default_factory=tuple)
    avoid_venues: Tuple[Venue, ...] = field(default_factory=tuple)
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    client_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate order parameters"""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require limit_price")
        
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require stop_price")
        
        if self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None or self.stop_price is None:
                raise ValueError("Stop-limit orders require both limit_price and stop_price")
        
        if self.limit_price is not None and self.limit_price <= Decimal('0'):
            raise ValueError("Limit price must be positive")
        
        if self.stop_price is not None and self.stop_price <= Decimal('0'):
            raise ValueError("Stop price must be positive")
    
    def is_market_order(self) -> bool:
        """Check if order is market order"""
        return self.order_type == OrderType.MARKET
    
    def is_limit_order(self) -> bool:
        """Check if order is limit order"""
        return self.order_type == OrderType.LIMIT
    
    def is_buy(self) -> bool:
        """Check if order is buy side"""
        return self.side == OrderSide.BUY
    
    def is_sell(self) -> bool:
        """Check if order is sell side"""
        return self.side == OrderSide.SELL
    
    def is_active(self) -> bool:
        """Check if order is in active state"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.ROUTED,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILL
        ]
    
    def is_terminal(self) -> bool:
        """Check if order is in terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    def requires_routing(self) -> bool:
        """Check if order needs routing decision"""
        return self.status == OrderStatus.PENDING and len(self.preferred_venues) == 0


@dataclass(frozen=True)
class VenueQuote:
    """
    Quote from specific venue
    
    Immutable market data snapshot
    """
    venue: Venue
    bid: Decimal
    ask: Decimal
    bid_size: int
    ask_size: int
    spread_bps: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate quote"""
        if self.bid < Decimal('0') or self.ask < Decimal('0'):
            raise ValueError("Prices must be non-negative")
        
        if self.bid > self.ask:
            raise ValueError(f"Bid ({self.bid}) cannot exceed ask ({self.ask})")
        
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("Sizes must be non-negative")
    
    def get_mid_price(self) -> Decimal:
        """Calculate mid price"""
        return (self.bid + self.ask) / Decimal('2')
    
    def get_spread(self) -> Decimal:
        """Calculate absolute spread"""
        return self.ask - self.bid
    
    def has_liquidity_for(self, side: OrderSide, quantity: int) -> bool:
        """Check if venue has sufficient liquidity"""
        if side == OrderSide.BUY:
            return self.ask_size >= quantity
        else:
            return self.bid_size >= quantity


@dataclass(frozen=True)
class RoutingDecision:
    """
    Smart routing decision
    
    Immutable routing analysis with rationale
    """
    order_id: str
    primary_venue: Venue
    backup_venues: Tuple[Venue, ...]
    
    # Expected execution quality
    expected_fill_price: Decimal
    expected_fill_probability: Decimal
    expected_slippage_bps: Decimal
    expected_latency_ms: Decimal
    
    # Analysis
    rationale: str
    confidence: Decimal  # 0-1
    
    # Alternatives considered
    alternatives_evaluated: int
    
    # Metadata
    decision_time_ms: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate routing decision"""
        if not (Decimal('0') <= self.expected_fill_probability <= Decimal('1')):
            raise ValueError("Fill probability must be between 0 and 1")
        
        if not (Decimal('0') <= self.confidence <= Decimal('1')):
            raise ValueError("Confidence must be between 0 and 1")
        
        if self.expected_slippage_bps < Decimal('0'):
            raise ValueError("Slippage must be non-negative")
    
    def is_high_confidence(self) -> bool:
        """Check if routing decision has high confidence"""
        return self.confidence >= Decimal('0.80')
    
    def has_backup_plan(self) -> bool:
        """Check if routing has backup venues"""
        return len(self.backup_venues) > 0
    
    def get_total_venues(self) -> int:
        """Get total number of venues in routing plan"""
        return 1 + len(self.backup_venues)


@dataclass(frozen=True)
class ExecutionReport:
    """
    Execution report for order
    
    Immutable record of execution
    """
    order_id: str
    venue: Venue
    
    # Execution details
    fill_price: Decimal
    fill_quantity: int
    remaining_quantity: int
    
    # Quality metrics
    actual_slippage_bps: Decimal
    execution_latency_ms: Decimal
    
    # Status
    status: OrderStatus
    is_complete: bool
    
    # Analysis
    execution_quality_score: Decimal  # 0-1, higher = better
    beat_nbbo: bool
    price_improvement_bps: Optional[Decimal] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    commission: Decimal = Decimal('0')
    exchange_fees: Decimal = Decimal('0')
    
    def __post_init__(self):
        """Validate execution report"""
        if self.fill_quantity < 0 or self.remaining_quantity < 0:
            raise ValueError("Quantities must be non-negative")
        
        if self.fill_price <= Decimal('0'):
            raise ValueError("Fill price must be positive")
        
        if not (Decimal('0') <= self.execution_quality_score <= Decimal('1')):
            raise ValueError("Quality score must be between 0 and 1")
    
    def get_fill_percentage(self) -> Decimal:
        """Calculate fill percentage"""
        total = self.fill_quantity + self.remaining_quantity
        if total > 0:
            return (Decimal(str(self.fill_quantity)) / Decimal(str(total))) * Decimal('100')
        return Decimal('0')
    
    def get_total_cost(self) -> Decimal:
        """Calculate total cost including fees"""
        return (self.fill_price * Decimal(str(self.fill_quantity)) * Decimal('100') +
                self.commission + self.exchange_fees)
    
    def get_average_cost_per_contract(self) -> Decimal:
        """Calculate average cost per contract including fees"""
        if self.fill_quantity > 0:
            return self.get_total_cost() / (Decimal(str(self.fill_quantity)) * Decimal('100'))
        return Decimal('0')
    
    def exceeds_quality_threshold(self, min_score: Decimal = Decimal('0.70')) -> bool:
        """Check if execution meets quality threshold"""
        return self.execution_quality_score >= min_score


@dataclass(frozen=True)
class ExecutionStatistics:
    """
    Aggregated execution statistics
    
    Immutable performance metrics
    """
    total_orders: int
    filled_orders: int
    partial_fills: int
    rejected_orders: int
    
    # Execution quality
    average_slippage_bps: Decimal
    average_fill_price_improvement_bps: Decimal
    average_latency_ms: Decimal
    average_quality_score: Decimal
    
    # Fill statistics
    fill_rate: Decimal
    average_fill_percentage: Decimal
    
    # Venue statistics
    best_venue: Venue
    venue_distribution: Dict[Venue, int]
    
    # Time period
    start_time: datetime
    end_time: datetime
    
    def __post_init__(self):
        """Validate statistics"""
        if self.total_orders < 0:
            raise ValueError("Total orders must be non-negative")
        
        if not (Decimal('0') <= self.fill_rate <= Decimal('1')):
            raise ValueError("Fill rate must be between 0 and 1")
    
    def get_rejection_rate(self) -> Decimal:
        """Calculate rejection rate"""
        if self.total_orders > 0:
            return Decimal(str(self.rejected_orders)) / Decimal(str(self.total_orders))
        return Decimal('0')
    
    def is_high_quality(
        self,
        min_fill_rate: Decimal = Decimal('0.95'),
        max_slippage: Decimal = Decimal('2.0'),
        min_quality: Decimal = Decimal('0.80')
    ) -> bool:
        """Check if execution meets high quality standards"""
        return (
            self.fill_rate >= min_fill_rate and
            self.average_slippage_bps <= max_slippage and
            self.average_quality_score >= min_quality
        )


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("EXECUTION DOMAIN VALUE OBJECTS")
    print("="*60)
    
    # Create order
    print("\n→ Creating Limit Order:")
    order = Order(
        order_id="ORD-001",
        symbol="SPY241115C00450000",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=Decimal('5.50'),
        time_in_force=TimeInForce.DAY,
        urgency=Urgency.NORMAL
    )
    
    print(f"   Order ID: {order.order_id}")
    print(f"   Symbol: {order.symbol}")
    print(f"   Side: {order.side.value}")
    print(f"   Quantity: {order.quantity}")
    print(f"   Limit: ${order.limit_price}")
    print(f"   Is buy: {'YES' if order.is_buy() else 'NO'}")
    print(f"   Is active: {'YES' if order.is_active() else 'NO'}")
    print(f"   Requires routing: {'YES' if order.requires_routing() else 'NO'}")
    
    # Create venue quote
    print("\n→ Venue Quote:")
    quote = VenueQuote(
        venue=Venue.CBOE,
        bid=Decimal('5.48'),
        ask=Decimal('5.52'),
        bid_size=150,
        ask_size=200,
        spread_bps=Decimal('72.5')
    )
    
    print(f"   Venue: {quote.venue.value}")
    print(f"   Bid: ${quote.bid} x {quote.bid_size}")
    print(f"   Ask: ${quote.ask} x {quote.ask_size}")
    print(f"   Mid: ${quote.get_mid_price()}")
    print(f"   Spread: {quote.spread_bps} bps")
    print(f"   Has liquidity: {'YES' if quote.has_liquidity_for(OrderSide.BUY, 100) else 'NO'}")
    
    # Create routing decision
    print("\n→ Routing Decision:")
    routing = RoutingDecision(
        order_id="ORD-001",
        primary_venue=Venue.CBOE,
        backup_venues=(Venue.ISE, Venue.PHLX),
        expected_fill_price=Decimal('5.51'),
        expected_fill_probability=Decimal('0.92'),
        expected_slippage_bps=Decimal('1.8'),
        expected_latency_ms=Decimal('2.5'),
        rationale="Best price with high fill probability",
        confidence=Decimal('0.88'),
        alternatives_evaluated=10,
        decision_time_ms=Decimal('0.85')
    )
    
    print(f"   Primary: {routing.primary_venue.value}")
    print(f"   Backups: {[v.value for v in routing.backup_venues]}")
    print(f"   Expected price: ${routing.expected_fill_price}")
    print(f"   Fill probability: {routing.expected_fill_probability:.1%}")
    print(f"   Expected slippage: {routing.expected_slippage_bps} bps")
    print(f"   Confidence: {routing.confidence:.1%}")
    print(f"   High confidence: {'YES' if routing.is_high_confidence() else 'NO'}")
    print(f"   Decision time: {routing.decision_time_ms}ms")
    
    # Create execution report
    print("\n→ Execution Report:")
    execution = ExecutionReport(
        order_id="ORD-001",
        venue=Venue.CBOE,
        fill_price=Decimal('5.50'),
        fill_quantity=100,
        remaining_quantity=0,
        actual_slippage_bps=Decimal('0.0'),
        execution_latency_ms=Decimal('2.3'),
        status=OrderStatus.FILLED,
        is_complete=True,
        execution_quality_score=Decimal('0.95'),
        beat_nbbo=True,
        price_improvement_bps=Decimal('3.6'),
        commission=Decimal('5.00'),
        exchange_fees=Decimal('1.50')
    )
    
    print(f"   Venue: {execution.venue.value}")
    print(f"   Fill price: ${execution.fill_price}")
    print(f"   Fill quantity: {execution.fill_quantity}")
    print(f"   Fill percentage: {execution.get_fill_percentage():.1f}%")
    print(f"   Actual slippage: {execution.actual_slippage_bps} bps")
    print(f"   Latency: {execution.execution_latency_ms}ms")
    print(f"   Quality score: {execution.execution_quality_score:.2f}")
    print(f"   Beat NBBO: {'YES' if execution.beat_nbbo else 'NO'}")
    print(f"   Price improvement: {execution.price_improvement_bps} bps")
    print(f"   Total cost: ${execution.get_total_cost():,.2f}")
    print(f"   Avg cost/contract: ${execution.get_average_cost_per_contract():.2f}")
    
    # Execution statistics
    print("\n→ Execution Statistics:")
    stats = ExecutionStatistics(
        total_orders=1000,
        filled_orders=972,
        partial_fills=18,
        rejected_orders=10,
        average_slippage_bps=Decimal('1.2'),
        average_fill_price_improvement_bps=Decimal('2.8'),
        average_latency_ms=Decimal('2.1'),
        average_quality_score=Decimal('0.89'),
        fill_rate=Decimal('0.972'),
        average_fill_percentage=Decimal('99.5'),
        best_venue=Venue.CBOE,
        venue_distribution={Venue.CBOE: 450, Venue.ISE: 350, Venue.PHLX: 172},
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 12, 31)
    )
    
    print(f"   Total orders: {stats.total_orders:,}")
    print(f"   Fill rate: {stats.fill_rate:.1%}")
    print(f"   Rejection rate: {stats.get_rejection_rate():.1%}")
    print(f"   Avg slippage: {stats.average_slippage_bps} bps")
    print(f"   Avg improvement: {stats.average_fill_price_improvement_bps} bps")
    print(f"   Avg latency: {stats.average_latency_ms}ms")
    print(f"   Avg quality: {stats.average_quality_score:.2f}")
    print(f"   Best venue: {stats.best_venue.value}")
    print(f"   High quality: {'✓ YES' if stats.is_high_quality() else '✗ NO'}")
    
    print("\n" + "="*60)
    print("✓ Immutable order objects")
    print("✓ Self-validating")
    print("✓ Rich domain behavior")
    print("✓ Type-safe with Decimal")
    print("✓ Complete order lifecycle")
    print("✓ Execution quality tracking")
    print("\nDOMAIN-DRIVEN DESIGN FOR ORDER EXECUTION")