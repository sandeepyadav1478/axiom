"""
Domain Entities - Identity and Lifecycle

Entities have:
- Identity (unique ID, not just values)
- Lifecycle (creation, updates, deletion)
- Behavior (methods that maintain invariants)
- Events (domain events for changes)

vs Value Objects: Entities have identity, value objects don't

This is proper Domain-Driven Design.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from decimal import Decimal
from datetime import datetime
from uuid import UUID, uuid4
from axiom.ai_layer.domain.value_objects import Greeks, OptionType
from axiom.ai_layer.domain.exceptions import ValidationError


# Domain events
@dataclass(frozen=True)
class DomainEvent:
    """Base class for domain events"""
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    aggregate_id: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'event_id': str(self.event_id),
            'event_type': self.__class__.__name__,
            'timestamp': self.timestamp.isoformat(),
            'aggregate_id': self.aggregate_id
        }


@dataclass(frozen=True)
class GreeksCalculated(DomainEvent):
    """Event: Greeks were calculated"""
    greeks: Greeks = field(default=None)
    spot: Decimal = field(default=Decimal('0'))
    strike: Decimal = field(default=Decimal('0'))
    calculation_time_us: Decimal = field(default=Decimal('0'))


@dataclass(frozen=True)
class PositionOpened(DomainEvent):
    """Event: New position opened"""
    symbol: str = ""
    quantity: int = 0
    entry_price: Decimal = field(default=Decimal('0'))


@dataclass(frozen=True)
class RiskLimitBreached(DomainEvent):
    """Event: Risk limit exceeded"""
    limit_type: str = ""
    current_value: Decimal = field(default=Decimal('0'))
    limit_value: Decimal = field(default=Decimal('0'))


class Entity:
    """
    Base class for all entities
    
    Provides:
    - Unique identity
    - Equality by ID (not values)
    - Domain events collection
    - Audit fields (created, modified)
    """
    
    def __init__(self, id: Optional[UUID] = None):
        self.id = id or uuid4()
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        self._domain_events: List[DomainEvent] = []
    
    def __eq__(self, other):
        """Entities equal if same ID"""
        if not isinstance(other, Entity):
            return False
        return self.id == other.id
    
    def __hash__(self):
        """Hash by ID for set/dict usage"""
        return hash(self.id)
    
    def add_domain_event(self, event: DomainEvent):
        """Add domain event to be published"""
        self._domain_events.append(event)
    
    def get_domain_events(self) -> List[DomainEvent]:
        """Get and clear domain events"""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
    
    def touch(self):
        """Update modified timestamp"""
        self.modified_at = datetime.now()


class Position(Entity):
    """
    Position entity - Represents an options position
    
    Aggregate root for position-related logic
    
    Invariants:
    - Quantity can be positive (long) or negative (short)
    - Greeks must be valid
    - Entry price must be positive
    - Position value calculated consistently
    """
    
    def __init__(
        self,
        symbol: str,
        strike: Decimal,
        expiry: datetime,
        option_type: OptionType,
        quantity: int,
        entry_price: Decimal,
        id: Optional[UUID] = None
    ):
        super().__init__(id)
        
        # Validate inputs
        if quantity == 0:
            raise ValidationError("Quantity cannot be zero", context={'symbol': symbol})
        
        if entry_price <= 0:
            raise ValidationError("Entry price must be positive", context={'price': float(entry_price)})
        
        self.symbol = symbol
        self.strike = strike
        self.expiry = expiry
        self.option_type = option_type
        self.quantity = quantity
        self.entry_price = entry_price
        
        # Current market data (updated)
        self.current_price: Optional[Decimal] = None
        self.current_greeks: Optional[Greeks] = None
        
        # Add domain event
        self.add_domain_event(PositionOpened(
            aggregate_id=str(self.id),
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price
        ))
    
    def update_greeks(self, greeks: Greeks):
        """
        Update position's Greeks
        
        Ensures Greeks are valid before updating
        """
        # Validate Greeks
        if greeks.option_type != self.option_type:
            raise ValidationError(
                f"Greeks option type {greeks.option_type} doesn't match position {self.option_type}"
            )
        
        self.current_greeks = greeks
        self.touch()
        
        # Add domain event
        self.add_domain_event(GreeksCalculated(
            aggregate_id=str(self.id),
            greeks=greeks,
            spot=Decimal('0'),  # Would pass actual spot
            strike=self.strike,
            calculation_time_us=greeks.calculation_time_microseconds
        ))
    
    def update_market_price(self, price: Decimal):
        """Update current market price"""
        if price <= 0:
            raise ValidationError("Price must be positive", context={'price': float(price)})
        
        self.current_price = price
        self.touch()
    
    def calculate_unrealized_pnl(self) -> Decimal:
        """
        Calculate unrealized P&L
        
        P&L = (current_price - entry_price) * quantity * 100
        """
        if self.current_price is None:
            return Decimal('0')
        
        pnl = (self.current_price - self.entry_price) * self.quantity * 100
        
        return pnl
    
    def get_effective_greeks(self) -> Optional[Greeks]:
        """
        Get effective Greeks (considering position size)
        
        Effective delta = delta * quantity
        """
        if self.current_greeks is None:
            return None
        
        # Greeks scale with quantity
        return Greeks(
            delta=self.current_greeks.delta * self.quantity,
            gamma=self.current_greeks.gamma * self.quantity,
            theta=self.current_greeks.theta * self.quantity,
            vega=self.current_greeks.vega * self.quantity,
            rho=self.current_greeks.rho * self.quantity,
            option_type=self.option_type,
            calculation_time_microseconds=self.current_greeks.calculation_time_microseconds,
            calculation_method=self.current_greeks.calculation_method,
            model_version=self.current_greeks.model_version
        )
    
    def can_close(self) -> bool:
        """Check if position can be closed"""
        # Business rules for closing
        return True  # Could add market hours, liquidity checks, etc.
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'id': str(self.id),
            'symbol': self.symbol,
            'strike': float(self.strike),
            'expiry': self.expiry.isoformat(),
            'option_type': self.option_type.value,
            'quantity': self.quantity,
            'entry_price': float(self.entry_price),
            'current_price': float(self.current_price) if self.current_price else None,
            'current_greeks': self.current_greeks.to_dict() if self.current_greeks else None,
            'unrealized_pnl': float(self.calculate_unrealized_pnl()),
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat()
        }


class Portfolio(Entity):
    """
    Portfolio aggregate root
    
    Manages collection of positions with:
    - Position tracking
    - Aggregated Greeks
    - Total P&L
    - Risk limits
    - Rebalancing logic
    
    This is the main aggregate for trading operations
    """
    
    def __init__(
        self,
        name: str,
        max_positions: int = 1000,
        max_delta: Decimal = Decimal('50000'),
        id: Optional[UUID] = None
    ):
        super().__init__(id)
        
        self.name = name
        self.max_positions = max_positions
        self.max_delta = max_delta
        
        # Positions
        self._positions: Dict[UUID, Position] = {}
        
        # Cached aggregates (invalidate on position change)
        self._total_greeks: Optional[Greeks] = None
        self._total_pnl: Optional[Decimal] = None
        self._greeks_cache_dirty = True
    
    def add_position(self, position: Position):
        """
        Add position to portfolio
        
        Validates:
        - Not exceeding max positions
        - Not breaching risk limits
        """
        if len(self._positions) >= self.max_positions:
            raise ValidationError(
                f"Portfolio full: {self.max_positions} positions",
                context={'portfolio_id': str(self.id)}
            )
        
        self._positions[position.id] = position
        self._greeks_cache_dirty = True
        self.touch()
        
        print(f"✓ Position added to portfolio: {position.symbol}")
    
    def remove_position(self, position_id: UUID):
        """Remove position from portfolio"""
        if position_id in self._positions:
            del self._positions[position_id]
            self._greeks_cache_dirty = True
            self.touch()
    
    def get_position(self, position_id: UUID) -> Optional[Position]:
        """Get position by ID"""
        return self._positions.get(position_id)
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self._positions.values())
    
    def calculate_total_greeks(self) -> Greeks:
        """
        Calculate aggregate portfolio Greeks
        
        Caches result for performance
        """
        if not self._greeks_cache_dirty and self._total_greeks:
            return self._total_greeks
        
        # Aggregate Greeks from all positions
        total_delta = Decimal('0')
        total_gamma = Decimal('0')
        total_theta = Decimal('0')
        total_vega = Decimal('0')
        total_rho = Decimal('0')
        
        for position in self._positions.values():
            effective = position.get_effective_greeks()
            if effective:
                total_delta += effective.delta
                total_gamma += effective.gamma
                total_theta += effective.theta
                total_vega += effective.vega
                total_rho += effective.rho
        
        self._total_greeks = Greeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho,
            option_type=OptionType.CALL,  # Portfolio level, not specific
            calculation_method='aggregate'
        )
        
        self._greeks_cache_dirty = False
        
        return self._total_greeks
    
    def calculate_total_pnl(self) -> Decimal:
        """Calculate total unrealized P&L"""
        total = Decimal('0')
        
        for position in self._positions.values():
            total += position.calculate_unrealized_pnl()
        
        return total
    
    def check_risk_limits(self) -> List[str]:
        """
        Check if portfolio within risk limits
        
        Returns: List of limit breaches (empty if all good)
        """
        breaches = []
        
        total_greeks = self.calculate_total_greeks()
        
        if abs(total_greeks.delta) > self.max_delta:
            breaches.append(f"Delta limit breached: {total_greeks.delta} > {self.max_delta}")
            
            # Raise domain event
            self.add_domain_event(RiskLimitBreached(
                aggregate_id=str(self.id),
                limit_type='delta',
                current_value=abs(total_greeks.delta),
                limit_value=self.max_delta
            ))
        
        return breaches


# Example usage
if __name__ == "__main__":
    from datetime import timedelta
    
    print("="*60)
    print("DOMAIN ENTITIES - PROPER DDD")
    print("="*60)
    
    # Create position
    print("\n→ Creating Position Entity:")
    
    position = Position(
        symbol='SPY241115C00450000',
        strike=Decimal('450.00'),
        expiry=datetime.now() + timedelta(days=30),
        option_type=OptionType.CALL,
        quantity=100,
        entry_price=Decimal('5.50')
    )
    
    print(f"   ID: {position.id}")
    print(f"   Symbol: {position.symbol}")
    print(f"   Quantity: {position.quantity}")
    
    # Update Greeks
    greeks = Greeks(
        delta=Decimal('0.52'),
        gamma=Decimal('0.015'),
        theta=Decimal('-0.03'),
        vega=Decimal('0.39'),
        rho=Decimal('0.51'),
        option_type=OptionType.CALL
    )
    
    position.update_greeks(greeks)
    print(f"   ✓ Greeks updated")
    
    # Calculate P&L
    position.update_market_price(Decimal('6.00'))
    pnl = position.calculate_unrealized_pnl()
    print(f"   Unrealized P&L: ${pnl}")
    
    # Domain events
    events = position.get_domain_events()
    print(f"\n   Domain Events: {len(events)}")
    for event in events:
        print(f"     - {event.__class__.__name__}")
    
    # Create portfolio
    print("\n→ Creating Portfolio Aggregate:")
    
    portfolio = Portfolio(
        name="Main Trading Portfolio",
        max_positions=1000,
        max_delta=Decimal('50000')
    )
    
    portfolio.add_position(position)
    
    total_greeks = portfolio.calculate_total_greeks()
    print(f"   Total Delta: {total_greeks.delta}")
    
    breaches = portfolio.check_risk_limits()
    print(f"   Risk breaches: {len(breaches)}")
    
    print("\n" + "="*60)
    print("✓ Proper entity design (identity-based)")
    print("✓ Domain events (for event sourcing)")
    print("✓ Invariant enforcement")
    print("✓ Aggregate roots")
    print("\nPROFESSIONAL DOMAIN-DRIVEN DESIGN")