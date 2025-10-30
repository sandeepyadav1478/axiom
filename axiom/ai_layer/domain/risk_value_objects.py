"""
Risk Management Domain Value Objects

Immutable value objects for risk management domain.
Following DDD principles - these are the building blocks of our risk domain.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on bad data)
- Rich behavior (not just data bags)
- Type-safe (using Decimal for money)

These are NOT DTOs - they have business logic and validation.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum


class RiskSeverity(str, Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VaRMethod(str, Enum):
    """Value at Risk calculation methods"""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


@dataclass(frozen=True)
class PortfolioGreeks:
    """
    Aggregated Greeks for entire portfolio
    
    Immutable, validated, uses Decimal for precision
    """
    total_delta: Decimal
    total_gamma: Decimal
    total_vega: Decimal
    total_theta: Decimal
    total_rho: Decimal
    
    # Metadata
    position_count: int
    calculation_time_microseconds: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate on creation"""
        if self.position_count < 0:
            raise ValueError("Position count cannot be negative")
        
        if self.calculation_time_microseconds < Decimal('0'):
            raise ValueError("Calculation time cannot be negative")
    
    def is_within_limits(
        self,
        delta_limit: Decimal,
        gamma_limit: Decimal,
        vega_limit: Decimal
    ) -> bool:
        """Check if Greeks are within risk limits"""
        return (
            abs(self.total_delta) <= delta_limit and
            abs(self.total_gamma) <= gamma_limit and
            abs(self.total_vega) <= vega_limit
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'total_delta': float(self.total_delta),
            'total_gamma': float(self.total_gamma),
            'total_vega': float(self.total_vega),
            'total_theta': float(self.total_theta),
            'total_rho': float(self.total_rho),
            'position_count': self.position_count,
            'calculation_time_us': float(self.calculation_time_microseconds),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass(frozen=True)
class VaRMetrics:
    """
    Value at Risk metrics
    
    Immutable, multiple calculation methods for cross-validation
    """
    # Multiple VaR calculations (cross-validate)
    parametric_var: Decimal
    historical_var: Decimal
    monte_carlo_var: Decimal
    conditional_var: Decimal  # CVaR (expected shortfall)
    
    # Parameters
    confidence_level: Decimal  # e.g., 0.99 for 99%
    time_horizon_days: int  # e.g., 1 for 1-day VaR
    
    # Metadata
    num_simulations: Optional[int] = None  # For Monte Carlo
    calculation_time_microseconds: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate on creation"""
        if not (Decimal('0') < self.confidence_level < Decimal('1')):
            raise ValueError("Confidence level must be between 0 and 1")
        
        if self.time_horizon_days <= 0:
            raise ValueError("Time horizon must be positive")
        
        # VaR should be positive (represents potential loss)
        for var_value in [self.parametric_var, self.historical_var, 
                          self.monte_carlo_var, self.conditional_var]:
            if var_value < Decimal('0'):
                raise ValueError(f"VaR values must be positive (loss amount), got {var_value}")
    
    def get_conservative_var(self) -> Decimal:
        """
        Return most conservative (highest) VaR
        
        For risk management, always err on the side of caution
        """
        return max(
            self.parametric_var,
            self.historical_var,
            self.monte_carlo_var
        )
    
    def get_method_agreement(self) -> Decimal:
        """
        Check agreement between methods
        
        Returns: Coefficient of variation (lower = better agreement)
        """
        import statistics
        values = [
            float(self.parametric_var),
            float(self.historical_var),
            float(self.monte_carlo_var)
        ]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        return Decimal(str(stdev / mean if mean > 0 else 0))


@dataclass(frozen=True)
class RiskLimits:
    """
    Risk limits configuration
    
    Immutable limits for portfolio
    """
    max_delta: Decimal
    max_gamma: Decimal
    max_vega: Decimal
    max_theta: Decimal
    max_var_1day: Decimal
    max_notional_exposure: Decimal
    
    # Alert thresholds (percentage of limit)
    warning_threshold_pct: Decimal = Decimal('0.80')  # 80% = warning
    critical_threshold_pct: Decimal = Decimal('0.95')  # 95% = critical
    
    def __post_init__(self):
        """Validate limits are positive"""
        for limit in [self.max_delta, self.max_gamma, self.max_vega,
                      self.max_theta, self.max_var_1day, self.max_notional_exposure]:
            if limit <= Decimal('0'):
                raise ValueError("Risk limits must be positive")
    
    def get_utilization(self, current_value: Decimal, limit: Decimal) -> Decimal:
        """Calculate utilization percentage"""
        return abs(current_value) / limit if limit > Decimal('0') else Decimal('0')
    
    def check_breach(self, current_value: Decimal, limit: Decimal) -> Optional[str]:
        """
        Check if value breaches limit
        
        Returns: None if OK, severity level if breached
        """
        utilization = self.get_utilization(current_value, limit)
        
        if utilization >= Decimal('1.0'):
            return RiskSeverity.CRITICAL.value
        elif utilization >= self.critical_threshold_pct:
            return RiskSeverity.HIGH.value
        elif utilization >= self.warning_threshold_pct:
            return RiskSeverity.MEDIUM.value
        
        return None


@dataclass(frozen=True)
class StressTestResult:
    """
    Result from stress testing scenario
    
    Immutable snapshot of portfolio under stress
    """
    scenario_name: str
    
    # Market shocks applied
    spot_shock_pct: Decimal
    volatility_shock_pct: Decimal
    rate_shock_bps: Decimal
    
    # Resulting metrics
    portfolio_pnl: Decimal
    greeks: PortfolioGreeks
    var_metrics: VaRMetrics
    
    # Severity
    severity: RiskSeverity
    limit_breaches: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_catastrophic(self, max_loss: Decimal) -> bool:
        """Check if scenario results in catastrophic loss"""
        return self.portfolio_pnl < -abs(max_loss)


@dataclass(frozen=True)
class RiskAlert:
    """
    Risk alert notification
    
    Immutable alert for limit breaches
    """
    alert_id: str
    alert_type: str  # 'delta_breach', 'var_breach', etc.
    severity: RiskSeverity
    
    message: str
    current_value: Decimal
    limit_value: Decimal
    utilization_pct: Decimal
    
    portfolio_id: Optional[str] = None
    position_ids: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    
    def requires_immediate_action(self) -> bool:
        """Check if alert requires immediate action"""
        return self.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("RISK DOMAIN VALUE OBJECTS")
    print("="*60)
    
    # Create portfolio Greeks
    print("\n→ Portfolio Greeks (immutable):")
    greeks = PortfolioGreeks(
        total_delta=Decimal('2500.75'),
        total_gamma=Decimal('125.50'),
        total_vega=Decimal('15000.00'),
        total_theta=Decimal('-450.25'),
        total_rho=Decimal('800.00'),
        position_count=150,
        calculation_time_microseconds=Decimal('2500.0')
    )
    print(f"   Delta: {greeks.total_delta}")
    print(f"   Positions: {greeks.position_count}")
    print(f"   Time: {greeks.calculation_time_microseconds}us")
    
    # Create VaR metrics
    print("\n→ VaR Metrics (multiple methods):")
    var = VaRMetrics(
        parametric_var=Decimal('125000.00'),
        historical_var=Decimal('130000.00'),
        monte_carlo_var=Decimal('132500.00'),
        conditional_var=Decimal('175000.00'),
        confidence_level=Decimal('0.99'),
        time_horizon_days=1,
        num_simulations=10000
    )
    print(f"   Parametric: ${var.parametric_var:,.0f}")
    print(f"   Historical: ${var.historical_var:,.0f}")
    print(f"   Monte Carlo: ${var.monte_carlo_var:,.0f}")
    print(f"   Conservative: ${var.get_conservative_var():,.0f}")
    print(f"   Method agreement: {var.get_method_agreement():.4f}")
    
    # Create risk limits
    print("\n→ Risk Limits:")
    limits = RiskLimits(
        max_delta=Decimal('10000'),
        max_gamma=Decimal('500'),
        max_vega=Decimal('50000'),
        max_theta=Decimal('2000'),
        max_var_1day=Decimal('500000'),
        max_notional_exposure=Decimal('10000000')
    )
    
    # Check utilization
    delta_util = limits.get_utilization(greeks.total_delta, limits.max_delta)
    print(f"   Delta utilization: {delta_util:.1%}")
    
    breach = limits.check_breach(greeks.total_delta, limits.max_delta)
    print(f"   Delta breach: {breach if breach else 'None'}")
    
    # Check if within limits
    within = greeks.is_within_limits(
        limits.max_delta,
        limits.max_gamma,
        limits.max_vega
    )
    print(f"   Within limits: {'✓ YES' if within else '✗ NO'}")
    
    # Create risk alert
    print("\n→ Risk Alert:")
    alert = RiskAlert(
        alert_id="ALERT-001",
        alert_type="delta_breach",
        severity=RiskSeverity.HIGH,
        message="Delta approaching limit",
        current_value=Decimal('9500'),
        limit_value=Decimal('10000'),
        utilization_pct=Decimal('0.95')
    )
    print(f"   Type: {alert.alert_type}")
    print(f"   Severity: {alert.severity.value}")
    print(f"   Immediate action: {'YES' if alert.requires_immediate_action() else 'NO'}")
    
    print("\n" + "="*60)
    print("✓ Immutable value objects")
    print("✓ Self-validating")
    print("✓ Rich domain behavior")
    print("✓ Type-safe with Decimal")
    print("\nDOMAIN-DRIVEN DESIGN FOR RISK MANAGEMENT")