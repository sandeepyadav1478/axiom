"""
Portfolio Analytics Domain Value Objects

Immutable value objects for analytics and performance analysis domain.
Following DDD principles - these capture P&L, attribution, and performance metrics.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on invalid metrics)
- Rich behavior (performance analysis, attribution logic)
- Type-safe (using Decimal for precision, Enum for categories)

These represent analytics as first-class domain concepts.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from enum import Enum
import statistics


class PnLCategory(str, Enum):
    """P&L categorization"""
    DELTA = "delta"  # From price movement
    GAMMA = "gamma"  # From convexity
    VEGA = "vega"  # From volatility
    THETA = "theta"  # From time decay
    RHO = "rho"  # From interest rates


class TimePeriod(str, Enum):
    """Analysis time period"""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class PerformanceRating(str, Enum):
    """Performance rating"""
    EXCELLENT = "excellent"  # Top quartile
    GOOD = "good"  # Above average
    AVERAGE = "average"  # Market performance
    BELOW_AVERAGE = "below_average"
    POOR = "poor"  # Bottom quartile


@dataclass(frozen=True)
class PnLSnapshot:
    """
    Point-in-time P&L snapshot
    
    Immutable P&L state at specific moment
    """
    # Total P&L
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    
    # Greeks attribution
    delta_pnl: Decimal
    gamma_pnl: Decimal
    vega_pnl: Decimal
    theta_pnl: Decimal
    rho_pnl: Decimal
    
    # By strategy
    strategy_pnl: Dict[str, Decimal]
    
    # By position
    position_pnl: Dict[str, Decimal]
    
    # Metrics
    pnl_volatility: Decimal
    max_drawdown_today: Decimal
    high_water_mark: Decimal
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    calculation_time_microseconds: Decimal = Decimal('0')
    
    def __post_init__(self):
        """Validate P&L snapshot"""
        # Total should equal sum
        calculated_total = self.realized_pnl + self.unrealized_pnl
        if abs(calculated_total - self.total_pnl) > Decimal('0.01'):
            raise ValueError(f"P&L mismatch: {calculated_total} != {self.total_pnl}")
    
    def get_greeks_attribution_pct(self) -> Dict[PnLCategory, Decimal]:
        """Get P&L attribution by Greek as percentages"""
        if abs(self.total_pnl) < Decimal('0.01'):
            return {cat: Decimal('0') for cat in PnLCategory}
        
        return {
            PnLCategory.DELTA: (self.delta_pnl / self.total_pnl) * Decimal('100'),
            PnLCategory.GAMMA: (self.gamma_pnl / self.total_pnl) * Decimal('100'),
            PnLCategory.VEGA: (self.vega_pnl / self.total_pnl) * Decimal('100'),
            PnLCategory.THETA: (self.theta_pnl / self.total_pnl) * Decimal('100'),
            PnLCategory.RHO: (self.rho_pnl / self.total_pnl) * Decimal('100')
        }
    
    def get_dominant_greek(self) -> PnLCategory:
        """Get Greek contributing most to P&L"""
        greeks = {
            PnLCategory.DELTA: abs(self.delta_pnl),
            PnLCategory.GAMMA: abs(self.gamma_pnl),
            PnLCategory.VEGA: abs(self.vega_pnl),
            PnLCategory.THETA: abs(self.theta_pnl),
            PnLCategory.RHO: abs(self.rho_pnl)
        }
        return max(greeks, key=greeks.get)
    
    def is_profitable(self) -> bool:
        """Check if snapshot shows profit"""
        return self.total_pnl > Decimal('0')
    
    def get_realized_pct(self) -> Decimal:
        """Get percentage of P&L that is realized"""
        if abs(self.total_pnl) > Decimal('0.01'):
            return (self.realized_pnl / self.total_pnl) * Decimal('100')
        return Decimal('0')


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Performance analysis metrics
    
    Immutable performance snapshot with risk-adjusted returns
    """
    # Returns
    total_return: Decimal
    annualized_return: Decimal
    
    # Risk-adjusted metrics
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    
    # Risk metrics
    volatility: Decimal
    max_drawdown: Decimal
    var_95: Decimal
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    
    # Profitability
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    profit_factor: Decimal
    
    # Consistency
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: Decimal  # Net profit / Max drawdown
    
    # Period
    start_date: datetime
    end_date: datetime
    days_traded: int
    
    # Metadata
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate performance metrics"""
        if self.total_trades < 0:
            raise ValueError("Total trades must be non-negative")
        
        if not (Decimal('0') <= self.win_rate <= Decimal('1')):
            raise ValueError("Win rate must be between 0 and 1")
        
        if self.volatility < Decimal('0'):
            raise ValueError("Volatility must be non-negative")
    
    def get_loss_rate(self) -> Decimal:
        """Calculate loss rate"""
        if self.total_trades > 0:
            return Decimal(str(self.losing_trades)) / Decimal(str(self.total_trades))
        return Decimal('0')
    
    def get_rating(self) -> PerformanceRating:
        """Get performance rating based on metrics"""
        # Excellent: Sharpe > 2, Win rate > 65%, Positive return
        if (self.sharpe_ratio > Decimal('2.0') and 
            self.win_rate > Decimal('0.65') and 
            self.total_return > Decimal('0')):
            return PerformanceRating.EXCELLENT
        
        # Good: Sharpe > 1.5, Win rate > 55%
        elif (self.sharpe_ratio > Decimal('1.5') and 
              self.win_rate > Decimal('0.55') and
              self.total_return > Decimal('0')):
            return PerformanceRating.GOOD
        
        # Average: Sharpe > 1.0, Win rate > 50%
        elif (self.sharpe_ratio > Decimal('1.0') and 
              self.win_rate > Decimal('0.50')):
            return PerformanceRating.AVERAGE
        
        # Below average: Positive but weak
        elif self.total_return > Decimal('0'):
            return PerformanceRating.BELOW_AVERAGE
        
        # Poor: Negative return
        else:
            return PerformanceRating.POOR
    
    def is_risk_adjusted_profitable(self, min_sharpe: Decimal = Decimal('1.0')) -> bool:
        """Check if performance is risk-adjusted profitable"""
        return self.sharpe_ratio >= min_sharpe and self.total_return > Decimal('0')
    
    def has_acceptable_drawdown(self, max_dd: Decimal = Decimal('0.20')) -> bool:
        """Check if drawdown is acceptable"""
        return abs(self.max_drawdown) <= max_dd


@dataclass(frozen=True)
class GreeksAttribution:
    """
    P&L attribution by Greek
    
    Immutable attribution analysis
    """
    time_period: TimePeriod
    
    # Attribution by Greek
    delta_contribution: Decimal
    gamma_contribution: Decimal
    vega_contribution: Decimal
    theta_contribution: Decimal
    rho_contribution: Decimal
    
    # Total P&L
    total_pnl: Decimal
    
    # Percentages
    delta_pct: Decimal
    gamma_pct: Decimal
    vega_pct: Decimal
    theta_pct: Decimal
    rho_pct: Decimal
    
    # Analysis
    dominant_greek: PnLCategory
    diversification_score: Decimal  # 0-1, higher = more diversified
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate attribution"""
        # Percentages should sum to ~100%
        total_pct = (self.delta_pct + self.gamma_pct + self.vega_pct + 
                     self.theta_pct + self.rho_pct)
        if abs(total_pct - Decimal('100')) > Decimal('5'):  # Allow 5% tolerance
            raise ValueError(f"Attribution percentages should sum to 100%, got {total_pct}%")
        
        if not (Decimal('0') <= self.diversification_score <= Decimal('1')):
            raise ValueError("Diversification score must be between 0 and 1")
    
    def is_delta_dominated(self) -> bool:
        """Check if P&L is dominated by delta"""
        return abs(self.delta_pct) > Decimal('70')
    
    def is_theta_strategy(self) -> bool:
        """Check if this is a theta (time decay) strategy"""
        return abs(self.theta_pct) > Decimal('50')
    
    def is_vega_strategy(self) -> bool:
        """Check if this is a vega (volatility) strategy"""
        return abs(self.vega_pct) > Decimal('50')


@dataclass(frozen=True)
class AnalyticsReport:
    """
    Complete analytics report
    
    Immutable comprehensive analysis
    """
    report_id: str
    report_period: TimePeriod
    
    # P&L
    pnl_snapshot: PnLSnapshot
    
    # Performance
    performance_metrics: PerformanceMetrics
    
    # Attribution
    greeks_attribution: GreeksAttribution
    
    # Insights
    key_insights: Tuple[str, ...]
    recommendations: Tuple[str, ...]
    
    # Quality
    data_quality_score: Decimal  # 0-1
    confidence: Decimal  # 0-1
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "analytics_agent"
    
    def __post_init__(self):
        """Validate report"""
        if not (Decimal('0') <= self.data_quality_score <= Decimal('1')):
            raise ValueError("Data quality score must be between 0 and 1")
        
        if not (Decimal('0') <= self.confidence <= Decimal('1')):
            raise ValueError("Confidence must be between 0 and 1")
    
    def is_high_quality(self, min_quality: Decimal = Decimal('0.80')) -> bool:
        """Check if report meets quality threshold"""
        return self.data_quality_score >= min_quality and self.confidence >= min_quality
    
    def get_overall_rating(self) -> PerformanceRating:
        """Get overall performance rating"""
        return self.performance_metrics.get_rating()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'report_id': self.report_id,
            'period': self.report_period.value,
            'pnl': {
                'total': float(self.pnl_snapshot.total_pnl),
                'realized': float(self.pnl_snapshot.realized_pnl),
                'unrealized': float(self.pnl_snapshot.unrealized_pnl)
            },
            'performance': {
                'return': float(self.performance_metrics.total_return),
                'sharpe': float(self.performance_metrics.sharpe_ratio),
                'win_rate': float(self.performance_metrics.win_rate),
                'rating': self.get_overall_rating().value
            },
            'attribution': {
                'delta': float(self.greeks_attribution.delta_pct),
                'gamma': float(self.greeks_attribution.gamma_pct),
                'vega': float(self.greeks_attribution.vega_pct),
                'theta': float(self.greeks_attribution.theta_pct),
                'dominant': self.greeks_attribution.dominant_greek.value
            },
            'insights': list(self.key_insights),
            'recommendations': list(self.recommendations),
            'quality': float(self.data_quality_score),
            'confidence': float(self.confidence)
        }


# Example usage
if __name__ == "__main__":
    from axiom.ai_layer.infrastructure.observability import Logger
    
    logger = Logger("analytics_domain_test")
    
    logger.info("test_starting", test="ANALYTICS DOMAIN VALUE OBJECTS")
    
    # Create P&L snapshot
    logger.info("creating_pnl_snapshot")
    
    pnl = PnLSnapshot(
        realized_pnl=Decimal('12500.00'),
        unrealized_pnl=Decimal('7800.00'),
        total_pnl=Decimal('20300.00'),
        delta_pnl=Decimal('15000.00'),
        gamma_pnl=Decimal('2500.00'),
        vega_pnl=Decimal('1800.00'),
        theta_pnl=Decimal('1000.00'),
        rho_pnl=Decimal('0.00'),
        strategy_pnl={'delta_neutral': Decimal('12000'), 'iron_condor': Decimal('8300')},
        position_pnl={'pos1': Decimal('15000'), 'pos2': Decimal('5300')},
        pnl_volatility=Decimal('2500.00'),
        max_drawdown_today=Decimal('-1200.00'),
        high_water_mark=Decimal('22000.00'),
        calculation_time_microseconds=Decimal('850')
    )
    
    logger.info(
        "pnl_snapshot_created",
        total_pnl=float(pnl.total_pnl),
        profitable=pnl.is_profitable(),
        realized_pct=float(pnl.get_realized_pct()),
        dominant_greek=pnl.get_dominant_greek().value
    )
    
    # Create performance metrics
    logger.info("creating_performance_metrics")
    
    performance = PerformanceMetrics(
        total_return=Decimal('0.28'),
        annualized_return=Decimal('0.35'),
        sharpe_ratio=Decimal('1.85'),
        sortino_ratio=Decimal('2.10'),
        calmar_ratio=Decimal('3.50'),
        volatility=Decimal('0.12'),
        max_drawdown=Decimal('-0.08'),
        var_95=Decimal('5000.00'),
        total_trades=250,
        winning_trades=158,
        losing_trades=92,
        win_rate=Decimal('0.632'),
        average_win=Decimal('850.00'),
        average_loss=Decimal('480.00'),
        largest_win=Decimal('5200.00'),
        largest_loss=Decimal('-2100.00'),
        profit_factor=Decimal('1.77'),
        consecutive_wins=8,
        consecutive_losses=4,
        recovery_factor=Decimal('3.50'),
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        days_traded=252
    )
    
    logger.info(
        "performance_metrics_created",
        sharpe_ratio=float(performance.sharpe_ratio),
        win_rate=float(performance.win_rate),
        rating=performance.get_rating().value,
        risk_adjusted_profitable=performance.is_risk_adjusted_profitable()
    )
    
    # Create Greeks attribution
    logger.info("creating_greeks_attribution")
    
    attribution = GreeksAttribution(
        time_period=TimePeriod.DAILY,
        delta_contribution=Decimal('15000.00'),
        gamma_contribution=Decimal('2500.00'),
        vega_contribution=Decimal('1800.00'),
        theta_contribution=Decimal('1000.00'),
        rho_contribution=Decimal('0.00'),
        total_pnl=Decimal('20300.00'),
        delta_pct=Decimal('73.9'),
        gamma_pct=Decimal('12.3'),
        vega_pct=Decimal('8.9'),
        theta_pct=Decimal('4.9'),
        rho_pct=Decimal('0.0'),
        dominant_greek=PnLCategory.DELTA,
        diversification_score=Decimal('0.45')
    )
    
    logger.info(
        "attribution_created",
        dominant_greek=attribution.dominant_greek.value,
        delta_dominated=attribution.is_delta_dominated(),
        diversification=float(attribution.diversification_score)
    )
    
    # Create complete report
    logger.info("creating_analytics_report")
    
    report = AnalyticsReport(
        report_id="REPORT-2024-10-30",
        report_period=TimePeriod.DAILY,
        pnl_snapshot=pnl,
        performance_metrics=performance,
        greeks_attribution=attribution,
        key_insights=(
            "P&L dominated by delta (directional moves)",
            "Sharpe ratio 1.85 indicates strong risk-adjusted returns",
            "Win rate 63.2% suggests edge in strategy selection"
        ),
        recommendations=(
            "Consider adding more theta strategies for diversification",
            "Monitor position concentration in delta-neutral strategy",
            "Excellent performance - maintain current approach"
        ),
        data_quality_score=Decimal('0.95'),
        confidence=Decimal('0.92')
    )
    
    logger.info(
        "report_created",
        report_id=report.report_id,
        rating=report.get_overall_rating().value,
        high_quality=report.is_high_quality(),
        insights_count=len(report.key_insights)
    )
    
    logger.info(
        "test_complete",
        artifacts_created=[
            "Immutable analytics objects",
            "Self-validating",
            "Rich domain behavior",
            "Type-safe with Decimal",
            "Performance analysis built-in",
            "Attribution logic",
            "Proper logging (no print)"
        ]
    )