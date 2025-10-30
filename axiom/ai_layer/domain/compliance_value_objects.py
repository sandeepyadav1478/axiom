"""
Regulatory Compliance Domain Value Objects

Immutable value objects for regulatory compliance domain.
Following DDD principles - these capture compliance checks, reports, and violations.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on compliance violations)
- Rich behavior (compliance checking, report generation)
- Type-safe (using Decimal for precision, Enum for regulations)

These represent compliance as a first-class domain concept.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from enum import Enum


class Regulation(str, Enum):
    """Regulatory frameworks"""
    SEC_15C3_3 = "sec_15c3_3"  # Customer Protection
    FINRA_4210 = "finra_4210"  # Margin Requirements
    DODD_FRANK = "dodd_frank"  # Swap Reporting
    MIFID_II = "mifid_ii"  # EU Transaction Reporting
    EMIR = "emir"  # EU Market Infrastructure
    REG_T = "reg_t"  # Federal Reserve


class ComplianceSeverity(str, Enum):
    """Compliance issue severity"""
    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


class ReportType(str, Enum):
    """Regulatory report types"""
    DAILY_POSITION = "daily_position"
    LOPR = "lopr"  # Large Options Position Report
    BLUE_SHEET = "blue_sheet"
    BEST_EXECUTION = "best_execution"
    MARGIN_REPORT = "margin_report"
    AUDIT_TRAIL = "audit_trail"


@dataclass(frozen=True)
class ComplianceCheck:
    """
    Single compliance check result
    
    Immutable compliance validation
    """
    check_id: str
    regulation: Regulation
    check_type: str
    
    # Result
    passed: bool
    severity: ComplianceSeverity
    
    # Details
    message: str
    actual_value: Optional[Decimal] = None
    limit_value: Optional[Decimal] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    checked_by: str = "compliance_agent"
    
    def __post_init__(self):
        """Validate compliance check"""
        if self.actual_value is not None and self.limit_value is not None:
            if self.actual_value < Decimal('0') or self.limit_value < Decimal('0'):
                raise ValueError("Values must be non-negative")
    
    def is_violation(self) -> bool:
        """Check if this is a compliance violation"""
        return not self.passed and self.severity in [ComplianceSeverity.VIOLATION, ComplianceSeverity.CRITICAL]
    
    def requires_immediate_action(self) -> bool:
        """Check if violation requires immediate action"""
        return self.severity == ComplianceSeverity.CRITICAL
    
    def get_utilization_pct(self) -> Optional[Decimal]:
        """Calculate utilization percentage if applicable"""
        if self.actual_value is not None and self.limit_value is not None and self.limit_value > Decimal('0'):
            return (self.actual_value / self.limit_value) * Decimal('100')
        return None


@dataclass(frozen=True)
class PositionLimits:
    """
    Position limit configuration
    
    Immutable regulatory limits
    """
    # Position limits
    max_contracts_per_underlying: int
    max_delta_exposure: Decimal
    max_gamma_exposure: Decimal
    max_notional_exposure: Decimal
    
    # Concentration limits
    max_position_concentration_pct: Decimal  # Max % of portfolio in one position
    max_sector_concentration_pct: Decimal  # Max % in one sector
    
    # Margin limits
    max_margin_utilization_pct: Decimal
    
    # Regulatory
    large_position_threshold: int  # LOPR threshold
    
    def __post_init__(self):
        """Validate limits"""
        if self.max_contracts_per_underlying <= 0:
            raise ValueError("Max contracts must be positive")
        
        if not (Decimal('0') < self.max_position_concentration_pct <= Decimal('100')):
            raise ValueError("Concentration must be between 0 and 100%")
    
    def check_position_size(self, quantity: int) -> ComplianceCheck:
        """Check if position size is compliant"""
        passed = abs(quantity) <= self.max_contracts_per_underlying
        
        return ComplianceCheck(
            check_id=f"POS-{abs(quantity)}",
            regulation=Regulation.FINRA_4210,
            check_type="position_size",
            passed=passed,
            severity=ComplianceSeverity.VIOLATION if not passed else ComplianceSeverity.INFO,
            message=f"Position size {'compliant' if passed else 'exceeds limit'}",
            actual_value=Decimal(str(abs(quantity))),
            limit_value=Decimal(str(self.max_contracts_per_underlying))
        )
    
    def requires_lopr(self, quantity: int) -> bool:
        """Check if position requires LOPR filing"""
        return abs(quantity) > self.large_position_threshold


@dataclass(frozen=True)
class ComplianceReport:
    """
    Regulatory compliance report
    
    Immutable audit record
    """
    report_id: str
    report_type: ReportType
    regulation: Regulation
    
    # Period
    report_date: datetime
    period_start: datetime
    period_end: datetime
    
    # Entity
    reporting_entity: str
    
    # Data
    report_data: Dict
    
    # Compliance status
    compliant: bool
    issues: Tuple[ComplianceCheck, ...]
    warnings: Tuple[ComplianceCheck, ...]
    
    # Format
    format: str  # 'json', 'xml', 'csv', 'fixed_width'
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "compliance_agent"
    
    def __post_init__(self):
        """Validate report"""
        if self.period_end < self.period_start:
            raise ValueError("Period end must be after period start")
    
    def get_issue_count(self) -> int:
        """Get total number of compliance issues"""
        return len(self.issues)
    
    def get_warning_count(self) -> int:
        """Get total number of warnings"""
        return len(self.warnings)
    
    def get_critical_issues(self) -> List[ComplianceCheck]:
        """Get critical compliance issues"""
        return [check for check in self.issues if check.severity == ComplianceSeverity.CRITICAL]
    
    def requires_filing(self) -> bool:
        """Check if report requires regulatory filing"""
        return self.report_type in [ReportType.LOPR, ReportType.BLUE_SHEET, ReportType.DAILY_POSITION]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export"""
        return {
            'report_id': self.report_id,
            'type': self.report_type.value,
            'regulation': self.regulation.value,
            'date': self.report_date.isoformat(),
            'entity': self.reporting_entity,
            'compliant': self.compliant,
            'issues': len(self.issues),
            'warnings': len(self.warnings),
            'data': self.report_data,
            'format': self.format,
            'generated_at': self.generated_at.isoformat()
        }


@dataclass(frozen=True)
class AuditTrail:
    """
    Audit trail entry
    
    Immutable compliance audit record
    """
    trail_id: str
    event_type: str  # 'trade', 'order', 'modification', 'cancellation'
    
    # Event details
    event_data: Dict
    
    # Actor
    user_id: str
    agent_name: str
    
    # Timestamp (critical for audit)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Compliance
    compliant: bool = True
    compliance_notes: str = ""
    
    def __post_init__(self):
        """Validate audit trail"""
        if not self.event_type:
            raise ValueError("Event type required for audit trail")
        
        if not self.user_id:
            raise ValueError("User ID required for audit trail")
    
    def is_trade_event(self) -> bool:
        """Check if this is a trade event"""
        return self.event_type == 'trade'
    
    def is_modification(self) -> bool:
        """Check if this is a modification event"""
        return self.event_type == 'modification'


@dataclass(frozen=True)
class ComplianceStatistics:
    """
    Aggregated compliance statistics
    
    Immutable compliance performance metrics
    """
    total_checks: int
    passed_checks: int
    failed_checks: int
    
    # By severity
    critical_violations: int
    violations: int
    warnings: int
    
    # Reports
    reports_generated: int
    reports_filed: int
    
    # Audit
    audit_trail_entries: int
    
    # Period
    start_date: datetime
    end_date: datetime
    
    def __post_init__(self):
        """Validate statistics"""
        if self.total_checks < 0:
            raise ValueError("Total checks must be non-negative")
    
    def get_compliance_rate(self) -> Decimal:
        """Calculate compliance rate"""
        if self.total_checks > 0:
            return Decimal(str(self.passed_checks)) / Decimal(str(self.total_checks))
        return Decimal('1.0')
    
    def has_critical_violations(self) -> bool:
        """Check if there are critical violations"""
        return self.critical_violations > 0
    
    def is_compliant(self) -> bool:
        """Check overall compliance"""
        return self.failed_checks == 0


# Example usage
if __name__ == "__main__":
    from axiom.ai_layer.infrastructure.observability import Logger
    
    logger = Logger("compliance_domain_test")
    
    logger.info("test_starting", test="COMPLIANCE DOMAIN VALUE OBJECTS")
    
    # Create compliance check
    logger.info("creating_compliance_check")
    
    check = ComplianceCheck(
        check_id="CHK-001",
        regulation=Regulation.FINRA_4210,
        check_type="position_limit",
        passed=False,
        severity=ComplianceSeverity.VIOLATION,
        message="Position exceeds limit",
        actual_value=Decimal('12000'),
        limit_value=Decimal('10000')
    )
    
    logger.info(
        "check_created",
        passed=check.passed,
        severity=check.severity.value,
        is_violation=check.is_violation(),
        utilization=float(check.get_utilization_pct()) if check.get_utilization_pct() else None
    )
    
    # Create position limits
    logger.info("creating_position_limits")
    
    limits = PositionLimits(
        max_contracts_per_underlying=10000,
        max_delta_exposure=Decimal('50000'),
        max_gamma_exposure=Decimal('2000'),
        max_notional_exposure=Decimal('10000000'),
        max_position_concentration_pct=Decimal('25'),
        max_sector_concentration_pct=Decimal('40'),
        max_margin_utilization_pct=Decimal('80'),
        large_position_threshold=10000
    )
    
    size_check = limits.check_position_size(12000)
    logger.info(
        "position_checked",
        compliant=size_check.passed,
        requires_lopr=limits.requires_lopr(12000)
    )
    
    # Create compliance report
    logger.info("creating_compliance_report")
    
    report = ComplianceReport(
        report_id="RPT-2024-10-30",
        report_type=ReportType.DAILY_POSITION,
        regulation=Regulation.SEC_15C3_3,
        report_date=datetime.utcnow(),
        period_start=datetime(2024, 10, 30, 0, 0),
        period_end=datetime(2024, 10, 30, 23, 59),
        reporting_entity="AXIOM_DERIVATIVES",
        report_data={'positions': 150, 'notional': 5000000},
        compliant=False,
        issues=(check,),
        warnings=(),
        format='json'
    )
    
    logger.info(
        "report_created",
        report_id=report.report_id,
        compliant=report.compliant,
        issue_count=report.get_issue_count(),
        requires_filing=report.requires_filing()
    )
    
    # Create audit trail
    logger.info("creating_audit_trail")
    
    audit = AuditTrail(
        trail_id="AUDIT-001",
        event_type="trade",
        event_data={'symbol': 'SPY_C_450', 'quantity': 100, 'price': 5.50},
        user_id="trader_001",
        agent_name="execution_agent",
        compliant=True,
        compliance_notes="Trade within limits"
    )
    
    logger.info(
        "audit_trail_created",
        event_type=audit.event_type,
        is_trade=audit.is_trade_event()
    )
    
    # Create statistics
    logger.info("creating_statistics")
    
    stats = ComplianceStatistics(
        total_checks=1000,
        passed_checks=972,
        failed_checks=28,
        critical_violations=2,
        violations=15,
        warnings=11,
        reports_generated=30,
        reports_filed=30,
        audit_trail_entries=5000,
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 30)
    )
    
    logger.info(
        "statistics_created",
        compliance_rate=float(stats.get_compliance_rate()),
        has_critical=stats.has_critical_violations(),
        overall_compliant=stats.is_compliant()
    )
    
    logger.info(
        "test_complete",
        artifacts_created=[
            "Immutable compliance objects",
            "Self-validating checks",
            "Rich domain behavior",
            "Type-safe with Decimal",
            "Audit trail support",
            "Report generation",
            "Proper logging (no print)"
        ]
    )