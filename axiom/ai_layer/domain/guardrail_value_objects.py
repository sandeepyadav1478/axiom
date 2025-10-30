"""
Safety Guardrail Domain Value Objects

Immutable value objects for AI safety and guardrail domain.
Following DDD principles - these capture safety checks, validations, and veto decisions.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on unsafe actions)
- Rich behavior (safety analysis, risk assessment)
- Type-safe (using Decimal for precision, Enum for risk levels)

These represent AI safety as a first-class domain concept.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level assessment"""
    LOW = "low"  # Safe, can proceed
    MEDIUM = "medium"  # Needs validation
    HIGH = "high"  # Multiple checks required
    CRITICAL = "critical"  # Human approval needed


class ValidationCheckType(str, Enum):
    """Types of validation checks"""
    RANGE_CHECK = "range_check"
    CROSS_VALIDATION = "cross_validation"
    SANITY_CHECK = "sanity_check"
    ANOMALY_DETECTION = "anomaly_detection"
    RULE_CHECK = "rule_check"


class ActionDecision(str, Enum):
    """Guardrail decision"""
    APPROVED = "approved"
    BLOCKED = "blocked"
    REQUIRES_HUMAN = "requires_human"
    ALTERNATIVE_SUGGESTED = "alternative_suggested"


@dataclass(frozen=True)
class SafetyCheck:
    """
    Single safety validation check
    
    Immutable safety check result
    """
    check_id: str
    check_type: ValidationCheckType
    check_name: str
    
    # Result
    passed: bool
    severity: RiskLevel
    
    # Details
    message: str
    expected_value: Optional[Decimal] = None
    actual_value: Optional[Decimal] = None
    deviation_pct: Optional[Decimal] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate safety check"""
        if self.deviation_pct is not None and self.deviation_pct < Decimal('0'):
            raise ValueError("Deviation percentage must be non-negative")
    
    def is_critical_failure(self) -> bool:
        """Check if this is a critical safety failure"""
        return not self.passed and self.severity == RiskLevel.CRITICAL
    
    def get_deviation_severity(self) -> Optional[str]:
        """Get severity based on deviation"""
        if self.deviation_pct is None:
            return None
        
        if self.deviation_pct > Decimal('50'):
            return "extreme"
        elif self.deviation_pct > Decimal('20'):
            return "high"
        elif self.deviation_pct > Decimal('10'):
            return "moderate"
        else:
            return "low"


@dataclass(frozen=True)
class ValidationResult:
    """
    Complete validation result
    
    Immutable comprehensive safety assessment
    """
    action_type: str
    agent_name: str
    
    # Overall result
    passed: bool
    decision: ActionDecision
    risk_level: RiskLevel
    
    # Checks performed
    checks: Tuple[SafetyCheck, ...]
    passed_checks: int
    failed_checks: int
    
    # Issues
    critical_issues: Tuple[str, ...]
    issues: Tuple[str, ...]
    warnings: Tuple[str, ...]
    
    # Recommendations
    recommendations: Tuple[str, ...]
    alternative_action: Optional[Dict] = None
    
    # Performance
    validation_time_ms: Decimal = Decimal('0')
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate result"""
        if self.passed_checks + self.failed_checks != len(self.checks):
            raise ValueError("Check count mismatch")
    
    def requires_human_approval(self) -> bool:
        """Check if action requires human approval"""
        return (
            self.decision == ActionDecision.REQUIRES_HUMAN or
            len(self.critical_issues) > 0 or
            self.risk_level == RiskLevel.CRITICAL
        )
    
    def is_safe_to_proceed(self) -> bool:
        """Check if action is safe to proceed"""
        return self.passed and self.decision == ActionDecision.APPROVED
    
    def get_critical_check_failures(self) -> List[SafetyCheck]:
        """Get all critical check failures"""
        return [c for c in self.checks if c.is_critical_failure()]
    
    def get_approval_confidence(self) -> Decimal:
        """Calculate confidence in approval decision"""
        if len(self.checks) > 0:
            return Decimal(str(self.passed_checks)) / Decimal(str(len(self.checks)))
        return Decimal('0')


@dataclass(frozen=True)
class SafetyRule:
    """
    Safety rule definition
    
    Immutable safety constraint
    """
    rule_id: str
    rule_name: str
    description: str
    
    # Rule parameters
    threshold_value: Decimal
    comparison: str  # 'lt', 'le', 'eq', 'ge', 'gt'
    
    # Severity if violated
    violation_severity: RiskLevel
    
    # Action
    block_on_violation: bool
    escalate_on_violation: bool
    
    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate rule"""
        valid_comparisons = ['lt', 'le', 'eq', 'ge', 'gt']
        if self.comparison not in valid_comparisons:
            raise ValueError(f"Invalid comparison: {self.comparison}")
    
    def evaluate(self, value: Decimal) -> bool:
        """Evaluate rule against value"""
        if self.comparison == 'lt':
            return value < self.threshold_value
        elif self.comparison == 'le':
            return value <= self.threshold_value
        elif self.comparison == 'eq':
            return value == self.threshold_value
        elif self.comparison == 'ge':
            return value >= self.threshold_value
        elif self.comparison == 'gt':
            return value > self.threshold_value
        return False


@dataclass(frozen=True)
class GuardrailStatistics:
    """
    Aggregated guardrail statistics
    
    Immutable safety metrics
    """
    total_validations: int
    approved_actions: int
    blocked_actions: int
    human_escalations: int
    
    # By risk level
    critical_blocks: int
    high_risk_blocks: int
    medium_risk_blocks: int
    
    # Performance
    average_validation_time_ms: Decimal
    
    # Effectiveness
    false_positives: int  # Blocked safe actions
    false_negatives: int  # Approved unsafe actions (should be 0)
    
    # Period
    start_date: datetime
    end_date: datetime
    
    def __post_init__(self):
        """Validate statistics"""
        if self.total_validations < 0:
            raise ValueError("Total validations must be non-negative")
    
    def get_block_rate(self) -> Decimal:
        """Calculate block rate"""
        if self.total_validations > 0:
            return Decimal(str(self.blocked_actions)) / Decimal(str(self.total_validations))
        return Decimal('0')
    
    def get_approval_rate(self) -> Decimal:
        """Calculate approval rate"""
        if self.total_validations > 0:
            return Decimal(str(self.approved_actions)) / Decimal(str(self.total_validations))
        return Decimal('0')
    
    def is_effective(self) -> bool:
        """Check if guardrail is effective (no false negatives)"""
        return self.false_negatives == 0


# Example usage
if __name__ == "__main__":
    from axiom.ai_layer.infrastructure.observability import Logger
    
    logger = Logger("guardrail_domain_test")
    
    logger.info("test_starting", test="GUARDRAIL DOMAIN VALUE OBJECTS")
    
    # Create safety check
    logger.info("creating_safety_check")
    
    check = SafetyCheck(
        check_id="SAFE-001",
        check_type=ValidationCheckType.RANGE_CHECK,
        check_name="delta_range_validation",
        passed=False,
        severity=RiskLevel.CRITICAL,
        message="Delta out of valid range",
        expected_value=Decimal('1.0'),
        actual_value=Decimal('1.5'),
        deviation_pct=Decimal('50.0')
    )
    
    logger.info(
        "check_created",
        passed=check.passed,
        critical=check.is_critical_failure(),
        deviation_severity=check.get_deviation_severity()
    )
    
    # Create validation result
    logger.info("creating_validation_result")
    
    result = ValidationResult(
        action_type="calculate_greeks",
        agent_name="pricing_agent",
        passed=False,
        decision=ActionDecision.BLOCKED,
        risk_level=RiskLevel.CRITICAL,
        checks=(check,),
        passed_checks=0,
        failed_checks=1,
        critical_issues=("Delta out of range",),
        issues=("Delta validation failed",),
        warnings=(),
        recommendations=("Use Black-Scholes fallback",),
        validation_time_ms=Decimal('0.85')
    )
    
    logger.info(
        "result_created",
        passed=result.passed,
        decision=result.decision.value,
        requires_human=result.requires_human_approval(),
        safe_to_proceed=result.is_safe_to_proceed()
    )
    
    # Create safety rule
    logger.info("creating_safety_rule")
    
    rule = SafetyRule(
        rule_id="RULE-001",
        rule_name="max_single_trade",
        description="Maximum contracts per trade",
        threshold_value=Decimal('10000'),
        comparison='le',
        violation_severity=RiskLevel.CRITICAL,
        block_on_violation=True,
        escalate_on_violation=True
    )
    
    passes = rule.evaluate(Decimal('5000'))
    logger.info(
        "rule_created",
        rule=rule.rule_name,
        threshold=float(rule.threshold_value),
        evaluation=passes
    )
    
    # Create statistics
    logger.info("creating_statistics")
    
    stats = GuardrailStatistics(
        total_validations=10000,
        approved_actions=9250,
        blocked_actions=750,
        human_escalations=25,
        critical_blocks=15,
        high_risk_blocks=235,
        medium_risk_blocks=500,
        average_validation_time_ms=Decimal('0.75'),
        false_positives=10,
        false_negatives=0,
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 30)
    )
    
    logger.info(
        "statistics_created",
        block_rate=float(stats.get_block_rate()),
        approval_rate=float(stats.get_approval_rate()),
        effective=stats.is_effective()
    )
    
    logger.info(
        "test_complete",
        artifacts_created=[
            "Immutable safety objects",
            "Self-validating",
            "Rich domain behavior",
            "Type-safe with Decimal",
            "Risk assessment",
            "Decision tracking",
            "Proper logging (no print)"
        ]
    )