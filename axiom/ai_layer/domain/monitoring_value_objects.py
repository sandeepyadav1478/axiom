"""
System Monitoring Domain Value Objects

Immutable value objects for system monitoring and observability domain.
Following DDD principles - these capture health checks, metrics, alerts, and anomalies.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on invalid metrics)
- Rich behavior (anomaly detection, SLA tracking)
- Type-safe (using Decimal for precision, Enum for states)

These represent system health as a first-class domain concept.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from enum import Enum
from collections import deque


class HealthStatus(str, Enum):
    """Agent health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Metric categorization"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    QUEUE_DEPTH = "queue_depth"


@dataclass(frozen=True)
class AgentHealthCheck:
    """
    Health check result for single agent
    
    Immutable health assessment
    """
    agent_name: str
    status: HealthStatus
    
    # Metrics
    latency_ms: Decimal
    error_rate: Decimal
    requests_processed: int
    
    # Resource usage
    cpu_usage_pct: Optional[Decimal] = None
    memory_usage_mb: Optional[Decimal] = None
    
    # Status details
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: Decimal = Decimal('0')
    
    # Issues
    issues: Tuple[str, ...] = field(default_factory=tuple)
    
    # Metadata
    checked_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate health check"""
        if not (Decimal('0') <= self.error_rate <= Decimal('1')):
            raise ValueError("Error rate must be between 0 and 1")
        
        if self.latency_ms < Decimal('0'):
            raise ValueError("Latency must be non-negative")
        
        if self.requests_processed < 0:
            raise ValueError("Requests processed must be non-negative")
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy"""
        return self.status == HealthStatus.HEALTHY
    
    def is_degraded(self) -> bool:
        """Check if agent is degraded"""
        return self.status == HealthStatus.DEGRADED
    
    def is_down(self) -> bool:
        """Check if agent is down"""
        return self.status == HealthStatus.DOWN
    
    def requires_attention(self) -> bool:
        """Check if agent requires immediate attention"""
        return self.status in [HealthStatus.DEGRADED, HealthStatus.DOWN]
    
    def get_availability(self, total_time_seconds: Decimal) -> Decimal:
        """Calculate availability percentage"""
        if total_time_seconds > Decimal('0'):
            return (self.uptime_seconds / total_time_seconds) * Decimal('100')
        return Decimal('0')


@dataclass(frozen=True)
class MetricSnapshot:
    """
    Single metric measurement
    
    Immutable metric value at point in time
    """
    agent_name: str
    metric_type: MetricType
    metric_name: str
    value: Decimal
    
    # Statistical context
    baseline_value: Optional[Decimal] = None
    stddev_from_baseline: Optional[Decimal] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate metric"""
        if self.value < Decimal('0'):
            raise ValueError("Metric value must be non-negative")
    
    def is_anomalous(self, threshold_stddev: Decimal = Decimal('3.0')) -> bool:
        """Check if metric is anomalous (>3 standard deviations from baseline)"""
        if self.stddev_from_baseline is not None:
            return abs(self.stddev_from_baseline) > threshold_stddev
        return False
    
    def get_deviation_from_baseline(self) -> Optional[Decimal]:
        """Calculate deviation from baseline"""
        if self.baseline_value is not None:
            return self.value - self.baseline_value
        return None


@dataclass(frozen=True)
class Alert:
    """
    System alert
    
    Immutable alert notification
    """
    alert_id: str
    severity: AlertSeverity
    
    # Source
    agent_name: str
    metric_name: Optional[str] = None
    
    # Details
    message: str
    current_value: Optional[Decimal] = None
    threshold_value: Optional[Decimal] = None
    
    # Status
    acknowledged: bool = False
    resolved: bool = False
    
    # Timing
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate alert"""
        if self.acknowledged and self.acknowledged_at is None:
            raise ValueError("Acknowledged alerts must have acknowledged_at timestamp")
        
        if self.resolved and self.resolved_at is None:
            raise ValueError("Resolved alerts must have resolved_at timestamp")
    
    def is_active(self) -> bool:
        """Check if alert is still active"""
        return not self.resolved
    
    def is_critical(self) -> bool:
        """Check if alert is critical"""
        return self.severity == AlertSeverity.CRITICAL
    
    def get_duration(self) -> timedelta:
        """Get alert duration"""
        end_time = self.resolved_at if self.resolved else datetime.utcnow()
        return end_time - self.triggered_at
    
    def requires_immediate_action(self) -> bool:
        """Check if alert requires immediate action"""
        return self.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] and self.is_active()


@dataclass(frozen=True)
class SystemHealth:
    """
    Overall system health assessment
    
    Immutable system-wide health snapshot
    """
    # Agent health
    agent_health: Dict[str, AgentHealthCheck]
    
    # Overall status
    overall_status: HealthStatus
    healthy_agents: int
    degraded_agents: int
    down_agents: int
    
    # Alerts
    active_alerts: Tuple[Alert, ...]
    critical_alerts: int
    
    # Metrics
    system_latency_p50_ms: Decimal
    system_latency_p95_ms: Decimal
    system_latency_p99_ms: Decimal
    system_error_rate: Decimal
    system_throughput_rps: Decimal
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate system health"""
        total = self.healthy_agents + self.degraded_agents + self.down_agents
        if total != len(self.agent_health):
            raise ValueError("Agent count mismatch")
    
    def get_availability_pct(self) -> Decimal:
        """Calculate system availability percentage"""
        total = self.healthy_agents + self.degraded_agents + self.down_agents
        if total > 0:
            return (Decimal(str(self.healthy_agents)) / Decimal(str(total))) * Decimal('100')
        return Decimal('0')
    
    def is_system_healthy(self) -> bool:
        """Check if overall system is healthy"""
        return self.overall_status == HealthStatus.HEALTHY and self.down_agents == 0
    
    def requires_escalation(self) -> bool:
        """Check if situation requires escalation"""
        return self.critical_alerts > 0 or self.down_agents > 0


# Example usage
if __name__ == "__main__":
    from axiom.ai_layer.infrastructure.observability import Logger
    
    logger = Logger("monitoring_domain_test")
    
    logger.info("test_starting", test="MONITORING DOMAIN VALUE OBJECTS")
    
    # Create agent health check
    logger.info("creating_health_check")
    
    health = AgentHealthCheck(
        agent_name="pricing_agent",
        status=HealthStatus.HEALTHY,
        latency_ms=Decimal('0.85'),
        error_rate=Decimal('0.0001'),
        requests_processed=10000,
        cpu_usage_pct=Decimal('15.5'),
        memory_usage_mb=Decimal('250'),
        uptime_seconds=Decimal('86400'),
        issues=()
    )
    
    logger.info(
        "health_check_created",
        agent=health.agent_name,
        status=health.status.value,
        healthy=health.is_healthy(),
        availability=float(health.get_availability(Decimal('86400')))
    )
    
    # Create metric snapshot
    logger.info("creating_metric_snapshot")
    
    metric = MetricSnapshot(
        agent_name="risk_agent",
        metric_type=MetricType.LATENCY,
        metric_name="portfolio_risk_latency",
        value=Decimal('4.2'),
        baseline_value=Decimal('5.0'),
        stddev_from_baseline=Decimal('-0.4')
    )
    
    logger.info(
        "metric_created",
        metric=metric.metric_name,
        value=float(metric.value),
        anomalous=metric.is_anomalous()
    )
    
    # Create alert
    logger.info("creating_alert")
    
    alert = Alert(
        alert_id="ALERT-001",
        severity=AlertSeverity.WARNING,
        agent_name="execution_agent",
        metric_name="slippage_bps",
        message="High slippage detected",
        current_value=Decimal('8.5'),
        threshold_value=Decimal('5.0'),
        acknowledged=False,
        resolved=False
    )
    
    logger.info(
        "alert_created",
        severity=alert.severity.value,
        active=alert.is_active(),
        requires_action=alert.requires_immediate_action()
    )
    
    # Create system health
    logger.info("creating_system_health")
    
    system = SystemHealth(
        agent_health={'pricing': health},
        overall_status=HealthStatus.HEALTHY,
        healthy_agents=11,
        degraded_agents=1,
        down_agents=0,
        active_alerts=(alert,),
        critical_alerts=0,
        system_latency_p50_ms=Decimal('2.5'),
        system_latency_p95_ms=Decimal('8.0'),
        system_latency_p99_ms=Decimal('15.0'),
        system_error_rate=Decimal('0.002'),
        system_throughput_rps=Decimal('1000')
    )
    
    logger.info(
        "system_health_created",
        overall_status=system.overall_status.value,
        availability=float(system.get_availability_pct()),
        system_healthy=system.is_system_healthy(),
        requires_escalation=system.requires_escalation()
    )
    
    logger.info(
        "test_complete",
        artifacts_created=[
            "Immutable monitoring objects",
            "Self-validating metrics",
            "Rich domain behavior",
            "Type-safe with Decimal",
            "Health tracking",
            "Alert management",
            "Proper logging (no print)"
        ]
    )