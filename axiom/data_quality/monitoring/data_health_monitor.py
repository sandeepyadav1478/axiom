"""
Data Health Monitoring System - Production Operations

Real-time monitoring of data quality metrics with alerting.
Critical for production data reliability and operations excellence.

Monitors:
- Data quality metrics (continuous)
- Anomaly rates
- Data freshness
- Pipeline health
- SLA compliance

This ensures data quality doesn't degrade in production!
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time


class HealthStatus(Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Single health metric measurement."""
    
    metric_name: str
    value: float
    threshold: float
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if metric is within healthy range."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


@dataclass
class DataHealthAlert:
    """Data health alert."""
    
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    affected_component: str
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)
    auto_remediation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "title": self.title,
            "description": self.description,
            "affected_component": self.affected_component,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "recommendations": self.recommendations,
            "auto_remediation": self.auto_remediation
        }


class DataHealthMonitor:
    """
    Data health monitoring system.
    
    Continuously monitors data quality and generates alerts
    when metrics degrade below thresholds.
    
    SLA Targets:
    - Data Quality Score: >= 85%
    - Data Freshness: < 1 hour
    - Anomaly Rate: < 1%
    - Validation Pass Rate: >= 95%
    """
    
    def __init__(self):
        """Initialize health monitor."""
        # SLA thresholds
        self.sla_thresholds = {
            "quality_score": 85.0,
            "data_freshness_hours": 1.0,
            "anomaly_rate_percent": 1.0,
            "validation_pass_rate": 95.0,
            "completeness_percent": 98.0
        }
        
        # Health metrics history
        self.metrics_history: List[HealthMetric] = []
        
        # Active alerts
        self.active_alerts: List[DataHealthAlert] = []
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring state
        self.monitoring_enabled = True
        self.last_check: Optional[datetime] = None
    
    def check_health(
        self,
        quality_score: float,
        anomaly_count: int,
        total_records: int,
        validation_results: Optional[List] = None,
        data_timestamp: Optional[datetime] = None
    ) -> Dict[str, HealthMetric]:
        """
        Perform comprehensive health check.
        
        Args:
            quality_score: Overall data quality score (0-100)
            anomaly_count: Number of anomalies detected
            total_records: Total number of records
            validation_results: Validation results
            data_timestamp: Timestamp of latest data
        
        Returns:
            Dictionary of health metrics
        """
        metrics = {}
        
        # 1. Quality Score Metric
        metrics["quality_score"] = HealthMetric(
            metric_name="quality_score",
            value=quality_score,
            threshold=self.sla_thresholds["quality_score"],
            status=self._get_status(
                quality_score,
                self.sla_thresholds["quality_score"],
                higher_is_better=True
            )
        )
        
        # 2. Anomaly Rate Metric
        anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0
        metrics["anomaly_rate"] = HealthMetric(
            metric_name="anomaly_rate",
            value=anomaly_rate,
            threshold=self.sla_thresholds["anomaly_rate_percent"],
            status=self._get_status(
                anomaly_rate,
                self.sla_thresholds["anomaly_rate_percent"],
                higher_is_better=False
            )
        )
        
        # 3. Data Freshness Metric
        if data_timestamp:
            age_hours = (datetime.now() - data_timestamp).total_seconds() / 3600
            metrics["data_freshness"] = HealthMetric(
                metric_name="data_freshness",
                value=age_hours,
                threshold=self.sla_thresholds["data_freshness_hours"],
                status=self._get_status(
                    age_hours,
                    self.sla_thresholds["data_freshness_hours"],
                    higher_is_better=False
                ),
                metadata={"age_hours": age_hours}
            )
        
        # 4. Validation Pass Rate
        if validation_results:
            passed = sum(1 for r in validation_results if r.passed)
            total = len(validation_results)
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            metrics["validation_pass_rate"] = HealthMetric(
                metric_name="validation_pass_rate",
                value=pass_rate,
                threshold=self.sla_thresholds["validation_pass_rate"],
                status=self._get_status(
                    pass_rate,
                    self.sla_thresholds["validation_pass_rate"],
                    higher_is_better=True
                )
            )
        
        # Store metrics history
        self.metrics_history.extend(metrics.values())
        self.last_check = datetime.now()
        
        # Generate alerts if needed
        self._generate_alerts(metrics)
        
        return metrics
    
    def _get_status(
        self,
        value: float,
        threshold: float,
        higher_is_better: bool
    ) -> HealthStatus:
        """Determine health status based on value and threshold."""
        
        if higher_is_better:
            if value >= threshold:
                return HealthStatus.HEALTHY
            elif value >= threshold * 0.9:
                return HealthStatus.DEGRADED
            elif value >= threshold * 0.75:
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.CRITICAL
        else:
            if value <= threshold:
                return HealthStatus.HEALTHY
            elif value <= threshold * 1.5:
                return HealthStatus.DEGRADED
            elif value <= threshold * 2.0:
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.CRITICAL
    
    def _generate_alerts(
        self,
        metrics: Dict[str, HealthMetric]
    ) -> None:
        """Generate alerts for unhealthy metrics."""
        
        for metric in metrics.values():
            if metric.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                alert = self._create_alert(metric)
                self.active_alerts.append(alert)
                
                # Trigger alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        print(f"Alert callback error: {e}")
    
    def _create_alert(self, metric: HealthMetric) -> DataHealthAlert:
        """Create alert from unhealthy metric."""
        
        alert_level = AlertLevel.CRITICAL if metric.status == HealthStatus.CRITICAL else AlertLevel.ERROR
        
        alert_id = f"{metric.metric_name}_{int(time.time())}"
        
        recommendations = self._get_recommendations(metric)
        
        return DataHealthAlert(
            alert_id=alert_id,
            level=alert_level,
            title=f"Data Health: {metric.metric_name} {metric.status.value}",
            description=f"{metric.metric_name} is {metric.value:.2f}, threshold is {metric.threshold:.2f}",
            affected_component=metric.metric_name,
            metric_value=metric.value,
            threshold_value=metric.threshold,
            recommendations=recommendations
        )
    
    def _get_recommendations(self, metric: HealthMetric) -> List[str]:
        """Get recommendations based on metric."""
        
        recommendations = []
        
        if metric.metric_name == "quality_score":
            if metric.value < 85:
                recommendations.append("Review data validation results for common failures")
                recommendations.append("Check data source quality")
                recommendations.append("Increase data cleaning efforts")
        
        elif metric.metric_name == "anomaly_rate":
            if metric.value > 1.0:
                recommendations.append("Investigate source of anomalies")
                recommendations.append("Review data collection process")
                recommendations.append("Check for system/market events")
        
        elif metric.metric_name == "data_freshness":
            if metric.value > 1.0:
                recommendations.append("Check data pipeline execution")
                recommendations.append("Verify data source availability")
                recommendations.append("Review ingestion schedule")
        
        return recommendations
    
    def register_alert_callback(self, callback: Callable[[DataHealthAlert], None]) -> None:
        """Register callback function to be called when alerts are generated."""
        self.alert_callbacks.append(callback)
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.metrics_history:
            return HealthStatus.HEALTHY
        
        # Get recent metrics (last hour)
        recent = [
            m for m in self.metrics_history
            if (datetime.now() - m.timestamp).seconds < 3600
        ]
        
        if not recent:
            return HealthStatus.HEALTHY
        
        # Check for any critical status
        if any(m.status == HealthStatus.CRITICAL for m in recent):
            return HealthStatus.CRITICAL
        
        # Check for multiple unhealthy
        unhealthy_count = sum(1 for m in recent if m.status == HealthStatus.UNHEALTHY)
        if unhealthy_count > 2:
            return HealthStatus.UNHEALTHY
        
        # Check for degraded
        if any(m.status == HealthStatus.DEGRADED for m in recent):
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None
    ) -> List[DataHealthAlert]:
        """Get active alerts, optionally filtered by level."""
        
        if level:
            return [a for a in self.active_alerts if a.level == level]
        return self.active_alerts
    
    def clear_alerts(self) -> int:
        """Clear all active alerts. Returns count of cleared alerts."""
        count = len(self.active_alerts)
        self.active_alerts = []
        return count
    
    def get_health_dashboard(self) -> Dict[str, Any]:
        """
        Get health dashboard data for visualization.
        
        Returns:
            Complete health status for dashboard display
        """
        overall_health = self.get_overall_health()
        
        # Get recent metrics
        recent_metrics = {}
        for metric_name in ["quality_score", "anomaly_rate", "data_freshness", "validation_pass_rate"]:
            recent = [
                m for m in self.metrics_history
                if m.metric_name == metric_name
                and (datetime.now() - m.timestamp).seconds < 3600
            ]
            if recent:
                recent_metrics[metric_name] = {
                    "current_value": recent[-1].value,
                    "threshold": recent[-1].threshold,
                    "status": recent[-1].status.value,
                    "trend": "improving" if len(recent) > 1 and recent[-1].value > recent[-2].value else "stable"
                }
        
        return {
            "overall_health": overall_health.value,
            "metrics": recent_metrics,
            "active_alerts": {
                "total": len(self.active_alerts),
                "critical": len([a for a in self.active_alerts if a.level == AlertLevel.CRITICAL]),
                "error": len([a for a in self.active_alerts if a.level == AlertLevel.ERROR]),
                "warning": len([a for a in self.active_alerts if a.level == AlertLevel.WARNING])
            },
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "monitoring_enabled": self.monitoring_enabled
        }


# Singleton instance
_health_monitor: Optional[DataHealthMonitor] = None


def get_health_monitor() -> DataHealthMonitor:
    """Get or create singleton health monitor."""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = DataHealthMonitor()
    
    return _health_monitor


if __name__ == "__main__":
    # Example usage
    monitor = get_health_monitor()
    
    # Register alert callback
    def handle_alert(alert: DataHealthAlert):
        print(f"🚨 ALERT [{alert.level.value}]: {alert.title}")
        print(f"   {alert.description}")
        if alert.recommendations:
            print(f"   Recommendations: {alert.recommendations[0]}")
    
    monitor.register_alert_callback(handle_alert)
    
    # Simulate health check
    print("Data Health Monitoring Test")
    print("=" * 60)
    
    # Good health
    metrics = monitor.check_health(
        quality_score=92.0,
        anomaly_count=5,
        total_records=1000,
        data_timestamp=datetime.now()
    )
    
    print(f"Overall Health: {monitor.get_overall_health().value}")
    
    for name, metric in metrics.items():
        print(f"{name}: {metric.value:.2f} ({metric.status.value})")
    
    # Simulate degraded health
    print("\n--- Simulating Quality Degradation ---")
    
    metrics = monitor.check_health(
        quality_score=75.0,  # Below threshold!
        anomaly_count=50,     # High anomalies!
        total_records=1000,
        data_timestamp=datetime.now() - timedelta(hours=3)  # Stale data!
    )
    
    dashboard = monitor.get_health_dashboard()
    print(f"\nDashboard Status: {dashboard['overall_health']}")
    print(f"Active Alerts: {dashboard['active_alerts']['total']}")
    print(f"Critical Alerts: {dashboard['active_alerts']['critical']}")
    
    print("\n✅ Health monitoring system operational!")