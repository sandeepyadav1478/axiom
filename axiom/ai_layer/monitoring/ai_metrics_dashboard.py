"""
AI Metrics Dashboard for Production Monitoring
Track LangGraph, DSPy, Claude performance in real-time

Architecture: Prometheus metrics + time-series tracking
Data Science: Statistical monitoring, drift detection
Production: SLA tracking, cost optimization metrics

Metrics Tracked:
- Agent execution times
- Claude API calls and costs
- Graph query performance
- Pipeline success rates
- Data quality scores
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, Any
import time
from datetime import datetime
import os


# ================================================================
# Prometheus Metrics Definitions
# ================================================================

# Agent execution metrics
agent_executions = Counter(
    'langgraph_agent_executions_total',
    'Total agent executions',
    ['agent_name', 'workflow', 'status']
)

agent_duration = Histogram(
    'langgraph_agent_duration_seconds',
    'Agent execution duration',
    ['agent_name', 'workflow'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Claude API metrics
claude_api_calls = Counter(
    'claude_api_calls_total',
    'Total Claude API calls',
    ['model', 'cache_hit']
)

claude_api_cost = Counter(
    'claude_api_cost_usd',
    'Claude API cost in USD',
    ['model']
)

claude_api_tokens = Counter(
    'claude_api_tokens_total',
    'Claude API tokens used',
    ['model', 'token_type']  # input or output
)

# Graph query metrics
neo4j_queries = Counter(
    'neo4j_queries_total',
    'Total Neo4j queries',
    ['query_type', 'status']
)

neo4j_query_duration = Histogram(
    'neo4j_query_duration_seconds',
    'Neo4j query duration',
    ['query_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# Pipeline metrics
pipeline_runs = Counter(
    'pipeline_runs_total',
    'Total pipeline runs',
    ['pipeline_name', 'status']
)

pipeline_duration = Histogram(
    'pipeline_duration_seconds',
    'Pipeline execution duration',
    ['pipeline_name'],
    buckets=[1, 5, 10, 30, 60, 300, 600]
)

# Data quality metrics
data_quality_score = Gauge(
    'data_quality_score',
    'Current data quality score (0-100)',
    ['table_name']
)

data_anomalies = Counter(
    'data_anomalies_total',
    'Data anomalies detected',
    ['severity', 'anomaly_type']
)

# System health
system_health = Gauge(
    'system_health_score',
    'Overall system health (0-1)',
    ['component']
)


class AIMetricsCollector:
    """
    Collect and expose AI/ML metrics for monitoring.
    
    Production: Prometheus integration
    Dashboards: Grafana visualization
    Alerts: Threshold-based alerting
    """
    
    def __init__(self, port: int = 9091):
        """Initialize metrics collector."""
        self.port = port
        self.start_time = time.time()
        
        # Start Prometheus HTTP server
        try:
            start_http_server(port)
            print(f"Metrics server started on port {port}")
        except Exception as e:
            print(f"Warning: Metrics server already running or port in use: {e}")
    
    def record_agent_execution(
        self,
        agent_name: str,
        workflow: str,
        duration_seconds: float,
        success: bool
    ):
        """Record agent execution metrics."""
        status = 'success' if success else 'failure'
        
        agent_executions.labels(
            agent_name=agent_name,
            workflow=workflow,
            status=status
        ).inc()
        
        agent_duration.labels(
            agent_name=agent_name,
            workflow=workflow
        ).observe(duration_seconds)
    
    def record_claude_call(
        self,
        model: str,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool = False
    ):
        """Record Claude API usage."""
        cache_status = 'hit' if cache_hit else 'miss'
        
        claude_api_calls.labels(model=model, cache_hit=cache_status).inc()
        claude_api_cost.labels(model=model).inc(cost_usd)
        claude_api_tokens.labels(model=model, token_type='input').inc(input_tokens)
        claude_api_tokens.labels(model=model, token_type='output').inc(output_tokens)
    
    def record_neo4j_query(
        self,
        query_type: str,
        duration_seconds: float,
        success: bool
    ):
        """Record Neo4j query metrics."""
        status = 'success' if success else 'failure'
        
        neo4j_queries.labels(query_type=query_type, status=status).inc()
        neo4j_query_duration.labels(query_type=query_type).observe(duration_seconds)
    
    def record_pipeline_run(
        self,
        pipeline_name: str,
        duration_seconds: float,
        success: bool
    ):
        """Record pipeline execution."""
        status = 'success' if success else 'failure'
        
        pipeline_runs.labels(pipeline_name=pipeline_name, status=status).inc()
        pipeline_duration.labels(pipeline_name=pipeline_name).observe(duration_seconds)
    
    def update_data_quality(
        self,
        table_name: str,
        quality_score: float
    ):
        """Update data quality gauge."""
        data_quality_score.labels(table_name=table_name).set(quality_score)
    
    def record_anomaly(
        self,
        severity: str,
        anomaly_type: str
    ):
        """Record data anomaly detection."""
        data_anomalies.labels(severity=severity, anomaly_type=anomaly_type).inc()
    
    def update_system_health(
        self,
        component: str,
        health_score: float  # 0-1
    ):
        """Update system health gauge."""
        system_health.labels(component=component).set(health_score)
    
    def get_uptime_seconds(self) -> float:
        """Get service uptime."""
        return time.time() - self.start_time


# Singleton instance
_metrics_collector = None

def get_metrics_collector() -> AIMetricsCollector:
    """Get or create metrics collector singleton."""
    global _metrics_collector
    
    if _metrics_collector is None:
        port = int(os.getenv('METRICS_PORT', '9091'))
        _metrics_collector = AIMetricsCollector(port=port)
    
    return _metrics_collector


# ================================================================
# Context Managers for Automatic Metric Collection
# ================================================================

class track_agent_execution:
    """Context manager to track agent execution automatically."""
    
    def __init__(self, agent_name: str, workflow: str):
        self.agent_name = agent_name
        self.workflow = workflow
        self.start_time = None
        self.collector = get_metrics_collector()
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        self.collector.record_agent_execution(
            self.agent_name,
            self.workflow,
            duration,
            success
        )
        
        return False  # Don't suppress exceptions


class track_claude_call:
    """Context manager to track Claude API calls."""
    
    def __init__(self, model: str = "claude-sonnet-4"):
        self.model = model
        self.collector = get_metrics_collector()
    
    def record(self, cost: float, input_tokens: int, output_tokens: int, cache_hit: bool = False):
        """Record Claude call metrics."""
        self.collector.record_claude_call(
            self.model,
            cost,
            input_tokens,
            output_tokens,
            cache_hit
        )


__all__ = [
    'AIMetricsCollector',
    'get_metrics_collector',
    'track_agent_execution',
    'track_claude_call'
]