"""Prometheus MCP Server Implementation.

Provides Prometheus metrics collection and monitoring through MCP protocol:
- PromQL queries (instant and range)
- Alert management
- Metric recording
- Target monitoring
- Health checks
"""

import asyncio
import logging
import time
from typing import Any, Optional
from datetime import datetime, timedelta

try:
    import aiohttp
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    aiohttp = None
    Counter = None
    Gauge = None
    Histogram = None
    Summary = None
    CollectorRegistry = None

logger = logging.getLogger(__name__)


class PrometheusMCPServer:
    """Prometheus MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "Required libraries not installed. "
                "Install with: pip install aiohttp prometheus-client"
            )
        
        self.config = config
        self.url = config.get("url", "http://localhost:9090")
        self.auth_token = config.get("auth_token")
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._registry = CollectorRegistry()
        self._custom_metrics: dict[str, Any] = {}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is created.

        Returns:
            Aiohttp client session
        """
        if self._session is None or self._session.closed:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            self._session = aiohttp.ClientSession(headers=headers)
            logger.info(f"Connected to Prometheus at {self.url}")
        
        return self._session

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info("Prometheus session closed")

    async def query(
        self,
        promql: str,
        timeout: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute instant PromQL query.

        Args:
            promql: PromQL query string
            timeout: Query timeout (e.g., "30s")

        Returns:
            Query results
        """
        start_time = time.time()
        
        try:
            session = await self._ensure_session()
            
            params = {"query": promql}
            if timeout:
                params["timeout"] = timeout
            
            async with session.get(
                f"{self.url}/api/v1/query",
                params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get("status") != "success":
                    return {
                        "success": False,
                        "error": data.get("error", "Query failed"),
                        "query": promql,
                    }
                
                result = data.get("data", {}).get("result", [])
                result_type = data.get("data", {}).get("resultType")
                
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "success": True,
                    "query": promql,
                    "result_type": result_type,
                    "result": result,
                    "count": len(result),
                    "latency_ms": latency_ms,
                }

        except aiohttp.ClientError as e:
            logger.error(f"Prometheus query failed: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {str(e)}",
                "query": promql,
            }
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return {
                "success": False,
                "error": f"Failed to execute query: {str(e)}",
                "query": promql,
            }

    async def query_range(
        self,
        promql: str,
        start: str,
        end: str,
        step: str = "15s",
        timeout: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute range PromQL query.

        Args:
            promql: PromQL query string
            start: Start time (RFC3339 or Unix timestamp)
            end: End time (RFC3339 or Unix timestamp)
            step: Query resolution step (e.g., "15s", "1m")
            timeout: Query timeout

        Returns:
            Query results with time series
        """
        start_time = time.time()
        
        try:
            session = await self._ensure_session()
            
            params = {
                "query": promql,
                "start": start,
                "end": end,
                "step": step,
            }
            if timeout:
                params["timeout"] = timeout
            
            async with session.get(
                f"{self.url}/api/v1/query_range",
                params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get("status") != "success":
                    return {
                        "success": False,
                        "error": data.get("error", "Query failed"),
                        "query": promql,
                    }
                
                result = data.get("data", {}).get("result", [])
                result_type = data.get("data", {}).get("resultType")
                
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "success": True,
                    "query": promql,
                    "start": start,
                    "end": end,
                    "step": step,
                    "result_type": result_type,
                    "result": result,
                    "count": len(result),
                    "latency_ms": latency_ms,
                }

        except aiohttp.ClientError as e:
            logger.error(f"Prometheus range query failed: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {str(e)}",
                "query": promql,
            }
        except Exception as e:
            logger.error(f"Failed to execute range query: {e}")
            return {
                "success": False,
                "error": f"Failed to execute range query: {str(e)}",
                "query": promql,
            }

    async def create_alert(
        self,
        name: str,
        expr: str,
        duration: str = "5m",
        severity: str = "warning",
        description: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create an alert rule (returns rule definition).

        Args:
            name: Alert name
            expr: PromQL expression for alert condition
            duration: Alert firing duration
            severity: Alert severity (info, warning, critical)
            description: Alert description

        Returns:
            Alert rule definition
        """
        try:
            alert_rule = {
                "alert": name,
                "expr": expr,
                "for": duration,
                "labels": {
                    "severity": severity,
                },
                "annotations": {
                    "summary": description or f"Alert: {name}",
                    "description": description or f"Alert triggered by: {expr}",
                },
            }
            
            return {
                "success": True,
                "alert": name,
                "rule": alert_rule,
                "message": "Alert rule definition created. Add to Prometheus configuration to activate.",
            }

        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            return {
                "success": False,
                "error": f"Failed to create alert: {str(e)}",
                "alert": name,
            }

    async def list_alerts(self) -> dict[str, Any]:
        """List active alerts.

        Returns:
            List of active alerts
        """
        try:
            session = await self._ensure_session()
            
            async with session.get(
                f"{self.url}/api/v1/alerts"
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get("status") != "success":
                    return {
                        "success": False,
                        "error": data.get("error", "Failed to list alerts"),
                    }
                
                alerts = data.get("data", {}).get("alerts", [])
                
                # Organize alerts by state
                by_state = {}
                for alert in alerts:
                    state = alert.get("state", "unknown")
                    if state not in by_state:
                        by_state[state] = []
                    by_state[state].append(alert)
                
                return {
                    "success": True,
                    "alerts": alerts,
                    "count": len(alerts),
                    "by_state": by_state,
                }

        except aiohttp.ClientError as e:
            logger.error(f"Failed to list alerts: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Failed to list alerts: {e}")
            return {
                "success": False,
                "error": f"Failed to list alerts: {str(e)}",
            }

    async def get_metrics(
        self,
        metric_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get current metric values.

        Args:
            metric_name: Specific metric name (None for all)

        Returns:
            Metric values
        """
        try:
            session = await self._ensure_session()
            
            # Get metadata for metrics
            async with session.get(
                f"{self.url}/api/v1/metadata"
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get("status") != "success":
                    return {
                        "success": False,
                        "error": data.get("error", "Failed to get metrics"),
                    }
                
                metadata = data.get("data", {})
                
                # Filter by metric name if provided
                if metric_name:
                    metadata = {
                        k: v for k, v in metadata.items()
                        if k == metric_name
                    }
                
                return {
                    "success": True,
                    "metrics": metadata,
                    "count": len(metadata),
                }

        except aiohttp.ClientError as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "success": False,
                "error": f"Failed to get metrics: {str(e)}",
            }

    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: str = "gauge",
        labels: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Record a custom metric (using prometheus_client).

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type (counter, gauge, histogram, summary)
            labels: Metric labels

        Returns:
            Recording result
        """
        try:
            labels = labels or {}
            
            # Create or get metric
            if name not in self._custom_metrics:
                if metric_type == "counter":
                    metric = Counter(
                        name,
                        f"Custom counter metric: {name}",
                        labelnames=list(labels.keys()),
                        registry=self._registry,
                    )
                elif metric_type == "gauge":
                    metric = Gauge(
                        name,
                        f"Custom gauge metric: {name}",
                        labelnames=list(labels.keys()),
                        registry=self._registry,
                    )
                elif metric_type == "histogram":
                    metric = Histogram(
                        name,
                        f"Custom histogram metric: {name}",
                        labelnames=list(labels.keys()),
                        registry=self._registry,
                    )
                elif metric_type == "summary":
                    metric = Summary(
                        name,
                        f"Custom summary metric: {name}",
                        labelnames=list(labels.keys()),
                        registry=self._registry,
                    )
                else:
                    return {
                        "success": False,
                        "error": f"Unknown metric type: {metric_type}",
                        "name": name,
                    }
                
                self._custom_metrics[name] = metric
            else:
                metric = self._custom_metrics[name]
            
            # Record value
            if labels:
                metric = metric.labels(**labels)
            
            if metric_type == "counter":
                metric.inc(value)
            elif metric_type == "gauge":
                metric.set(value)
            elif metric_type in ["histogram", "summary"]:
                metric.observe(value)
            
            return {
                "success": True,
                "name": name,
                "value": value,
                "type": metric_type,
                "labels": labels,
            }

        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
            return {
                "success": False,
                "error": f"Failed to record metric: {str(e)}",
                "name": name,
            }

    async def get_targets(self) -> dict[str, Any]:
        """Get scrape targets.

        Returns:
            List of scrape targets
        """
        try:
            session = await self._ensure_session()
            
            async with session.get(
                f"{self.url}/api/v1/targets"
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get("status") != "success":
                    return {
                        "success": False,
                        "error": data.get("error", "Failed to get targets"),
                    }
                
                targets = data.get("data", {})
                active_targets = targets.get("activeTargets", [])
                dropped_targets = targets.get("droppedTargets", [])
                
                return {
                    "success": True,
                    "active_targets": active_targets,
                    "dropped_targets": dropped_targets,
                    "active_count": len(active_targets),
                    "dropped_count": len(dropped_targets),
                }

        except aiohttp.ClientError as e:
            logger.error(f"Failed to get targets: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Failed to get targets: {e}")
            return {
                "success": False,
                "error": f"Failed to get targets: {str(e)}",
            }


def get_server_definition() -> dict[str, Any]:
    """Get Prometheus MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "prometheus",
        "category": "monitoring",
        "description": "Prometheus metrics collection and monitoring (queries, alerts, targets)",
        "tools": [
            {
                "name": "query",
                "description": "Execute instant PromQL query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "promql": {
                            "type": "string",
                            "description": "PromQL query string (e.g., 'up', 'rate(http_requests_total[5m])')",
                        },
                        "timeout": {
                            "type": "string",
                            "description": "Query timeout (e.g., '30s')",
                        },
                    },
                    "required": ["promql"],
                },
            },
            {
                "name": "query_range",
                "description": "Execute range PromQL query for time series",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "promql": {
                            "type": "string",
                            "description": "PromQL query string",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start time (RFC3339 or Unix timestamp)",
                        },
                        "end": {
                            "type": "string",
                            "description": "End time (RFC3339 or Unix timestamp)",
                        },
                        "step": {
                            "type": "string",
                            "description": "Query resolution step (e.g., '15s', '1m')",
                            "default": "15s",
                        },
                        "timeout": {
                            "type": "string",
                            "description": "Query timeout",
                        },
                    },
                    "required": ["promql", "start", "end"],
                },
            },
            {
                "name": "create_alert",
                "description": "Create alert rule definition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Alert name",
                        },
                        "expr": {
                            "type": "string",
                            "description": "PromQL expression for alert condition",
                        },
                        "duration": {
                            "type": "string",
                            "description": "Alert firing duration (e.g., '5m')",
                            "default": "5m",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warning", "critical"],
                            "description": "Alert severity",
                            "default": "warning",
                        },
                        "description": {
                            "type": "string",
                            "description": "Alert description",
                        },
                    },
                    "required": ["name", "expr"],
                },
            },
            {
                "name": "list_alerts",
                "description": "List active alerts",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "get_metrics",
                "description": "Get current metric values and metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "Specific metric name (omit for all metrics)",
                        }
                    },
                },
            },
            {
                "name": "record_metric",
                "description": "Record a custom metric value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Metric name",
                        },
                        "value": {
                            "type": "number",
                            "description": "Metric value",
                        },
                        "metric_type": {
                            "type": "string",
                            "enum": ["counter", "gauge", "histogram", "summary"],
                            "description": "Metric type",
                            "default": "gauge",
                        },
                        "labels": {
                            "type": "object",
                            "description": "Metric labels (key-value pairs)",
                        },
                    },
                    "required": ["name", "value"],
                },
            },
            {
                "name": "get_targets",
                "description": "Get Prometheus scrape targets",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "critical",
            "category": "monitoring",
            "requires": ["aiohttp", "prometheus-client>=0.19.0"],
            "performance_target": "<50ms for queries",
        },
    }