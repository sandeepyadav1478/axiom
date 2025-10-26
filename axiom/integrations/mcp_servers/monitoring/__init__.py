"""Monitoring MCP Servers.

Provides monitoring and observability tools through MCP:
- Prometheus metrics and queries
- Grafana dashboards
- Logging aggregation
- Health checks
"""

from typing import Any

__all__ = ["prometheus_server"]


def get_available_servers() -> list[str]:
    """Get list of available monitoring servers.
    
    Returns:
        List of server names
    """
    return ["prometheus"]