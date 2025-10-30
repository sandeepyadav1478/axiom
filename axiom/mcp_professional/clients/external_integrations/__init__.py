"""MCP Servers Integration Package.

Comprehensive MCP (Model Context Protocol) ecosystem for Axiom platform,
providing standardized interfaces to all operational aspects.

Available Categories:
- Data: Financial data providers
- Storage: Databases and caches (PostgreSQL, Redis, etc.)
- Filesystem: File operations (read, write, search)
- DevOps: Git, Docker, Kubernetes, CI/CD
- Cloud: AWS, GCP, Azure
- Communication: Slack, Email, SMS
- Monitoring: Prometheus, Grafana, Logging
- ML Ops: Model serving, training, MLflow
- Code Quality: Linting, testing, security
- Business Intel: Analytics, reporting
- Research: Papers, patents, legal
"""

from axiom.mcp_professional.clients.external_integrations.config import MCPEcosystemConfig, MCPServerSettings, mcp_settings
from axiom.mcp_professional.clients.external_integrations.manager import (
    MCPCategory,
    MCPResource,
    MCPServer,
    MCPServerStatus,
    MCPTool,
    UnifiedMCPManager,
    mcp_manager,
)
from axiom.mcp_professional.clients.external_integrations.registry import (
    MCPRegistry,
    MCPServerFactory,
    create_registry,
)

# Week 1 Critical Servers
from axiom.mcp_professional.clients.external_integrations.filesystem.server import FilesystemMCPServer
from axiom.mcp_professional.clients.external_integrations.devops.git_server import GitMCPServer
from axiom.mcp_professional.clients.external_integrations.storage.postgres_server import PostgreSQLMCPServer
from axiom.mcp_professional.clients.external_integrations.communication.slack_server import SlackMCPServer

__all__ = [
    # Core components
    "MCPCategory",
    "MCPResource",
    "MCPServer",
    "MCPServerStatus",
    "MCPTool",
    "UnifiedMCPManager",
    "mcp_manager",
    "MCPRegistry",
    "MCPServerFactory",
    "create_registry",
    "MCPEcosystemConfig",
    "MCPServerSettings",
    "mcp_settings",
    # Server implementations
    "FilesystemMCPServer",
    "GitMCPServer",
    "PostgreSQLMCPServer",
    "SlackMCPServer",
]

__version__ = "1.0.0"
__author__ = "Axiom Platform Team"