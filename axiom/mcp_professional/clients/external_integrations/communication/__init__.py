"""Communication MCP servers package."""

from axiom.mcp_professional.clients.external_integrations.communication.slack_server import (
    SlackMCPServer,
    get_server_definition,
)

__all__ = ["SlackMCPServer", "get_server_definition"]