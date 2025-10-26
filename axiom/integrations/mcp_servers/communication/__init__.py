"""Communication MCP servers package."""

from axiom.integrations.mcp_servers.communication.slack_server import (
    SlackMCPServer,
    get_server_definition,
)

__all__ = ["SlackMCPServer", "get_server_definition"]