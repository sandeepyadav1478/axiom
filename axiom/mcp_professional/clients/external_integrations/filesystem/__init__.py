"""Filesystem MCP server package."""

from axiom.mcp_professional.clients.external_integrations.filesystem.server import (
    FilesystemMCPServer,
    get_server_definition,
)

__all__ = ["FilesystemMCPServer", "get_server_definition"]