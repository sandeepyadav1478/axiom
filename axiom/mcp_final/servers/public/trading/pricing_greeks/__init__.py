"""
Pricing Greeks MCP Server Package

Industry-standard MCP server for ultra-fast option Greeks calculation.

This package provides complete MCP server implementation:
- server.py: Main MCP server
- config.json: MCP configuration
- README.md: Documentation
- Dockerfile: Container deployment
- requirements.txt: Dependencies

Compatible with: Claude Desktop, Cline, any MCP client
"""

__version__ = "1.0.0"
__author__ = "Axiom Derivatives"
__mcp_protocol_version__ = "1.0.0"

from .server import PricingGreeksMCPServer

__all__ = ['PricingGreeksMCPServer']