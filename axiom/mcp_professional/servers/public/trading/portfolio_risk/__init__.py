"""
Portfolio Risk MCP Server Package

Industry-standard MCP server for real-time portfolio risk management.
"""

__version__ = "1.0.0"
__author__ = "Axiom Derivatives"
__mcp_protocol_version__ = "1.0.0"

from .server import PortfolioRiskMCPServer

__all__ = ['PortfolioRiskMCPServer']