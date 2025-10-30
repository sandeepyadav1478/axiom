"""
Smart Execution MCP Server Package

Intelligent order routing and execution across 10 venues.
"""

__version__ = "1.0.0"
__author__ = "Axiom Derivatives"
__mcp_protocol_version__ = "1.0.0"

from .server import SmartExecutionMCPServer

__all__ = ['SmartExecutionMCPServer']