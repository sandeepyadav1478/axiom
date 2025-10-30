"""
Strategy Generation MCP Server Package

AI-powered trading strategy generation using Reinforcement Learning.
"""

__version__ = "1.0.0"
__author__ = "Axiom Derivatives"
__mcp_protocol_version__ = "1.0.0"

from .server import StrategyGenerationMCPServer

__all__ = ['StrategyGenerationMCPServer']