"""
Axiom External Integrations Module
Handles all external service integrations for Investment Banking Analytics
"""

# AI Provider integrations
from .ai_providers import (
    AIMessage,
    AIProviderFactory,
    AIResponse,
    BaseAIProvider,
    ClaudeProvider,
    OpenAIProvider,
    SGLangProvider,
    get_ai_provider,
    get_layer_provider,
    provider_factory,
)

# Search tool integrations
from .search_tools.firecrawl_client import FirecrawlClient
from .search_tools.mcp_adapter import InvestmentBankingMCPAdapter, mcp_adapter
from .search_tools.tavily_client import TavilyClient

# Data source integrations
from .data_sources.finance.base_financial_provider import BaseFinancialProvider
from .data_sources.finance.openbb_provider import OpenBBProvider

__all__ = [
    # AI Providers
    "BaseAIProvider",
    "AIMessage", 
    "AIResponse",
    "OpenAIProvider",
    "ClaudeProvider",
    "SGLangProvider",
    "AIProviderFactory",
    "provider_factory",
    "get_ai_provider",
    "get_layer_provider",
    
    # Search Tools
    "TavilyClient",
    "FirecrawlClient", 
    "InvestmentBankingMCPAdapter",
    "mcp_adapter",
    
    # Financial Data
    "BaseFinancialProvider",
    "OpenBBProvider",
]