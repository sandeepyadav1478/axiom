"""
AI Client Integrations Module
Multi-provider AI integration for Investment Banking Analytics
"""

from .base_ai_provider import AIMessage, AIProviderError, AIResponse, BaseAIProvider
from .claude_provider import ClaudeProvider
from .openai_provider import OpenAIProvider
from .provider_factory import (
    AIProviderFactory,
    get_ai_provider,
    get_layer_provider,
    provider_factory,
    test_providers,
)
from .sglang_provider import SGLangProvider

__all__ = [
    "BaseAIProvider",
    "AIMessage",
    "AIResponse",
    "AIProviderError",
    "OpenAIProvider",
    "ClaudeProvider",
    "SGLangProvider",
    "AIProviderFactory",
    "provider_factory",
    "get_ai_provider",
    "get_layer_provider",
    "test_providers",
]
