"""
AI Client Integrations Module
Multi-provider AI integration for Investment Banking Analytics
"""

from .base_ai_provider import BaseAIProvider, AIMessage, AIResponse, AIProviderError
from .openai_provider import OpenAIProvider
from .claude_provider import ClaudeProvider
from .sglang_provider import SGLangProvider
from .provider_factory import AIProviderFactory, provider_factory, get_ai_provider, get_layer_provider, test_providers

__all__ = [
    'BaseAIProvider',
    'AIMessage',
    'AIResponse',
    'AIProviderError',
    'OpenAIProvider',
    'ClaudeProvider',
    'SGLangProvider',
    'AIProviderFactory',
    'provider_factory',
    'get_ai_provider',
    'get_layer_provider',
    'test_providers'
]