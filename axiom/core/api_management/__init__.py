"""
API Management Module for Investment Banking Analytics
Simple API key failover system for quota exhaustion scenarios
"""

from .failover_key_manager import (
    FailoverKeyManager,
    ProviderKeyConfig,
    failover_manager,
)

__all__ = [
    "FailoverKeyManager",
    "ProviderKeyConfig", 
    "failover_manager",
]