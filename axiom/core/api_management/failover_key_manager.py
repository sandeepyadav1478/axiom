"""
Simple API Key Failover System for Quota Exhaustion
When provider fails due to quota exhaustion, automatically try next API key
"""

import logging
from typing import Any

from pydantic import BaseModel, Field


class ProviderKeyConfig(BaseModel):
    """Configuration for provider API keys with failover."""
    
    provider_name: str = Field(..., description="Provider name (openai, claude, tavily)")
    api_keys: list[str] = Field(..., description="List of API keys for this provider")
    current_key_index: int = Field(default=0, description="Current active key index")
    failed_keys: set[str] = Field(default_factory=set, description="Keys that have failed")
    rotation_enabled: bool = Field(default=True, description="Enable/disable rotation")


class FailoverKeyManager:
    """Simple key failover manager for quota exhaustion scenarios."""
    
    def __init__(self, enable_rotation: bool = True):
        """
        Initialize failover key manager.
        
        Args:
            enable_rotation: Global setting to enable/disable key rotation
        """
        self.providers: dict[str, ProviderKeyConfig] = {}
        self.rotation_enabled = enable_rotation
        self.logger = logging.getLogger("failover_key_manager")
    
    def configure_provider(
        self, 
        provider_name: str, 
        api_keys: list[str] | str,
        enable_rotation: bool = True
    ):
        """
        Configure API keys for a provider.
        
        Args:
            provider_name: Provider name (openai, claude, tavily, firecrawl)
            api_keys: Single key string or list of keys
            enable_rotation: Enable rotation for this specific provider
        """
        
        keys_list = [api_keys] if isinstance(api_keys, str) else api_keys
        
        self.providers[provider_name] = ProviderKeyConfig(
            provider_name=provider_name,
            api_keys=keys_list,
            rotation_enabled=enable_rotation and self.rotation_enabled
        )
        
        self.logger.info(
            f"Configured {provider_name}: {len(keys_list)} keys, "
            f"rotation {'enabled' if enable_rotation else 'disabled'}"
        )
    
    def get_active_key(self, provider_name: str) -> str | None:
        """Get current active API key for provider."""
        
        if provider_name not in self.providers:
            return None
        
        config = self.providers[provider_name]
        
        # If rotation disabled, always return first key
        if not config.rotation_enabled:
            return config.api_keys[0] if config.api_keys else None
        
        # Find available (non-failed) keys
        available_keys = [k for k in config.api_keys if k not in config.failed_keys]
        
        if not available_keys:
            self.logger.error(f"All {provider_name} keys exhausted!")
            return None
        
        # Return current key in round-robin fashion
        return available_keys[config.current_key_index % len(available_keys)]
    
    def handle_quota_failure(self, provider_name: str, failed_key: str) -> str | None:
        """
        Handle quota exhaustion by failing over to next key.
        
        Args:
            provider_name: Provider that failed
            failed_key: The specific API key that hit quota
            
        Returns:
            Next available API key or None if all exhausted
        """
        
        if provider_name not in self.providers:
            self.logger.warning(f"Provider {provider_name} not configured for rotation")
            return failed_key  # Return original key if not configured
        
        config = self.providers[provider_name]
        
        # If rotation disabled, don't rotate
        if not config.rotation_enabled:
            self.logger.info(f"Rotation disabled for {provider_name}, keeping same key")
            return failed_key
        
        # Mark key as failed
        config.failed_keys.add(failed_key)
        
        # Find next available key
        available_keys = [k for k in config.api_keys if k not in config.failed_keys]
        
        if not available_keys:
            self.logger.critical(f"All {provider_name} API keys exhausted!")
            return None
        
        # Move to next key
        config.current_key_index = (config.current_key_index + 1) % len(available_keys)
        next_key = available_keys[config.current_key_index]
        
        self.logger.info(f"Failover: {provider_name} rotated to next API key")
        return next_key
    
    def is_quota_error(self, error_message: str) -> bool:
        """Check if error message indicates quota exhaustion."""
        
        quota_indicators = [
            "quota", "limit", "exceeded", "exhausted", "rate limit",
            "too many requests", "insufficient_quota", "usage limit"
        ]
        
        error_lower = error_message.lower()
        return any(indicator in error_lower for indicator in quota_indicators)
    
    def enable_rotation(self, provider_name: str | None = None):
        """Enable rotation globally or for specific provider."""
        
        if provider_name:
            if provider_name in self.providers:
                self.providers[provider_name].rotation_enabled = True
                self.logger.info(f"Enabled rotation for {provider_name}")
        else:
            self.rotation_enabled = True
            for config in self.providers.values():
                config.rotation_enabled = True
            self.logger.info("Enabled rotation for all providers")
    
    def disable_rotation(self, provider_name: str | None = None):
        """Disable rotation globally or for specific provider."""
        
        if provider_name:
            if provider_name in self.providers:
                self.providers[provider_name].rotation_enabled = False
                self.logger.info(f"Disabled rotation for {provider_name}")
        else:
            self.rotation_enabled = False
            for config in self.providers.values():
                config.rotation_enabled = False
            self.logger.info("Disabled rotation for all providers")
    
    def get_status(self) -> dict[str, Any]:
        """Get current failover system status."""
        
        return {
            "global_rotation_enabled": self.rotation_enabled,
            "configured_providers": list(self.providers.keys()),
            "provider_status": {
                name: {
                    "total_keys": len(config.api_keys),
                    "available_keys": len([k for k in config.api_keys if k not in config.failed_keys]),
                    "failed_keys": len(config.failed_keys),
                    "rotation_enabled": config.rotation_enabled,
                    "current_key_index": config.current_key_index
                }
                for name, config in self.providers.items()
            }
        }
    
    def reset_failed_keys(self, provider_name: str | None = None):
        """Reset failed keys (useful after daily quota reset)."""
        
        if provider_name:
            if provider_name in self.providers:
                self.providers[provider_name].failed_keys.clear()
                self.logger.info(f"Reset failed keys for {provider_name}")
        else:
            for config in self.providers.values():
                config.failed_keys.clear()
            self.logger.info("Reset failed keys for all providers")


# Global failover manager instance
failover_manager = FailoverKeyManager()