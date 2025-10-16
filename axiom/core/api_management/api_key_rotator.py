"""
Simple API Key Rotation for Quota Exhaustion Failover
When API quota exhausted, automatically try next key for same provider
"""

from typing import Any
from datetime import datetime
import logging

from pydantic import BaseModel, Field


class APIKeyPool(BaseModel):
    """Simple API key pool for a provider with failover."""
    
    provider_name: str = Field(..., description="Provider name")
    keys: list[str] = Field(..., description="List of API keys")
    current_index: int = Field(default=0, description="Current key index")
    failed_keys: set[str] = Field(default_factory=set, description="Failed keys")
    rotation_enabled: bool = Field(default=True, description="Enable key rotation")
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_current_key(self) -> str | None:
        """Get current active API key."""
        if not self.keys or not self.rotation_enabled:
            return self.keys[0] if self.keys else None
        
        # Find next available key
        available_keys = [k for k in self.keys if k not in self.failed_keys]
        
        if not available_keys:
            logging.warning(f"All {self.provider_name} keys exhausted!")
            return None
        
        return available_keys[self.current_index % len(available_keys)]
    
    def rotate_to_next_key(self, failed_key: str, reason: str = "quota_exhausted") -> str | None:
        """Rotate to next available key when current key fails."""
        
        if not self.rotation_enabled:
            logging.warning(f"Key rotation disabled for {self.provider_name}")
            return self.keys[0] if self.keys else None
        
        # Mark current key as failed
        if failed_key in self.keys:
            self.failed_keys.add(failed_key)
            logging.info(f"Marked {self.provider_name} key as failed: {reason}")
        
        # Move to next key
        available_keys = [k for k in self.keys if k not in self.failed_keys]
        
        if not available_keys:
            logging.error(f"All {self.provider_name} API keys exhausted!")
            return None
        
        # Simple round-robin to next available key
        self.current_index = (self.current_index + 1) % len(available_keys)
        next_key = available_keys[self.current_index]
        
        logging.info(f"Rotated {self.provider_name} to next API key")
        return next_key


class APIKeyRotator:
    """Simple API key rotation manager for all providers."""
    
    def __init__(self):
        self.key_pools: dict[str, APIKeyPool] = {}
        self.rotation_enabled = True
        self.logger = logging.getLogger("api_key_rotator")
    
    def add_provider_keys(
        self, 
        provider_name: str, 
        api_keys: list[str] | str,
        rotation_enabled: bool = True
    ):
        """Add API keys for a provider."""
        
        keys_list = [api_keys] if isinstance(api_keys, str) else api_keys
        
        self.key_pools[provider_name] = APIKeyPool(
            provider_name=provider_name,
            keys=keys_list,
            rotation_enabled=rotation_enabled
        )
        
        self.logger.info(
            f"Added {len(keys_list)} API keys for {provider_name} "
            f"(rotation: {'enabled' if rotation_enabled else 'disabled'})"
        )
    
    def get_current_key(self, provider_name: str) -> str | None:
        """Get current API key for provider."""
        
        if provider_name not in self.key_pools:
            return None
        
        return self.key_pools[provider_name].get_current_key()
    
    def handle_api_failure(
        self, 
        provider_name: str, 
        failed_key: str,
        error_message: str = ""
    ) -> str | None:
        """
        Handle API failure by rotating to next key if quota exhausted.
        
        Args:
            provider_name: Name of provider (openai, claude, tavily, etc.)
            failed_key: The API key that failed
            error_message: Error message to determine if it's quota exhaustion
            
        Returns:
            Next available API key or None if all exhausted
        """
        
        if provider_name not in self.key_pools:
            return None
        
        error_lower = error_message.lower()
        
        # Check if error is quota/limit related (eligible for rotation)
        quota_errors = [
            "quota", "limit", "exceeded", "exhausted", "rate limit",
            "too many requests", "insufficient_quota"
        ]
        
        is_quota_error = any(term in error_lower for term in quota_errors)
        
        if is_quota_error:
            self.logger.info(f"Quota exhaustion detected for {provider_name}: {error_message}")
            return self.key_pools[provider_name].rotate_to_next_key(failed_key, "quota_exhausted")
        else:
            # Non-quota error (invalid key, network issue, etc.) - don't rotate
            self.logger.warning(f"Non-quota error for {provider_name}: {error_message}")
            return failed_key  # Keep using same key
    
    def enable_rotation(self, provider_name: str | None = None):
        """Enable rotation for specific provider or all providers."""
        
        if provider_name:
            if provider_name in self.key_pools:
                self.key_pools[provider_name].rotation_enabled = True
                self.logger.info(f"Enabled rotation for {provider_name}")
        else:
            self.rotation_enabled = True
            for pool in self.key_pools.values():
                pool.rotation_enabled = True
            self.logger.info("Enabled rotation for all providers")
    
    def disable_rotation(self, provider_name: str | None = None):
        """Disable rotation for specific provider or all providers."""
        
        if provider_name:
            if provider_name in self.key_pools:
                self.key_pools[provider_name].rotation_enabled = False
                self.logger.info(f"Disabled rotation for {provider_name}")
        else:
            self.rotation_enabled = False
            for pool in self.key_pools.values():
                pool.rotation_enabled = False
            self.logger.info("Disabled rotation for all providers")
    
    def get_rotation_status(self) -> dict[str, Any]:
        """Get rotation status for all providers."""
        
        return {
            "global_rotation_enabled": self.rotation_enabled,
            "providers": {
                name: {
                    "total_keys": len(pool.keys),
                    "available_keys": len([k for k in pool.keys if k not in pool.failed_keys]),
                    "failed_keys": len(pool.failed_keys),
                    "rotation_enabled": pool.rotation_enabled,
                    "current_key_index": pool.current_index
                }
                for name, pool in self.key_pools.items()
            }
        }
    
    def reset_failed_keys(self, provider_name: str | None = None):
        """Reset failed keys (useful after quota reset period)."""
        
        if provider_name:
            if provider_name in self.key_pools:
                self.key_pools[provider_name].failed_keys.clear()
                self.logger.info(f"Reset failed keys for {provider_name}")
        else:
            for pool in self.key_pools.values():
                pool.failed_keys.clear()
            self.logger.info("Reset failed keys for all providers")


# Global API key rotator instance
api_key_rotator = APIKeyRotator()