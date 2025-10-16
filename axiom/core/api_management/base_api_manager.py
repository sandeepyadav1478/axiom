"""
Base API Key Management Framework for Investment Banking Analytics
Provides intelligent API key rotation, quota monitoring, and failover capabilities
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RotationStrategy(Enum):
    """API key rotation strategies."""
    
    ROUND_ROBIN = "round_robin"  # Simple round-robin rotation
    WEIGHTED = "weighted"        # Weighted by quota/performance
    FAILOVER = "failover"       # Only rotate on failure
    INTELLIGENT = "intelligent"  # AI-driven rotation based on usage patterns


class APIKeyStatus(Enum):
    """API key status tracking."""
    
    ACTIVE = "active"           # Key is working normally
    QUOTA_EXHAUSTED = "quota_exhausted"  # Quota limit reached
    RATE_LIMITED = "rate_limited"        # Rate limit hit
    FAILED = "failed"                    # Key failed/invalid
    SUSPENDED = "suspended"              # Temporarily disabled


class QuotaUsage(BaseModel):
    """API key quota usage tracking."""
    
    key_id: str = Field(..., description="Unique key identifier")
    provider: str = Field(..., description="Provider name (openai, claude, etc.)")
    
    # Usage statistics
    tokens_used: int = Field(default=0, description="Total tokens consumed")
    requests_made: int = Field(default=0, description="Total requests made")
    
    # Quota limits
    daily_token_limit: int | None = Field(None, description="Daily token limit")
    daily_request_limit: int | None = Field(None, description="Daily request limit")
    rate_limit_per_minute: int | None = Field(None, description="Rate limit per minute")
    
    # Time tracking
    last_used: datetime = Field(default_factory=datetime.now)
    reset_time: datetime | None = Field(None, description="When quotas reset")
    
    # Performance metrics
    avg_response_time: float = Field(default=0.0, description="Average response time (ms)")
    success_rate: float = Field(default=1.0, description="Success rate (0-1)")
    
    # Status
    status: APIKeyStatus = Field(default=APIKeyStatus.ACTIVE)
    failure_count: int = Field(default=0, description="Consecutive failure count")


class APIKeyManager(ABC):
    """Abstract base class for managing multiple API keys with intelligent rotation."""
    
    def __init__(
        self,
        provider_name: str,
        keys: list[str] | str,
        rotation_strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN,
        max_retries: int = 3,
        failure_threshold: int = 3,
    ):
        """
        Initialize API key manager.
        
        Args:
            provider_name: Name of the API provider (openai, claude, tavily, etc.)
            keys: List of API keys or single key string
            rotation_strategy: Strategy for key rotation
            max_retries: Maximum retry attempts before marking key as failed
            failure_threshold: Consecutive failures before suspending key
        """
        self.provider_name = provider_name
        self.keys = [keys] if isinstance(keys, str) else keys
        self.rotation_strategy = rotation_strategy
        self.max_retries = max_retries
        self.failure_threshold = failure_threshold
        
        # Initialize key tracking
        self.key_usage: dict[str, QuotaUsage] = {}
        self.current_key_index = 0
        self.failed_keys: set[str] = set()
        self.suspended_keys: set[str] = set()
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = datetime.now()
        
        # Initialize usage tracking for all keys
        self._initialize_key_tracking()
        
        # Setup logging
        self.logger = logging.getLogger(f"api_manager.{provider_name}")
    
    def _initialize_key_tracking(self):
        """Initialize usage tracking for all API keys."""
        for i, key in enumerate(self.keys):
            key_id = self._get_key_id(key, i)
            self.key_usage[key_id] = QuotaUsage(
                key_id=key_id,
                provider=self.provider_name
            )
    
    def _get_key_id(self, key: str, index: int) -> str:
        """Generate unique key identifier from API key."""
        # Use last 8 characters for identification (safe for logging)
        return f"{self.provider_name}_{index}_{key[-8:] if len(key) >= 8 else key}"
    
    @abstractmethod
    async def test_key_validity(self, api_key: str) -> tuple[bool, str]:
        """
        Test if an API key is valid and working.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def estimate_quota_remaining(self, key_id: str) -> float:
        """
        Estimate remaining quota for a key (0.0 to 1.0).
        
        Returns:
            Float representing percentage of quota remaining
        """
        pass
    
    @abstractmethod
    async def get_quota_usage(self, api_key: str) -> dict[str, Any]:
        """
        Get current quota usage from provider API.
        
        Returns:
            Dictionary with usage statistics from provider
        """
        pass
    
    def get_active_key(self) -> str | None:
        """Get currently active API key based on rotation strategy."""
        
        if not self.keys:
            self.logger.error("No API keys available")
            return None
        
        if self.rotation_strategy == RotationStrategy.ROUND_ROBIN:
            return self._get_round_robin_key()
        elif self.rotation_strategy == RotationStrategy.WEIGHTED:
            return self._get_weighted_key()
        elif self.rotation_strategy == RotationStrategy.FAILOVER:
            return self._get_failover_key()
        elif self.rotation_strategy == RotationStrategy.INTELLIGENT:
            return self._get_intelligent_key()
        else:
            return self.keys[0]  # Default to first key
    
    def _get_round_robin_key(self) -> str | None:
        """Simple round-robin key selection."""
        available_keys = [k for k in self.keys if k not in self.failed_keys]
        
        if not available_keys:
            self.logger.warning("No available keys for round-robin rotation")
            return None
        
        key = available_keys[self.current_key_index % len(available_keys)]
        self.current_key_index = (self.current_key_index + 1) % len(available_keys)
        return key
    
    def _get_weighted_key(self) -> str | None:
        """Weighted key selection based on quota remaining and performance."""
        available_keys = [k for k in self.keys if k not in self.failed_keys]
        
        if not available_keys:
            return None
        
        best_key = None
        best_score = -1.0
        
        for key in available_keys:
            key_id = self._find_key_id(key)
            if not key_id:
                continue
                
            usage = self.key_usage[key_id]
            
            # Calculate weighted score (quota remaining * success rate)
            quota_remaining = self.estimate_quota_remaining(key_id)
            performance_score = usage.success_rate
            
            # Combined score favors both quota availability and performance
            weighted_score = (quota_remaining * 0.7) + (performance_score * 0.3)
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_key = key
        
        return best_key
    
    def _get_failover_key(self) -> str | None:
        """Failover strategy - use first working key, only rotate on failure."""
        for key in self.keys:
            if key not in self.failed_keys:
                return key
        
        self.logger.error("All keys failed in failover strategy")
        return None
    
    def _get_intelligent_key(self) -> str | None:
        """AI-driven key selection based on usage patterns and predictions."""
        # Simplified intelligent selection - can be enhanced with ML models
        available_keys = [k for k in self.keys if k not in self.failed_keys]
        
        if not available_keys:
            return None
        
        # Consider time of day, historical usage patterns, quota cycles
        current_hour = datetime.now().hour
        
        # Simple heuristic: prefer keys with recent successful usage during similar hours
        best_key = available_keys[0]
        best_recent_success = 0
        
        for key in available_keys:
            key_id = self._find_key_id(key)
            if not key_id:
                continue
                
            usage = self.key_usage[key_id]
            
            # Score based on recent success during similar time periods
            time_similarity_score = 1.0 - abs(usage.last_used.hour - current_hour) / 24.0
            recent_success_score = usage.success_rate * time_similarity_score
            
            if recent_success_score > best_recent_success:
                best_recent_success = recent_success_score
                best_key = key
        
        return best_key
    
    def _find_key_id(self, api_key: str) -> str | None:
        """Find key ID for given API key."""
        for key_id, usage in self.key_usage.items():
            if api_key[-8:] in key_id:  # Match by last 8 characters
                return key_id
        return None
    
    async def rotate_key(self, reason: str, failed_key: str | None = None) -> str | None:
        """
        Rotate to next available API key.
        
        Args:
            reason: Reason for rotation (quota_exhausted, rate_limited, failed, etc.)
            failed_key: The key that failed (will be marked as failed)
            
        Returns:
            Next available API key or None if all keys exhausted
        """
        
        if failed_key:
            self.mark_key_failed(failed_key, reason)
        
        new_key = self.get_active_key()
        
        if new_key:
            self.logger.info(f"Rotated API key for {self.provider_name}: {reason}")
            await self._log_rotation_event(reason, failed_key, new_key)
        else:
            self.logger.critical(f"All API keys exhausted for {self.provider_name}")
            await self._alert_all_keys_exhausted()
        
        return new_key
    
    def mark_key_failed(self, api_key: str, reason: str):
        """Mark an API key as failed with reason."""
        key_id = self._find_key_id(api_key)
        if not key_id:
            return
        
        usage = self.key_usage[key_id]
        usage.failure_count += 1
        
        if reason == "quota_exhausted":
            usage.status = APIKeyStatus.QUOTA_EXHAUSTED
            # Predict reset time (typically daily)
            usage.reset_time = datetime.now() + timedelta(days=1)
        elif reason == "rate_limited":
            usage.status = APIKeyStatus.RATE_LIMITED
            # Rate limits typically reset within an hour
            usage.reset_time = datetime.now() + timedelta(hours=1)
        else:
            usage.status = APIKeyStatus.FAILED
            if usage.failure_count >= self.failure_threshold:
                self.failed_keys.add(api_key)
                self.logger.warning(f"Key {key_id} marked as failed after {usage.failure_count} failures")
    
    def track_successful_usage(
        self,
        api_key: str,
        tokens_used: int = 0,
        response_time: float = 0.0
    ):
        """Track successful API key usage for quota monitoring."""
        key_id = self._find_key_id(api_key)
        if not key_id:
            return
        
        usage = self.key_usage[key_id]
        
        # Update usage statistics
        usage.tokens_used += tokens_used
        usage.requests_made += 1
        usage.last_used = datetime.now()
        
        # Update performance metrics
        if response_time > 0:
            if usage.avg_response_time == 0:
                usage.avg_response_time = response_time
            else:
                # Exponential moving average
                usage.avg_response_time = (usage.avg_response_time * 0.8) + (response_time * 0.2)
        
        # Update success rate
        total_attempts = usage.requests_made + usage.failure_count
        usage.success_rate = usage.requests_made / max(total_attempts, 1)
        
        # Reset status if key was previously failed but now working
        if usage.status in [APIKeyStatus.FAILED, APIKeyStatus.RATE_LIMITED]:
            usage.status = APIKeyStatus.ACTIVE
            if api_key in self.failed_keys:
                self.failed_keys.remove(api_key)
                self.logger.info(f"Key {key_id} recovered and marked as active")
    
    def get_usage_statistics(self) -> dict[str, Any]:
        """Get comprehensive usage statistics across all keys."""
        
        total_tokens = sum(usage.tokens_used for usage in self.key_usage.values())
        total_requests = sum(usage.requests_made for usage in self.key_usage.values())
        
        active_keys = len([k for k in self.keys if k not in self.failed_keys])
        failed_keys = len(self.failed_keys)
        
        uptime = datetime.now() - self.start_time
        
        return {
            "provider": self.provider_name,
            "total_keys": len(self.keys),
            "active_keys": active_keys,
            "failed_keys": failed_keys,
            "suspension_rate": failed_keys / max(len(self.keys), 1),
            "total_tokens_used": total_tokens,
            "total_requests": total_requests,
            "uptime_hours": uptime.total_seconds() / 3600,
            "avg_tokens_per_request": total_tokens / max(total_requests, 1),
            "rotation_strategy": self.rotation_strategy.value,
            "key_usage_details": [
                {
                    "key_id": usage.key_id,
                    "status": usage.status.value,
                    "tokens_used": usage.tokens_used,
                    "requests_made": usage.requests_made,
                    "success_rate": usage.success_rate,
                    "avg_response_time": usage.avg_response_time,
                    "last_used": usage.last_used.isoformat(),
                    "quota_remaining_estimate": self.estimate_quota_remaining(usage.key_id)
                }
                for usage in self.key_usage.values()
            ]
        }
    
    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all API keys."""
        
        health_results = {
            "provider": self.provider_name,
            "overall_health": "healthy",
            "timestamp": datetime.now().isoformat(),
            "key_results": []
        }
        
        unhealthy_keys = 0
        
        for key in self.keys:
            key_id = self._find_key_id(key)
            
            # Test key validity
            is_valid, error_msg = await self.test_key_validity(key)
            
            key_health = {
                "key_id": key_id,
                "status": "healthy" if is_valid else "unhealthy",
                "error": error_msg if not is_valid else None,
                "usage": self.key_usage.get(key_id, {})
            }
            
            health_results["key_results"].append(key_health)
            
            if not is_valid:
                unhealthy_keys += 1
        
        # Overall health assessment
        health_ratio = (len(self.keys) - unhealthy_keys) / max(len(self.keys), 1)
        
        if health_ratio >= 0.8:
            health_results["overall_health"] = "healthy"
        elif health_ratio >= 0.5:
            health_results["overall_health"] = "degraded"
        else:
            health_results["overall_health"] = "unhealthy"
            
        return health_results
    
    async def _log_rotation_event(self, reason: str, failed_key: str | None, new_key: str):
        """Log key rotation event for audit purposes."""
        
        failed_key_id = self._find_key_id(failed_key) if failed_key else None
        new_key_id = self._find_key_id(new_key)
        
        rotation_event = {
            "timestamp": datetime.now().isoformat(),
            "provider": self.provider_name,
            "rotation_reason": reason,
            "failed_key_id": failed_key_id,
            "new_key_id": new_key_id,
            "rotation_strategy": self.rotation_strategy.value
        }
        
        # Log for audit trail (could be sent to monitoring system)
        self.logger.info(f"API key rotation: {rotation_event}")
    
    async def _alert_all_keys_exhausted(self):
        """Alert when all API keys are exhausted."""
        
        alert = {
            "severity": "critical",
            "provider": self.provider_name,
            "message": f"All API keys exhausted for {self.provider_name}",
            "timestamp": datetime.now().isoformat(),
            "impact": "Service disruption - no working API keys available",
            "action_required": "Add new API keys or wait for quota reset"
        }
        
        # Critical alert - could integrate with PagerDuty, Slack, etc.
        self.logger.critical(f"CRITICAL ALERT: {alert}")
        
        # In production, trigger alerting system
        # await send_critical_alert(alert)
    
    def reset_failed_keys(self):
        """Reset failed keys (useful after quota reset time)."""
        
        current_time = datetime.now()
        recovered_keys = []
        
        for key_id, usage in self.key_usage.items():
            # Check if reset time has passed
            if usage.reset_time and current_time >= usage.reset_time:
                usage.status = APIKeyStatus.ACTIVE
                usage.failure_count = 0
                usage.reset_time = None
                
                # Remove from failed keys set
                key = next((k for k in self.keys if self._find_key_id(k) == key_id), None)
                if key and key in self.failed_keys:
                    self.failed_keys.remove(key)
                    recovered_keys.append(key_id)
        
        if recovered_keys:
            self.logger.info(f"Recovered keys after quota reset: {recovered_keys}")
    
    def get_provider_status(self) -> dict[str, Any]:
        """Get comprehensive provider status for monitoring dashboards."""
        
        stats = self.get_usage_statistics()
        
        return {
            "provider_name": self.provider_name,
            "service_status": "operational" if stats["active_keys"] > 0 else "degraded",
            "key_availability": f"{stats['active_keys']}/{stats['total_keys']}",
            "rotation_strategy": self.rotation_strategy.value,
            "performance_metrics": {
                "total_requests": stats["total_requests"],
                "total_tokens": stats["total_tokens"],
                "avg_tokens_per_request": stats["avg_tokens_per_request"],
                "uptime_hours": stats["uptime_hours"]
            },
            "health_score": (stats["active_keys"] / max(stats["total_keys"], 1)) * 100
        }


class APIManagerError(Exception):
    """Custom exception for API manager errors."""
    
    def __init__(self, provider: str, message: str, key_id: str = None):
        self.provider = provider
        self.key_id = key_id
        super().__init__(f"[{provider}] {message}")