"""
Redis Cache Integration Layer.

Provides high-performance caching for:
- Hot features (real-time access)
- Latest market data (sub-millisecond latency)
- Frequently accessed company data
- Session data and temporary results
"""

import logging
import json
from typing import Any, Optional, List, Dict
from datetime import timedelta
import pickle

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache integration for high-performance data access.
    
    Use Cases:
    - Feature caching (avoid recomputation)
    - Latest price caching (real-time trading)
    - Company data caching (M&A analysis)
    - Session state caching
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0
    ):
        """
        Initialize Redis client.
        
        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database number
        """
        try:
            import redis
            self.redis = redis
        except ImportError:
            raise ImportError(
                "Redis not installed. Install with: pip install redis"
            )
        
        self.client = redis.Redis(
            host=host,
            port=port,
            password=password or "axiom_redis",
            db=db,
            decode_responses=False  # We'll handle encoding
        )
        
        logger.info(f"Initialized Redis cache at {host}:{port}")
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: str = "json"
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = no expiry)
            serialize: Serialization method ('json' or 'pickle')
            
        Returns:
            True if successful
        """
        try:
            # Serialize value
            if serialize == "json":
                serialized = json.dumps(value)
            elif serialize == "pickle":
                serialized = pickle.dumps(value)
            else:
                serialized = str(value)
            
            if ttl:
                return self.client.setex(key, ttl, serialized)
            else:
                return self.client.set(key, serialized)
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def get(
        self,
        key: str,
        default: Any = None,
        deserialize: str = "json"
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            deserialize: Deserialization method ('json' or 'pickle')
            
        Returns:
            Cached value or default
        """
        try:
            value = self.client.get(key)
            
            if value is None:
                return default
            
            # Deserialize
            if deserialize == "json":
                return json.loads(value)
            elif deserialize == "pickle":
                return pickle.loads(value)
            else:
                return value.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return default
    
    def delete(self, *keys: str) -> int:
        """Delete keys from cache."""
        return self.client.delete(*keys)
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.client.exists(key) > 0
    
    def ttl(self, key: str) -> int:
        """Get time-to-live for key."""
        return self.client.ttl(key)
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key."""
        return self.client.expire(key, seconds)
    
    # Feature Caching Methods
    
    def cache_feature(
        self,
        symbol: str,
        feature_name: str,
        value: float,
        ttl: int = 3600  # 1 hour default
    ) -> bool:
        """
        Cache a computed feature.
        
        Args:
            symbol: Asset symbol
            feature_name: Feature name
            value: Feature value
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        key = f"feature:{symbol}:{feature_name}"
        return self.set(key, value, ttl=ttl)
    
    def get_feature(
        self,
        symbol: str,
        feature_name: str
    ) -> Optional[float]:
        """
        Get cached feature.
        
        Args:
            symbol: Asset symbol
            feature_name: Feature name
            
        Returns:
            Feature value or None
        """
        key = f"feature:{symbol}:{feature_name}"
        return self.get(key)
    
    def cache_features_bulk(
        self,
        symbol: str,
        features: Dict[str, float],
        ttl: int = 3600
    ) -> int:
        """
        Cache multiple features at once.
        
        Args:
            symbol: Asset symbol
            features: Dictionary of {feature_name: value}
            ttl: Time-to-live in seconds
            
        Returns:
            Number of features cached
        """
        pipe = self.client.pipeline()
        count = 0
        
        for feature_name, value in features.items():
            key = f"feature:{symbol}:{feature_name}"
            pipe.setex(key, ttl, json.dumps(value))
            count += 1
        
        pipe.execute()
        
        logger.info(f"Bulk cached {count} features for {symbol}")
        return count
    
    def get_features_bulk(
        self,
        symbol: str,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Get multiple cached features at once.
        
        Args:
            symbol: Asset symbol
            feature_names: List of feature names
            
        Returns:
            Dictionary of {feature_name: value}
        """
        keys = [f"feature:{symbol}:{name}" for name in feature_names]
        values = self.client.mget(keys)
        
        results = {}
        for i, (name, value) in enumerate(zip(feature_names, values)):
            if value is not None:
                results[name] = json.loads(value)
        
        logger.info(f"Retrieved {len(results)}/{len(feature_names)} cached features for {symbol}")
        return results
    
    # Price Data Caching
    
    def cache_latest_price(
        self,
        symbol: str,
        price: float,
        ttl: int = 60  # 1 minute for real-time
    ) -> bool:
        """Cache latest price."""
        key = f"price:latest:{symbol}"
        return self.set(key, price, ttl=ttl)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get cached latest price."""
        key = f"price:latest:{symbol}"
        return self.get(key)
    
    # Health Check
    
    def health_check(self) -> bool:
        """Check Redis health."""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def info(self) -> Dict[str, Any]:
        """Get Redis info."""
        try:
            return self.client.info()
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    def close(self):
        """Close Redis connection."""
        self.client.close()


# Export
__all__ = [
    "RedisCache",
]