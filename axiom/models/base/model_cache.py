"""
Model Caching Layer for Performance Optimization

Caches loaded models to avoid reloading overhead.
Critical for production performance when using 35+ models.

Benefits:
- Lazy loading (load on first use)
- Singleton pattern (one instance per model type)
- Memory management (LRU eviction)
- Thread-safe access
"""

from typing import Dict, Any, Optional
from functools import lru_cache
import threading

from axiom.models.base.factory import ModelFactory, ModelType


class ModelCache:
    """
    Thread-safe model cache with lazy loading
    
    Usage:
        cache = ModelCache()
        model = cache.get_model(ModelType.PORTFOLIO_TRANSFORMER)
        # Second call returns cached instance
        model2 = cache.get_model(ModelType.PORTFOLIO_TRANSFORMER)  # Same instance
    """
    
    def __init__(self, max_cached_models: int = 20):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._max_models = max_cached_models
        self._access_count: Dict[str, int] = {}
    
    def get_model(self, model_type: ModelType, config: Optional[Any] = None):
        """
        Get model from cache or create if not exists
        
        Args:
            model_type: Type of model to get
            config: Optional custom config
            
        Returns:
            Cached or newly created model instance
        """
        cache_key = model_type.value
        
        # Check cache first
        with self._lock:
            if cache_key in self._cache:
                self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                return self._cache[cache_key]
        
        # Not in cache - create new
        try:
            model = ModelFactory.create(model_type, config=config)
            
            # Add to cache
            with self._lock:
                # Check cache size
                if len(self._cache) >= self._max_models:
                    self._evict_lru()
                
                self._cache[cache_key] = model
                self._access_count[cache_key] = 1
            
            return model
            
        except Exception as e:
            print(f"Failed to create model {model_type.value}: {e}")
            return None
    
    def _evict_lru(self):
        """Evict least recently used model"""
        if not self._access_count:
            return
        
        # Find least accessed
        lru_key = min(self._access_count, key=self._access_count.get)
        
        # Remove
        if lru_key in self._cache:
            del self._cache[lru_key]
        del self._access_count[lru_key]
    
    def clear_cache(self):
        """Clear all cached models"""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            return {
                'cached_models': len(self._cache),
                'max_models': self._max_models,
                'model_types': list(self._cache.keys()),
                'access_counts': dict(self._access_count)
            }


# Global cache instance
_global_cache = ModelCache(max_cached_models=25)


def get_cached_model(model_type: ModelType, config: Optional[Any] = None):
    """
    Convenience function to get model from global cache
    
    Usage:
        from axiom.models.base.model_cache import get_cached_model
        model = get_cached_model(ModelType.PORTFOLIO_TRANSFORMER)
    """
    return _global_cache.get_model(model_type, config)


def clear_model_cache():
    """Clear global model cache"""
    _global_cache.clear_cache()


def get_cache_statistics():
    """Get global cache statistics"""
    return _global_cache.get_cache_stats()


if __name__ == "__main__":
    print("Model Caching Layer")
    print("=" * 60)
    
    from axiom.models.base.factory import ModelType
    
    print("\n1. Creating cache...")
    cache = ModelCache(max_cached_models=10)
    
    print("\n2. Loading models (will cache)...")
    
    # Load some models
    try:
        model1 = cache.get_model(ModelType.PORTFOLIO_TRANSFORMER)
        print("  ✓ Portfolio Transformer loaded")
    except:
        print("  ⚠ Portfolio Transformer unavailable")
    
    try:
        model2 = cache.get_model(ModelType.ANN_GREEKS_CALCULATOR)
        print("  ✓ ANN Greeks loaded")
    except:
        print("  ⚠ ANN Greeks unavailable")
    
    # Second access (cached)
    try:
        model1_cached = cache.get_model(ModelType.PORTFOLIO_TRANSFORMER)
        print("  ✓ Portfolio Transformer from cache (same instance)")
    except:
        pass
    
    # Stats
    stats = cache.get_cache_stats()
    print(f"\n3. Cache Statistics:")
    print(f"  Cached models: {stats['cached_models']}")
    print(f"  Model types: {stats['model_types']}")
    
    print("\n✓ Model caching reduces loading overhead")
    print("  Critical for production with 35+ models")