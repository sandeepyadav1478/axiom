"""
Redis Real-Time Cache

Sub-millisecond caching and pub/sub for real-time data streaming.

Uses the `redis-py` library (used by Airbnb, Instagram, Twitter)
for production-grade caching and messaging.
"""

import asyncio
import json
import time
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

try:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    RedisError = Exception
    RedisConnectionError = Exception

from axiom.streaming.config import StreamingConfig


logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    publishes: int = 0
    avg_latency_ms: float = 0.0
    total_operations: int = 0


class RealTimeCache:
    """
    Redis-based real-time data cache with pub/sub.
    
    Features:
    - Sub-millisecond read/write
    - Pub/Sub for real-time updates
    - Time-series data storage
    - Sorted sets for order books
    - Atomic operations
    - Connection pooling
    
    Uses `redis-py` for production-grade caching.
    Performance Target: <1ms cache read/write
    """
    
    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize Redis cache.
        
        Args:
            config: Streaming configuration
            redis_url: Redis connection URL (overrides config)
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis library not installed. "
                "Install with: pip install redis"
            )
        
        self.config = config or StreamingConfig()
        self.redis_url = redis_url or self.config.redis_url
        
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[Any] = None
        self.stats = CacheStats()
        
        self._subscribers: Dict[str, List[Callable]] = {}
        self._running = False
        
        logger.info(f"Redis cache initialized with URL: {self.redis_url}")
    
    async def connect(self):
        """Establish Redis connection."""
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=self.config.redis_decode_responses,
                max_connections=self.config.redis_max_connections,
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Redis connection established")
            
        except RedisConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.pubsub:
            await self.pubsub.close()
        
        if self.redis:
            await self.redis.close()
        
        logger.info("Redis connection closed")
    
    # Price Data Operations
    
    async def set_price(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[float] = None,
    ):
        """
        Store latest price with timestamp.
        
        Args:
            symbol: Instrument symbol
            price: Current price
            timestamp: Unix timestamp (defaults to current time)
        """
        timestamp = timestamp or time.time()
        
        start_time = time.time()
        
        try:
            # Store in sorted set (time-series)
            await self.redis.zadd(
                f"prices:{symbol}",
                {str(timestamp): price}
            )
            
            # Store latest price separately for quick access
            await self.redis.set(
                f"price:latest:{symbol}",
                price,
                ex=self.config.redis_ttl
            )
            
            # Publish update
            await self.redis.publish(
                f"price_updates:{symbol}",
                json.dumps({
                    'symbol': symbol,
                    'price': price,
                    'timestamp': timestamp
                })
            )
            
            self.stats.sets += 1
            self.stats.publishes += 1
            
        except RedisError as e:
            logger.error(f"Error setting price for {symbol}: {e}")
            raise
        
        finally:
            if self.config.log_latency:
                latency = (time.time() - start_time) * 1000
                self._update_latency(latency)
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get most recent price for symbol.
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            Latest price or None
        """
        start_time = time.time()
        
        try:
            price = await self.redis.get(f"price:latest:{symbol}")
            
            if price:
                self.stats.hits += 1
                return float(price)
            else:
                self.stats.misses += 1
                return None
                
        except RedisError as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
        
        finally:
            if self.config.log_latency:
                latency = (time.time() - start_time) * 1000
                self._update_latency(latency)
    
    async def get_price_history(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, float]]:
        """
        Get price history for symbol.
        
        Args:
            symbol: Instrument symbol
            start_time: Start timestamp (Unix)
            end_time: End timestamp (Unix)
            limit: Maximum number of results
        
        Returns:
            List of price history entries
        """
        try:
            if start_time and end_time:
                results = await self.redis.zrangebyscore(
                    f"prices:{symbol}",
                    start_time,
                    end_time,
                    withscores=True,
                    start=0,
                    num=limit
                )
            else:
                results = await self.redis.zrange(
                    f"prices:{symbol}",
                    -limit,
                    -1,
                    withscores=True
                )
            
            return [
                {'timestamp': float(score), 'price': float(value)}
                for value, score in results
            ]
            
        except RedisError as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []
    
    # Market Data Operations
    
    async def set_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: int,
        ask_size: int,
        timestamp: Optional[float] = None,
    ):
        """
        Store latest quote data.
        
        Args:
            symbol: Instrument symbol
            bid: Bid price
            ask: Ask price
            bid_size: Bid size
            ask_size: Ask size
            timestamp: Unix timestamp
        """
        timestamp = timestamp or time.time()
        
        quote_data = {
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'timestamp': timestamp,
            'spread': ask - bid,
        }
        
        try:
            await self.redis.hset(
                f"quote:{symbol}",
                mapping={k: json.dumps(v) for k, v in quote_data.items()}
            )
            
            await self.redis.expire(f"quote:{symbol}", self.config.redis_ttl)
            
            self.stats.sets += 1
            
        except RedisError as e:
            logger.error(f"Error setting quote for {symbol}: {e}")
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest quote for symbol.
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            Quote data or None
        """
        try:
            quote = await self.redis.hgetall(f"quote:{symbol}")
            
            if quote:
                self.stats.hits += 1
                return {k: json.loads(v) for k, v in quote.items()}
            else:
                self.stats.misses += 1
                return None
                
        except RedisError as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    # Order Book Operations
    
    async def update_order_book(
        self,
        symbol: str,
        bids: List[tuple],
        asks: List[tuple],
    ):
        """
        Update order book using sorted sets.
        
        Args:
            symbol: Instrument symbol
            bids: List of (price, size) tuples for bids
            asks: List of (price, size) tuples for asks
        """
        try:
            # Clear existing order book
            await self.redis.delete(f"orderbook:bids:{symbol}")
            await self.redis.delete(f"orderbook:asks:{symbol}")
            
            # Add bids (sorted by price descending)
            if bids:
                await self.redis.zadd(
                    f"orderbook:bids:{symbol}",
                    {json.dumps({'price': p, 'size': s}): -p for p, s in bids}
                )
            
            # Add asks (sorted by price ascending)
            if asks:
                await self.redis.zadd(
                    f"orderbook:asks:{symbol}",
                    {json.dumps({'price': p, 'size': s}): p for p, s in asks}
                )
            
            self.stats.sets += 1
            
        except RedisError as e:
            logger.error(f"Error updating order book for {symbol}: {e}")
    
    async def get_order_book(
        self,
        symbol: str,
        depth: int = 10,
    ) -> Dict[str, List[Dict]]:
        """
        Get order book for symbol.
        
        Args:
            symbol: Instrument symbol
            depth: Order book depth
        
        Returns:
            Dictionary with 'bids' and 'asks'
        """
        try:
            bids = await self.redis.zrange(
                f"orderbook:bids:{symbol}",
                0,
                depth - 1
            )
            
            asks = await self.redis.zrange(
                f"orderbook:asks:{symbol}",
                0,
                depth - 1
            )
            
            return {
                'bids': [json.loads(b) for b in bids],
                'asks': [json.loads(a) for a in asks],
            }
            
        except RedisError as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    # Pub/Sub Operations
    
    async def subscribe_prices(
        self,
        symbol: str,
        callback: Callable,
    ):
        """
        Subscribe to real-time price updates.
        
        Args:
            symbol: Instrument symbol
            callback: Async callback function
        """
        channel = f"price_updates:{symbol}"
        
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        
        self._subscribers[channel].append(callback)
        
        # Start subscriber if not running
        if not self._running:
            await self._start_subscriber()
    
    async def _start_subscriber(self):
        """Start pub/sub subscriber."""
        if self.pubsub:
            return
        
        self.pubsub = self.redis.pubsub()
        self._running = True
        
        # Subscribe to all registered channels
        for channel in self._subscribers.keys():
            await self.pubsub.subscribe(channel)
        
        # Start listening task
        asyncio.create_task(self._listen_messages())
        
        logger.info("Pub/sub subscriber started")
    
    async def _listen_messages(self):
        """Listen for pub/sub messages."""
        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel']
                    data = json.loads(message['data'])
                    
                    # Call all callbacks for this channel
                    if channel in self._subscribers:
                        for callback in self._subscribers[channel]:
                            try:
                                await callback(data)
                            except Exception as e:
                                logger.error(f"Error in subscriber callback: {e}")
                                
        except Exception as e:
            logger.error(f"Error listening to pub/sub messages: {e}")
            self._running = False
    
    # Cache Management
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "prices:*")
        
        Returns:
            Number of keys deleted
        """
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.redis.delete(*keys)
                self.stats.deletes += deleted
                return deleted
            
            return 0
            
        except RedisError as e:
            logger.error(f"Error deleting pattern {pattern}: {e}")
            return 0
    
    async def clear_symbol_data(self, symbol: str):
        """
        Clear all cached data for a symbol.
        
        Args:
            symbol: Instrument symbol
        """
        patterns = [
            f"prices:{symbol}",
            f"price:latest:{symbol}",
            f"quote:{symbol}",
            f"orderbook:bids:{symbol}",
            f"orderbook:asks:{symbol}",
        ]
        
        for pattern in patterns:
            await self.delete_pattern(pattern)
    
    # Statistics
    
    def _update_latency(self, latency_ms: float):
        """Update average latency metric."""
        self.stats.total_operations += 1
        self.stats.avg_latency_ms = (
            0.9 * self.stats.avg_latency_ms + 0.1 * latency_ms
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        hit_rate = (
            self.stats.hits / (self.stats.hits + self.stats.misses)
            if (self.stats.hits + self.stats.misses) > 0
            else 0.0
        )
        
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate': hit_rate,
            'sets': self.stats.sets,
            'deletes': self.stats.deletes,
            'publishes': self.stats.publishes,
            'avg_latency_ms': self.stats.avg_latency_ms,
            'total_operations': self.stats.total_operations,
            'active_subscriptions': len(self._subscribers),
        }
    
    async def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False