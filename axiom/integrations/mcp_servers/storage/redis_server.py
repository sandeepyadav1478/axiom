"""Redis MCP Server Implementation.

Provides Redis cache and pub/sub operations through MCP protocol:
- Key-value operations (get, set, delete)
- Pub/sub messaging
- Sorted sets for time-series data
- Cache statistics and health monitoring
- TTL management
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional

try:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    RedisError = Exception
    RedisConnectionError = Exception

logger = logging.getLogger(__name__)


class RedisMCPServer:
    """Redis MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis library not installed. "
                "Install with: pip install redis[hiredis]"
            )
        
        self.config = config
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self.max_connections = config.get("max_connections", 50)
        
        self._redis: Optional["aioredis.Redis"] = None
        self._pubsub: Optional[Any] = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "publishes": 0,
            "total_operations": 0,
        }

    async def _ensure_connection(self) -> "aioredis.Redis":
        """Ensure Redis connection is established.

        Returns:
            Redis client instance
        """
        if self._redis is None:
            connection_kwargs = {
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "encoding": "utf-8",
                "decode_responses": True,
                "max_connections": self.max_connections,
            }
            
            if self.password:
                connection_kwargs["password"] = self.password
            
            self._redis = aioredis.Redis(**connection_kwargs)
            
            # Test connection
            await self._redis.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        
        return self._redis

    async def close(self):
        """Close Redis connection."""
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None
        
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")

    async def get_value(
        self,
        key: str,
        deserialize: bool = True,
    ) -> dict[str, Any]:
        """Get value from Redis cache.

        Args:
            key: Cache key
            deserialize: Deserialize JSON value

        Returns:
            Retrieved value and metadata
        """
        start_time = time.time()
        
        try:
            redis = await self._ensure_connection()
            value = await redis.get(key)
            
            if value is not None:
                self._stats["hits"] += 1
                self._stats["total_operations"] += 1
                
                if deserialize and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "success": True,
                    "key": key,
                    "value": value,
                    "found": True,
                    "latency_ms": latency_ms,
                }
            else:
                self._stats["misses"] += 1
                self._stats["total_operations"] += 1
                
                return {
                    "success": True,
                    "key": key,
                    "value": None,
                    "found": False,
                    "latency_ms": (time.time() - start_time) * 1000,
                }

        except RedisError as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return {
                "success": False,
                "error": f"Redis error: {str(e)}",
                "key": key,
            }
        except Exception as e:
            logger.error(f"Failed to get value for key {key}: {e}")
            return {
                "success": False,
                "error": f"Failed to get value: {str(e)}",
                "key": key,
            }

    async def set_value(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True,
    ) -> dict[str, Any]:
        """Set value in Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for no expiration)
            serialize: Serialize value as JSON

        Returns:
            Operation result
        """
        start_time = time.time()
        
        try:
            redis = await self._ensure_connection()
            
            # Serialize if needed
            if serialize and not isinstance(value, str):
                value = json.dumps(value)
            
            # Set with optional TTL
            if ttl:
                await redis.setex(key, ttl, value)
            else:
                await redis.set(key, value)
            
            self._stats["sets"] += 1
            self._stats["total_operations"] += 1
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "key": key,
                "ttl": ttl,
                "latency_ms": latency_ms,
            }

        except RedisError as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return {
                "success": False,
                "error": f"Redis error: {str(e)}",
                "key": key,
            }
        except Exception as e:
            logger.error(f"Failed to set value for key {key}: {e}")
            return {
                "success": False,
                "error": f"Failed to set value: {str(e)}",
                "key": key,
            }

    async def delete_key(self, key: str) -> dict[str, Any]:
        """Delete key from Redis.

        Args:
            key: Cache key to delete

        Returns:
            Operation result
        """
        try:
            redis = await self._ensure_connection()
            deleted = await redis.delete(key)
            
            self._stats["deletes"] += deleted
            self._stats["total_operations"] += 1
            
            return {
                "success": True,
                "key": key,
                "deleted": deleted > 0,
            }

        except RedisError as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return {
                "success": False,
                "error": f"Redis error: {str(e)}",
                "key": key,
            }
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return {
                "success": False,
                "error": f"Failed to delete key: {str(e)}",
                "key": key,
            }

    async def publish_message(
        self,
        channel: str,
        message: Any,
        serialize: bool = True,
    ) -> dict[str, Any]:
        """Publish message to Redis pub/sub channel.

        Args:
            channel: Channel name
            message: Message to publish
            serialize: Serialize message as JSON

        Returns:
            Operation result
        """
        try:
            redis = await self._ensure_connection()
            
            # Serialize if needed
            if serialize and not isinstance(message, str):
                message = json.dumps(message)
            
            subscribers = await redis.publish(channel, message)
            
            self._stats["publishes"] += 1
            self._stats["total_operations"] += 1
            
            return {
                "success": True,
                "channel": channel,
                "subscribers": subscribers,
            }

        except RedisError as e:
            logger.error(f"Redis publish failed for channel {channel}: {e}")
            return {
                "success": False,
                "error": f"Redis error: {str(e)}",
                "channel": channel,
            }
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            return {
                "success": False,
                "error": f"Failed to publish: {str(e)}",
                "channel": channel,
            }

    async def subscribe_channel(
        self,
        channel: str,
        timeout: int = 10,
    ) -> dict[str, Any]:
        """Subscribe to Redis pub/sub channel and get messages.

        Args:
            channel: Channel name
            timeout: Timeout in seconds

        Returns:
            Received messages
        """
        try:
            redis = await self._ensure_connection()
            pubsub = redis.pubsub()
            
            await pubsub.subscribe(channel)
            
            messages = []
            start_time = time.time()
            
            while (time.time() - start_time) < timeout:
                try:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=1.0
                    )
                    
                    if message and message["type"] == "message":
                        data = message["data"]
                        try:
                            data = json.loads(data)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        
                        messages.append({
                            "channel": message["channel"],
                            "data": data,
                        })
                
                except asyncio.TimeoutError:
                    if messages:
                        break
                    continue
            
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            
            return {
                "success": True,
                "channel": channel,
                "messages": messages,
                "count": len(messages),
            }

        except RedisError as e:
            logger.error(f"Redis subscribe failed for channel {channel}: {e}")
            return {
                "success": False,
                "error": f"Redis error: {str(e)}",
                "channel": channel,
            }
        except Exception as e:
            logger.error(f"Failed to subscribe to channel {channel}: {e}")
            return {
                "success": False,
                "error": f"Failed to subscribe: {str(e)}",
                "channel": channel,
            }

    async def zadd(
        self,
        key: str,
        score: float,
        member: Any,
        serialize: bool = True,
    ) -> dict[str, Any]:
        """Add member to sorted set with score.

        Args:
            key: Sorted set key
            score: Score for ordering
            member: Member to add
            serialize: Serialize member as JSON

        Returns:
            Operation result
        """
        try:
            redis = await self._ensure_connection()
            
            # Serialize if needed
            if serialize and not isinstance(member, str):
                member = json.dumps(member)
            
            added = await redis.zadd(key, {member: score})
            
            self._stats["sets"] += 1
            self._stats["total_operations"] += 1
            
            return {
                "success": True,
                "key": key,
                "score": score,
                "added": added > 0,
            }

        except RedisError as e:
            logger.error(f"Redis zadd failed for key {key}: {e}")
            return {
                "success": False,
                "error": f"Redis error: {str(e)}",
                "key": key,
            }
        except Exception as e:
            logger.error(f"Failed to zadd to key {key}: {e}")
            return {
                "success": False,
                "error": f"Failed to zadd: {str(e)}",
                "key": key,
            }

    async def zrange(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        withscores: bool = True,
        reverse: bool = False,
        deserialize: bool = True,
    ) -> dict[str, Any]:
        """Get members from sorted set.

        Args:
            key: Sorted set key
            start: Start index
            end: End index (-1 for all)
            withscores: Include scores
            reverse: Reverse order (highest to lowest)
            deserialize: Deserialize JSON members

        Returns:
            Sorted set members
        """
        try:
            redis = await self._ensure_connection()
            
            if reverse:
                results = await redis.zrevrange(
                    key, start, end, withscores=withscores
                )
            else:
                results = await redis.zrange(
                    key, start, end, withscores=withscores
                )
            
            self._stats["hits"] += 1
            self._stats["total_operations"] += 1
            
            if withscores:
                members = []
                for member, score in results:
                    if deserialize and isinstance(member, str):
                        try:
                            member = json.loads(member)
                        except json.JSONDecodeError:
                            pass
                    members.append({
                        "member": member,
                        "score": score,
                    })
            else:
                members = []
                for member in results:
                    if deserialize and isinstance(member, str):
                        try:
                            member = json.loads(member)
                        except json.JSONDecodeError:
                            pass
                    members.append(member)
            
            return {
                "success": True,
                "key": key,
                "members": members,
                "count": len(members),
            }

        except RedisError as e:
            logger.error(f"Redis zrange failed for key {key}: {e}")
            return {
                "success": False,
                "error": f"Redis error: {str(e)}",
                "key": key,
            }
        except Exception as e:
            logger.error(f"Failed to zrange key {key}: {e}")
            return {
                "success": False,
                "error": f"Failed to zrange: {str(e)}",
                "key": key,
            }

    async def get_stats(self) -> dict[str, Any]:
        """Get Redis statistics.

        Returns:
            Cache statistics
        """
        try:
            redis = await self._ensure_connection()
            
            # Get Redis info
            info = await redis.info()
            
            # Calculate hit rate
            total_ops = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_ops
                if total_ops > 0
                else 0.0
            )
            
            return {
                "success": True,
                "stats": {
                    "hits": self._stats["hits"],
                    "misses": self._stats["misses"],
                    "hit_rate": hit_rate,
                    "sets": self._stats["sets"],
                    "deletes": self._stats["deletes"],
                    "publishes": self._stats["publishes"],
                    "total_operations": self._stats["total_operations"],
                },
                "redis_info": {
                    "version": info.get("redis_version"),
                    "uptime_seconds": info.get("uptime_in_seconds"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory": info.get("used_memory_human"),
                    "total_commands_processed": info.get("total_commands_processed"),
                },
            }

        except RedisError as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {
                "success": False,
                "error": f"Redis error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "success": False,
                "error": f"Failed to get stats: {str(e)}",
            }


def get_server_definition() -> dict[str, Any]:
    """Get Redis MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "redis",
        "category": "storage",
        "description": "Redis cache and pub/sub operations (key-value, sorted sets, messaging)",
        "tools": [
            {
                "name": "get_value",
                "description": "Get value from Redis cache",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Cache key",
                        },
                        "deserialize": {
                            "type": "boolean",
                            "description": "Deserialize JSON value",
                            "default": True,
                        },
                    },
                    "required": ["key"],
                },
            },
            {
                "name": "set_value",
                "description": "Set value in Redis cache with optional TTL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Cache key",
                        },
                        "value": {
                            "description": "Value to cache (string, number, object, array)",
                        },
                        "ttl": {
                            "type": "integer",
                            "description": "Time-to-live in seconds (optional)",
                        },
                        "serialize": {
                            "type": "boolean",
                            "description": "Serialize value as JSON",
                            "default": True,
                        },
                    },
                    "required": ["key", "value"],
                },
            },
            {
                "name": "delete_key",
                "description": "Delete key from Redis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Cache key to delete",
                        }
                    },
                    "required": ["key"],
                },
            },
            {
                "name": "publish_message",
                "description": "Publish message to Redis pub/sub channel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "description": "Channel name",
                        },
                        "message": {
                            "description": "Message to publish",
                        },
                        "serialize": {
                            "type": "boolean",
                            "description": "Serialize message as JSON",
                            "default": True,
                        },
                    },
                    "required": ["channel", "message"],
                },
            },
            {
                "name": "subscribe_channel",
                "description": "Subscribe to Redis pub/sub channel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "description": "Channel name",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                            "default": 10,
                        },
                    },
                    "required": ["channel"],
                },
            },
            {
                "name": "zadd",
                "description": "Add member to sorted set with score (for time-series data)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Sorted set key",
                        },
                        "score": {
                            "type": "number",
                            "description": "Score for ordering (e.g., timestamp)",
                        },
                        "member": {
                            "description": "Member to add",
                        },
                        "serialize": {
                            "type": "boolean",
                            "description": "Serialize member as JSON",
                            "default": True,
                        },
                    },
                    "required": ["key", "score", "member"],
                },
            },
            {
                "name": "zrange",
                "description": "Get members from sorted set (time-series data)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Sorted set key",
                        },
                        "start": {
                            "type": "integer",
                            "description": "Start index",
                            "default": 0,
                        },
                        "end": {
                            "type": "integer",
                            "description": "End index (-1 for all)",
                            "default": -1,
                        },
                        "withscores": {
                            "type": "boolean",
                            "description": "Include scores",
                            "default": True,
                        },
                        "reverse": {
                            "type": "boolean",
                            "description": "Reverse order (highest to lowest)",
                            "default": False,
                        },
                        "deserialize": {
                            "type": "boolean",
                            "description": "Deserialize JSON members",
                            "default": True,
                        },
                    },
                    "required": ["key"],
                },
            },
            {
                "name": "get_stats",
                "description": "Get Redis cache statistics",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "critical",
            "category": "storage",
            "requires": ["redis[hiredis]>=5.0.1"],
            "performance_target": "<2ms per operation",
        },
    }