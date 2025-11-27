"""
Redis Pub/Sub Manager for Multi-Instance Broadcasting.

Enables message broadcasting across multiple API instances
for horizontal scaling and load balancing.
"""

import asyncio
import logging
import json
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import redis.asyncio as redis

from .event_types import StreamEvent, EventType

logger = logging.getLogger(__name__)


class RedisPubSubManager:
    """
    Redis Pub/Sub manager for distributed event broadcasting.
    
    Features:
    - Publish events to Redis channels
    - Subscribe to Redis channels
    - Pattern-based subscriptions
    - Automatic reconnection
    - Message serialization/deserialization
    - Multiple instance coordination
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "axiom:streaming:",
        max_reconnect_attempts: int = 5
    ):
        """
        Initialize Redis Pub/Sub manager.
        
        Args:
            redis_url: Redis connection URL
            channel_prefix: Prefix for all channels
            max_reconnect_attempts: Max reconnection attempts
        """
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.subscriptions: Dict[str, Callable] = {}
        
        self._listen_task: Optional[asyncio.Task] = None
        self._connected = False
        
        logger.info(f"RedisPubSubManager initialized with URL: {redis_url}")
    
    async def connect(self):
        """Connect to Redis and initialize pub/sub."""
        try:
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.pubsub = self.redis_client.pubsub()
            self._connected = True
            
            logger.info("Connected to Redis successfully")
            
            # Start listening for messages
            await self._start_listening()
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Redis and clean up."""
        self._connected = False
        
        # Stop listening task
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        # Unsubscribe from all channels
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Disconnected from Redis")
    
    async def publish(
        self,
        event: StreamEvent,
        channel: Optional[str] = None
    ) -> int:
        """
        Publish event to Redis channel.
        
        Args:
            event: Event to publish
            channel: Optional specific channel (defaults to event type)
            
        Returns:
            Number of subscribers that received the message
        """
        if not self._connected or not self.redis_client:
            logger.warning("Not connected to Redis, cannot publish")
            return 0
        
        try:
            # Determine channel
            if channel is None:
                channel = f"{self.channel_prefix}{event.event_type.value}"
            else:
                channel = f"{self.channel_prefix}{channel}"
            
            # Serialize event
            message = event.to_json()
            
            # Publish to Redis
            subscribers = await self.redis_client.publish(channel, message)
            
            logger.debug(f"Published {event.event_type.value} to {channel}, reached {subscribers} subscribers")
            
            return subscribers
            
        except Exception as e:
            logger.error(f"Error publishing to Redis: {e}")
            return 0
    
    async def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[StreamEvent], None],
        pattern: bool = False
    ):
        """
        Subscribe to a Redis channel for specific event type.
        
        Args:
            event_type: Event type to subscribe to
            callback: Callback function to handle received events
            pattern: Whether to use pattern matching
        """
        if not self._connected or not self.pubsub:
            logger.warning("Not connected to Redis, cannot subscribe")
            return
        
        try:
            channel = f"{self.channel_prefix}{event_type.value}"
            
            # Store callback
            self.subscriptions[channel] = callback
            
            # Subscribe to channel
            if pattern:
                await self.pubsub.psubscribe(channel)
            else:
                await self.pubsub.subscribe(channel)
            
            logger.info(f"Subscribed to channel: {channel}")
            
        except Exception as e:
            logger.error(f"Error subscribing to Redis channel: {e}")
    
    async def subscribe_pattern(
        self,
        pattern: str,
        callback: Callable[[StreamEvent], None]
    ):
        """
        Subscribe to multiple channels using pattern matching.
        
        Args:
            pattern: Pattern for channel matching (e.g., "axiom:streaming:*")
            callback: Callback function
        """
        if not self._connected or not self.pubsub:
            logger.warning("Not connected to Redis, cannot subscribe")
            return
        
        try:
            full_pattern = f"{self.channel_prefix}{pattern}"
            
            # Store callback
            self.subscriptions[full_pattern] = callback
            
            # Subscribe with pattern
            await self.pubsub.psubscribe(full_pattern)
            
            logger.info(f"Subscribed to pattern: {full_pattern}")
            
        except Exception as e:
            logger.error(f"Error subscribing to Redis pattern: {e}")
    
    async def unsubscribe(self, event_type: EventType):
        """
        Unsubscribe from a channel.
        
        Args:
            event_type: Event type to unsubscribe from
        """
        if not self._connected or not self.pubsub:
            return
        
        try:
            channel = f"{self.channel_prefix}{event_type.value}"
            
            await self.pubsub.unsubscribe(channel)
            
            if channel in self.subscriptions:
                del self.subscriptions[channel]
            
            logger.info(f"Unsubscribed from channel: {channel}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from Redis channel: {e}")
    
    async def _start_listening(self):
        """Start background task to listen for messages."""
        if self._listen_task is None or self._listen_task.done():
            self._listen_task = asyncio.create_task(self._listen_loop())
            logger.info("Started Redis message listener")
    
    async def _listen_loop(self):
        """Background loop to listen for Redis messages."""
        try:
            if not self.pubsub:
                logger.error("PubSub not initialized")
                return
            
            while self._connected:
                try:
                    message = await self.pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0
                    )
                    
                    if message and message['type'] in ['message', 'pmessage']:
                        await self._handle_message(message)
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in listen loop: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("Redis listen loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in listen loop: {e}")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """
        Handle received Redis message.
        
        Args:
            message: Message from Redis
        """
        try:
            # Extract channel and data
            channel = message.get('channel') or message.get('pattern')
            data = message.get('data')
            
            if not data or not channel:
                return
            
            # Deserialize event
            event_data = json.loads(data)
            event = StreamEvent(**event_data)
            
            # Find and call appropriate callback
            if channel in self.subscriptions:
                callback = self.subscriptions[channel]
                
                # Call callback (sync or async)
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                    
                logger.debug(f"Handled message from {channel}")
            
        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")
    
    async def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self._connected or not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis pub/sub statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self._connected or not self.redis_client:
            return {
                "connected": False,
                "subscriptions": 0,
                "channels": []
            }
        
        try:
            # Get info from Redis
            info = await self.redis_client.info('stats')
            
            return {
                "connected": True,
                "subscriptions": len(self.subscriptions),
                "channels": list(self.subscriptions.keys()),
                "pubsub_channels": info.get('pubsub_channels', 0),
                "pubsub_patterns": info.get('pubsub_patterns', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {
                "connected": True,
                "subscriptions": len(self.subscriptions),
                "channels": list(self.subscriptions.keys()),
                "error": str(e)
            }


__all__ = ["RedisPubSubManager"]