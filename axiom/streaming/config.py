"""
Streaming Configuration

Configuration management for real-time data streaming infrastructure.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class StreamingConfig:
    """
    Configuration for real-time streaming infrastructure.
    
    Attributes:
        WebSocket Configuration:
            reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Initial reconnection delay (seconds)
            ping_interval: WebSocket ping interval (seconds)
            connection_timeout: Connection timeout (seconds)
            max_reconnect_delay: Maximum reconnection delay (seconds)
        
        Data Providers:
            polygon_api_key: Polygon.io API key
            alpaca_api_key: Alpaca API key
            alpaca_secret: Alpaca secret key
            binance_api_key: Binance API key
            binance_secret: Binance secret key
        
        Redis Configuration:
            redis_url: Redis connection URL
            redis_ttl: Time-to-live for cached data (seconds)
            redis_max_connections: Maximum Redis connection pool size
        
        Processing Configuration:
            batch_size: Event batch size for processing
            batch_timeout: Batch processing timeout (seconds)
            max_queue_size: Maximum event queue size
            worker_threads: Number of worker threads
        
        Risk Monitoring:
            risk_check_interval: Risk check interval (seconds)
            var_limit: VaR limit (percentage, e.g., 0.02 = 2%)
            position_limit_pct: Maximum position size (percentage)
            drawdown_limit: Maximum drawdown limit (percentage)
        
        Performance:
            enable_metrics: Enable performance metrics collection
            log_latency: Log operation latency
            metrics_interval: Metrics reporting interval (seconds)
    """
    
    # WebSocket Configuration
    reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    ping_interval: int = 30
    connection_timeout: int = 10
    heartbeat_interval: int = 30
    
    # Data Providers
    polygon_api_key: Optional[str] = None
    alpaca_api_key: Optional[str] = None
    alpaca_secret: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_secret: Optional[str] = None
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_ttl: int = 86400  # 24 hours
    redis_max_connections: int = 50
    redis_decode_responses: bool = True
    
    # Processing Configuration
    batch_size: int = 100
    batch_timeout: float = 0.1
    max_queue_size: int = 10000
    worker_threads: int = 4
    enable_async: bool = True
    
    # Risk Monitoring
    risk_check_interval: int = 1  # seconds
    var_limit: float = 0.02  # 2%
    position_limit_pct: float = 0.25  # 25% max per position
    drawdown_limit: float = 0.10  # 10% maximum drawdown
    margin_requirement: float = 0.30  # 30% margin requirement
    
    # Performance & Monitoring
    enable_metrics: bool = True
    log_latency: bool = True
    metrics_interval: int = 60  # seconds
    log_level: str = "INFO"
    
    # Feature Flags
    enable_order_book: bool = False
    enable_level2_data: bool = False
    enable_news_feed: bool = False
    enable_options_data: bool = False
    
    # Rate Limiting
    rate_limit_per_second: int = 100
    rate_limit_burst: int = 200
    
    def __post_init__(self):
        """Load configuration from environment variables if not set."""
        # Load API keys from environment
        if not self.polygon_api_key:
            self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        
        if not self.alpaca_api_key:
            self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        
        if not self.alpaca_secret:
            self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.binance_api_key:
            self.binance_api_key = os.getenv('BINANCE_API_KEY')
        
        if not self.binance_secret:
            self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        
        # Load Redis URL from environment
        redis_env = os.getenv('REDIS_URL')
        if redis_env:
            self.redis_url = redis_env
    
    @classmethod
    def from_env(cls) -> 'StreamingConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            StreamingConfig instance
        """
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StreamingConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            StreamingConfig instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            # WebSocket
            'reconnect_attempts': self.reconnect_attempts,
            'reconnect_delay': self.reconnect_delay,
            'max_reconnect_delay': self.max_reconnect_delay,
            'ping_interval': self.ping_interval,
            'connection_timeout': self.connection_timeout,
            
            # Data Providers (exclude secrets)
            'polygon_api_key': '***' if self.polygon_api_key else None,
            'alpaca_api_key': '***' if self.alpaca_api_key else None,
            'binance_api_key': '***' if self.binance_api_key else None,
            
            # Redis
            'redis_url': self.redis_url,
            'redis_ttl': self.redis_ttl,
            'redis_max_connections': self.redis_max_connections,
            
            # Processing
            'batch_size': self.batch_size,
            'batch_timeout': self.batch_timeout,
            'max_queue_size': self.max_queue_size,
            
            # Risk
            'var_limit': self.var_limit,
            'position_limit_pct': self.position_limit_pct,
            'drawdown_limit': self.drawdown_limit,
            
            # Performance
            'enable_metrics': self.enable_metrics,
            'log_latency': self.log_latency,
        }
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if valid, False otherwise
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate numeric ranges
        if self.reconnect_attempts < 1:
            raise ValueError("reconnect_attempts must be >= 1")
        
        if self.reconnect_delay < 0:
            raise ValueError("reconnect_delay must be >= 0")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        if self.var_limit < 0 or self.var_limit > 1:
            raise ValueError("var_limit must be between 0 and 1")
        
        if self.position_limit_pct < 0 or self.position_limit_pct > 1:
            raise ValueError("position_limit_pct must be between 0 and 1")
        
        return True


# Global default configuration
DEFAULT_CONFIG = StreamingConfig()