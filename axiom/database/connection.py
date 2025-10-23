"""
Database connection management with connection pooling.

Features:
- Connection pooling for performance
- Automatic reconnection
- Health checks
- Transaction management
- Thread-safe operations
"""

import logging
from contextlib import contextmanager
from typing import Optional, Generator
from urllib.parse import quote_plus

from sqlalchemy import create_engine, event, exc, pool
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from ..config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    PostgreSQL database connection manager with pooling.
    
    Features:
    - Connection pooling for high performance
    - Automatic connection recycling
    - Health monitoring
    - Thread-safe operations
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ):
        """
        Initialize database connection.
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Number of connections in pool
            max_overflow: Maximum overflow connections
            pool_timeout: Timeout for getting connection
            pool_recycle: Recycle connections after N seconds
            echo: Echo SQL statements (debug mode)
        """
        self.database_url = database_url or self._build_database_url()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo or settings.debug
        
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        
    def _build_database_url(self) -> str:
        """Build database URL from settings."""
        # Get database configuration from environment
        db_user = getattr(settings, 'postgres_user', 'axiom')
        db_password = getattr(settings, 'postgres_password', 'axiom_password')
        db_host = getattr(settings, 'postgres_host', 'localhost')
        db_port = getattr(settings, 'postgres_port', 5432)
        db_name = getattr(settings, 'postgres_db', 'axiom_finance')
        
        # URL encode password to handle special characters
        encoded_password = quote_plus(db_password)
        
        return f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    
    def connect(self) -> Engine:
        """
        Create database engine with connection pooling.
        
        Returns:
            SQLAlchemy Engine
        """
        if self._engine is not None:
            return self._engine
        
        logger.info(f"Connecting to PostgreSQL database...")
        
        try:
            # Create engine with connection pooling
            self._engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=True,  # Enable connection health checks
                echo=self.echo,
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "axiom_analytics",
                },
            )
            
            # Set up event listeners for connection management
            self._setup_event_listeners()
            
            # Test connection
            with self._engine.connect() as conn:
                conn.execute("SELECT 1")
            
            logger.info("Successfully connected to PostgreSQL database")
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
            )
            
            return self._engine
            
        except exc.SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _setup_event_listeners(self):
        """Setup event listeners for connection management."""
        
        @event.listens_for(self._engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Called when connection is created."""
            logger.debug("New database connection established")
            
            # Set connection parameters
            with dbapi_conn.cursor() as cursor:
                cursor.execute("SET TIME ZONE 'UTC'")
                cursor.execute("SET statement_timeout = '30s'")
        
        @event.listens_for(self._engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Called when connection is checked out from pool."""
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self._engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Called when connection is returned to pool."""
            logger.debug("Connection returned to pool")
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            SQLAlchemy Session
        """
        if self._session_factory is None:
            self.connect()
        
        return self._session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.
        
        Usage:
            with db.session_scope() as session:
                session.add(obj)
                # Automatic commit on success, rollback on error
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if database is healthy
        """
        try:
            if self._engine is None:
                self.connect()
            
            with self._engine.connect() as conn:
                result = conn.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_pool_status(self) -> dict:
        """
        Get connection pool status.
        
        Returns:
            Dictionary with pool statistics
        """
        if self._engine is None or not hasattr(self._engine.pool, 'size'):
            return {
                "connected": False,
                "pool_size": 0,
                "checked_out": 0,
                "overflow": 0,
            }
        
        pool = self._engine.pool
        return {
            "connected": True,
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "timeout": pool.timeout(),
        }
    
    def close(self):
        """Close database connection and dispose of pool."""
        if self._engine is not None:
            logger.info("Closing database connection pool")
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None


def get_db() -> DatabaseConnection:
    """
    Get global database connection instance.
    
    Returns:
        DatabaseConnection instance
    """
    global _db_connection
    
    if _db_connection is None:
        _db_connection = DatabaseConnection()
        _db_connection.connect()
    
    return _db_connection


def close_db():
    """Close global database connection."""
    global _db_connection
    
    if _db_connection is not None:
        _db_connection.close()
        _db_connection = None


# Export
__all__ = [
    "DatabaseConnection",
    "get_db",
    "close_db",
]