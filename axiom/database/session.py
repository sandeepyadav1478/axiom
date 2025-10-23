"""
Session management with transaction support.

Features:
- Session lifecycle management
- Transaction context managers
- Batch operations
- Query helpers
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional, Type, TypeVar, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from .connection import get_db
from .models import Base

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Base)


class SessionManager:
    """
    Database session manager with transaction support.
    
    Provides:
    - Session lifecycle management
    - Transaction management
    - Batch operations
    - Query helpers
    """
    
    def __init__(self, session: Optional[Session] = None):
        """
        Initialize session manager.
        
        Args:
            session: Existing session or create new one
        """
        self._session = session
        self._owns_session = session is None
    
    @property
    def session(self) -> Session:
        """Get or create session."""
        if self._session is None:
            db = get_db()
            self._session = db.get_session()
        return self._session
    
    def commit(self):
        """Commit current transaction."""
        try:
            self.session.commit()
            logger.debug("Transaction committed")
        except SQLAlchemyError as e:
            logger.error(f"Commit failed: {e}")
            self.rollback()
            raise
    
    def rollback(self):
        """Rollback current transaction."""
        self.session.rollback()
        logger.debug("Transaction rolled back")
    
    def flush(self):
        """Flush pending changes without committing."""
        self.session.flush()
    
    def close(self):
        """Close session if owned by manager."""
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None
    
    # CRUD Operations
    
    def add(self, obj: Base) -> Base:
        """
        Add object to session.
        
        Args:
            obj: Model instance
            
        Returns:
            Added object
        """
        self.session.add(obj)
        return obj
    
    def add_all(self, objects: List[Base]):
        """
        Add multiple objects to session.
        
        Args:
            objects: List of model instances
        """
        self.session.add_all(objects)
    
    def delete(self, obj: Base):
        """
        Delete object from database.
        
        Args:
            obj: Model instance to delete
        """
        self.session.delete(obj)
    
    def get(self, model: Type[T], id: Any) -> Optional[T]:
        """
        Get object by primary key.
        
        Args:
            model: Model class
            id: Primary key value
            
        Returns:
            Model instance or None
        """
        return self.session.get(model, id)
    
    def query(self, model: Type[T]) -> Any:
        """
        Create query for model.
        
        Args:
            model: Model class
            
        Returns:
            Query object
        """
        return self.session.query(model)
    
    def execute(self, statement: Any) -> Any:
        """
        Execute raw SQL statement.
        
        Args:
            statement: SQL statement
            
        Returns:
            Result
        """
        return self.session.execute(statement)
    
    # Batch Operations
    
    def bulk_insert(self, objects: List[Base], return_defaults: bool = False):
        """
        Bulk insert objects.
        
        Args:
            objects: List of model instances
            return_defaults: Whether to return generated defaults
        """
        self.session.bulk_save_objects(
            objects,
            return_defaults=return_defaults
        )
        logger.info(f"Bulk inserted {len(objects)} objects")
    
    def bulk_update(self, model: Type[T], updates: List[Dict[str, Any]]):
        """
        Bulk update objects.
        
        Args:
            model: Model class
            updates: List of update dictionaries with 'id' key
        """
        self.session.bulk_update_mappings(model, updates)
        logger.info(f"Bulk updated {len(updates)} {model.__name__} objects")
    
    # Transaction Context
    
    @contextmanager
    def transaction(self) -> Generator[Session, None, None]:
        """
        Transaction context manager.
        
        Usage:
            with session_manager.transaction() as session:
                session.add(obj)
                # Auto-commit on success, rollback on error
        """
        try:
            yield self.session
            self.commit()
        except Exception as e:
            self.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.rollback()
        self.close()


def get_session() -> Session:
    """
    Get a database session.
    
    Returns:
        SQLAlchemy Session
    """
    db = get_db()
    return db.get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope.
    
    Usage:
        with session_scope() as session:
            session.add(obj)
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Export
__all__ = [
    "SessionManager",
    "get_session",
    "session_scope",
]