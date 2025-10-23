"""
Database migration management system.

Features:
- Schema versioning
- Migration execution
- Rollback support
- Migration history tracking
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass

from sqlalchemy import text, Table, Column, Integer, String, DateTime, MetaData
from sqlalchemy.exc import SQLAlchemyError

from .connection import get_db
from .models import Base

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Migration definition."""
    version: int
    name: str
    up: Callable
    down: Optional[Callable] = None
    description: str = ""


class MigrationManager:
    """
    Database migration manager.
    
    Handles:
    - Schema creation
    - Migration execution
    - Version tracking
    - Rollback operations
    """
    
    def __init__(self):
        """Initialize migration manager."""
        self.db = get_db()
        self.migrations: List[Migration] = []
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Create migration tracking table if it doesn't exist."""
        with self.db.session_scope() as session:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    version INTEGER NOT NULL UNIQUE,
                    name VARCHAR(255) NOT NULL,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """))
            session.commit()
        
        logger.info("Migration tracking table ready")
    
    def register_migration(
        self,
        version: int,
        name: str,
        up: Callable,
        down: Optional[Callable] = None,
        description: str = ""
    ):
        """
        Register a migration.
        
        Args:
            version: Migration version number
            name: Migration name
            up: Upgrade function
            down: Downgrade function (optional)
            description: Migration description
        """
        migration = Migration(
            version=version,
            name=name,
            up=up,
            down=down,
            description=description
        )
        self.migrations.append(migration)
        logger.debug(f"Registered migration: {version}_{name}")
    
    def get_current_version(self) -> int:
        """Get current database schema version."""
        with self.db.session_scope() as session:
            result = session.execute(text(
                "SELECT MAX(version) FROM schema_migrations"
            ))
            version = result.scalar()
            return version or 0
    
    def get_applied_migrations(self) -> List[int]:
        """Get list of applied migration versions."""
        with self.db.session_scope() as session:
            result = session.execute(text(
                "SELECT version FROM schema_migrations ORDER BY version"
            ))
            return [row[0] for row in result]
    
    def migrate(self, target_version: Optional[int] = None):
        """
        Run migrations up to target version.
        
        Args:
            target_version: Target version (None = latest)
        """
        current_version = self.get_current_version()
        
        if target_version is None:
            target_version = max([m.version for m in self.migrations]) if self.migrations else 0
        
        if current_version >= target_version:
            logger.info(f"Database already at version {current_version}")
            return
        
        # Sort migrations by version
        sorted_migrations = sorted(self.migrations, key=lambda m: m.version)
        
        # Execute pending migrations
        for migration in sorted_migrations:
            if migration.version <= current_version:
                continue
            
            if migration.version > target_version:
                break
            
            self._execute_migration(migration, direction='up')
    
    def rollback(self, target_version: Optional[int] = None):
        """
        Rollback migrations to target version.
        
        Args:
            target_version: Target version (None = previous version)
        """
        current_version = self.get_current_version()
        
        if target_version is None:
            # Rollback one version
            target_version = current_version - 1
        
        if target_version < 0:
            target_version = 0
        
        if current_version <= target_version:
            logger.info(f"Already at or below version {target_version}")
            return
        
        # Sort migrations by version (descending)
        sorted_migrations = sorted(self.migrations, key=lambda m: m.version, reverse=True)
        
        # Execute rollback migrations
        applied = self.get_applied_migrations()
        for migration in sorted_migrations:
            if migration.version not in applied:
                continue
            
            if migration.version <= target_version:
                break
            
            self._execute_migration(migration, direction='down')
    
    def _execute_migration(self, migration: Migration, direction: str = 'up'):
        """Execute a single migration."""
        logger.info(f"{'Applying' if direction == 'up' else 'Rolling back'} migration: {migration.version}_{migration.name}")
        
        try:
            with self.db.session_scope() as session:
                if direction == 'up':
                    # Execute upgrade
                    migration.up(session)
                    
                    # Record migration
                    session.execute(text("""
                        INSERT INTO schema_migrations (version, name, description)
                        VALUES (:version, :name, :description)
                    """), {
                        'version': migration.version,
                        'name': migration.name,
                        'description': migration.description
                    })
                else:
                    # Execute downgrade
                    if migration.down is None:
                        raise ValueError(f"Migration {migration.version} has no downgrade")
                    
                    migration.down(session)
                    
                    # Remove migration record
                    session.execute(text("""
                        DELETE FROM schema_migrations WHERE version = :version
                    """), {'version': migration.version})
                
                session.commit()
            
            logger.info(f"Migration {migration.version}_{migration.name} {'applied' if direction == 'up' else 'rolled back'}")
            
        except Exception as e:
            logger.error(f"Migration {migration.version}_{migration.name} failed: {e}")
            raise
    
    def init_schema(self):
        """Initialize database schema from models."""
        logger.info("Initializing database schema...")
        
        try:
            # Create all tables from models
            Base.metadata.create_all(bind=self.db._engine)
            logger.info("Database schema initialized successfully")
            
            # Record as migration 0
            with self.db.session_scope() as session:
                # Check if already recorded
                result = session.execute(text(
                    "SELECT 1 FROM schema_migrations WHERE version = 0"
                ))
                if not result.scalar():
                    session.execute(text("""
                        INSERT INTO schema_migrations (version, name, description)
                        VALUES (0, 'init_schema', 'Initial schema creation')
                    """))
                    session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise
    
    def drop_all(self):
        """Drop all database tables (DANGEROUS!)."""
        logger.warning("Dropping all database tables...")
        
        try:
            Base.metadata.drop_all(bind=self.db._engine)
            
            # Drop migration table
            with self.db.session_scope() as session:
                session.execute(text("DROP TABLE IF EXISTS schema_migrations"))
                session.commit()
            
            logger.info("All tables dropped")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    def status(self) -> dict:
        """Get migration status."""
        current_version = self.get_current_version()
        applied_migrations = self.get_applied_migrations()
        
        total_migrations = len(self.migrations)
        latest_version = max([m.version for m in self.migrations]) if self.migrations else 0
        
        return {
            'current_version': current_version,
            'latest_version': latest_version,
            'applied_migrations': len(applied_migrations),
            'total_migrations': total_migrations,
            'pending_migrations': total_migrations - len(applied_migrations),
            'is_up_to_date': current_version >= latest_version
        }


# Global migration manager
_migration_manager: Optional[MigrationManager] = None


def get_migration_manager() -> MigrationManager:
    """Get global migration manager."""
    global _migration_manager
    
    if _migration_manager is None:
        _migration_manager = MigrationManager()
    
    return _migration_manager


# Export
__all__ = [
    "Migration",
    "MigrationManager",
    "get_migration_manager",
]