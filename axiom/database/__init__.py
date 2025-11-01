"""
Database infrastructure for institutional-grade data management.

Provides:
- PostgreSQL integration for structured financial data
- Vector DB integration for semantic search
- Connection pooling and query optimization
- Migration management
- Transaction support
"""

from .connection import DatabaseConnection, get_db
from .models import (
    Base,
    PriceData,
    PortfolioPosition,
    Trade,
    CompanyFundamental,
    VaRCalculation,
    PerformanceMetric,
    PortfolioOptimization,
    FeatureData,
    ValidationResult,
    PipelineRun,
    DataLineage,
)
from .session import SessionManager, get_session
from .vector_store import VectorStore, VectorStoreType
from .integrations import (
    VaRIntegration,
    PortfolioIntegration,
    MarketDataIntegration,
    VectorIntegration
)
from .feature_integration import FeatureIntegration
from .quality_integration import QualityIntegration, PipelineIntegration
from .cache_integration import RedisCache
from .graph_integration import Neo4jGraph
from .multi_db_coordinator import MultiDatabaseCoordinator
from .migrations import MigrationManager, get_migration_manager

__all__ = [
    "DatabaseConnection",
    "get_db",
    "Base",
    "PriceData",
    "PortfolioPosition",
    "Trade",
    "CompanyFundamental",
    "VaRCalculation",
    "PerformanceMetric",
    "PortfolioOptimization",
    "FeatureData",
    "ValidationResult",
    "PipelineRun",
    "DataLineage",
    "SessionManager",
    "get_session",
    "VectorStore",
    "VectorStoreType",
    "VaRIntegration",
    "PortfolioIntegration",
    "MarketDataIntegration",
    "VectorIntegration",
    "FeatureIntegration",
    "QualityIntegration",
    "PipelineIntegration",
    "RedisCache",
    "Neo4jGraph",
    "MultiDatabaseCoordinator",
    "MigrationManager",
    "get_migration_manager",
]

# Convenience function
get_db_session = get_session