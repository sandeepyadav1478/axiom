"""
Multi-Database Coordinator.

Orchestrates multiple database types for optimal performance:
- PostgreSQL: Structured financial data (price, trades, fundamentals)
- Vector DB: Semantic search, company similarity, document embeddings  
- Redis: Hot data caching, real-time features, session state

This is how REAL production systems work - using the right database for each use case!
"""

import logging
from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from .connection import get_db
from .session import SessionManager
from .models import PriceData, CompanyFundamental, FeatureData
from .integrations import MarketDataIntegration, VectorIntegration
from .feature_integration import FeatureIntegration
from .cache_integration import RedisCache
from .vector_store import VectorStoreType

logger = logging.getLogger(__name__)


class MultiDatabaseCoordinator:
    """
    Coordinates multiple databases for optimal performance.
    
    Architecture:
    - PostgreSQL → Structured data (historical, transactions)
    - Vector DB → Semantic search (documents, company similarity)
    - Redis → Hot data (real-time, frequently accessed)
    
    This matches real-world financial platforms (Bloomberg, FactSet, etc.)
    """
    
    def __init__(
        self,
        use_cache: bool = True,
        use_vector_db: bool = True,
        vector_store_type: VectorStoreType = VectorStoreType.CHROMA
    ):
        """
        Initialize multi-database coordinator.
        
        Args:
            use_cache: Enable Redis caching
            use_vector_db: Enable vector database
            vector_store_type: Type of vector database
        """
        # PostgreSQL (always enabled)
        self.session = SessionManager()
        self.market_data = MarketDataIntegration(self.session)
        self.features = FeatureIntegration(self.session)
        
        # Redis cache (optional but recommended)
        self.cache: Optional[RedisCache] = None
        if use_cache:
            try:
                self.cache = RedisCache()
                if self.cache.health_check():
                    logger.info("✅ Redis cache enabled")
                else:
                    logger.warning("⚠️  Redis cache unhealthy, running without cache")
                    self.cache = None
            except Exception as e:
                logger.warning(f"⚠️  Redis not available: {e}, running without cache")
                self.cache = None
        
        # Vector DB (optional but recommended for semantic search)
        self.vector: Optional[VectorIntegration] = None
        if use_vector_db:
            try:
                self.vector = VectorIntegration(
                    session_manager=self.session,
                    vector_store_type=vector_store_type.value
                )
                logger.info(f"✅ Vector DB enabled ({vector_store_type.value})")
            except Exception as e:
                logger.warning(f"⚠️  Vector DB not available: {e}")
                self.vector = None
    
    # ============================================
    # Price Data (PostgreSQL + Redis Cache)
    # ============================================
    
    def get_latest_price(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[Decimal]:
        """
        Get latest price with caching.
        
        Flow:
        1. Check Redis cache (sub-millisecond)
        2. If miss, query PostgreSQL
        3. Cache result in Redis
        
        Args:
            symbol: Asset symbol
            use_cache: Use Redis cache
            
        Returns:
            Latest price or None
        """
        # Try cache first (FAST!)
        if use_cache and self.cache:
            cached = self.cache.get_latest_price(symbol)
            if cached is not None:
                logger.debug(f"Cache HIT for {symbol} price")
                return Decimal(str(cached))
        
        # Query PostgreSQL (slower but authoritative)
        latest = self.session.query(PriceData).filter(
            PriceData.symbol == symbol
        ).order_by(PriceData.timestamp.desc()).first()
        
        if latest:
            price = latest.close
            
            # Cache for next time
            if self.cache:
                self.cache.cache_latest_price(symbol, float(price), ttl=60)
                logger.debug(f"Cached price for {symbol}")
            
            return price
        
        return None
    
    def get_price_history(
        self,
        symbol: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get price history from PostgreSQL.
        
        Args:
            symbol: Asset symbol
            days: Number of days
            
        Returns:
            DataFrame with OHLCV data
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        prices = self.session.query(PriceData).filter(
            PriceData.symbol == symbol,
            PriceData.timestamp >= cutoff
        ).order_by(PriceData.timestamp.asc()).all()
        
        if not prices:
            return pd.DataFrame()
        
        data = [{
            'timestamp': p.timestamp,
            'open': float(p.open),
            'high': float(p.high),
            'low': float(p.low),
            'close': float(p.close),
            'volume': float(p.volume),
        } for p in prices]
        
        df = pd.DataFrame(data).set_index('timestamp')
        
        logger.info(f"Retrieved {len(df)} price records for {symbol} from PostgreSQL")
        
        return df
    
    # ============================================
    # Features (PostgreSQL + Redis Cache)
    # ============================================
    
    def get_feature(
        self,
        symbol: str,
        feature_name: str,
        use_cache: bool = True
    ) -> Optional[float]:
        """
        Get latest feature with caching.
        
        Flow:
        1. Check Redis cache (microsecond latency)
        2. If miss, query PostgreSQL
        3. Cache result
        
        Args:
            symbol: Asset symbol
            feature_name: Feature name
            use_cache: Use Redis cache
            
        Returns:
            Feature value or None
        """
        # Try cache
        if use_cache and self.cache:
            cached = self.cache.get_feature(symbol, feature_name)
            if cached is not None:
                logger.debug(f"Cache HIT for {symbol}/{feature_name}")
                return cached
        
        # Query PostgreSQL
        features = self.features.get_latest_features(symbol, [feature_name])
        value = features.get(feature_name)
        
        # Cache it
        if value is not None and self.cache:
            self.cache.cache_feature(symbol, feature_name, value, ttl=300)  # 5 min
        
        return value
    
    def get_features_batch(
        self,
        symbol: str,
        feature_names: List[str],
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Get multiple features with caching.
        
        Args:
            symbol: Asset symbol
            feature_names: List of feature names
            use_cache: Use Redis cache
            
        Returns:
            Dictionary of {feature_name: value}
        """
        results = {}
        
        # Try cache first
        if use_cache and self.cache:
            cached = self.cache.get_features_bulk(symbol, feature_names)
            results.update(cached)
            
            # Find cache misses
            cache_misses = [name for name in feature_names if name not in cached]
        else:
            cache_misses = feature_names
        
        # Query PostgreSQL for misses
        if cache_misses:
            db_results = self.features.get_latest_features(symbol, cache_misses)
            results.update(db_results)
            
            # Cache them
            if self.cache and db_results:
                self.cache.cache_features_bulk(symbol, db_results, ttl=300)
        
        logger.info(f"Retrieved {len(results)}/{len(feature_names)} features for {symbol}")
        logger.info(f"  Cache: {len(cached) if use_cache and self.cache else 0}, DB: {len(cache_misses)}")
        
        return results
    
    # ============================================
    # Company Similarity (Vector DB)
    # ============================================
    
    def find_similar_companies(
        self,
        query_symbol: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar companies using vector similarity search.
        
        Uses:
        - Company embeddings in Vector DB
        - Semantic similarity based on fundamentals, news, filings
        
        Args:
            query_symbol: Company to find similarities for
            top_k: Number of similar companies
            
        Returns:
            List of similar companies with scores
        """
        if not self.vector:
            logger.warning("Vector DB not available, cannot perform similarity search")
            return []
        
        # Get company fundamental data
        company = self.session.query(CompanyFundamental).filter(
            CompanyFundamental.symbol == query_symbol
        ).order_by(CompanyFundamental.report_date.desc()).first()
        
        if not company:
            logger.warning(f"No fundamental data found for {query_symbol}")
            return []
        
        # Create query embedding from company data
        # (In production, this would use company description, filings, etc.)
        query_text = f"{company.company_name} {company.sector} {company.industry}"
        
        # Search for similar companies
        # (Requires embeddings to be pre-computed and stored)
        collection_name = "companies_embeddings"
        
        try:
            # This requires actual embeddings to be stored first
            # For now, return placeholder
            logger.info(f"Similarity search for {query_symbol} in vector DB")
            return []
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
    
    # ============================================
    # Document Search (Vector DB)
    # ============================================
    
    def search_documents(
        self,
        query_text: str,
        document_type: str = "sec_filing",
        top_k: int = 10,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across documents.
        
        Uses Vector DB for:
        - SEC filings search
        - Research paper search
        - News article search
        
        Args:
            query_text: Search query
            document_type: Type of documents
            top_k: Number of results
            symbol: Filter by company
            
        Returns:
            List of relevant documents
        """
        if not self.vector:
            logger.warning("Vector DB not available")
            return []
        
        # In production, you'd:
        # 1. Create embedding for query_text
        # 2. Search vector DB for similar embeddings
        # 3. Return matched documents
        
        logger.info(f"Document search: '{query_text}' in {document_type}")
        
        # Placeholder - requires embeddings to be pre-computed
        return []
    
    # ============================================
    # Health Monitoring
    # ============================================
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all databases.
        
        Returns:
            Dictionary of {db_name: is_healthy}
        """
        health = {
            'postgresql': get_db().health_check(),
            'redis': self.cache.health_check() if self.cache else False,
            'vector_db': self.vector.vector_store.health_check() if self.vector else False,
        }
        
        logger.info(f"Database health: {health}")
        
        return health
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all databases.
        
        Returns:
            Dictionary with stats from each database
        """
        stats = {}
        
        # PostgreSQL stats
        try:
            stats['postgresql'] = {
                'price_records': self.session.query(PriceData).count(),
                'features': self.session.query(FeatureData).count(),
                'companies': self.session.query(CompanyFundamental).count(),
            }
        except Exception as e:
            stats['postgresql'] = {'error': str(e)}
        
        # Redis stats
        if self.cache:
            try:
                info = self.cache.info()
                stats['redis'] = {
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'total_keys': self.cache.client.dbsize(),
                }
            except Exception as e:
                stats['redis'] = {'error': str(e)}
        
        # Vector DB stats
        if self.vector:
            try:
                collections = self.vector.vector_store.list_collections()
                stats['vector_db'] = {
                    'collections': len(collections),
                    'collection_names': collections,
                }
            except Exception as e:
                stats['vector_db'] = {'error': str(e)}
        
        return stats


# Export
__all__ = [
    "MultiDatabaseCoordinator",
]