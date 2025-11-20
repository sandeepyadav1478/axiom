"""
Lightweight Data Ingestion Pipeline
Minimal dependencies - no heavy ML imports
Direct database connections only

This is a lightweight version specifically for containerized deployment.
Uses only essential dependencies to avoid dependency hell.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
import os

import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Numeric, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Base = declarative_base()


class PriceData(Base):
    """Lightweight price data model."""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Numeric(precision=20, scale=6))
    high = Column(Numeric(precision=20, scale=6))
    low = Column(Numeric(precision=20, scale=6))
    close = Column(Numeric(precision=20, scale=6))
    volume = Column(Numeric(precision=20, scale=2))
    source = Column(String(50))


class LightweightPipeline:
    """
    Lightweight data ingestion pipeline with minimal dependencies.
    
    Features:
    - Direct PostgreSQL connection
    - Redis caching
    - Neo4j graph storage
    - Real-time data from yfinance
    - No heavy ML dependencies
    """
    
    def __init__(self):
        """Initialize database connections."""
        
        # PostgreSQL - use environment variables
        pg_host = os.getenv('POSTGRES_HOST', 'postgres')
        pg_port = os.getenv('POSTGRES_PORT', '5432')
        pg_user = os.getenv('POSTGRES_USER', 'axiom')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'axiom_password')
        pg_db = os.getenv('POSTGRES_DB', 'axiom_finance')
        
        pg_url = f'postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}'
        self.engine = create_engine(pg_url)
        # Don't create tables - use existing schema
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Redis
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Neo4j
        try:
            neo4j_uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
            neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
            neo4j_password = os.getenv('NEO4J_PASSWORD', 'axiom_neo4j')
            self.neo4j = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            logger.info("✅ Neo4j connected")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")
            self.neo4j = None
        
        logger.info("Lightweight pipeline initialized")
    
    async def ingest_realtime_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Ingest real-time prices using yfinance.
        
        Flow:
        1. Fetch from yfinance
        2. Store in PostgreSQL
        3. Cache in Redis
        4. Update Neo4j graph
        """
        logger.info(f"Ingesting prices for {len(symbols)} symbols")
        
        metrics = {
            'symbols_processed': 0,
            'records_stored': 0,
            'records_cached': 0,
            'errors': []
        }
        
        for symbol in symbols:
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                
                if hist.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                latest = hist.iloc[-1]
                
                # Store in PostgreSQL using raw SQL to avoid schema issues
                from sqlalchemy import text
                self.session.execute(text("""
                    INSERT INTO price_data (symbol, timestamp, timeframe, open, high, low, close, volume, source)
                    VALUES (:symbol, :timestamp, 'DAY_1', :open, :high, :low, :close, :volume, :source)
                """), {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'open': float(latest['Open']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'close': float(latest['Close']),
                    'volume': int(latest['Volume']),
                    'source': 'yfinance'
                })
                self.session.commit()
                metrics['records_stored'] += 1
                
                # Cache in Redis
                if self.redis_client:
                    cache_key = f"price:{symbol}:latest"
                    cache_data = {
                        'close': float(latest['Close']),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.redis_client.hset(cache_key, mapping=cache_data)
                    self.redis_client.expire(cache_key, 60)
                    metrics['records_cached'] += 1
                
                # Update Neo4j
                if self.neo4j:
                    with self.neo4j.session() as session:
                        session.run(
                            """
                            MERGE (s:Stock {symbol: $symbol})
                            SET s.last_price = $price,
                                s.last_updated = datetime($timestamp)
                            """,
                            symbol=symbol,
                            price=float(latest['Close']),
                            timestamp=datetime.now().isoformat()
                        )
                
                metrics['symbols_processed'] += 1
                logger.info(f"✅ {symbol}: ${latest['Close']:.2f}")
                
            except Exception as e:
                logger.error(f"Error ingesting {symbol}: {e}")
                metrics['errors'].append(f"{symbol}: {str(e)}")
        
        return metrics
    
    async def run_continuous_mode(self):
        """
        Run pipeline in continuous mode.
        
        Reads configuration from environment:
        - SYMBOLS: Comma-separated list
        - PIPELINE_INTERVAL: Update interval in seconds
        """
        symbols_str = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOGL')
        symbols = [s.strip() for s in symbols_str.split(',')]
        interval = int(os.getenv('PIPELINE_INTERVAL', '60'))
        
        logger.info(f"Starting continuous mode: {symbols} every {interval}s")
        
        while True:
            try:
                metrics = await self.ingest_realtime_prices(symbols)
                logger.info(f"Cycle complete: {metrics['symbols_processed']}/{len(symbols)} processed")
                
                if metrics['errors']:
                    logger.warning(f"Errors: {metrics['errors']}")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping pipeline")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(interval)
    
    def cleanup(self):
        """Cleanup connections."""
        if self.session:
            self.session.close()
        if self.neo4j:
            self.neo4j.close()
        logger.info("Pipeline cleanup complete")


if __name__ == "__main__":
    async def main():
        pipeline = LightweightPipeline()
        try:
            await pipeline.run_continuous_mode()
        finally:
            pipeline.cleanup()
    
    asyncio.run(main())