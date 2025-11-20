"""
Neo4j Graph Client for Pipeline Data Integration
Handles all graph database operations for pipelines
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)


class Neo4jGraphClient:
    """
    Unified Neo4j client for all pipelines.
    
    Features:
    - Connection management
    - Schema validation
    - Common graph operations
    - Query helpers
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """Initialize Neo4j connection."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'axiom_neo4j')
        
        self.driver: Driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        
        logger.info(f"✅ Neo4j connected: {self.uri}")
        
        # Initialize schema
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Create indexes and constraints for optimal performance."""
        with self.driver.session() as session:
            # Stock indexes
            session.run("""
                CREATE INDEX stock_symbol IF NOT EXISTS
                FOR (s:Stock) ON (s.symbol)
            """)
            
            # Company indexes
            session.run("""
                CREATE INDEX company_symbol IF NOT EXISTS
                FOR (c:Company) ON (c.symbol)
            """)
            
            # Sector indexes
            session.run("""
                CREATE INDEX sector_name IF NOT EXISTS
                FOR (s:Sector) ON (s.name)
            """)
            
            # Event indexes
            session.run("""
                CREATE INDEX event_date IF NOT EXISTS
                FOR (e:MarketEvent) ON (e.date)
            """)
            
            logger.info("✅ Neo4j indexes created")
    
    def create_stock_node(
        self,
        symbol: str,
        name: str,
        last_price: float,
        **additional_props
    ) -> Dict[str, Any]:
        """Create or update a Stock node."""
        with self.driver.session() as session:
            result = session.run("""
                MERGE (s:Stock {symbol: $symbol})
                SET s.name = $name,
                    s.last_price = $last_price,
                    s.last_updated = datetime(),
                    s += $props
                RETURN s
            """, 
                symbol=symbol,
                name=name,
                last_price=last_price,
                props=additional_props
            )
            
            record = result.single()
            return dict(record['s']) if record else {}
    
    def create_company_node(
        self,
        symbol: str,
        name: str,
        sector: str,
        industry: str,
        **fundamentals
    ) -> Dict[str, Any]:
        """Create or update a Company node."""
        with self.driver.session() as session:
            result = session.run("""
                MERGE (c:Company {symbol: $symbol})
                SET c.name = $name,
                    c.sector = $sector,
                    c.industry = $industry,
                    c.last_updated = datetime(),
                    c += $fundamentals
                RETURN c
            """,
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry,
                fundamentals=fundamentals
            )
            
            record = result.single()
            return dict(record['c']) if record else {}
    
    def link_company_to_sector(self, symbol: str, sector_name: str):
        """Create BELONGS_TO relationship."""
        with self.driver.session() as session:
            session.run("""
                MATCH (c:Company {symbol: $symbol})
                MERGE (s:Sector {name: $sector})
                MERGE (c)-[:BELONGS_TO]->(s)
            """, symbol=symbol, sector=sector_name)
            
            logger.debug(f"{symbol} → BELONGS_TO → {sector_name}")
    
    def create_competitor_relationship(
        self,
        symbol1: str,
        symbol2: str,
        intensity: float = 0.5
    ):
        """Create COMPETES_WITH relationship."""
        with self.driver.session() as session:
            session.run("""
                MATCH (c1:Company {symbol: $symbol1})
                MATCH (c2:Company {symbol: $symbol2})
                MERGE (c1)-[r:COMPETES_WITH]-(c2)
                SET r.intensity = $intensity,
                    r.last_updated = datetime()
            """,
                symbol1=symbol1,
                symbol2=symbol2,
                intensity=intensity
            )
            
            logger.debug(f"{symbol1} COMPETES_WITH {symbol2} (intensity: {intensity})")
    
    def create_correlation_relationship(
        self,
        symbol1: str,
        symbol2: str,
        coefficient: float,
        period_days: int = 30
    ):
        """Create CORRELATED_WITH relationship."""
        with self.driver.session() as session:
            session.run("""
                MATCH (s1:Stock {symbol: $symbol1})
                MATCH (s2:Stock {symbol: $symbol2})
                MERGE (s1)-[r:CORRELATED_WITH]-(s2)
                SET r.coefficient = $coefficient,
                    r.period_days = $period_days,
                    r.calculated_at = datetime()
            """,
                symbol1=symbol1,
                symbol2=symbol2,
                coefficient=coefficient,
                period_days=period_days
            )
    
    def create_market_event(
        self,
        event_type: str,
        date: datetime,
        description: str,
        **additional_props
    ) -> Dict[str, Any]:
        """Create MarketEvent node."""
        with self.driver.session() as session:
            result = session.run("""
                CREATE (e:MarketEvent {
                    type: $type,
                    date: date($date),
                    description: $description
                })
                SET e += $props
                RETURN e
            """,
                type=event_type,
                date=date.isoformat(),
                description=description,
                props=additional_props
            )
            
            record = result.single()
            return dict(record['e']) if record else {}
    
    def link_event_to_company(
        self,
        event_id: int,
        symbol: str,
        impact_score: float = 0.5
    ):
        """Link MarketEvent to Company."""
        with self.driver.session() as session:
            session.run("""
                MATCH (e:MarketEvent)
                WHERE id(e) = $event_id
                MATCH (c:Company {symbol: $symbol})
                MERGE (c)-[r:AFFECTED_BY]->(e)
                SET r.impact_score = $impact_score
            """,
                event_id=event_id,
                symbol=symbol,
                impact_score=impact_score
            )
    
    def execute_cypher(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute arbitrary Cypher query."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self.driver.session() as session:
            # Count nodes by type
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as node_type, count(n) as count
                ORDER BY count DESC
            """)
            
            # Count relationships by type
            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)
            
            return {
                'nodes': [dict(r) for r in node_counts],
                'relationships': [dict(r) for r in rel_counts],
                'total_nodes': sum(r['count'] for r in node_counts),
                'total_relationships': sum(r['count'] for r in rel_counts)
            }
    
    def close(self):
        """Close driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


# Singleton instance
_client: Optional[Neo4jGraphClient] = None


def get_neo4j_client() -> Neo4jGraphClient:
    """Get or create singleton Neo4j client."""
    global _client
    
    if _client is None:
        _client = Neo4jGraphClient()
    
    return _client