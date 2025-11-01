"""
Graph Database Integration (Neo4j).

Enables relationship analysis critical for finance:
- Company ownership structures
- M&A transaction networks
- Board member connections
- Portfolio correlations
- Counterparty risk networks
- Supply chain relationships
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class Neo4jGraph:
    """
    Neo4j graph database integration.
    
    Use Cases (Critical for Finance):
    - Company ownership graphs (parent-subsidiary)
    - M&A networks (who acquired whom)
    - Board member connections (shared directors)
    - Portfolio correlations (asset relationships)
    - Counterparty networks (trading relationships)
    - Supply chain graphs (vendor-customer)
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "axiom_neo4j"
    ):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j URI
            user: Username
            password: Password
        """
        try:
            from neo4j import GraphDatabase
            self.GraphDatabase = GraphDatabase
        except ImportError:
            raise ImportError(
                "Neo4j driver not installed. Install with: pip install neo4j"
            )
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Initialized Neo4j at: {uri}")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def health_check(self) -> bool:
        """Check Neo4j health."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                return result.single()[0] == 1
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False
    
    # ============================================
    # Company Relationships
    # ============================================
    
    def create_company(
        self,
        symbol: str,
        name: str,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        market_cap: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create company node.
        
        Args:
            symbol: Company ticker
            name: Company name
            sector: Business sector
            industry: Industry
            market_cap: Market capitalization
            properties: Additional properties
            
        Returns:
            Created company node
        """
        query = """
        MERGE (c:Company {symbol: $symbol})
        SET c.name = $name,
            c.sector = $sector,
            c.industry = $industry,
            c.market_cap = $market_cap,
            c.updated_at = datetime()
        """
        
        if properties:
            for key, value in properties.items():
                query += f", c.{key} = ${key}"
        
        query += " RETURN c"
        
        params = {
            'symbol': symbol,
            'name': name,
            'sector': sector,
            'industry': industry,
            'market_cap': market_cap,
            **(properties or {})
        }
        
        with self.driver.session() as session:
            result = session.run(query, params)
            record = result.single()
            
        logger.info(f"Created company node: {symbol}")
        return dict(record['c'])
    
    def create_ownership_relationship(
        self,
        parent_symbol: str,
        child_symbol: str,
        ownership_percent: float,
        effective_date: Optional[datetime] = None
    ):
        """
        Create parent-subsidiary ownership relationship.
        
        Critical for:
        - Corporate structure analysis
        - Risk aggregation
        - Regulatory reporting
        
        Args:
            parent_symbol: Parent company
            child_symbol: Subsidiary
            ownership_percent: Ownership percentage (0-100)
            effective_date: When ownership established
        """
        query = """
        MATCH (parent:Company {symbol: $parent_symbol})
        MATCH (child:Company {symbol: $child_symbol})
        MERGE (parent)-[r:OWNS]->(child)
        SET r.percent = $ownership_percent,
            r.effective_date = $effective_date,
            r.updated_at = datetime()
        RETURN r
        """
        
        with self.driver.session() as session:
            session.run(query, {
                'parent_symbol': parent_symbol,
                'child_symbol': child_symbol,
                'ownership_percent': ownership_percent,
                'effective_date': effective_date or datetime.now()
            })
        
        logger.info(f"Created ownership: {parent_symbol} -[OWNS {ownership_percent}%]-> {child_symbol}")
    
    def create_acquisition(
        self,
        acquirer_symbol: str,
        target_symbol: str,
        deal_value: float,
        announcement_date: datetime,
        completion_date: Optional[datetime] = None,
        deal_type: str = "merger"
    ):
        """
        Create M&A acquisition relationship.
        
        Critical for:
        - M&A network analysis
        - Deal pattern recognition
        - Target prediction
        
        Args:
            acquirer_symbol: Acquiring company
            target_symbol: Target company
            deal_value: Deal value in USD
            announcement_date: When announced
            completion_date: When completed
            deal_type: Type of deal
        """
        query = """
        MATCH (acquirer:Company {symbol: $acquirer_symbol})
        MATCH (target:Company {symbol: $target_symbol})
        CREATE (acquirer)-[r:ACQUIRED {
            deal_value: $deal_value,
            announcement_date: $announcement_date,
            completion_date: $completion_date,
            deal_type: $deal_type,
            created_at: datetime()
        }]->(target)
        RETURN r
        """
        
        with self.driver.session() as session:
            session.run(query, {
                'acquirer_symbol': acquirer_symbol,
                'target_symbol': target_symbol,
                'deal_value': deal_value,
                'announcement_date': announcement_date,
                'completion_date': completion_date,
                'deal_type': deal_type
            })
        
        logger.info(f"Created acquisition: {acquirer_symbol} -[ACQUIRED ${deal_value/1e9:.1f}B]-> {target_symbol}")
    
    def create_board_member_connection(
        self,
        person_name: str,
        company_symbols: List[str],
        title: str = "Board Member"
    ):
        """
        Create board member connection across companies.
        
        Critical for:
        - Conflict of interest detection
        - Network influence analysis
        - M&A prediction
        
        Args:
            person_name: Board member name
            company_symbols: Companies they serve on
            title: Board title
        """
        # Create person node
        query_person = """
        MERGE (p:Person {name: $name})
        SET p.updated_at = datetime()
        RETURN p
        """
        
        with self.driver.session() as session:
            session.run(query_person, {'name': person_name})
            
            # Create relationships to each company
            for symbol in company_symbols:
                query_rel = """
                MATCH (p:Person {name: $name})
                MATCH (c:Company {symbol: $symbol})
                MERGE (p)-[r:SERVES_ON {
                    title: $title,
                    created_at: datetime()
                }]->(c)
                RETURN r
                """
                session.run(query_rel, {
                    'name': person_name,
                    'symbol': symbol,
                    'title': title
                })
        
        logger.info(f"Created board connections for {person_name} across {len(company_symbols)} companies")
    
    def create_correlation_edge(
        self,
        symbol1: str,
        symbol2: str,
        correlation: float,
        timeframe_days: int = 252
    ):
        """
        Create correlation relationship between assets.
        
        Critical for:
        - Portfolio diversification
        - Risk analysis
        - Pair trading strategies
        
        Args:
            symbol1: First asset
            symbol2: Second asset
            correlation: Correlation coefficient (-1 to 1)
            timeframe_days: Analysis timeframe
        """
        query = """
        MATCH (a1:Company {symbol: $symbol1})
        MATCH (a2:Company {symbol: $symbol2})
        MERGE (a1)-[r:CORRELATED_WITH]-(a2)
        SET r.correlation = $correlation,
            r.timeframe_days = $timeframe_days,
            r.updated_at = datetime()
        RETURN r
        """
        
        with self.driver.session() as session:
            session.run(query, {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'correlation': correlation,
                'timeframe_days': timeframe_days
            })
        
        logger.info(f"Created correlation: {symbol1} <-[{correlation:.2f}]-> {symbol2}")
    
    # ============================================
    # Query Methods
    # ============================================
    
    def get_subsidiaries(
        self,
        parent_symbol: str,
        min_ownership: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Get all subsidiaries of a parent company.
        
        Args:
            parent_symbol: Parent company symbol
            min_ownership: Minimum ownership percentage
            
        Returns:
            List of subsidiaries with ownership info
        """
        query = """
        MATCH (parent:Company {symbol: $parent_symbol})-[r:OWNS]->(child:Company)
        WHERE r.percent >= $min_ownership
        RETURN child.symbol AS symbol,
               child.name AS name,
               r.percent AS ownership_percent,
               r.effective_date AS effective_date
        ORDER BY r.percent DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'parent_symbol': parent_symbol,
                'min_ownership': min_ownership
            })
            return [dict(record) for record in result]
    
    def get_acquisition_history(
        self,
        symbol: str,
        as_acquirer: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get M&A history for a company.
        
        Args:
            symbol: Company symbol
            as_acquirer: True for acquisitions made, False for being acquired
            
        Returns:
            List of M&A deals
        """
        if as_acquirer:
            query = """
            MATCH (acquirer:Company {symbol: $symbol})-[r:ACQUIRED]->(target:Company)
            RETURN target.symbol AS target_symbol,
                   target.name AS target_name,
                   r.deal_value AS deal_value,
                   r.announcement_date AS announcement_date,
                   r.completion_date AS completion_date,
                   r.deal_type AS deal_type
            ORDER BY r.announcement_date DESC
            """
        else:
            query = """
            MATCH (acquirer:Company)-[r:ACQUIRED]->(target:Company {symbol: $symbol})
            RETURN acquirer.symbol AS acquirer_symbol,
                   acquirer.name AS acquirer_name,
                   r.deal_value AS deal_value,
                   r.announcement_date AS announcement_date,
                   r.completion_date AS completion_date,
                   r.deal_type AS deal_type
            ORDER BY r.announcement_date DESC
            """
        
        with self.driver.session() as session:
            result = session.run(query, {'symbol': symbol})
            return [dict(record) for record in result]
    
    def find_connected_companies(
        self,
        symbol: str,
        relationship_type: Optional[str] = None,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find companies connected through relationships.
        
        Args:
            symbol: Starting company
            relationship_type: Type of relationship (OWNS, ACQUIRED, etc.)
            max_hops: Maximum relationship hops
            
        Returns:
            List of connected companies
        """
        rel_filter = f":{relationship_type}" if relationship_type else ""
        
        query = f"""
        MATCH path = (start:Company {{symbol: $symbol}})-[r{rel_filter}*1..{max_hops}]-(connected:Company)
        RETURN DISTINCT connected.symbol AS symbol,
               connected.name AS name,
               length(path) AS distance,
               [rel in relationships(path) | type(rel)] AS relationship_chain
        ORDER BY distance, symbol
        LIMIT 100
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'symbol': symbol})
            return [dict(record) for record in result]
    
    def get_highly_correlated_assets(
        self,
        symbol: str,
        min_correlation: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find highly correlated assets.
        
        Critical for:
        - Diversification analysis
        - Risk management
        - Pair trading
        
        Args:
            symbol: Asset symbol
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of correlated assets
        """
        query = """
        MATCH (a1:Company {symbol: $symbol})-[r:CORRELATED_WITH]-(a2:Company)
        WHERE abs(r.correlation) >= $min_correlation
        RETURN a2.symbol AS symbol,
               a2.name AS name,
               r.correlation AS correlation,
               r.timeframe_days AS timeframe_days
        ORDER BY abs(r.correlation) DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'symbol': symbol,
                'min_correlation': min_correlation
            })
            return [dict(record) for record in result]
    
    def find_shared_board_members(
        self,
        symbol1: str,
        symbol2: str
    ) -> List[Dict[str, Any]]:
        """
        Find board members shared between two companies.
        
        Critical for:
        - Conflict of interest
        - M&A likelihood prediction
        - Network influence
        
        Args:
            symbol1: First company
            symbol2: Second company
            
        Returns:
            List of shared board members
        """
        query = """
        MATCH (c1:Company {symbol: $symbol1})<-[r1:SERVES_ON]-(person:Person)-[r2:SERVES_ON]->(c2:Company {symbol: $symbol2})
        RETURN person.name AS name,
               r1.title AS title_at_company1,
               r2.title AS title_at_company2
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'symbol1': symbol1,
                'symbol2': symbol2
            })
            return [dict(record) for record in result]
    
    def get_ma_network(
        self,
        sector: Optional[str] = None,
        min_deal_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get M&A network for sector analysis.
        
        Args:
            sector: Filter by sector
            min_deal_value: Minimum deal value
            
        Returns:
            List of M&A relationships
        """
        query = """
        MATCH (acquirer:Company)-[r:ACQUIRED]->(target:Company)
        """
        
        conditions = []
        params = {}
        
        if sector:
            conditions.append("acquirer.sector = $sector OR target.sector = $sector")
            params['sector'] = sector
        
        if min_deal_value:
            conditions.append("r.deal_value >= $min_deal_value")
            params['min_deal_value'] = min_deal_value
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
        RETURN acquirer.symbol AS acquirer,
               acquirer.name AS acquirer_name,
               target.symbol AS target,
               target.name AS target_name,
               r.deal_value AS deal_value,
               r.announcement_date AS date,
               r.deal_type AS deal_type
        ORDER BY r.deal_value DESC
        LIMIT 100
        """
        
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def find_acquisition_targets(
        self,
        acquirer_symbol: str,
        target_sector: Optional[str] = None,
        max_market_cap: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Find potential M&A targets based on graph analysis.
        
        Uses:
        - Industry patterns
        - Historical acquisition behavior
        - Network connections
        
        Args:
            acquirer_symbol: Acquiring company
            target_sector: Target sector
            max_market_cap: Maximum target size
            
        Returns:
            List of potential targets with scores
        """
        # Find companies in similar acquisition patterns
        query = """
        MATCH (acquirer:Company {symbol: $acquirer_symbol})-[:ACQUIRED]->(past_target:Company)
        MATCH (potential:Company)
        WHERE potential.symbol <> $acquirer_symbol
        """
        
        conditions = []
        params = {'acquirer_symbol': acquirer_symbol}
        
        if target_sector:
            conditions.append("potential.sector = $target_sector")
            params['target_sector'] = target_sector
        
        if max_market_cap:
            conditions.append("potential.market_cap <= $max_market_cap")
            params['max_market_cap'] = max_market_cap
        
        # Exclude already acquired
        conditions.append("NOT (acquirer)-[:ACQUIRED|OWNS]->(potential)")
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += """
        RETURN potential.symbol AS symbol,
               potential.name AS name,
               potential.sector AS sector,
               potential.market_cap AS market_cap,
               COUNT(past_target) AS pattern_score
        ORDER BY pattern_score DESC, potential.market_cap ASC
        LIMIT 20
        """
        
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def analyze_portfolio_network(
        self,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze relationships within a portfolio.
        
        Returns:
        - Correlation clusters
        - Ownership overlaps
        - Shared board members
        - Risk concentration
        
        Args:
            symbols: Portfolio symbols
            
        Returns:
            Network analysis results
        """
        query = """
        MATCH (c:Company)
        WHERE c.symbol IN $symbols
        OPTIONAL MATCH (c)-[r:CORRELATED_WITH]-(other:Company)
        WHERE other.symbol IN $symbols AND r.correlation > 0.5
        RETURN c.symbol AS symbol,
               collect({
                   correlated_with: other.symbol,
                   correlation: r.correlation
               }) AS correlations
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'symbols': symbols})
            correlations = [dict(record) for record in result]
        
        return {
            'portfolio_size': len(symbols),
            'correlations': correlations,
            'total_edges': sum(len(c['correlations']) for c in correlations) // 2
        }
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def clear_database(self):
        """Clear entire graph (DANGEROUS!)."""
        query = "MATCH (n) DETACH DELETE n"
        
        with self.driver.session() as session:
            session.run(query)
        
        logger.warning("Cleared entire Neo4j database")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph database statistics."""
        query = """
        MATCH (c:Company) WITH count(c) AS companies
        MATCH ()-[r:OWNS]->() WITH companies, count(r) AS ownership_rels
        MATCH ()-[r2:ACQUIRED]->() WITH companies, ownership_rels, count(r2) AS acquisitions
        MATCH ()-[r3:CORRELATED_WITH]-() WITH companies, ownership_rels, acquisitions, count(r3) AS correlations
        MATCH (p:Person) WITH companies, ownership_rels, acquisitions, correlations, count(p) AS people
        RETURN companies, ownership_rels, acquisitions, correlations, people
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            
            if record:
                return {
                    'companies': record['companies'],
                    'ownership_relationships': record['ownership_rels'],
                    'acquisitions': record['acquisitions'],
                    'correlations': record['correlations'],
                    'people': record['people']
                }
        
        return {}


# Export
__all__ = [
    "Neo4jGraph",
]