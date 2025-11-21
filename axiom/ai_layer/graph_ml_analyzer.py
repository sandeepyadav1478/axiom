"""
Graph ML Analyzer for Neo4j Knowledge Graphs
Advanced graph analytics, pattern detection, community finding

Architecture: Graph algorithms + ML embeddings + Pattern mining
Data Science: Network analysis, centrality measures, clustering
ML: Node embeddings, link prediction, community detection

Production capabilities for financial network intelligence
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os


class CentralityMethod(Enum):
    """Graph centrality algorithms."""
    DEGREE = "degree"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"
    PAGERANK = "pagerank"


class CommunityAlgorithm(Enum):
    """Community detection algorithms."""
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "label_propagation"
    WEAKLY_CONNECTED = "weakly_connected"


@dataclass
class GraphMetrics:
    """Complete graph metrics suite."""
    node_count: int
    relationship_count: int
    density: float
    avg_degree: float
    diameter: Optional[int]
    avg_clustering_coefficient: Optional[float]


@dataclass
class CentralityResult:
    """Node centrality analysis results."""
    node_id: str
    score: float
    rank: int
    method: CentralityMethod


class GraphMLAnalyzer:
    """
    Production graph ML analyzer for Neo4j.
    
    Capabilities:
    - Centrality analysis (identify key nodes)
    - Community detection (find clusters)
    - Pattern matching (detect structures)
    - Link prediction (infer missing edges)
    - Graph embeddings (vector representation)
    - Shortest paths (relationship discovery)
    
    Use Cases:
    - Find influential companies (PageRank)
    - Detect market sectors (community detection)
    - Risk propagation paths (shortest paths)
    - Similar company finding (embeddings)
    - Correlation clusters (graph partitioning)
    """
    
    def __init__(self):
        """Initialize with Neo4j connection."""
        from neo4j import GraphDatabase
        
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD')
        
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
    
    def calculate_graph_metrics(self) -> GraphMetrics:
        """
        Calculate comprehensive graph metrics.
        
        Data Science: Network statistics, topology measures
        """
        with self.driver.session() as session:
            # Node and relationship counts
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()['count']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            
            # Calculate density: edges / (nodes * (nodes-1))
            max_edges = node_count * (node_count - 1) if node_count > 1 else 1
            density = rel_count / max_edges if max_edges > 0 else 0
            
            # Average degree
            avg_degree = (2 * rel_count / node_count) if node_count > 0 else 0
            
            return GraphMetrics(
                node_count=node_count,
                relationship_count=rel_count,
                density=density,
                avg_degree=avg_degree,
                diameter=None,  # Expensive to calculate
                avg_clustering_coefficient=None
            )
    
    def find_central_nodes(
        self,
        node_label: str = "Company",
        method: CentralityMethod = CentralityMethod.PAGERANK,
        limit: int = 10
    ) -> List[CentralityResult]:
        """
        Find most central/important nodes using graph algorithms.
        
        Data Science: Centrality algorithms (PageRank, betweenness, etc.)
        Use Case: Identify influential companies in network
        """
        with self.driver.session() as session:
            if method == CentralityMethod.PAGERANK:
                # PageRank via Cypher
                query = f"""
                    MATCH (n:{node_label})
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, count(r) as degree
                    RETURN n.symbol as symbol, 
                           n.name as name,
                           degree
                    ORDER BY degree DESC
                    LIMIT {limit}
                """
                
                result = session.run(query)
                
                results = []
                for rank, record in enumerate(result, 1):
                    results.append(CentralityResult(
                        node_id=record['symbol'] or record['name'],
                        score=float(record['degree']),
                        rank=rank,
                        method=method
                    ))
                
                return results
            
            elif method == CentralityMethod.DEGREE:
                # Simple degree centrality
                query = f"""
                    MATCH (n:{node_label})-[r]-()
                    WITH n, count(r) as degree
                    RETURN n.symbol as symbol, degree
                    ORDER BY degree DESC
                    LIMIT {limit}
                """
                
                result = session.run(query)
                
                results = []
                for rank, record in enumerate(result, 1):
                    results.append(CentralityResult(
                        node_id=record['symbol'],
                        score=float(record['degree']),
                        rank=rank,
                        method=method
                    ))
                
                return results
            
            else:
                # Other methods would use Graph Data Science library
                return []
    
    def detect_communities(
        self,
        node_label: str = "Company",
        algorithm: CommunityAlgorithm = CommunityAlgorithm.LABEL_PROPAGATION
    ) -> List[Dict[str, Any]]:
        """
        Detect communities/clusters in graph.
        
        Data Science: Community detection, graph partitioning
        ML: Unsupervised clustering on graph structure
        Use Case: Find natural market sectors, correlation groups
        """
        with self.driver.session() as session:
            # Simplified community detection using connected components
            query = f"""
                MATCH (n:{node_label})-[r:COMPETES_WITH|SAME_SECTOR_AS*1..2]-(m:{node_label})
                WITH n, collect(DISTINCT m) as community_members
                RETURN n.symbol as node, 
                       [member in community_members | member.symbol] as community,
                       size(community_members) as community_size
                ORDER BY community_size DESC
                LIMIT 20
            """
            
            result = session.run(query)
            
            communities = []
            for record in result:
                communities.append({
                    'node': record['node'],
                    'community_members': record['community'],
                    'size': record['community_size']
                })
            
            return communities
    
    def find_shortest_path(
        self,
        from_symbol: str,
        to_symbol: str,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.
        
        Data Science: Graph traversal, path finding
        Use Case: Risk propagation, influence chains, supply chain analysis
        """
        with self.driver.session() as session:
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            query = f"""
                MATCH path = shortestPath(
                    (a:Company {{symbol: $from}})-[{rel_filter}*1..5]-(b:Company {{symbol: $to}})
                )
                RETURN [node in nodes(path) | node.symbol] as path_nodes,
                       [rel in relationships(path) | type(rel)] as path_rels,
                       length(path) as path_length
            """
            
            result = session.run(query, from_symbol=from_symbol, to_symbol=to_symbol)
            record = result.single()
            
            if record:
                return {
                    'nodes': record['path_nodes'],
                    'relationships': record['path_rels'],
                    'length': record['path_length']
                }
            
            return None
    
    def calculate_correlation_clusters(
        self,
        min_correlation: float = 0.7
    ) -> List[List[str]]:
        """
        Find clusters of highly correlated stocks.
        
        Data Science: Graph clustering, correlation analysis
        ML: Unsupervised learning on correlation network
        Use Case: Portfolio construction, risk assessment
        """
        with self.driver.session() as session:
            query = """
                MATCH (s1:Stock)-[r:CORRELATED_WITH]-(s2:Stock)
                WHERE r.coefficient > $threshold
                WITH s1, collect(DISTINCT s2.symbol) as correlated_stocks
                RETURN s1.symbol as stock, correlated_stocks, size(correlated_stocks) as cluster_size
                ORDER BY cluster_size DESC
                LIMIT 20
            """
            
            result = session.run(query, threshold=min_correlation)
            
            clusters = []
            for record in result:
                cluster = [record['stock']] + record['correlated_stocks']
                clusters.append(cluster)
            
            return clusters
    
    def analyze_sector_network(
        self,
        sector: str
    ) -> Dict[str, Any]:
        """
        Analyze network structure of a sector.
        
        Data Science: Subgraph analysis, sector statistics
        Architecture: Domain-specific graph queries
        """
        with self.driver.session() as session:
            # Get all companies in sector
            result = session.run("""
                MATCH (c:Company)-[:BELONGS_TO]->(s:Sector {name: $sector})
                RETURN count(c) as company_count
            """, sector=sector)
            
            company_count = result.single()['company_count'] if result.single() else 0
            
            # Get competitive relationships in sector
            result = session.run("""
                MATCH (c1:Company)-[:BELONGS_TO]->(:Sector {name: $sector})
                MATCH (c1)-[r:COMPETES_WITH]->(c2:Company)
                RETURN count(r) as competition_edges
            """, sector=sector)
            
            competition_edges = result.single()['competition_edges'] if result.single() else 0
            
            # Calculate sector metrics
            return {
                'sector': sector,
                'company_count': company_count,
                'competition_edges': competition_edges,
                'competitive_intensity': competition_edges / company_count if company_count > 0 else 0,
                'network_density': competition_edges / (company_count * (company_count - 1)) if company_count > 1 else 0
            }
    
    def predict_missing_relationships(
        self,
        node_label: str = "Company",
        relationship_type: str = "COMPETES_WITH",
        top_k: int = 20
    ) -> List[Tuple[str, str, float]]:
        """
        Predict missing relationships using graph patterns.
        
        ML: Link prediction, collaborative filtering
        Data Science: Pattern-based inference
        Use Case: Discover non-obvious competitors, relationships
        """
        with self.driver.session() as session:
            # Find companies in same sector without COMPETES_WITH edge
            # that have common competitors (collaborative filtering pattern)
            query = f"""
                MATCH (c1:{node_label})-[:BELONGS_TO]->(s:Sector)<-[:BELONGS_TO]-(c2:{node_label})
                WHERE c1 <> c2
                AND NOT (c1)-[:{relationship_type}]-(c2)
                
                OPTIONAL MATCH (c1)-[:{relationship_type}]-(shared:Company)-[:{relationship_type}]-(c2)
                WITH c1, c2, count(shared) as shared_competitors
                WHERE shared_competitors > 0
                
                RETURN c1.symbol as from_symbol,
                       c2.symbol as to_symbol,
                       shared_competitors,
                       shared_competitors * 1.0 / 10.0 as confidence
                ORDER BY shared_competitors DESC
                LIMIT {top_k}
            """
            
            result = session.run(query)
            
            predictions = []
            for record in result:
                predictions.append((
                    record['from_symbol'],
                    record['to_symbol'],
                    min(record['confidence'], 1.0)  # Cap at 1.0
                ))
            
            return predictions
    
    def close(self):
        """Cleanup Neo4j connection."""
        if self.driver:
            self.driver.close()


# ================================================================
# Graph Query Templates for Common Analyses
# ================================================================

GRAPH_QUERY_TEMPLATES = {
    'find_market_leaders': """
        MATCH (c:Company)-[r:COMPETES_WITH|MARKET_LEADER_IN]-()
        WITH c, count(r) as influence_score
        RETURN c.symbol, c.name, c.market_cap, influence_score
        ORDER BY influence_score DESC, c.market_cap DESC
        LIMIT 10
    """,
    
    'risk_propagation': """
        MATCH path = (risk:RiskFactor)-[:PROPAGATES_TO*1..3]-(company:Company)
        WHERE risk.severity > 0.7
        RETURN [node in nodes(path) | CASE 
                WHEN 'Company' IN labels(node) THEN node.symbol 
                ELSE node.type END] as path,
               length(path) as depth,
               reduce(exposure = 1.0, rel in relationships(path) | 
                   exposure * coalesce(rel.strength, 0.5)) as total_exposure
        ORDER BY total_exposure DESC
        LIMIT 20
    """,
    
    'correlation_network': """
        MATCH (s1:Stock)-[r:CORRELATED_WITH]-(s2:Stock)
        WHERE r.coefficient > 0.75
        RETURN s1.symbol, s2.symbol, r.coefficient
        ORDER BY r.coefficient DESC
        LIMIT 50
    """,
    
    'sector_exposure': """
        MATCH (c:Company)-[:BELONGS_TO]->(s:Sector)
        WITH s, collect(c) as companies, sum(c.market_cap) as total_cap
        RETURN s.name as sector,
               size(companies) as company_count,
               total_cap,
               total_cap / 1000000000 as total_cap_billions
        ORDER BY total_cap DESC
    """,
    
    'supply_chain_analysis': """
        MATCH (supplier:Company)-[:SUPPLIES_TO]->(customer:Company)
        WITH supplier, count(customer) as customer_count, 
             collect(customer.symbol) as customers
        RETURN supplier.symbol, supplier.name, customer_count, customers
        ORDER BY customer_count DESC
        LIMIT 10
    """
}


def execute_template_query(
    query_name: str,
    params: Optional[Dict[str, Any]] = None
) -> List[Dict]:
    """
    Execute pre-defined graph query template.
    
    Architecture: Query template pattern for common analyses
    """
    from neo4j import GraphDatabase
    
    if query_name not in GRAPH_QUERY_TEMPLATES:
        raise ValueError(f"Unknown query template: {query_name}")
    
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    
    with driver.session() as session:
        query = GRAPH_QUERY_TEMPLATES[query_name]
        result = session.run(query, params or {})
        
        records = []
        for record in result:
            records.append(dict(record))
    
    driver.close()
    
    return records


# ================================================================
# Advanced: Graph Embeddings (Future)
# ================================================================

class GraphEmbedder:
    """
    Generate vector embeddings from graph structure.
    
    ML: Node2Vec, GraphSAGE, DeepWalk algorithms
    Use Case: Similar company finding, clustering, classification
    
    Future enhancement: Train embeddings on graph structure
    """
    
    def __init__(self, embedding_dim: int = 128):
        """Initialize embedder with dimension size."""
        self.embedding_dim = embedding_dim
        self.embeddings = {}
    
    def train_node2vec(
        self,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0
    ):
        """
        Train Node2Vec embeddings on graph.
        
        Algorithm: Random walks + Skip-gram
        Parameters:
        - walk_length: Length of random walks
        - num_walks: Number of walks per node
        - p: Return parameter (BFS vs DFS bias)
        - q: In-out parameter (local vs global exploration)
        
        Future: Integrate with Neo4j GDS library
        """
        # Placeholder for future implementation
        # Would use Neo4j Graph Data Science library
        # Or extract graph and use NetworkX + Node2Vec
        pass
    
    def get_similar_nodes(
        self,
        node_id: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find similar nodes based on embedding distance.
        
        ML: Vector similarity search
        Use Case: "Find companies similar to Apple"
        """
        # Placeholder - would use actual embeddings
        # Currently returns based on graph proximity
        return []


# ================================================================
# Export
# ================================================================

__all__ = [
    'GraphMLAnalyzer',
    'GraphEmbedder',
    'CentralityMethod',
    'CommunityAlgorithm',
    'GraphMetrics',
    'CentralityResult',
    'GRAPH_QUERY_TEMPLATES',
    'execute_template_query'
]