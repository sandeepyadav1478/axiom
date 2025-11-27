"""
Graph Enhancement for RAG Retrieval.

Uses Neo4j to add relationship context to retrieved documents:
- Company relationships (M&A, ownership, board connections)
- Deal networks and patterns
- Industry relationships
- Enriches retrieval with graph-based insights
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ...database.graph_integration import Neo4jGraph

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """Graph-based context for a document."""
    
    # Company relationships
    related_companies: List[Dict[str, Any]] = field(default_factory=list)
    ma_history: List[Dict[str, Any]] = field(default_factory=list)
    ownership_structure: List[Dict[str, Any]] = field(default_factory=list)
    
    # Network insights
    shared_board_members: List[Dict[str, Any]] = field(default_factory=list)
    industry_connections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Deal patterns
    similar_deals: List[Dict[str, Any]] = field(default_factory=list)
    acquisition_targets: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    graph_hops: int = 0
    relationship_count: int = 0


class GraphEnhancer:
    """
    Enhance RAG retrieval with Neo4j graph context.
    
    Features:
    - Automatic company extraction from text
    - Multi-hop relationship traversal
    - Deal pattern identification
    - Network-based insights
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "axiom_neo4j",
        max_graph_hops: int = 2,
        max_relationships: int = 20
    ):
        """
        Initialize graph enhancer.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            max_graph_hops: Maximum relationship hops
            max_relationships: Max relationships to return
        """
        self.max_graph_hops = max_graph_hops
        self.max_relationships = max_relationships
        
        try:
            self.graph = Neo4jGraph(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password
            )
            
            # Test connection
            if not self.graph.health_check():
                logger.warning("Neo4j connection unhealthy")
            else:
                logger.info("Connected to Neo4j for graph enhancement")
                
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            self.graph = None
    
    def enhance_with_graph_context(
        self,
        text: str,
        companies: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphContext:
        """
        Add graph context to a document.
        
        Args:
            text: Document text
            companies: List of company symbols/names mentioned
            metadata: Additional metadata
            
        Returns:
            Graph context with relationships
        """
        if not self.graph or not companies:
            return GraphContext()
        
        context = GraphContext()
        
        try:
            # Get relationships for each company
            for company in companies[:5]:  # Limit to 5 companies
                # Get M&A history
                ma_deals = self.graph.get_acquisition_history(
                    symbol=company,
                    as_acquirer=True
                )
                context.ma_history.extend(ma_deals[:5])
                
                # Get subsidiaries
                subsidiaries = self.graph.get_subsidiaries(
                    parent_symbol=company,
                    min_ownership=50.0
                )
                context.ownership_structure.extend(subsidiaries[:5])
                
                # Get related companies
                related = self.graph.find_connected_companies(
                    symbol=company,
                    max_hops=self.max_graph_hops
                )
                context.related_companies.extend(related[:10])
            
            # Count total relationships
            context.relationship_count = (
                len(context.ma_history) +
                len(context.ownership_structure) +
                len(context.related_companies)
            )
            
            context.graph_hops = self.max_graph_hops
            
            logger.debug(f"Enhanced with {context.relationship_count} graph relationships")
            
        except Exception as e:
            logger.error(f"Failed to enhance with graph context: {e}")
        
        return context
    
    def find_deal_patterns(
        self,
        acquirer: str,
        target_sector: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find M&A deal patterns for a company.
        
        Args:
            acquirer: Acquiring company symbol
            target_sector: Target sector filter
            
        Returns:
            List of potential targets based on patterns
        """
        if not self.graph:
            return []
        
        try:
            targets = self.graph.find_acquisition_targets(
                acquirer_symbol=acquirer,
                target_sector=target_sector
            )
            
            return targets[:self.max_relationships]
            
        except Exception as e:
            logger.error(f"Failed to find deal patterns: {e}")
            return []
    
    def get_industry_network(
        self,
        sector: str,
        min_deal_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get M&A network for a sector.
        
        Args:
            sector: Industry sector
            min_deal_value: Minimum deal value filter
            
        Returns:
            List of M&A relationships in sector
        """
        if not self.graph:
            return []
        
        try:
            network = self.graph.get_ma_network(
                sector=sector,
                min_deal_value=min_deal_value
            )
            
            return network[:self.max_relationships]
            
        except Exception as e:
            logger.error(f"Failed to get industry network: {e}")
            return []
    
    def enrich_companies_context(
        self,
        companies: List[str]
    ) -> Dict[str, Any]:
        """
        Get comprehensive context for multiple companies.
        
        Args:
            companies: List of company symbols
            
        Returns:
            Dictionary with combined context
        """
        if not self.graph or not companies:
            return {}
        
        context = {
            "companies": [],
            "relationships": [],
            "ma_activity": [],
            "network_insights": {}
        }
        
        try:
            for company in companies:
                # Company info
                company_data = {
                    "symbol": company,
                    "ma_count": 0,
                    "subsidiaries": 0,
                    "connections": 0
                }
                
                # Get M&A count
                ma_deals = self.graph.get_acquisition_history(company, True)
                company_data["ma_count"] = len(ma_deals)
                context["ma_activity"].extend(ma_deals[:3])
                
                # Get subsidiaries count
                subs = self.graph.get_subsidiaries(company)
                company_data["subsidiaries"] = len(subs)
                
                # Get connections
                connected = self.graph.find_connected_companies(company, max_hops=1)
                company_data["connections"] = len(connected)
                
                context["companies"].append(company_data)
            
            # Find shared relationships
            if len(companies) >= 2:
                shared = self.graph.find_shared_board_members(
                    companies[0],
                    companies[1]
                )
                context["network_insights"]["shared_board"] = shared
            
        except Exception as e:
            logger.error(f"Failed to enrich companies context: {e}")
        
        return context
    
    def get_relationship_summary(
        self,
        company: str
    ) -> Dict[str, Any]:
        """
        Get relationship summary for a company.
        
        Args:
            company: Company symbol
            
        Returns:
            Summary of relationships
        """
        if not self.graph:
            return {}
        
        try:
            summary = {
                "company": company,
                "acquisitions_made": [],
                "subsidiaries": [],
                "related_companies": [],
                "total_relationships": 0
            }
            
            # Get acquisitions
            acquisitions = self.graph.get_acquisition_history(company, True)
            summary["acquisitions_made"] = acquisitions[:5]
            
            # Get subsidiaries
            subs = self.graph.get_subsidiaries(company)
            summary["subsidiaries"] = subs[:5]
            
            # Get related
            related = self.graph.find_connected_companies(company, max_hops=1)
            summary["related_companies"] = related[:10]
            
            # Count total
            summary["total_relationships"] = (
                len(acquisitions) + len(subs) + len(related)
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get relationship summary: {e}")
            return {}
    
    def close(self):
        """Close Neo4j connection."""
        if self.graph:
            self.graph.close()


__all__ = ["GraphEnhancer", "GraphContext"]