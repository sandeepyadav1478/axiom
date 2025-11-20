"""
Company Graph Builder Pipeline
Uses LangGraph + Claude to build intelligent company relationship graphs

This pipeline demonstrates:
- LangGraph workflow orchestration
- Claude-powered relationship extraction
- Neo4j graph building
- Multi-agent collaboration
"""

import asyncio
import logging
from typing import Dict, List, Any, TypedDict, Optional
import os
import json

import yfinance as yf
from langgraph.graph import StateGraph, END

# Import shared utilities (will be available in container)
import sys
sys.path.insert(0, '/app')
from axiom.pipelines.shared.neo4j_client import get_neo4j_client
from axiom.pipelines.shared.langgraph_base import BaseLangGraphPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompanyGraphState(TypedDict):
    """State for company graph building workflow."""
    symbol: str
    company_data: Dict[str, Any]
    competitors: List[str]
    sector_companies: List[str]
    cypher_queries: List[str]
    neo4j_updates: int
    success: bool
    error: Optional[str]


class CompanyGraphBuilderPipeline(BaseLangGraphPipeline):
    """
    LangGraph-powered pipeline for building company knowledge graphs.
    
    Workflow:
    1. Fetch company fundamentals (yfinance)
    2. Extract competitors using Claude AI
    3. Identify sector peers using Claude AI
    4. Generate Cypher queries using Claude AI
    5. Execute graph updates in Neo4j
    6. Validate graph consistency
    """
    
    def __init__(self):
        """Initialize Company Graph Builder."""
        super().__init__("CompanyGraphBuilder")
        
        # Get Neo4j client
        self.neo4j = get_neo4j_client()
    
    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        workflow = StateGraph(CompanyGraphState)
        
        # Define agent nodes
        workflow.add_node("fetch_data", self.fetch_company_data)
        workflow.add_node("extract_competitors", self.extract_competitors_with_claude)
        workflow.add_node("identify_sector_peers", self.identify_sector_peers_with_claude)
        workflow.add_node("generate_cypher", self.generate_cypher_with_claude)
        workflow.add_node("execute_neo4j", self.execute_neo4j_updates)
        workflow.add_node("validate", self.validate_graph)
        
        # Define workflow edges
        workflow.add_edge("fetch_data", "extract_competitors")
        workflow.add_edge("extract_competitors", "identify_sector_peers")
        workflow.add_edge("identify_sector_peers", "generate_cypher")
        workflow.add_edge("generate_cypher", "execute_neo4j")
        workflow.add_edge("execute_neo4j", "validate")
        workflow.add_edge("validate", END)
        
        # Set entry point
        workflow.set_entry_point("fetch_data")
        
        return workflow
    
    def fetch_company_data(self, state: CompanyGraphState) -> CompanyGraphState:
        """Agent 1: Fetch company fundamentals from yfinance."""
        symbol = state['symbol']
        logger.info(f"Fetching data for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            state['company_data'] = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'description': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'country': info.get('country', ''),
            }
            
            logger.info(f"✅ Fetched: {state['company_data']['name']}")
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            state['error'] = str(e)
            state['success'] = False
        
        return state
    
    def extract_competitors_with_claude(self, state: CompanyGraphState) -> CompanyGraphState:
        """Agent 2: Use Claude to identify direct competitors."""
        company = state['company_data']
        
        prompt = f"""Analyze this company and identify its top 5 direct competitors:

Company: {company['name']}
Symbol: {company['symbol']}
Sector: {company['sector']}
Industry: {company['industry']}
Description: {company['description'][:500]}

Requirements:
- Return ONLY stock ticker symbols (e.g., AAPL, MSFT, GOOGL)
- Separate with commas
- No explanations or additional text
- Focus on direct market competitors
- If a company has no clear competitors, return empty

Example output: MSFT,GOOGL,META,AMZN"""
        
        try:
            response = self.invoke_claude(
                prompt,
                system="You are a financial analyst expert at identifying market competitors."
            )
            
            # Parse competitors from Claude's response
            competitors_text = response.strip()
            if competitors_text and competitors_text != 'empty':
                competitors = [c.strip() for c in competitors_text.split(',') if c.strip()]
                state['competitors'] = [c for c in competitors if c != company['symbol']]
            else:
                state['competitors'] = []
            
            logger.info(f"✅ Claude identified competitors: {state['competitors']}")
            
        except Exception as e:
            logger.error(f"Error extracting competitors: {e}")
            state['competitors'] = []
        
        return state
    
    def identify_sector_peers_with_claude(self, state: CompanyGraphState) -> CompanyGraphState:
        """Agent 3: Use Claude to identify other companies in same sector."""
        company = state['company_data']
        
        prompt = f"""Identify 5-10 other major companies in the same sector:

Sector: {company['sector']}
Industry: {company['industry']}

Requirements:
- Return ONLY stock ticker symbols
- Separate with commas
- Focus on largest market cap companies in this sector
- Exclude {company['symbol']} and {','.join(state['competitors'])}

Example output: NVDA,AMD,INTC,QCOM,TXN"""
        
        try:
            response = self.invoke_claude(
                prompt,
                system="You are a financial analyst expert at sector analysis."
            )
            
            peers = [c.strip() for c in response.strip().split(',') if c.strip()]
            state['sector_companies'] = peers
            
            logger.info(f"✅ Claude identified sector peers: {state['sector_companies']}")
            
        except Exception as e:
            logger.error(f"Error identifying sector peers: {e}")
            state['sector_companies'] = []
        
        return state
    
    def generate_cypher_with_claude(self, state: CompanyGraphState) -> CompanyGraphState:
        """Agent 4: Use Claude to generate Neo4j Cypher queries."""
        company = state['company_data']
        competitors = state['competitors']
        sector_companies = state['sector_companies']
        
        prompt = f"""Generate Neo4j Cypher queries to build a company knowledge graph.

Company Data:
{json.dumps(company, indent=2)}

Competitors: {competitors}
Sector Peers: {sector_companies}

Generate Cypher queries to:
1. CREATE/MERGE Company node with all properties
2. CREATE/MERGE Sector node
3. CREATE relationship: Company BELONGS_TO Sector
4. CREATE relationships: Company COMPETES_WITH each competitor (with intensity based on how direct the competition is)
5. CREATE relationships: Company SAME_SECTOR_AS each sector peer

Requirements:
- Use MERGE for nodes to avoid duplicates
- Use proper data types (integers for numbers, strings for text)
- Set intensity values between 0.0-1.0 for COMPETES_WITH
- Add timestamps using datetime()
- Return ONLY valid Cypher queries, one per line
- No explanations or comments

Example format:
MERGE (c:Company {{symbol: 'AAPL', name: 'Apple Inc.'}})
SET c.sector = 'Technology', c.market_cap = 2500000000000
MERGE (s:Sector {{name: 'Technology'}})
MERGE (c)-[:BELONGS_TO]->(s)"""
        
        try:
            response = self.invoke_claude(
                prompt,
                system="You are a Neo4j Cypher query expert. Generate only valid Cypher code."
            )
            
            # Parse Cypher queries (one per line, ignore empty lines)
            queries = [
                line.strip() 
                for line in response.strip().split('\n') 
                if line.strip() and not line.strip().startswith('#')
            ]
            
            state['cypher_queries'] = queries
            logger.info(f"✅ Claude generated {len(queries)} Cypher queries")
            
        except Exception as e:
            logger.error(f"Error generating Cypher: {e}")
            state['cypher_queries'] = []
            state['error'] = str(e)
        
        return state
    
    def execute_neo4j_updates(self, state: CompanyGraphState) -> CompanyGraphState:
        """Agent 5: Execute Cypher queries in Neo4j."""
        queries = state['cypher_queries']
        updates = 0
        
        for query in queries:
            try:
                # Execute query
                result = self.neo4j.execute_cypher(query)
                updates += 1
                logger.debug(f"Executed: {query[:80]}...")
                
            except Exception as e:
                logger.error(f"Query failed: {query[:80]}... Error: {e}")
        
        state['neo4j_updates'] = updates
        logger.info(f"✅ Executed {updates}/{len(queries)} Neo4j updates")
        
        return state
    
    def validate_graph(self, state: CompanyGraphState) -> CompanyGraphState:
        """Agent 6: Validate graph was updated correctly."""
        symbol = state['symbol']
        
        try:
            # Check if company node exists
            result = self.neo4j.execute_cypher("""
                MATCH (c:Company {symbol: $symbol})
                OPTIONAL MATCH (c)-[r]->(other)
                RETURN c.name as name, 
                       count(r) as relationship_count,
                       collect(type(r)) as relationship_types
            """, {'symbol': symbol})
            
            if result:
                data = result[0]
                logger.info(f"✅ Validation: {symbol} has {data['relationship_count']} relationships")
                logger.info(f"   Types: {set(data['relationship_types'])}")
                state['success'] = True
            else:
                logger.warning(f"⚠️ Company node not found for {symbol}")
                state['success'] = False
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            state['success'] = False
            state['error'] = str(e)
        
        return state
    
    async def process_item(self, item: str) -> Dict[str, Any]:
        """Process a single company symbol through the LangGraph workflow."""
        
        # Initialize state
        initial_state: CompanyGraphState = {
            'symbol': item,
            'company_data': {},
            'competitors': [],
            'sector_companies': [],
            'cypher_queries': [],
            'neo4j_updates': 0,
            'success': False,
            'error': None
        }
        
        # Run through LangGraph workflow
        final_state = self.app.invoke(initial_state)
        
        return {
            'symbol': item,
            'success': final_state['success'],
            'updates': final_state['neo4j_updates'],
            'error': final_state.get('error')
        }


async def main():
    """Main entry point for Company Graph Builder."""
    
    # Get symbols from environment
    symbols_str = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOGL')
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    logger.info("="*70)
    logger.info("COMPANY GRAPH BUILDER PIPELINE")
    logger.info("LangGraph + Claude + Neo4j")
    logger.info("="*70)
    
    # Initialize pipeline
    pipeline = CompanyGraphBuilderPipeline()
    
    # Run continuous mode
    await pipeline.run_continuous(symbols)


if __name__ == "__main__":
    asyncio.run(main())