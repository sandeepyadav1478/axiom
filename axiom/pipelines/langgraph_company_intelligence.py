"""
LangGraph Company Intelligence Pipeline
Intelligent company profiling with Claude AI

Purpose: Expand from 3 to 50 companies with AI-powered enrichment
Architecture: Multi-agent LangGraph workflow (not Airflow)
Advantages: No worker timeouts, parallel processing, adaptive intelligence

Workflow: Fetch → Profile → Compete → Products → Validate → Store
"""

import asyncio
import logging
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
import os
import json
import sys

from dotenv import load_dotenv
load_dotenv()

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import yfinance as yf
from neo4j import GraphDatabase
import psycopg2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


# ================================================================
# State Definition
# ================================================================

class CompanyState(TypedDict):
    """State for company intelligence workflow."""
    # Input
    symbol: str
    batch_number: int
    
    # Fetched data
    basic_data: Dict[str, Any]
    
    # Claude extractions
    business_profile: Dict[str, Any]
    competitors: List[str]
    products: List[str]
    risk_factors: List[str]
    
    # Quality
    quality_score: float
    quality_issues: List[str]
    
    # Storage results
    stored_postgres: bool
    stored_neo4j: bool
    
    # Workflow control
    messages: List[str]
    errors: List[str]
    success: bool


# ================================================================
# Company Intelligence Workflow
# ================================================================

class CompanyIntelligenceWorkflow:
    """
    LangGraph-powered company intelligence pipeline.
    
    Features:
    - Parallel processing (5 companies at once)
    - Claude-powered extraction
    - Multi-database storage
    - Self-validating quality
    - No Airflow dependencies
    """
    
    def __init__(self):
        """Initialize workflow with Claude."""
        self.claude = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            max_tokens=4096,
            temperature=0.1
        )
        
        # Database connections
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            auth=(
                os.getenv('NEO4J_USER', 'neo4j'),
                os.getenv('NEO4J_PASSWORD', 'axiom_neo4j_password')
            )
        )
        
        self.pg_conn_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'user': os.getenv('POSTGRES_USER', 'axiom'),
            'password': os.getenv('POSTGRES_PASSWORD', 'axiom_password'),
            'database': os.getenv('POSTGRES_DB', 'axiom_finance')
        }
        
        # Build workflow
        self.app = self._build_workflow()
        
        logger.info("✅ Company Intelligence Workflow initialized")
    
    def _build_workflow(self):
        """Build LangGraph workflow."""
        workflow = StateGraph(CompanyState)
        
        # Add agent nodes
        workflow.add_node("fetch_basic", self._fetch_basic_data)
        workflow.add_node("claude_profile", self._extract_business_profile)
        workflow.add_node("claude_competitors", self._extract_competitors)
        workflow.add_node("claude_products", self._extract_products)
        workflow.add_node("claude_risks", self._extract_risks)
        workflow.add_node("validate_quality", self._validate_quality)
        workflow.add_node("store_data", self._store_multi_database)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define flow
        workflow.set_entry_point("fetch_basic")
        workflow.add_edge("fetch_basic", "claude_profile")
        workflow.add_edge("claude_profile", "claude_competitors")
        workflow.add_edge("claude_competitors", "claude_products")
        workflow.add_edge("claude_products", "claude_risks")
        workflow.add_edge("claude_risks", "validate_quality")
        
        # Conditional: Re-enrich if quality low, else store
        workflow.add_conditional_edges(
            "validate_quality",
            lambda state: "store" if state['quality_score'] >= 0.7 else "re_enrich",
            {
                "store": "store_data",
                "re_enrich": "claude_profile"  # Loop back for better extraction
            }
        )
        
        workflow.add_edge("store_data", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    # ================================================================
    # Agent Nodes
    # ================================================================
    
    def _fetch_basic_data(self, state: CompanyState) -> CompanyState:
        """
        Agent 1: Fetch basic company data from yfinance.
        """
        symbol = state['symbol']
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            state['basic_data'] = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'business_summary': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'country': info.get('country', ''),
                'city': info.get('city', ''),
                'employees': info.get('fullTimeEmployees', 0),
                'market_cap': info.get('marketCap', 0),
                'revenue': info.get('totalRevenue', 0),
                'profit_margin': info.get('profitMargins', 0),
                'pe_ratio': info.get('trailingPE', 0)
            }
            
            state['messages'].append(f"✅ Fetched {symbol}: {state['basic_data']['name']}")
            logger.info(f"✅ {symbol}: {len(state['basic_data']['business_summary'])} char summary")
            
        except Exception as e:
            state['errors'].append(f"Fetch failed: {str(e)}")
            state['success'] = False
            logger.error(f"❌ Failed to fetch {symbol}: {e}")
        
        return state
    
    def _extract_business_profile(self, state: CompanyState) -> CompanyState:
        """
        Agent 2: Claude extracts business intelligence from text.
        """
        summary = state['basic_data'].get('business_summary', '')
        
        if not summary:
            state['business_profile'] = {}
            state['messages'].append("⚠️ No business summary to profile")
            return state
        
        try:
            prompt = f"""Analyze this company's business model:

{summary}

Extract and return ONLY a JSON object:
{{
    "business_model": "Brief description of how they make money",
    "target_markets": ["Market1", "Market2"],
    "competitive_advantages": ["Advantage1", "Advantage2"],
    "growth_drivers": ["Driver1", "Driver2"],
    "key_strengths": ["Strength1", "Strength2"]
}}"""
            
            response = self.claude.invoke([
                SystemMessage(content="You are a business analyst. Return ONLY valid JSON, no markdown, no explanations."),
                HumanMessage(content=prompt)
            ])
            
            # Parse JSON
            profile = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            state['business_profile'] = profile
            
            state['messages'].append(f"✅ Claude profiled business model")
            logger.info(f"✅ {state['symbol']}: Business profile extracted")
            
        except Exception as e:
            state['errors'].append(f"Profile extraction failed: {str(e)}")
            state['business_profile'] = {}
            logger.error(f"❌ Profile extraction failed for {state['symbol']}: {e}")
        
        return state
    
    def _extract_competitors(self, state: CompanyState) -> CompanyState:
        """
        Agent 3: Claude identifies competitors from business description.
        """
        name = state['basic_data'].get('name', '')
        sector = state['basic_data'].get('sector', '')
        industry = state['basic_data'].get('industry', '')
        summary = state['basic_data'].get('business_summary', '')
        
        try:
            prompt = f"""Identify the top 5 direct competitors for this company:

Company: {name}
Sector: {sector}
Industry: {industry}
Business: {summary[:500]}

Return ONLY a JSON array of competitor ticker symbols:
["COMP1", "COMP2", "COMP3", "COMP4", "COMP5"]

Use actual stock ticker symbols (e.g., AAPL, MSFT, GOOGL).
If unknown, use company name."""
            
            response = self.claude.invoke([
                SystemMessage(content="You are a market analyst. Return ONLY a JSON array of strings."),
                HumanMessage(content=prompt)
            ])
            
            competitors = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            state['competitors'] = competitors if isinstance(competitors, list) else []
            
            state['messages'].append(f"✅ Claude identified {len(state['competitors'])} competitors")
            logger.info(f"✅ {state['symbol']}: {len(state['competitors'])} competitors found")
            
        except Exception as e:
            state['errors'].append(f"Competitor extraction failed: {str(e)}")
            state['competitors'] = []
            logger.error(f"❌ Competitor extraction failed: {e}")
        
        return state
    
    def _extract_products(self, state: CompanyState) -> CompanyState:
        """
        Agent 4: Claude extracts key products/services.
        """
        summary = state['basic_data'].get('business_summary', '')
        
        if not summary:
            state['products'] = []
            return state
        
        try:
            prompt = f"""Extract the key products and services from this business:

{summary}

Return ONLY a JSON array of product/service names:
["Product1", "Product2", "Product3", "Product4", "Product5"]

Focus on the main revenue-generating offerings."""
            
            response = self.claude.invoke([
                SystemMessage(content="You are a product analyst. Return ONLY a JSON array."),
                HumanMessage(content=prompt)
            ])
            
            products = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            state['products'] = products if isinstance(products, list) else []
            
            state['messages'].append(f"✅ Claude identified {len(state['products'])} products")
            logger.info(f"✅ {state['symbol']}: {len(state['products'])} products found")
            
        except Exception as e:
            state['errors'].append(f"Product extraction failed: {str(e)}")
            state['products'] = []
            logger.error(f"❌ Product extraction failed: {e}")
        
        return state
    
    def _extract_risks(self, state: CompanyState) -> CompanyState:
        """
        Agent 5: Claude identifies risk factors.
        """
        summary = state['basic_data'].get('business_summary', '')
        sector = state['basic_data'].get('sector', '')
        
        try:
            prompt = f"""Identify top 5 risk factors for this company:

Sector: {sector}
Business: {summary[:500]}

Return ONLY a JSON array of risk factor descriptions:
["Risk1 description", "Risk2 description", ...]

Focus on material business risks."""
            
            response = self.claude.invoke([
                SystemMessage(content="You are a risk analyst. Return ONLY a JSON array."),
                HumanMessage(content=prompt)
            ])
            
            risks = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            state['risk_factors'] = risks if isinstance(risks, list) else []
            
            state['messages'].append(f"✅ Claude identified {len(state['risk_factors'])} risks")
            logger.info(f"✅ {state['symbol']}: {len(state['risk_factors'])} risks identified")
            
        except Exception as e:
            state['errors'].append(f"Risk extraction failed: {str(e)}")
            state['risk_factors'] = []
            logger.error(f"❌ Risk extraction failed: {e}")
        
        return state
    
    def _validate_quality(self, state: CompanyState) -> CompanyState:
        """
        Agent 6: Claude validates profile completeness and quality.
        """
        try:
            prompt = f"""Assess this company profile quality:

Basic Data: {list(state['basic_data'].keys())}
Business Profile: {list(state.get('business_profile', {}).keys())}
Competitors: {len(state.get('competitors', []))} found
Products: {len(state.get('products', []))} found
Risks: {len(state.get('risk_factors', []))} found

Rate quality 0-1 based on:
- Completeness (all fields populated?)
- Usefulness (enough for analysis?)
- Accuracy (makes business sense?)

Return JSON:
{{
    "score": 0.95,
    "issues": ["Missing X", "Incomplete Y"],
    "recommendation": "accept" or "re_enrich"
}}"""
            
            response = self.claude.invoke([
                SystemMessage(content="You are a data quality analyst. Return ONLY JSON."),
                HumanMessage(content=prompt)
            ])
            
            validation = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            
            state['quality_score'] = validation.get('score', 0.0)
            state['quality_issues'] = validation.get('issues', [])
            
            state['messages'].append(f"✅ Quality: {state['quality_score']:.0%}")
            logger.info(f"✅ {state['symbol']}: Quality {state['quality_score']:.0%}")
            
        except Exception as e:
            state['errors'].append(f"Validation failed: {str(e)}")
            state['quality_score'] = 0.5  # Assume medium quality
            state['quality_issues'] = []
        
        return state
    
    def _store_multi_database(self, state: CompanyState) -> CompanyState:
        """
        Agent 7: Store in PostgreSQL + Neo4j.
        """
        symbol = state['symbol']
        
        # Store in PostgreSQL
        try:
            conn = psycopg2.connect(**self.pg_conn_params)
            cur = conn.cursor()
            
            basic = state['basic_data']
            cur.execute("""
                INSERT INTO company_fundamentals 
                (symbol, report_date, fiscal_period, company_name, sector, industry, 
                 market_cap, revenue, pe_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, report_date, fiscal_period) DO UPDATE
                SET company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap = EXCLUDED.market_cap,
                    revenue = EXCLUDED.revenue,
                    pe_ratio = EXCLUDED.pe_ratio
            """, (
                symbol,
                datetime.now().date(),
                'CURRENT',
                basic.get('name'),
                basic.get('sector'),
                basic.get('industry'),
                basic.get('market_cap', 0),
                basic.get('revenue', 0),
                basic.get('pe_ratio', 0)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            state['stored_postgres'] = True
            logger.info(f"✅ {symbol}: Stored in PostgreSQL")
            
        except Exception as e:
            state['errors'].append(f"PostgreSQL storage failed: {str(e)}")
            state['stored_postgres'] = False
            logger.error(f"❌ PostgreSQL storage failed: {e}")
        
        # Store in Neo4j
        try:
            with self.neo4j_driver.session() as session:
                basic = state['basic_data']
                profile = state.get('business_profile', {})
                
                # Create company node
                session.run("""
                    MERGE (c:Company {symbol: $symbol})
                    SET c.name = $name,
                        c.sector = $sector,
                        c.industry = $industry,
                        c.business_summary = $summary,
                        c.website = $website,
                        c.country = $country,
                        c.market_cap = $market_cap,
                        c.employees = $employees,
                        c.revenue = $revenue,
                        c.business_model = $business_model,
                        c.target_markets = $target_markets,
                        c.updated_at = datetime()
                """, 
                    symbol=symbol,
                    name=basic.get('name'),
                    sector=basic.get('sector'),
                    industry=basic.get('industry'),
                    summary=basic.get('business_summary'),
                    website=basic.get('website'),
                    country=basic.get('country'),
                    market_cap=basic.get('market_cap', 0),
                    employees=basic.get('employees', 0),
                    revenue=basic.get('revenue', 0),
                    business_model=profile.get('business_model', ''),
                    target_markets=profile.get('target_markets', [])
                )
                
                # Create competitor relationships
                for comp in state.get('competitors', [])[:5]:  # Top 5
                    session.run("""
                        MATCH (c1:Company {symbol: $symbol})
                        MERGE (c2:Company {symbol: $competitor})
                        MERGE (c1)-[r:COMPETES_WITH]->(c2)
                        SET r.identified_by = 'claude',
                            r.updated_at = datetime()
                    """, symbol=symbol, competitor=comp)
                
                # Store products as properties
                if state.get('products'):
                    session.run("""
                        MATCH (c:Company {symbol: $symbol})
                        SET c.products = $products
                    """, symbol=symbol, products=state['products'])
                
                # Store risks
                if state.get('risk_factors'):
                    session.run("""
                        MATCH (c:Company {symbol: $symbol})
                        SET c.risk_factors = $risks
                    """, symbol=symbol, risks=state['risk_factors'])
            
            state['stored_neo4j'] = True
            state['messages'].append(f"✅ Stored in Neo4j with {len(state.get('competitors', []))} relationships")
            logger.info(f"✅ {symbol}: Stored in Neo4j with graph")
            
        except Exception as e:
            state['errors'].append(f"Neo4j storage failed: {str(e)}")
            state['stored_neo4j'] = False
            logger.error(f"❌ Neo4j storage failed: {e}")
        
        state['success'] = state['stored_postgres'] and state['stored_neo4j']
        return state
    
    def _handle_error(self, state: CompanyState) -> CompanyState:
        """Error handler node."""
        state['success'] = False
        logger.error(f"❌ {state['symbol']}: Workflow failed - {state['errors']}")
        return state
    
    # ================================================================
    # Processing Functions
    # ================================================================
    
    async def process_company(self, symbol: str, batch_number: int = 0) -> CompanyState:
        """
        Process single company through workflow.
        """
        initial_state: CompanyState = {
            'symbol': symbol,
            'batch_number': batch_number,
            'basic_data': {},
            'business_profile': {},
            'competitors': [],
            'products': [],
            'risk_factors': [],
            'quality_score': 0.0,
            'quality_issues': [],
            'stored_postgres': False,
            'stored_neo4j': False,
            'messages': [],
            'errors': [],
            'success': False
        }
        
        # Run workflow
        result = self.app.invoke(initial_state)
        return result
    
    async def process_batch(self, symbols: List[str], batch_size: int = 5) -> Dict[str, Any]:
        """
        Process companies in parallel batches.
        
        Args:
            symbols: Company symbols to process
            batch_size: Number of companies to process in parallel
            
        Returns:
            Batch processing summary
        """
        total = len(symbols)
        successful = 0
        failed = []
        
        # Process in batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            batch_num = i // batch_size
            
            logger.info(f"=== Processing Batch {batch_num} ({len(batch)} companies) ===")
            
            # Parallel processing
            tasks = [self.process_company(symbol, batch_num) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    failed.append(batch[j])
                    logger.error(f"❌ {batch[j]}: {result}")
                elif result.get('success'):
                    successful += 1
                    logger.info(f"✅ {batch[j]}: Complete")
                else:
                    failed.append(batch[j])
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(2)
        
        summary = {
            'total': total,
            'successful': successful,
            'failed': len(failed),
            'failed_symbols': failed,
            'success_rate': successful / total if total > 0 else 0
        }
        
        logger.info(f"=== Batch Processing Complete: {successful}/{total} successful ===")
        return summary
    
    def close(self):
        """Cleanup resources."""
        self.neo4j_driver.close()
        logger.info("✅ Workflow cleanup complete")


# ================================================================
# Main Execution
# ================================================================

async def main():
    """Run company intelligence pipeline."""
    
    # Company universe (50 companies)
    COMPANIES = [
        # Top 10 Mega-Cap
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JPM',
        # Next 10 Large-Cap Tech
        'V', 'MA', 'HD', 'PG', 'DIS', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC',
        # Next 10 Large-Cap Diversified
        'PFE', 'MRK', 'ABBV', 'KO', 'PEP', 'WMT', 'COST', 'CVX', 'XOM', 'BA',
        # Next 10 Financial/Healthcare
        'BAC', 'GS', 'C', 'WFC', 'MS', 'JNJ', 'LLY', 'ABBV', 'TMO', 'DHR',
        # Next 10 Diversified
        'CSCO', 'AVGO', 'TXN', 'QCOM', 'AMD', 'MU', 'AMAT', 'LRCX', 'KLAC', 'SNPS'
    ]
    
    logger.info(f"Starting LangGraph Company Intelligence Pipeline")
    logger.info(f"Target: {len(COMPANIES)} companies")
    logger.info(f"Parallel batch size: 5")
    
    # Initialize workflow
    workflow = CompanyIntelligenceWorkflow()
    
    try:
        # Process all companies
        summary = await workflow.process_batch(COMPANIES, batch_size=5)
        
        # Report results
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Companies: {summary['total']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        if summary['failed_symbols']:
            logger.info(f"Failed Symbols: {', '.join(summary['failed_symbols'])}")
        logger.info("="*60)
        
    finally:
        workflow.close()
    
    return summary


if __name__ == '__main__':
    # Run the pipeline
    summary = asyncio.run(main())
    
    print(f"\n🎉 Company Intelligence Pipeline Complete!")
    print(f"✅ {summary['successful']}/{summary['total']} companies profiled")
    print(f"\nCheck results in:")
    print(f"  - PostgreSQL: SELECT * FROM company_fundamentals;")
    print(f"  - Neo4j: MATCH (c:Company) RETURN c LIMIT 10;")