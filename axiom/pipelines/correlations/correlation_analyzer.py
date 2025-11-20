"""
Correlation Analyzer Pipeline
Uses LangGraph + Claude to find and explain market correlations

Features:
- Fetches historical price data from PostgreSQL
- Calculates correlation matrices
- Uses Claude to explain WHY correlations exist
- Creates CORRELATED_WITH relationships in Neo4j
"""

import asyncio
import logging
from typing import Dict, List, Any, TypedDict, Optional
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from langgraph.graph import StateGraph, END

import sys
sys.path.insert(0, '/app')
from axiom.pipelines.shared.neo4j_client import get_neo4j_client
from axiom.pipelines.shared.langgraph_base import BaseLangGraphPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorrelationState(TypedDict):
    """State for correlation analysis workflow."""
    symbols: List[str]
    price_data: pd.DataFrame
    correlation_matrix: pd.DataFrame
    significant_correlations: List[Dict]
    explanations: Dict[str, str]
    neo4j_updates: int
    success: bool
    error: Optional[str]


class CorrelationAnalyzerPipeline(BaseLangGraphPipeline):
    """
    LangGraph-powered pipeline for correlation analysis.
    
    Workflow:
    1. Fetch historical prices from PostgreSQL
    2. Calculate correlation matrix
    3. Filter significant correlations (>0.7)
    4. Ask Claude to explain each correlation
    5. Create CORRELATED_WITH edges in Neo4j
    6. Validate graph updates
    """
    
    def __init__(self):
        """Initialize Correlation Analyzer."""
        super().__init__("CorrelationAnalyzer")
        
        # Get database clients
        self.neo4j = get_neo4j_client()
        
        # PostgreSQL connection
        pg_host = os.getenv('POSTGRES_HOST', 'postgres')
        pg_user = os.getenv('POSTGRES_USER', 'axiom')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'axiom_secure_2024')
        pg_db = os.getenv('POSTGRES_DB', 'axiom_db')
        
        pg_url = f'postgresql://{pg_user}:{pg_password}@{pg_host}:5432/{pg_db}'
        self.pg_engine = create_engine(pg_url)
        
        logger.info("✅ PostgreSQL connected for historical data")
    
    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        workflow = StateGraph(CorrelationState)
        
        # Define agent nodes
        workflow.add_node("fetch_prices", self.fetch_historical_prices)
        workflow.add_node("calculate_correlations", self.calculate_correlation_matrix)
        workflow.add_node("filter_significant", self.filter_significant_correlations)
        workflow.add_node("explain_correlations", self.explain_with_claude)
        workflow.add_node("update_graph", self.update_neo4j_correlations)
        workflow.add_node("validate", self.validate_updates)
        
        # Define workflow edges
        workflow.add_edge("fetch_prices", "calculate_correlations")
        workflow.add_edge("calculate_correlations", "filter_significant")
        workflow.add_edge("filter_significant", "explain_correlations")
        workflow.add_edge("explain_correlations", "update_graph")
        workflow.add_edge("update_graph", "validate")
        workflow.add_edge("validate", END)
        
        # Set entry point
        workflow.set_entry_point("fetch_prices")
        
        return workflow
    
    def fetch_historical_prices(self, state: CorrelationState) -> CorrelationState:
        """Agent 1: Fetch 30-day price history from PostgreSQL."""
        symbols = state['symbols']
        logger.info(f"Fetching price history for {len(symbols)} symbols")
        
        try:
            # Query last 30 days of prices
            query = text("""
                SELECT symbol, timestamp, close
                FROM price_data
                WHERE symbol = ANY(:symbols)
                  AND timestamp >= NOW() - INTERVAL '30 days'
                ORDER BY symbol, timestamp
            """)
            
            # Execute query
            df = pd.read_sql(query, self.pg_engine, params={'symbols': symbols})
            
            # Pivot to get symbols as columns
            price_matrix = df.pivot(index='timestamp', columns='symbol', values='close')
            
            state['price_data'] = price_matrix
            logger.info(f"✅ Fetched {len(price_matrix)} days of price data")
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            state['price_data'] = pd.DataFrame()
            state['error'] = str(e)
        
        return state
    
    def calculate_correlation_matrix(self, state: CorrelationState) -> CorrelationState:
        """Agent 2: Calculate correlation matrix."""
        
        if state['price_data'].empty:
            state['correlation_matrix'] = pd.DataFrame()
            return state
        
        try:
            # Calculate Pearson correlation
            corr_matrix = state['price_data'].corr()
            state['correlation_matrix'] = corr_matrix
            
            logger.info(f"✅ Calculated correlation matrix ({corr_matrix.shape[0]} × {corr_matrix.shape[1]})")
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            state['correlation_matrix'] = pd.DataFrame()
        
        return state
    
    def filter_significant_correlations(self, state: CorrelationState) -> CorrelationState:
        """Agent 3: Filter for significant correlations (>0.7 or <-0.5)."""
        
        if state['correlation_matrix'].empty:
            state['significant_correlations'] = []
            return state
        
        significant = []
        corr_matrix = state['correlation_matrix']
        
        # Find significant correlations
        for i, symbol1 in enumerate(corr_matrix.columns):
            for j, symbol2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    coef = corr_matrix.loc[symbol1, symbol2]
                    
                    # Filter significant (strong positive or negative)
                    if abs(coef) > 0.7:
                        significant.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'coefficient': float(coef),
                            'strength': 'strong' if abs(coef) > 0.85 else 'moderate'
                        })
        
        state['significant_correlations'] = significant
        logger.info(f"✅ Found {len(significant)} significant correlations")
        
        return state
    
    def explain_with_claude(self, state: CorrelationState) -> CorrelationState:
        """Agent 4: Use Claude to explain WHY correlations exist."""
        
        explanations = {}
        
        for corr in state['significant_correlations'][:10]:  # Limit to top 10 to save API calls
            symbol1 = corr['symbol1']
            symbol2 = corr['symbol2']
            coef = corr['coefficient']
            
            prompt = f"""Explain why these two stocks are correlated:

Stock 1: {symbol1}
Stock 2: {symbol2}
Correlation Coefficient: {coef:.3f}

Provide a concise 1-2 sentence explanation of the most likely reason for this correlation.
Consider: sector, industry, supply chain, competitive dynamics, market cap, geographic exposure.

Example: "Both are large-cap tech companies with similar business models and customer bases."
"""
            
            try:
                response = self.invoke_claude(
                    prompt,
                    system="You are a financial analyst expert at explaining market correlations."
                )
                
                explanations[f"{symbol1}-{symbol2}"] = response.strip()
                logger.debug(f"Explained {symbol1}-{symbol2}: {response[:80]}...")
                
            except Exception as e:
                logger.error(f"Error getting explanation: {e}")
                explanations[f"{symbol1}-{symbol2}"] = "Explanation unavailable"
        
        state['explanations'] = explanations
        logger.info(f"✅ Claude explained {len(explanations)} correlations")
        
        return state
    
    def update_neo4j_correlations(self, state: CorrelationState) -> CorrelationState:
        """Agent 5: Create CORRELATED_WITH relationships in Neo4j."""
        
        updates = 0
        
        for corr in state['significant_correlations']:
            symbol1 = corr['symbol1']
            symbol2 = corr['symbol2']
            coefficient = corr['coefficient']
            
            # Get explanation if available
            explanation = state['explanations'].get(f"{symbol1}-{symbol2}", "")
            
            try:
                self.neo4j.execute_cypher("""
                    MERGE (s1:Stock {symbol: $symbol1})
                    MERGE (s2:Stock {symbol: $symbol2})
                    MERGE (s1)-[r:CORRELATED_WITH]-(s2)
                    SET r.coefficient = $coefficient,
                        r.period_days = 30,
                        r.calculated_at = datetime(),
                        r.explanation = $explanation,
                        r.strength = $strength
                """, {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'coefficient': coefficient,
                    'explanation': explanation,
                    'strength': corr['strength']
                })
                
                updates += 1
                
            except Exception as e:
                logger.error(f"Error creating correlation edge: {e}")
        
        state['neo4j_updates'] = updates
        logger.info(f"✅ Created {updates} correlation relationships")
        
        return state
    
    def validate_updates(self, state: CorrelationState) -> CorrelationState:
        """Agent 6: Validate correlations were added to graph."""
        
        try:
            # Count total correlation edges
            result = self.neo4j.execute_cypher("""
                MATCH ()-[r:CORRELATED_WITH]->()
                RETURN count(r) as total_correlations
            """)
            
            if result:
                total = result[0]['total_correlations']
                logger.info(f"✅ Validation: Graph now has {total} total correlations")
                state['success'] = True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            state['success'] = False
        
        return state
    
    async def process_item(self, item: List[str]) -> Dict[str, Any]:
        """Process a group of symbols to find correlations."""
        
        initial_state: CorrelationState = {
            'symbols': item,
            'price_data': pd.DataFrame(),
            'correlation_matrix': pd.DataFrame(),
            'significant_correlations': [],
            'explanations': {},
            'neo4j_updates': 0,
            'success': False,
            'error': None
        }
        
        # Run through LangGraph workflow
        final_state = self.app.invoke(initial_state)
        
        return {
            'symbols_analyzed': len(item),
            'correlations_found': len(final_state['significant_correlations']),
            'neo4j_updates': final_state['neo4j_updates'],
            'success': final_state['success']
        }


async def main():
    """Main entry point for Correlation Analyzer."""
    
    # Get symbols from environment (analyze as a group)
    symbols_str = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOGL')
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    logger.info("="*70)
    logger.info("CORRELATION ANALYZER PIPELINE")
    logger.info("LangGraph + Claude + Neo4j + Statistical Analysis")
    logger.info("="*70)
    
    # Initialize pipeline
    pipeline = CorrelationAnalyzerPipeline()
    
    # Run continuous mode (analyze all symbols together)
    await pipeline.run_continuous([symbols])


if __name__ == "__main__":
    asyncio.run(main())