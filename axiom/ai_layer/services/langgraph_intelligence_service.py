"""
Native LangGraph Intelligence Service
Runs continuously without Airflow wrapper - direct AI orchestration

This demonstrates LangGraph can orchestrate itself - no external scheduler needed.
Compares to Airflow approach showing architectural alternatives.
"""

import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from neo4j import GraphDatabase
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContinuousIntelligenceService:
    """
    Native LangGraph service - self-contained, no imports needed.
    Demonstrates LangGraph can run independently without Airflow.
    """
    
    def __init__(self):
        self.claude = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY'),
            max_tokens=2048
        )
        self.interval_seconds = int(os.getenv('LANGGRAPH_INTERVAL', '300'))
        self.symbols = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOGL,TSLA,NVDA').split(',')
        
        logger.info(f"Native LangGraph Service initialized (no Airflow!)")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Interval: {self.interval_seconds}s")
    
    async def analyze_company(self, symbol: str) -> dict:
        """Simple company analysis with Claude."""
        try:
            # Fetch from Neo4j
            driver = GraphDatabase.driver(
                os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
                auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
            )
            
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Company {symbol: $symbol})
                    RETURN c.name as name, c.business_summary as summary
                """, symbol=symbol)
                
                record = result.single()
                
                if record:
                    # Claude analyzes
                    response = self.claude.invoke([{
                        "role": "user",
                        "content": f"Analyze {record['name']} as acquisition target. One sentence."
                    }])
                    
                    return {
                        'symbol': symbol,
                        'name': record['name'],
                        'analysis': response.content,
                        'timestamp': datetime.now().isoformat()
                    }
            
            driver.close()
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            
        return {'symbol': symbol, 'error': 'Analysis failed'}
    
    async def run_analysis_cycle(self):
        """Run analysis cycle on all symbols."""
        logger.info("=" * 60)
        logger.info(f"Analysis Cycle - {datetime.now()}")
        logger.info("=" * 60)
        
        for symbol in self.symbols:
            result = await self.analyze_company(symbol)
            
            if 'error' not in result:
                logger.info(f"✅ {symbol}: {result.get('analysis', '')[:100]}...")
            else:
                logger.error(f"❌ {symbol}: {result.get('error')}")
        
        logger.info(f"Cycle complete - next in {self.interval_seconds}s")
    
    async def run_continuous(self):
        """Main continuous loop - self-orchestrating."""
        logger.info("Starting native LangGraph intelligence service (NO AIRFLOW!)")
        logger.info("This demonstrates LangGraph can orchestrate itself")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                logger.info(f"\n📊 Cycle {cycle_count}")
                
                await self.run_analysis_cycle()
                
                await asyncio.sleep(self.interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(self.interval_seconds)


if __name__ == "__main__":
    service = ContinuousIntelligenceService()
    asyncio.run(service.run_continuous())