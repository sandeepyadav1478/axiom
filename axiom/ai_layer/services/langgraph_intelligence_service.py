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


# Import our orchestrators
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ai_layer.langgraph_ma_orchestrator import MAOrchestrator


class ContinuousIntelligenceService:
    """
    Native LangGraph service running continuously.
    
    No Airflow, no external scheduler - LangGraph orchestrates itself.
    Demonstrates: Self-contained AI service architecture.
    """
    
    def __init__(self):
        self.ma_orchestrator = MAOrchestrator()
        self.interval_seconds = int(os.getenv('LANGGRAPH_INTERVAL', '300'))  # 5 minutes
        self.symbols = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOGL,TSLA,NVDA').split(',')
        
        logger.info(f"LangGraph Intelligence Service initialized")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Interval: {self.interval_seconds}s")
    
    async def run_ma_analysis_cycle(self):
        """Run M&A analysis on all symbols."""
        logger.info("=" * 60)
        logger.info(f"M&A Analysis Cycle - {datetime.now()}")
        logger.info("=" * 60)
        
        for symbol in self.symbols:
            try:
                result = self.ma_orchestrator.analyze_deal(
                    target=symbol,
                    analysis_type='acquisition_target'
                )
                
                logger.info(f"✅ {symbol}: {result['recommendation']} (confidence: {result['confidence']:.0%})")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
        
        logger.info(f"Cycle complete - next in {self.interval_seconds}s")
    
    async def run_continuous(self):
        """Main continuous loop - self-orchestrating."""
        logger.info("Starting continuous LangGraph intelligence service...")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                logger.info(f"\n📊 Cycle {cycle_count}")
                
                await self.run_ma_analysis_cycle()
                
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