"""
Market Events Tracker Pipeline
Uses LangGraph + Claude to track and classify market events

Tracks:
- Earnings announcements
- Fed decisions
- M&A activity
- Regulatory changes
- Major news
"""

import asyncio
import logging
from typing import Dict, List, Any, TypedDict, Optional
import os
from datetime import datetime, timedelta

import yfinance as yf
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


class EventTrackerState(TypedDict):
    """State for event tracking workflow."""
    symbol: str
    news_items: List[Dict]
    classified_events: List[Dict]
    affected_companies: List[str]
    impact_scores: Dict[str, float]
    neo4j_updates: int
    success: bool
    error: Optional[str]


class EventTrackerPipeline(BaseLangGraphPipeline):
    """
    LangGraph-powered pipeline for tracking market events.
    
    Workflow:
    1. Fetch news for symbol
    2. Classify events using Claude (earnings, fed, m&a, etc.)
    3. Identify affected companies using Claude
    4. Calculate impact scores using Claude
    5. Create event nodes in Neo4j
    6. Link events to affected companies
    """
    
    def __init__(self):
        """Initialize Event Tracker."""
        super().__init__("EventTracker")
        
        # Get Neo4j client
        self.neo4j = get_neo4j_client()
    
    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        workflow = StateGraph(EventTrackerState)
        
        # Define agent nodes
        workflow.add_node("fetch_news", self.fetch_company_news)
        workflow.add_node("classify_events", self.classify_events_with_claude)
        workflow.add_node("identify_affected", self.identify_affected_companies)
        workflow.add_node("calculate_impact", self.calculate_impact_scores)
        workflow.add_node("create_events", self.create_event_nodes)
        workflow.add_node("link_to_companies", self.link_events_to_companies)
        
        # Define workflow edges
        workflow.add_edge("fetch_news", "classify_events")
        workflow.add_edge("classify_events", "identify_affected")
        workflow.add_edge("identify_affected", "calculate_impact")
        workflow.add_edge("calculate_impact", "create_events")
        workflow.add_edge("create_events", "link_to_companies")
        workflow.add_edge("link_to_companies", END)
        
        # Set entry point
        workflow.set_entry_point("fetch_news")
        
        return workflow
    
    def fetch_company_news(self, state: EventTrackerState) -> EventTrackerState:
        """Agent 1: Fetch recent news for company."""
        symbol = state['symbol']
        logger.info(f"Fetching news for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news if hasattr(ticker, 'news') else []
            
            # Get recent news (last 7 days)
            cutoff = datetime.now() - timedelta(days=7)
            recent_news = []
            
            for item in news[:10]:  # Limit to 10 most recent
                recent_news.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'timestamp': item.get('providerPublishTime', 0)
                })
            
            state['news_items'] = recent_news
            logger.info(f"✅ Found {len(recent_news)} news items")
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            state['news_items'] = []
        
        return state
    
    def classify_events_with_claude(self, state: EventTrackerState) -> EventTrackerState:
        """Agent 2: Classify news into event types using Claude."""
        
        if not state['news_items']:
            state['classified_events'] = []
            return state
        
        # Prepare news for Claude
        news_text = "\n\n".join([
            f"[{i+1}] {item['title']}"
            for i, item in enumerate(state['news_items'])
        ])
        
        prompt = f"""Analyze these news items and classify them into event types:

{news_text}

For each news item, determine:
1. Event type: earnings, merger, acquisition, partnership, lawsuit, regulatory, product_launch, fed_decision, other
2. Relevance score: 0.0-1.0 (how market-moving is this event?)

Return JSON array format:
[
  {{"item": 1, "type": "earnings", "relevance": 0.9}},
  {{"item": 2, "type": "partnership", "relevance": 0.6}}
]

Return ONLY the JSON array, no explanation."""
        
        try:
            response = self.invoke_claude(
                prompt,
                system="You are a financial news analyst expert at event classification."
            )
            
            # Parse JSON response
            import json
            classified = json.loads(response.strip())
            
            # Combine with original news
            events = []
            for classification in classified:
                idx = classification['item'] - 1
                if idx < len(state['news_items']):
                    events.append({
                        **state['news_items'][idx],
                        'event_type': classification['type'],
                        'relevance': classification['relevance']
                    })
            
            state['classified_events'] = events
            logger.info(f"✅ Claude classified {len(events)} events")
            
        except Exception as e:
            logger.error(f"Error classifying events: {e}")
            state['classified_events'] = []
        
        return state
    
    def identify_affected_companies(self, state: EventTrackerState) -> EventTrackerState:
        """Agent 3: Identify which companies are affected by each event."""
        
        affected_set = set([state['symbol']])  # Primary company always affected
        
        for event in state['classified_events']:
            # Use Claude to identify affected companies for high-relevance events
            if event['relevance'] > 0.6:
                prompt = f"""This event occurred: "{event['title']}"
Event type: {event['event_type']}

Which other stock symbols might be significantly affected by this event?
Consider:
- Direct competitors
- Supply chain partners
- Same sector companies

Return ONLY stock ticker symbols, comma-separated.
Maximum 5 symbols.
If none affected, return "none".

Example: MSFT,GOOGL,META"""
                
                try:
                    response = self.invoke_claude(prompt)
                    if response.strip().lower() != 'none':
                        symbols = [s.strip() for s in response.strip().split(',')]
                        affected_set.update(symbols)
                except Exception as e:
                    logger.error(f"Error identifying affected companies: {e}")
        
        state['affected_companies'] = list(affected_set)
        logger.info(f"✅ Identified {len(state['affected_companies'])} affected companies")
        
        return state
    
    def calculate_impact_scores(self, state: EventTrackerState) -> EventTrackerState:
        """Agent 4: Calculate impact scores for each affected company."""
        
        impact_scores = {}
        
        for company in state['affected_companies']:
            # Primary company gets higher impact
            if company == state['symbol']:
                avg_relevance = sum(e['relevance'] for e in state['classified_events']) / max(len(state['classified_events']), 1)
                impact_scores[company] = min(avg_relevance, 1.0)
            else:
                # Secondary companies get lower impact
                impact_scores[company] = 0.3
        
        state['impact_scores'] = impact_scores
        logger.info(f"✅ Calculated impact scores for {len(impact_scores)} companies")
        
        return state
    
    def create_event_nodes(self, state: EventTrackerState) -> EventTrackerState:
        """Agent 5: Create MarketEvent nodes in Neo4j."""
        
        created = 0
        for event in state['classified_events']:
            try:
                self.neo4j.execute_cypher("""
                    CREATE (e:MarketEvent {
                        type: $type,
                        title: $title,
                        publisher: $publisher,
                        link: $link,
                        timestamp: datetime($timestamp),
                        relevance: $relevance,
                        primary_symbol: $symbol
                    })
                """, {
                    'type': event['event_type'],
                    'title': event['title'],
                    'publisher': event['publisher'],
                    'link': event['link'],
                    'timestamp': datetime.fromtimestamp(event['timestamp']).isoformat(),
                    'relevance': event['relevance'],
                    'symbol': state['symbol']
                })
                
                created += 1
                
            except Exception as e:
                logger.error(f"Error creating event node: {e}")
        
        state['neo4j_updates'] = created
        logger.info(f"✅ Created {created} event nodes")
        
        return state
    
    def link_events_to_companies(self, state: EventTrackerState) -> EventTrackerState:
        """Agent 6: Link events to affected companies."""
        
        links_created = 0
        
        for event in state['classified_events']:
            for company in state['affected_companies']:
                impact = state['impact_scores'].get(company, 0.0)
                
                try:
                    self.neo4j.execute_cypher("""
                        MATCH (e:MarketEvent {title: $title, primary_symbol: $primary})
                        MERGE (c:Company {symbol: $company})
                        MERGE (c)-[r:AFFECTED_BY]->(e)
                        SET r.impact_score = $impact,
                            r.created_at = datetime()
                    """, {
                        'title': event['title'],
                        'primary': state['symbol'],
                        'company': company,
                        'impact': impact
                    })
                    
                    links_created += 1
                    
                except Exception as e:
                    logger.error(f"Error linking event to company: {e}")
        
        logger.info(f"✅ Created {links_created} event-company links")
        state['success'] = True
        
        return state
    
    async def process_item(self, item: str) -> Dict[str, Any]:
        """Process a single symbol through the event tracking workflow."""
        
        initial_state: EventTrackerState = {
            'symbol': item,
            'news_items': [],
            'classified_events': [],
            'affected_companies': [],
            'impact_scores': {},
            'neo4j_updates': 0,
            'success': False,
            'error': None
        }
        
        # Run through LangGraph workflow
        final_state = self.app.invoke(initial_state)
        
        return {
            'symbol': item,
            'success': final_state['success'],
            'events_found': len(final_state['classified_events']),
            'neo4j_updates': final_state['neo4j_updates']
        }


async def main():
    """Main entry point for Event Tracker."""
    
    # Get symbols from environment
    symbols_str = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOGL')
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    logger.info("="*70)
    logger.info("MARKET EVENTS TRACKER PIPELINE")
    logger.info("LangGraph + Claude + Neo4j")
    logger.info("="*70)
    
    # Initialize pipeline
    pipeline = EventTrackerPipeline()
    
    # Run continuous mode
    await pipeline.run_continuous(symbols)


if __name__ == "__main__":
    asyncio.run(main())