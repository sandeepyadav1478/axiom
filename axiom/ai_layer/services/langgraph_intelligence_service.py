"""
LangGraph Intelligence Synthesis Service
Real-time analysis and insights from market data

Purpose: Continuously analyze data and generate actionable intelligence
Architecture: Multi-agent LangGraph with streaming results
Data Sources: PostgreSQL (prices) + Neo4j (graph) + Real-time streams

Workflow: Gather → Analyze → Synthesize → Stream insights
"""

import asyncio
import logging
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import json

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from neo4j import GraphDatabase
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ================================================================
# State Definition
# ================================================================

class IntelligenceState(TypedDict):
    """State for intelligence synthesis workflow."""
    # Input context
    analysis_type: str  # 'market_overview', 'stock_analysis', 'opportunity_scan', 'risk_alert'
    symbols: List[str]
    timeframe: str  # '1h', '1d', '1w', '1m'
    
    # Gathered data
    price_data: Dict[str, List[Dict]]
    company_profiles: Dict[str, Dict]
    graph_relationships: Dict[str, Any]
    news_events: List[Dict]
    
    # Analysis
    patterns_identified: List[Dict]
    correlations_found: List[Dict]
    risks_detected: List[Dict]
    opportunities_found: List[Dict]
    
    # Synthesis
    key_insights: List[str]
    recommendations: List[Dict]
    confidence: float
    
    # Output
    report: Dict[str, Any]
    summary: str
    
    # Workflow
    messages: List[str]
    errors: List[str]


# ================================================================
# Intelligence Synthesis Workflow
# ================================================================

class IntelligenceSynthesisService:
    """
    LangGraph-powered intelligence synthesis.
    
    Features:
    - Multi-agent analysis (pattern detection, risk assessment, opportunity finding)
    - Claude-powered reasoning
    - Graph-aware intelligence
    - Real-time streaming insights
    - Professional investment-grade reports
    """
    
    def __init__(self):
        """Initialize service."""
        self.claude = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            max_tokens=8192,
            temperature=0.1
        )
        
        # Database connections
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            auth=(
                os.getenv('NEO4J_USER', 'neo4j'),
                os.getenv('NEO4J_PASSWORD')
            )
        )
        
        self.pg_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'user': os.getenv('POSTGRES_USER', 'axiom'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'database': os.getenv('POSTGRES_DB', 'axiom_finance')
        }
        
        # Build workflow
        self.app = self._build_workflow()
        
        logger.info("✅ Intelligence Synthesis Service initialized")
    
    def _build_workflow(self):
        """Build intelligence synthesis workflow."""
        workflow = StateGraph(IntelligenceState)
        
        # Data gathering agents (parallel)
        workflow.add_node("gather_prices", self._gather_price_data)
        workflow.add_node("gather_companies", self._gather_company_data)
        workflow.add_node("gather_graph", self._gather_graph_intelligence)
        workflow.add_node("gather_news", self._gather_news_events)
        
        # Analysis agents (parallel)
        workflow.add_node("detect_patterns", self._detect_patterns)
        workflow.add_node("find_correlations", self._find_correlations)
        workflow.add_node("assess_risks", self._assess_risks)
        workflow.add_node("identify_opportunities", self._identify_opportunities)
        
        # Synthesis agent (sequential)
        workflow.add_node("synthesize_insights", self._synthesize_insights)
        workflow.add_node("generate_report", self._generate_report)
        
        # Flow: Parallel gathering → Parallel analysis → Sequential synthesis
        workflow.set_entry_point("gather_prices")
        
        workflow.add_edge("gather_prices", "gather_companies")
        workflow.add_edge("gather_prices", "gather_graph")
        workflow.add_edge("gather_prices", "gather_news")
        
        # All gathering must complete before analysis
        workflow.add_edge("gather_companies", "detect_patterns")
        workflow.add_edge("gather_graph", "detect_patterns")
        workflow.add_edge("gather_news", "detect_patterns")
        
        # Parallel analysis
        workflow.add_edge("detect_patterns", "find_correlations")
        workflow.add_edge("detect_patterns", "assess_risks")
        workflow.add_edge("detect_patterns", "identify_opportunities")
        
        # All analysis feeds synthesis
        workflow.add_edge("find_correlations", "synthesize_insights")
        workflow.add_edge("assess_risks", "synthesize_insights")
        workflow.add_edge("identify_opportunities", "synthesize_insights")
        
        workflow.add_edge("synthesize_insights", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    # ================================================================
    # Data Gathering Agents
    # ================================================================
    
    def _gather_price_data(self, state: IntelligenceState) -> IntelligenceState:
        """Gather recent price data from PostgreSQL."""
        symbols = state['symbols']
        timeframe = state.get('timeframe', '1d')
        
        # Map timeframe to hours
        hours_map = {'1h': 1, '1d': 24, '1w': 168, '1m': 720}
        hours = hours_map.get(timeframe, 24)
        
        try:
            conn = psycopg2.connect(**self.pg_params)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            price_data = {}
            
            for symbol in symbols:
                cur.execute("""
                    SELECT symbol, timestamp, close, volume
                    FROM price_data
                    WHERE symbol = %s
                    AND timestamp > NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """, (symbol, hours))
                
                rows = cur.fetchall()
                price_data[symbol] = [dict(row) for row in rows]
            
            cur.close()
            conn.close()
            
            state['price_data'] = price_data
            state['messages'].append(f"✅ Gathered price data for {len(symbols)} symbols")
            logger.info(f"✅ Price data: {sum(len(p) for p in price_data.values())} data points")
            
        except Exception as e:
            state['errors'].append(f"Price gathering failed: {str(e)}")
            state['price_data'] = {}
        
        return state
    
    def _gather_company_data(self, state: IntelligenceState) -> IntelligenceState:
        """Gather company profiles from PostgreSQL."""
        symbols = state['symbols']
        
        try:
            conn = psycopg2.connect(**self.pg_params)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            profiles = {}
            
            for symbol in symbols:
                cur.execute("""
                    SELECT *
                    FROM company_fundamentals
                    WHERE symbol = %s
                    ORDER BY report_date DESC
                    LIMIT 1
                """, (symbol,))
                
                row = cur.fetchone()
                if row:
                    profiles[symbol] = dict(row)
            
            cur.close()
            conn.close()
            
            state['company_profiles'] = profiles
            state['messages'].append(f"✅ Gathered profiles for {len(profiles)} companies")
            
        except Exception as e:
            state['errors'].append(f"Company gathering failed: {str(e)}")
            state['company_profiles'] = {}
        
        return state
    
    def _gather_graph_intelligence(self, state: IntelligenceState) -> IntelligenceState:
        """Gather relationship intelligence from Neo4j."""
        symbols = state['symbols']
        
        try:
            with self.neo4j_driver.session() as session:
                # Get company relationships
                result = session.run("""
                    MATCH (c:Company)
                    WHERE c.symbol IN $symbols
                    OPTIONAL MATCH (c)-[r:COMPETES_WITH]-(comp)
                    OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Sector)
                    RETURN c.symbol as symbol,
                           count(DISTINCT r) as competitor_count,
                           collect(DISTINCT comp.symbol) as competitors,
                           s.name as sector
                """, symbols=symbols)
                
                graph_data = {}
                for record in result:
                    symbol = record['symbol']
                    graph_data[symbol] = {
                        'competitor_count': record['competitor_count'],
                        'competitors': record['competitors'],
                        'sector': record['sector']
                    }
                
                state['graph_relationships'] = graph_data
                state['messages'].append(f"✅ Gathered graph relationships")
                
        except Exception as e:
            state['errors'].append(f"Graph gathering failed: {str(e)}")
            state['graph_relationships'] = {}
        
        return state
    
    def _gather_news_events(self, state: IntelligenceState) -> IntelligenceState:
        """Gather recent news events from Neo4j."""
        symbols = state['symbols']
        
        try:
            with self.neo4j_driver.session() as session:
                # Get recent events
                result = session.run("""
                    MATCH (c:Company)-[:MENTIONED_IN]->(e:Event)
                    WHERE c.symbol IN $symbols
                    AND e.timestamp > datetime() - duration('P1D')
                    RETURN e.title as title,
                           e.sentiment_score as sentiment,
                           e.timestamp as timestamp,
                           collect(c.symbol) as affected_companies
                    ORDER BY e.timestamp DESC
                    LIMIT 50
                """, symbols=symbols)
                
                events = []
                for record in result:
                    events.append({
                        'title': record['title'],
                        'sentiment': record['sentiment'],
                        'timestamp': record['timestamp'],
                        'companies': record['affected_companies']
                    })
                
                state['news_events'] = events
                state['messages'].append(f"✅ Gathered {len(events)} news events")
                
        except Exception as e:
            state['errors'].append(f"News gathering failed: {str(e)}")
            state['news_events'] = []
        
        return state
    
    # ================================================================
    # Analysis Agents (Claude-Powered)
    # ================================================================
    
    def _detect_patterns(self, state: IntelligenceState) -> IntelligenceState:
        """
        Claude analyzes price patterns.
        """
        price_data = state['price_data']
        
        if not price_data:
            state['patterns_identified'] = []
            return state
        
        try:
            # Prepare data summary for Claude
            data_summary = {}
            for symbol, prices in price_data.items():
                if len(prices) >= 2:
                    latest = prices[0]['close']
                    oldest = prices[-1]['close']
                    change = ((latest - oldest) / oldest) * 100
                    avg_volume = sum(p['volume'] for p in prices) / len(prices)
                    
                    data_summary[symbol] = {
                        'price_change': round(change, 2),
                        'current_price': latest,
                        'avg_volume': int(avg_volume),
                        'data_points': len(prices)
                    }
            
            prompt = f"""Analyze these market patterns:

{json.dumps(data_summary, indent=2)}

Identify significant patterns:
1. Strong trends (>5% move)
2. Unusual volume
3. Correlation clusters
4. Reversals or breakouts

Return JSON:
[
  {{
    "pattern": "strong_uptrend",
    "symbols": ["AAPL", "MSFT"],
    "confidence": 0.85,
    "description": "Tech sector showing coordinated strength"
  }},
  ...
]"""
            
            response = self.claude.invoke([
                SystemMessage(content="You are a quantitative analyst. Return ONLY JSON array."),
                HumanMessage(content=prompt)
            ])
            
            patterns = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            state['patterns_identified'] = patterns if isinstance(patterns, list) else []
            
            state['messages'].append(f"✅ Claude identified {len(state['patterns_identified'])} patterns")
            logger.info(f"✅ Patterns: {len(state['patterns_identified'])} found")
            
        except Exception as e:
            state['errors'].append(f"Pattern detection failed: {str(e)}")
            state['patterns_identified'] = []
        
        return state
    
    def _find_correlations(self, state: IntelligenceState) -> IntelligenceState:
        """Claude analyzes correlations and relationships."""
        price_data = state['price_data']
        graph_data = state['graph_relationships']
        
        try:
            prompt = f"""Analyze these relationships:

Price Performance:
{json.dumps({k: v[0] if v else {} for k, v in price_data.items()}, indent=2)}

Graph Relationships:
{json.dumps(graph_data, indent=2)}

Find:
1. Companies moving together (correlation)
2. Sector patterns
3. Competitive dynamics
4. Unexpected relationships

Return JSON array of findings."""
            
            response = self.claude.invoke([
                SystemMessage(content="You are a correlation analyst. Return ONLY JSON."),
                HumanMessage(content=prompt)
            ])
            
            correlations = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            state['correlations_found'] = correlations if isinstance(correlations, list) else []
            
            state['messages'].append(f"✅ Found {len(state['correlations_found'])} correlations")
            
        except Exception as e:
            state['errors'].append(f"Correlation analysis failed: {str(e)}")
            state['correlations_found'] = []
        
        return state
    
    def _assess_risks(self, state: IntelligenceState) -> IntelligenceState:
        """Claude assesses market risks."""
        news = state.get('news_events', [])
        patterns = state.get('patterns_identified', [])
        
        try:
            prompt = f"""Assess market risks based on:

Patterns: {json.dumps(patterns, indent=2)}
Recent News: {json.dumps(news[:10], indent=2)}

Identify:
1. Elevated risks
2. Market vulnerabilities
3. Systemic concerns
4. Company-specific threats

Return JSON:
[
  {{
    "risk_type": "volatility_spike",
    "affected": ["TSLA"],
    "severity": 0.7,
    "description": "Increased price volatility suggesting uncertainty"
  }},
  ...
]"""
            
            response = self.claude.invoke([
                SystemMessage(content="You are a risk analyst. Return ONLY JSON."),
                HumanMessage(content=prompt)
            ])
            
            risks = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            state['risks_detected'] = risks if isinstance(risks, list) else []
            
            state['messages'].append(f"✅ Assessed {len(state['risks_detected'])} risks")
            
        except Exception as e:
            state['errors'].append(f"Risk assessment failed: {str(e)}")
            state['risks_detected'] = []
        
        return state
    
    def _identify_opportunities(self, state: IntelligenceState) -> IntelligenceState:
        """Claude identifies investment opportunities."""
        patterns = state.get('patterns_identified', [])
        correlations = state.get('correlations_found', [])
        company_profiles = state.get('company_profiles', {})
        
        try:
            prompt = f"""Based on this analysis, identify investment opportunities:

Market Patterns: {json.dumps(patterns, indent=2)}
Correlations: {json.dumps(correlations[:5], indent=2)}
Company Data: {json.dumps({k: v.get('sector') for k, v in company_profiles.items()}, indent=2)}

Find:
1. Undervalued stocks
2. Sector rotations
3. Momentum plays
4. Mean reversion setups
5. Event-driven opportunities

Return JSON:
[
  {{
    "opportunity_type": "sector_rotation",
    "symbols": ["XYZ"],
    "confidence": 0.75,
    "rationale": "Tech showing relative strength vs financials",
    "time_horizon": "short_term"
  }},
  ...
]"""
            
            response = self.claude.invoke([
                SystemMessage(content="You are an investment strategist. Return ONLY JSON."),
                HumanMessage(content=prompt)
            ])
            
            opportunities = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            state['opportunities_found'] = opportunities if isinstance(opportunities, list) else []
            
            state['messages'].append(f"✅ Identified {len(state['opportunities_found'])} opportunities")
            
        except Exception as e:
            state['errors'].append(f"Opportunity identification failed: {str(e)}")
            state['opportunities_found'] = []
        
        return state
    
    # ================================================================
    # Synthesis Agent (Claude Reasoning)
    # ================================================================
    
    def _synthesize_insights(self, state: IntelligenceState) -> IntelligenceState:
        """
        Claude synthesizes all analyses into coherent insights.
        """
        try:
            prompt = f"""Synthesize these analyses into key insights:

Patterns Identified:
{json.dumps(state.get('patterns_identified', []), indent=2)}

Correlations Found:
{json.dumps(state.get('correlations_found', []), indent=2)}

Risks Detected:
{json.dumps(state.get('risks_detected', []), indent=2)}

Opportunities Found:
{json.dumps(state.get('opportunities_found', []), indent=2)}

Generate 5-7 key insights that:
1. Connect different analyses
2. Provide actionable intelligence
3. Highlight what matters most
4. Suggest next steps

Return JSON:
[
  "Insight 1: Market showing...",
  "Insight 2: Tech sector...",
  ...
]"""
            
            response = self.claude.invoke([
                SystemMessage(content="You are a senior market strategist. Provide clear, actionable insights."),
                HumanMessage(content=prompt)
            ])
            
            insights = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
            state['key_insights'] = insights if isinstance(insights, list) else []
            
            # Generate recommendations
            state['recommendations'] = [
                {
                    'action': 'monitor',
                    'symbols': state['symbols'],
                    'rationale': insight
                }
                for insight in state['key_insights'][:3]
            ]
            
            state['confidence'] = 0.80
            state['messages'].append(f"✅ Synthesized {len(state['key_insights'])} key insights")
            
        except Exception as e:
            state['errors'].append(f"Synthesis failed: {str(e)}")
            state['key_insights'] = []
            state['recommendations'] = []
            state['confidence'] = 0.0
        
        return state
    
    def _generate_report(self, state: IntelligenceState) -> IntelligenceState:
        """Generate professional intelligence report."""
        state['report'] = {
            'generated_at': datetime.now().isoformat(),
            'analysis_type': state.get('analysis_type', 'market_overview'),
            'timeframe': state.get('timeframe', '1d'),
            'symbols_analyzed': state['symbols'],
            
            'data_summary': {
                'price_points': sum(len(p) for p in state.get('price_data', {}).values()),
                'companies_profiled': len(state.get('company_profiles', {})),
                'relationships': len(state.get('graph_relationships', {})),
                'news_events': len(state.get('news_events', []))
            },
            
            'analysis_results': {
                'patterns': state.get('patterns_identified', []),
                'correlations': state.get('correlations_found', []),
                'risks': state.get('risks_detected', []),
                'opportunities': state.get('opportunities_found', [])
            },
            
            'intelligence': {
                'key_insights': state.get('key_insights', []),
                'recommendations': state.get('recommendations', []),
                'confidence': state.get('confidence', 0.0)
            },
            
            'metadata': {
                'workflow_messages': state.get('messages', []),
                'errors': state.get('errors', [])
            }
        }
        
        # Generate executive summary
        insights = state.get('key_insights', [])
        summary = f"""
MARKET INTELLIGENCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symbols: {', '.join(state['symbols'])}

KEY INSIGHTS:
{chr(10).join(f"{i+1}. {insight}" for i, insight in enumerate(insights))}

CONFIDENCE: {state.get('confidence', 0.0):.0%}
"""
        
        state['summary'] = summary
        state['messages'].append("✅ Generated intelligence report")
        
        return state
    
    # ================================================================
    # Public API
    # ================================================================
    
    async def analyze_market(
        self,
        symbols: List[str],
        analysis_type: str = 'market_overview',
        timeframe: str = '1d'
    ) -> Dict[str, Any]:
        """
        Run complete intelligence analysis.
        
        Args:
            symbols: Stock symbols to analyze
            analysis_type: Type of analysis
            timeframe: Data timeframe
            
        Returns:
            Complete intelligence report
        """
        initial_state: IntelligenceState = {
            'analysis_type': analysis_type,
            'symbols': symbols,
            'timeframe': timeframe,
            'price_data': {},
            'company_profiles': {},
            'graph_relationships': {},
            'news_events': [],
            'patterns_identified': [],
            'correlations_found': [],
            'risks_detected': [],
            'opportunities_found': [],
            'key_insights': [],
            'recommendations': [],
            'confidence': 0.0,
            'report': {},
            'summary': '',
            'messages': [],
            'errors': []
        }
        
        # Run workflow
        result = self.app.invoke(initial_state)
        
        return result['report']
    
    async def stream_insights(self, symbols: List[str], interval_seconds: int = 60):
        """
        Continuously generate and stream insights.
        
        Perfect for: Real-time dashboards, WebSocket feeds
        """
        logger.info(f"Starting continuous intelligence stream for {symbols}")
        
        while True:
            try:
                # Generate analysis
                report = await self.analyze_market(symbols)
                
                # Log insights
                for insight in report['intelligence']['key_insights']:
                    logger.info(f"💡 {insight}")
                
                # Yield for streaming
                yield report
                
                # Wait for next cycle
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Stopping intelligence stream")
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def close(self):
        """Cleanup resources."""
        self.neo4j_driver.close()


# ================================================================
# Main (Demo/Testing)
# ================================================================

async def main():
    """Demo: Generate intelligence report."""
    service = IntelligenceSynthesisService()
    
    # Analyze current holdings
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATING MARKET INTELLIGENCE")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"{'='*60}\n")
    
    try:
        report = await service.analyze_market(symbols, timeframe='1d')
        
        # Print summary
        print("\n" + report['summary'])
        
        # Print detailed insights
        print("\nDETAILED ANALYSIS:")
        print(f"  Patterns: {len(report['analysis_results']['patterns'])}")
        print(f"  Risks: {len(report['analysis_results']['risks'])}")
        print(f"  Opportunities: {len(report['analysis_results']['opportunities'])}")
        
        print(f"\nData Analyzed:")
        print(f"  Price Points: {report['data_summary']['price_points']}")
        print(f"  Companies: {report['data_summary']['companies_profiled']}")
        print(f"  News Events: {report['data_summary']['news_events']}")
        
        print(f"\nFull report available as JSON")
        
    finally:
        service.close()


if __name__ == '__main__':
    asyncio.run(main())