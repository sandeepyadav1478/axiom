# 🎯 LangGraph + Neo4j Multi-Pipeline Architecture

## Vision: Graph-Powered Quantitative Intelligence

Transform raw market data into an **intelligent knowledge graph** using LangGraph orchestration.

---

## 🏗️ Proposed Architecture

### Multi-Container Pipeline System (6 Specialized Pipelines)

```
┌─────────────────────────────────────────────────────────────────┐
│              LANGGRAPH ORCHESTRATION LAYER                      │
│         (Coordinates all pipelines intelligently)               │
└─────────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┬──────────────────┐
        │                 │                 │                  │
        ▼                 ▼                 ▼                  ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   PRICE     │   │  COMPANY    │   │ MARKET      │   │  OPTIONS    │
│  INGESTION  │   │   GRAPH     │   │  EVENTS     │   │   CHAIN     │
│             │   │   BUILDER   │   │  TRACKER    │   │  INGESTION  │
├─────────────┤   ├─────────────┤   ├─────────────┤   ├─────────────┤
│ Real-time   │   │ Fundamental │   │ Earnings,   │   │ Greeks,     │
│ OHLCV data  │   │ Relationships│   │ News, Fed   │   │ IV Surface  │
│ 60s cycle   │   │ Sector links│   │ 5min cycle  │   │ 1min cycle  │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
        │                 │                 │                  │
        └─────────────────┼─────────────────┴──────────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │      NEO4J GRAPH DB      │
              │   (Central Knowledge)    │
              ├──────────────────────────┤
              │ Nodes:                   │
              │ • Stock                  │
              │ • Company                │
              │ • Sector                 │
              │ • MarketEvent            │
              │ • Option                 │
              │ • Earnings               │
              │                          │
              │ Relationships:           │
              │ • BELONGS_TO_SECTOR      │
              │ • COMPETES_WITH          │
              │ • SUPPLIES_TO            │
              │ • AFFECTED_BY_EVENT      │
              │ • HAS_OPTION             │
              └──────────────────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │   LANGGRAPH ANALYSIS     │
              │                          │
              │ • Pattern detection      │
              │ • Correlation analysis   │
              │ • Risk propagation       │
              │ • Trading signals        │
              └──────────────────────────┘
```

---

## 📦 6 Specialized Pipeline Containers

### 1. **axiom-pipeline-prices** (Real-time Prices)
```yaml
Purpose: Ingest real-time OHLCV data
Sources: yfinance, Polygon, Finnhub
Interval: 60 seconds
Neo4j: Stock nodes, price updates
```

**Graph Pattern**:
```cypher
CREATE (s:Stock {
  symbol: 'AAPL',
  name: 'Apple Inc.',
  last_price: 150.25,
  last_updated: datetime()
})
```

### 2. **axiom-pipeline-companies** (Company Knowledge Graph)
```yaml
Purpose: Build comprehensive company relationship graph
Sources: Alpha Vantage, FMP, SEC filings
Interval: Daily (3600s)
Neo4j: Company nodes, sector relationships, supply chains
```

**Graph Pattern**:
```cypher
// Company node
CREATE (c:Company {
  symbol: 'AAPL',
  name: 'Apple Inc.',
  sector: 'Technology',
  industry: 'Consumer Electronics',
  market_cap: 2500000000000,
  employees: 164000
})

// Sector relationship
MATCH (c:Company {symbol: 'AAPL'})
MERGE (s:Sector {name: 'Technology'})
CREATE (c)-[:BELONGS_TO]->(s)

// Competitor relationships
MATCH (aapl:Company {symbol: 'AAPL'})
MATCH (msft:Company {symbol: 'MSFT'})
CREATE (aapl)-[:COMPETES_WITH {intensity: 0.8}]->(msft)

// Supply chain
MATCH (aapl:Company {symbol: 'AAPL'})
MATCH (tsmc:Company {symbol: 'TSM'})
CREATE (tsmc)-[:SUPPLIES_TO {product: 'chips'}]->(aapl)
```

### 3. **axiom-pipeline-events** (Market Events Tracker)
```yaml
Purpose: Track market-moving events
Sources: News APIs, Earnings calendars, Fed announcements
Interval: 5 minutes
Neo4j: MarketEvent nodes, event-stock relationships
```

**Graph Pattern**:
```cypher
// Earnings event
CREATE (e:MarketEvent {
  type: 'earnings',
  company: 'AAPL',
  date: date('2025-11-15'),
  eps_actual: 1.55,
  eps_estimate: 1.50,
  surprise: 0.05
})

// Link to stock
MATCH (s:Stock {symbol: 'AAPL'})
MATCH (e:MarketEvent {company: 'AAPL'})
CREATE (s)-[:HAS_EVENT]->(e)

// Fed announcement
CREATE (f:MarketEvent {
  type: 'fed_decision',
  date: date('2025-11-07'),
  decision: 'rate_cut',
  magnitude: 0.25
})

// Affects multiple stocks
MATCH (tech:Sector {name: 'Technology'})
MATCH (f:MarketEvent {type: 'fed_decision'})
CREATE (tech)-[:AFFECTED_BY {impact: 0.7}]->(f)
```

### 4. **axiom-pipeline-options** (Options Chain)
```yaml
Purpose: Ingest options data and Greeks
Sources: Polygon, Tradier
Interval: 60 seconds (active trading hours)
Neo4j: Option nodes, strike chains, volatility surface
```

**Graph Pattern**:
```cypher
// Option contract
CREATE (o:Option {
  underlying: 'AAPL',
  type: 'call',
  strike: 150,
  expiry: date('2025-12-20'),
  price: 5.25,
  volume: 15000,
  open_interest: 50000,
  implied_vol: 0.25,
  delta: 0.55,
  gamma: 0.02,
  theta: -0.05,
  vega: 0.12
})

// Link to underlying
MATCH (s:Stock {symbol: 'AAPL'})
MATCH (o:Option {underlying: 'AAPL'})
CREATE (s)-[:HAS_OPTION]->(o)

// Volatility surface
MATCH (s:Stock {symbol: 'AAPL'})
CREATE (v:VolatilitySurface {
  symbol: 'AAPL',
  timestamp: datetime(),
  surface_data: {...}  # JSON blob
})
CREATE (s)-[:HAS_VOL_SURFACE]->(v)
```

### 5. **axiom-pipeline-correlations** (Relationship Analyzer)
```yaml
Purpose: Calculate and update correlations
Sources: Internal (from PostgreSQL price_data)
Interval: 1 hour
Neo4j: Correlation edges, correlation strength
```

**Graph Pattern**:
```cypher
// Price correlation
MATCH (s1:Stock {symbol: 'AAPL'})
MATCH (s2:Stock {symbol: 'MSFT'})
CREATE (s1)-[:CORRELATED_WITH {
  coefficient: 0.85,
  period_days: 30,
  calculated_at: datetime()
}]->(s2)

// Volatility correlation
CREATE (s1)-[:VOL_CORRELATED {
  coefficient: 0.72,
  period_days: 30
}]->(s2)
```

### 6. **axiom-pipeline-risk-graph** (Risk Propagation)
```yaml
Purpose: Build risk propagation network
Sources: Internal analysis + external risk factors
Interval: 1 hour
Neo4j: Risk nodes, propagation edges
```

**Graph Pattern**:
```cypher
// Risk factor
CREATE (r:RiskFactor {
  type: 'geopolitical',
  region: 'China',
  severity: 0.7,
  description: 'Trade tensions'
})

// Risk propagation
MATCH (r:RiskFactor {type: 'geopolitical', region: 'China'})
MATCH (aapl:Company {symbol: 'AAPL'})
CREATE (aapl)-[:EXPOSED_TO {
  exposure: 0.35,  # 35% revenue from China
  channel: 'supply_chain'
}]->(r)
```

---

## 🧠 LangGraph Integration

### LangGraph Workflows (Each Pipeline Uses LangGraph)

#### Example: Company Graph Builder Pipeline

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# Define the graph workflow
class CompanyGraphState(TypedDict):
    symbol: str
    company_data: Dict
    relationships: List[Dict]
    neo4j_queries: List[str]
    status: str

# Build the workflow
workflow = StateGraph(CompanyGraphState)

# Nodes (LangGraph agents)
workflow.add_node("fetch_company_data", fetch_company_agent)
workflow.add_node("extract_relationships", relationship_agent)
workflow.add_node("build_graph_queries", cypher_generator_agent)
workflow.add_node("execute_neo4j", neo4j_executor_agent)
workflow.add_node("validate_graph", validation_agent)

# Edges (workflow flow)
workflow.add_edge("fetch_company_data", "extract_relationships")
workflow.add_edge("extract_relationships", "build_graph_queries")
workflow.add_edge("build_graph_queries", "execute_neo4j")
workflow.add_edge("execute_neo4j", "validate_graph")
workflow.add_edge("validate_graph", END)

# Set entry point
workflow.set_entry_point("fetch_company_data")

# Compile
graph = workflow.compile()

# Run for each company
for symbol in symbols:
    result = graph.invoke({
        "symbol": symbol,
        "status": "pending"
    })
```

#### LangGraph Agents for Each Pipeline:

**Pipeline 1: Prices**
- Agent 1: Fetch prices from multiple sources
- Agent 2: Validate and clean data
- Agent 3: Store in PostgreSQL
- Agent 4: Update Neo4j stock nodes
- Agent 5: Cache in Redis

**Pipeline 2: Company Graph**
- Agent 1: Fetch company fundamentals
- Agent 2: Extract key relationships (competitors, suppliers, customers)
- Agent 3: Generate Cypher queries
- Agent 4: Execute graph updates
- Agent 5: Validate graph consistency

**Pipeline 3: Market Events**
- Agent 1: Scrape news and events
- Agent 2: Classify event type and impact
- Agent 3: Link events to affected companies
- Agent 4: Create event nodes in Neo4j
- Agent 5: Calculate propagation paths

**Pipeline 4: Options**
- Agent 1: Fetch options chain
- Agent 2: Calculate Greeks
- Agent 3: Build volatility surface
- Agent 4: Create option nodes
- Agent 5: Link to underlying stocks

**Pipeline 5: Correlations**
- Agent 1: Query historical prices from PostgreSQL
- Agent 2: Calculate correlation matrices
- Agent 3: Identify significant correlations (>0.7)
- Agent 4: Create CORRELATED_WITH edges
- Agent 5: Update correlation strength over time

**Pipeline 6: Risk Graph**
- Agent 1: Identify risk factors
- Agent 2: Calculate company exposures
- Agent 3: Build risk propagation model
- Agent 4: Create risk nodes and edges
- Agent 5: Run graph algorithms (PageRank, centrality)

---

## 🎯 Neo4j-Centric Graph Model

### Complete Graph Schema

```cypher
// ========================================
// NODE TYPES
// ========================================

// 1. Stock (real-time prices)
(:Stock {
  symbol: string,
  name: string,
  exchange: string,
  last_price: float,
  last_volume: int,
  last_updated: datetime
})

// 2. Company (fundamentals)
(:Company {
  symbol: string,
  name: string,
  sector: string,
  industry: string,
  market_cap: float,
  employees: int,
  revenue: float,
  net_income: float,
  debt_to_equity: float,
  pe_ratio: float,
  pb_ratio: float,
  dividend_yield: float
})

// 3. Sector
(:Sector {
  name: string,
  description: string,
  total_market_cap: float,
  company_count: int
})

// 4. MarketEvent
(:MarketEvent {
  type: string,  // 'earnings', 'fed', 'merger', 'lawsuit'
  date: date,
  company: string,  // optional
  impact_score: float,
  description: text
})

// 5. Option
(:Option {
  underlying: string,
  type: string,  // 'call' or 'put'
  strike: float,
  expiry: date,
  price: float,
  implied_vol: float,
  delta: float,
  gamma: float,
  theta: float,
  vega: float,
  volume: int,
  open_interest: int
})

// 6. RiskFactor
(:RiskFactor {
  type: string,  // 'geopolitical', 'regulatory', 'economic'
  region: string,
  severity: float,  // 0-1
  description: text,
  probability: float
})

// 7. Analyst (for ratings/price targets)
(:Analyst {
  firm: string,
  analyst_name: string,
  accuracy_score: float,
  specialization: string
})

// 8. Pattern (technical patterns)
(:Pattern {
  pattern_type: string,  // 'head_shoulders', 'double_top'
  symbol: string,
  detected_at: datetime,
  confidence: float,
  target_price: float
})

// ========================================
// RELATIONSHIP TYPES
// ========================================

// Price/Company relationships
(:Stock)-[:IS]-(:Company)

// Sector relationships
(:Company)-[:BELONGS_TO]-(:Sector)
(:Sector)-[:PART_OF]-(:Sector)  // Sub-sectors

// Competitive relationships
(:Company)-[:COMPETES_WITH {intensity: float}]-(:Company)
(:Company)-[:MARKET_LEADER_IN]-(:Sector)

// Supply chain
(:Company)-[:SUPPLIES_TO {product: string, revenue_pct: float}]-(:Company)
(:Company)-[:CUSTOMER_OF]-(:Company)

// Market events
(:Stock)-[:HAD_EVENT]-(:MarketEvent)
(:Company)-[:AFFECTED_BY {impact: float}]-(:MarketEvent)
(:Sector)-[:IMPACTED_BY]-(:MarketEvent)

// Options
(:Stock)-[:HAS_OPTION]-(:Option)
(:Option)-[:PART_OF_STRATEGY {strategy: string}]-(:Option)

// Risk
(:Company)-[:EXPOSED_TO {exposure: float}]-(:RiskFactor)
(:RiskFactor)-[:PROPAGATES_TO {strength: float}]-(:RiskFactor)

// Correlations
(:Stock)-[:CORRELATED_WITH {coefficient: float, period: int}]-(:Stock)
(:Stock)-[:VOL_CORRELATED {coefficient: float}]-(:Stock)

// Analyst coverage
(:Analyst)-[:COVERS]-(:Stock)
(:Analyst)-[:ISSUED_RATING {
  rating: string,
  price_target: float,
  date: date
}]-(:Stock)

// Technical patterns
(:Stock)-[:EXHIBITS_PATTERN]-(:Pattern)
```

---

## 🔄 Workflow with LangGraph

### Master Orchestration Graph

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class MarketIntelligenceState(TypedDict):
    symbols: List[str]
    prices_fetched: bool
    graph_updated: bool
    events_processed: bool
    analysis_complete: bool
    recommendations: List[Dict]

# Main orchestration graph
orchestrator = StateGraph(MarketIntelligenceState)

# Add pipeline nodes
orchestrator.add_node("ingest_prices", prices_pipeline)
orchestrator.add_node("build_company_graph", company_pipeline)
orchestrator.add_node("track_events", events_pipeline)
orchestrator.add_node("analyze_options", options_pipeline)
orchestrator.add_node("calculate_correlations", correlation_pipeline)
orchestrator.add_node("assess_risk", risk_pipeline)
orchestrator.add_node("generate_insights", insight_agent)

# Define workflow
orchestrator.add_edge("ingest_prices", "build_company_graph")
orchestrator.add_edge("build_company_graph", "track_events")
orchestrator.add_edge("track_events", "analyze_options")
orchestrator.add_edge("analyze_options", "calculate_correlations")
orchestrator.add_edge("calculate_correlations", "assess_risk")
orchestrator.add_edge("assess_risk", "generate_insights")
orchestrator.add_edge("generate_insights", END)

orchestrator.set_entry_point("ingest_prices")

# Compile with memory
checkpointer = MemorySaver()
graph = orchestrator.compile(checkpointer=checkpointer)

# Run continuously
while True:
    result = graph.invoke(
        {"symbols": symbols},
        config={"configurable": {"thread_id": "market_intelligence"}}
    )
    await asyncio.sleep(60)
```

---

## 🎨 Neo4j Graph Queries (LangGraph-Powered)

### Query 1: Find Trading Opportunities

```cypher
// LangGraph agent generates this based on market conditions
MATCH (s:Stock)-[:CORRELATED_WITH {coefficient: >0.8}]->(corr:Stock)
WHERE s.last_price < s.moving_avg_50
  AND corr.last_price > corr.moving_avg_50
MATCH (s)-[:HAS_EVENT]->(e:MarketEvent {type: 'earnings'})
WHERE e.surprise > 0.1
RETURN s.symbol, s.last_price, 
       collect(corr.symbol) as correlated_stocks,
       e.eps_actual - e.eps_estimate as earnings_surprise
ORDER BY earnings_surprise DESC
LIMIT 10
```

### Query 2: Risk Propagation Analysis

```cypher
// LangGraph agent traces risk through supply chains
MATCH path = (risk:RiskFactor)-[:PROPAGATES_TO*1..3]-(company:Company)
WHERE risk.severity > 0.7
RETURN path, 
       length(path) as propagation_depth,
       company.symbol,
       reduce(exposure = 1.0, rel in relationships(path) | 
         exposure * rel.strength
       ) as total_exposure
ORDER BY total_exposure DESC
```

### Query 3: Sector Impact Analysis

```cypher
// LangGraph determines which sectors are most affected
MATCH (event:MarketEvent {type: 'fed_decision'})
MATCH (sector:Sector)<-[:BELONGS_TO]-(company:Company)
MATCH (company)-[:EXPOSED_TO]->(risk:RiskFactor)
WHERE risk.type = 'interest_rate'
RETURN sector.name,
       count(company) as companies_affected,
       avg(company.debt_to_equity) as avg_leverage,
       sum(company.market_cap) as total_market_cap
ORDER BY avg_leverage DESC
```

---

## 🚀 Implementation Roadmap

### Phase 1: Multi-Container Foundation (Week 1)
- [ ] Create 6 pipeline containers
- [ ] Each pipeline writes to Neo4j
- [ ] Basic graph schema implemented
- [ ] All containers running stably

### Phase 2: LangGraph Integration (Week 2)
- [ ] Add LangGraph to each pipeline
- [ ] Implement agent-based data processing
- [ ] Create Cypher query generators
- [ ] Add graph validation agents

### Phase 3: Advanced Graph Features (Week 3)
- [ ] Implement graph algorithms (PageRank, Community Detection)
- [ ] Build temporal graphs (time-series relationships)
- [ ] Add graph embeddings (GNN)
- [ ] Create graph-based recommendations

### Phase 4: Production Intelligence (Week 4)
- [ ] Real-time graph queries via API
- [ ] LangGraph-powered trading signals
- [ ] Risk propagation alerts
- [ ] Portfolio optimization using graph data

---

## 📁 File Structure for Multi-Pipeline

```
axiom/pipelines/
├── docker-compose.yml              # All 6 pipelines
├── shared/
│   ├── neo4j_client.py            # Shared Neo4j utilities
│   ├── langgraph_base.py          # Base LangGraph setup
│   └── graph_schema.py            # Graph model definitions
│
├── prices/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pipeline.py                # Price ingestion
│   └── langgraph_workflow.py     # LangGraph for prices
│
├── companies/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pipeline.py                # Company graph builder
│   └── langgraph_workflow.py     # Relationship extraction
│
├── events/
│   ├── Dockerfile  
│   ├── requirements.txt
│   ├── pipeline.py                # Event tracker
│   └── langgraph_workflow.py     # Event classification
│
├── options/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pipeline.py                # Options ingestion
│   └── langgraph_workflow.py     # Greeks calculation
│
├── correlations/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pipeline.py                # Correlation analyzer
│   └── langgraph_workflow.py     # Correlation detection
│
└── risk/
    ├── Dockerfile
    ├── requirements.txt
    ├── pipeline.py                # Risk graph builder
    └── langgraph_workflow.py     # Risk propagation
```

---

## 🎯 Next Steps

### Immediate Actions:

1. **Install LangGraph**:
```bash
uv pip install langgraph langchain-anthropic
```

2. **Create Multi-Pipeline docker-compose.yml**:
```yaml
services:
  prices-pipeline:
    ...
  companies-pipeline:
    ...
  events-pipeline:
    ...
  options-pipeline:
    ...
  correlations-pipeline:
    ...
  risk-pipeline:
    ...
```

3. **Implement First LangGraph Pipeline** (Company Graph Builder):
- Fetch company data
- Extract relationships using Claude
- Generate Cypher queries
- Build Neo4j graph
- Validate with graph queries

4. **Define Neo4j Graph Schema**:
```python
# axiom/pipelines/shared/graph_schema.py
class GraphSchema:
    nodes = {
        'Stock': ['symbol', 'name', 'last_price'],
        'Company': ['symbol', 'sector', 'market_cap'],
        'Sector': ['name', 'description'],
        # ... all node types
    }
    
    relationships = {
        'BELONGS_TO': ['Company', 'Sector'],
        'CORRELATED_WITH': ['Stock', 'Stock'],
        # ... all edge types
    }
```

---

## 💎 The Vision

Transform Axiom from a **data collector** into a **knowledge engine**:

1. **Data Layer**: Multiple pipelines ingest from various sources
2. **Graph Layer**: Neo4j stores rich relationships
3. **Intelligence Layer**: LangGraph orchestrates analysis
4. **Insight Layer**: Graph queries reveal hidden patterns

**Example Use Case**:
```
Question: "Which tech stocks are most exposed to China supply chain risks?"

LangGraph Workflow:
1. Query Neo4j for tech companies
2. Follow SUPPLIES_TO relationships to China
3. Calculate exposure percentages
4. Cross-reference with risk factors
5. Rank by total exposure
6. Generate trading recommendations

Result: AAPL (35% exposure), NVDA (28% exposure), ...
```

---

## 🔮 Advanced Features (Future)

### 1. Graph Neural Networks
Train GNNs on the Neo4j graph to predict:
- Price movements based on graph structure
- Risk propagation paths
- Optimal portfolio allocations

### 2. Temporal Graphs
Track how relationships change over time:
```cypher
(:Company)-[:CORRELATED_WITH {
  coefficient: [0.8, 0.85, 0.9],  // Last 3 months
  timestamps: [date1, date2, date3]
}]-(:Company)
```

### 3. Graph Embeddings
Convert graph structure to vector space for ML models

### 4. Real-Time Graph Streaming
Update graph in real-time as events occur

---

## 📊 Success Metrics

### Pipeline Performance:
- 6 containers all healthy
- Each pipeline < 10s execution time
- 99.9% uptime

### Graph Quality:
- 1000+ nodes (stocks, companies, sectors)
- 10,000+ relationships
- < 100ms query response time
- Full graph consistency

### LangGraph Effectiveness:
- Accurate relationship extraction (>95%)
- Valid Cypher query generation (100%)
- Actionable insights generated

---

**This is the path to turning Axiom into a true AI-powered quant intelligence platform.**

Ready to build this?