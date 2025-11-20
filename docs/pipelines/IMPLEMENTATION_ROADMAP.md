# 🚀 LangGraph + Neo4j Multi-Pipeline Implementation Roadmap

## 🎯 Objective
Transform Axiom into an **AI-powered quantitative intelligence platform** using:
- **LangGraph**: Agent orchestration for data processing
- **Neo4j**: Central knowledge graph for relationships
- **Multi-container pipelines**: Specialized, scalable architecture

---

## 📅 Implementation Plan

### Phase 1: Foundation (Days 1-2)
**Goal**: Set up LangGraph + expand Neo4j schema

#### Day 1: LangGraph Setup
- [ ] Install LangGraph in environment
- [ ] Create base LangGraph agent framework
- [ ] Test simple workflow with Claude API
- [ ] Verify checkpoint/memory functionality

#### Day 2: Neo4j Schema Design
- [ ] Define comprehensive node types (Stock, Company, Sector, Event, Option, Risk)
- [ ] Define relationship types (15+ relationship patterns)
- [ ] Create schema constraints and indexes
- [ ] Build schema validation queries

**Deliverables**:
```
axiom/pipelines/shared/
├── langgraph_base.py        # Base agent framework
├── neo4j_schema.py          # Complete graph schema
└── graph_utils.py           # Cypher query helpers
```

---

### Phase 2: First Multi-Pipeline (Days 3-4)
**Goal**: Split current pipeline into 3 specialized containers

#### Container 1: Price Ingestion (Keep current)
```yaml
axiom-pipeline-prices:
  Purpose: Real-time OHLCV data
  Interval: 60 seconds
  Neo4j: Stock nodes, price updates
  LangGraph: Price validation workflow
```

#### Container 2: Company Graph Builder (NEW)
```yaml
axiom-pipeline-companies:
  Purpose: Build company relationship graph
  Interval: Daily
  Neo4j: Company nodes, sectors, relationships
  LangGraph: Relationship extraction from fundamentals
```

#### Container 3: Market Events (NEW)
```yaml
axiom-pipeline-events:
  Purpose: Track market-moving events
  Interval: 5 minutes  
  Neo4j: Event nodes, event-stock links
  LangGraph: Event classification and impact analysis
```

**Deliverables**:
```
axiom/pipelines/
├── docker-compose-multi.yml     # 3 pipeline services
├── prices/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pipeline_with_langgraph.py
├── companies/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── company_graph_builder.py
└── events/
    ├── Dockerfile
    ├── requirements.txt
    └── event_tracker.py
```

---

### Phase 3: Neo4j Graph Enrichment (Days 5-6)
**Goal**: Build rich knowledge graph with relationships

#### Implement Relationship Types:

**Day 5: Competitive & Sector Relationships**
```cypher
// Competitive landscape
(:Company)-[:COMPETES_WITH {intensity: 0.85}]-(:Company)
(:Company)-[:BELONGS_TO]-(:Sector)
(:Sector)-[:PART_OF]-(:Sector)  // Tech → Software → SaaS

// Market positioning  
(:Company)-[:MARKET_LEADER_IN {rank: 1}]-(:Sector)
```

**Day 6: Supply Chain & Events**
```cypher
// Supply chains
(:Company)-[:SUPPLIES_TO {product: 'chips', revenue_pct: 0.45}]-(:Company)

// Event impacts
(:MarketEvent)-[:AFFECTS {impact_score: 0.8}]-(:Company)
(:MarketEvent)-[:TRIGGERS]-(:MarketEvent)  // Event chains
```

**Deliverables**:
- 50+ companies in graph
- 200+ relationships
- 5+ relationship types
- Graph visualization ready

---

### Phase 4: LangGraph Workflows (Days 7-9)
**Goal**: Implement AI-powered data processing in each pipeline

#### Day 7: Company Graph LangGraph Workflow

```python
from langgraph.graph import StateGraph

# State definition
class CompanyGraphState(TypedDict):
    symbol: str
    fundamental_data: Dict
    competitors: List[str]
    suppliers: List[str]
    cypher_queries: List[str]
    graph_updates: int

# Agent nodes
workflow = StateGraph(CompanyGraphState)

workflow.add_node("fetch_fundamentals", fetch_agent)
workflow.add_node("identify_competitors", competitor_agent)  # Uses Claude
workflow.add_node("extract_supply_chain", supply_chain_agent)  # Uses Claude
workflow.add_node("generate_cypher", cypher_agent)  # Uses Claude
workflow.add_node("execute_graph_updates", neo4j_agent)

# Flow
workflow.add_edge("fetch_fundamentals", "identify_competitors")
workflow.add_edge("identify_competitors", "extract_supply_chain")
workflow.add_edge("extract_supply_chain", "generate_cypher")
workflow.add_edge("generate_cypher", "execute_graph_updates")
```

**Key Innovation**: Claude analyzes company descriptions and generates:
- Competitor lists
- Supply chain relationships
- Industry classifications
- Risk exposures

#### Day 8: Event Tracker LangGraph Workflow

```python
# Event classification agent
class EventAnalysisState(TypedDict):
    raw_news: str
    event_type: str  # earnings, merger, fed, lawsuit
    affected_companies: List[str]
    impact_score: float
    cypher_queries: List[str]

workflow = StateGraph(EventAnalysisState)

workflow.add_node("classify_event", classification_agent)  # Claude
workflow.add_node("identify_affected", impact_agent)  # Claude
workflow.add_node("calculate_severity", severity_agent)
workflow.add_node("create_event_node", neo4j_agent)
workflow.add_node("link_to_companies", relationship_agent)
```

**Key Innovation**: Claude reads news/events and:
- Classifies event types
- Identifies affected companies
- Calculates impact scores
- Generates appropriate graph relationships

#### Day 9: Options LangGraph Workflow

```python
# Options strategy identification
class OptionsAnalysisState(TypedDict):
    options_chain: List[Dict]
    volatility_surface: Dict
    identified_strategies: List[str]
    risk_metrics: Dict
    graph_queries: List[str]

workflow = StateGraph(OptionsAnalysisState)

workflow.add_node("fetch_options", options_fetcher)
workflow.add_node("calculate_greeks", greeks_calculator)
workflow.add_node("build_vol_surface", volatility_agent)
workflow.add_node("identify_strategies", strategy_agent)  # Claude
workflow.add_node("create_option_nodes", neo4j_agent)
```

**Deliverables**:
- 3 complete LangGraph workflows
- Claude-powered analysis in each pipeline
- Graph updates from AI insights

---

### Phase 5: Advanced Analytics (Days 10-12)

#### Day 10: Correlation Pipeline with LangGraph

```python
# Auto-discover meaningful correlations
workflow.add_node("fetch_price_history", price_loader)
workflow.add_node("calculate_correlations", correlation_calculator)
workflow.add_node("filter_significant", significance_agent)  # Claude decides what's meaningful
workflow.add_node("explain_correlation", explanation_agent)  # Claude explains WHY
workflow.add_node("create_correlation_edges", neo4j_agent)
```

**Key Innovation**: Claude explains why stocks are correlated:
- "AAPL-MSFT: Both tech sector, similar customer base"
- "AAPL-TSM: Supply chain dependency for chips"
- "AAPL-GOOGL: Compete in smartphone OS market"

#### Day 11: Risk Propagation Pipeline

```python
# Build risk propagation network
workflow.add_node("identify_risks", risk_scanner)
workflow.add_node("calculate_exposure", exposure_agent)  # Claude analyzes
workflow.add_node("trace_propagation", propagation_agent)
workflow.add_node("assign_severity", severity_agent)  # Claude evaluates
workflow.add_node("create_risk_graph", neo4j_agent)
```

**Key Innovation**: Claude traces how risks propagate:
- China trade risk → AAPL (35% exposure) → Suppliers (TSM, etc.)
- Interest rate risk → High-debt companies → Sector impact

#### Day 12: Graph Query Optimization

Implement intelligent graph queries:
```cypher
// Find hidden opportunities
MATCH path = (s1:Stock)-[:CORRELATED_WITH*1..2]-(s2:Stock)
WHERE s1.last_price < s1.moving_avg_50
  AND s2.last_price > s2.moving_avg_50
RETURN path, 
       s1.symbol as undervalued,
       s2.symbol as correlated_strength,
       relationships(path) as connection_type
```

---

### Phase 6: Production Deployment (Days 13-14)

#### Day 13: Container Orchestration

**Deploy all 6 pipelines**:
```bash
docker compose -f axiom/pipelines/docker-compose-multi.yml up -d

# Verify all running:
docker ps | grep pipeline
# axiom-pipeline-prices          Up (healthy)
# axiom-pipeline-companies       Up (healthy)
# axiom-pipeline-events          Up (healthy)  
# axiom-pipeline-options         Up (healthy)
# axiom-pipeline-correlations    Up (healthy)
# axiom-pipeline-risk            Up (healthy)
```

#### Day 14: Monitoring & Validation

**Create monitoring dashboard**:
```python
# axiom/pipelines/monitoring/dashboard.py
class PipelineMonitor:
    def get_status(self):
        return {
            'prices': self.check_pipeline('prices'),
            'companies': self.check_pipeline('companies'),
            'events': self.check_pipeline('events'),
            'options': self.check_pipeline('options'),
            'correlations': self.check_pipeline('correlations'),
            'risk': self.check_pipeline('risk'),
            'graph_stats': self.get_neo4j_stats()
        }
    
    def get_neo4j_stats(self):
        return {
            'total_nodes': count_nodes(),
            'total_relationships': count_relationships(),
            'node_types': get_node_type_counts(),
            'relationship_types': get_relationship_type_counts(),
            'query_performance': benchmark_queries()
        }
```

**Deliverables**:
- 6 pipeline containers running
- Monitoring dashboard
- Graph health checks
- Performance benchmarks

---

## 🎓 Learning Path

### Week 1: LangGraph Fundamentals
```python
# Learn LangGraph basics
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Simple example
def analyze(state):
    # Use Claude to analyze data
    response = claude_client.messages.create(...)
    return {"analysis": response.content}

graph = StateGraph(dict)
graph.add_node("analyze", analyze)
graph.set_entry_point("analyze")
graph.set_finish_point("analyze")

app = graph.compile(checkpointer=MemorySaver())
result = app.invoke({"data": "AAPL financial data"})
```

### Week 2: Neo4j Graph Modeling
```cypher
// Practice building relationships
MATCH (aapl:Company {symbol: 'AAPL'})
MATCH (tech:Sector {name: 'Technology'})
CREATE (aapl)-[:BELONGS_TO]->(tech)

// Practice graph queries
MATCH path = (s:Stock)-[:CORRELATED_WITH*1..2]-(other:Stock)
RETURN path
LIMIT 10
```

### Week 3: Integration
- Combine LangGraph workflows with Neo4j updates
- Build end-to-end pipeline: Data → LangGraph → Neo4j → Insights

---

## 💡 Quick Win: First LangGraph Pipeline

**Let's start with Company Graph Builder** (simplest to implement):

### Step-by-Step Implementation:

```python
# axiom/pipelines/companies/company_graph_builder.py

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
import os

# Initialize Claude
claude = ChatAnthropic(
    model="claude-sonnet-4",
    api_key=os.getenv('ANTHROPIC_API_KEY')
)

# State
class CompanyState(TypedDict):
    symbol: str
    company_data: Dict
    competitors: List[str]
    cypher_queries: List[str]

# Agent 1: Fetch company data
def fetch_company(state):
    import yfinance as yf
    ticker = yf.Ticker(state['symbol'])
    state['company_data'] = ticker.info
    return state

# Agent 2: Identify competitors (CLAUDE MAGIC)
def identify_competitors(state):
    prompt = f"""
    Company: {state['company_data'].get('longName')}
    Sector: {state['company_data'].get('sector')}
    Industry: {state['company_data'].get('industry')}
    
    Identify the top 5 direct competitors. Return ONLY stock symbols, comma-separated.
    """
    
    response = claude.invoke(prompt)
    competitors = response.content.strip().split(',')
    state['competitors'] = [c.strip() for c in competitors]
    return state

# Agent 3: Generate Cypher queries (CLAUDE MAGIC)
def generate_cypher(state):
    prompt = f"""
    Generate Neo4j Cypher queries to:
    1. Create Company node for {state['symbol']}
    2. Link to Sector
    3. Create COMPETES_WITH relationships to: {state['competitors']}
    
    Use this data: {state['company_data']}
    Return ONLY valid Cypher queries, one per line.
    """
    
    response = claude.invoke(prompt)
    state['cypher_queries'] = response.content.strip().split('\n')
    return state

# Agent 4: Execute in Neo4j
def execute_neo4j(state):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    
    with driver.session() as session:
        for query in state['cypher_queries']:
            if query.strip():
                session.run(query)
    
    return state

# Build workflow
workflow = StateGraph(CompanyState)

workflow.add_node("fetch", fetch_company)
workflow.add_node("analyze", identify_competitors)
workflow.add_node("generate", generate_cypher)
workflow.add_node("execute", execute_neo4j)

workflow.add_edge("fetch", "analyze")
workflow.add_edge("analyze", "generate")
workflow.add_edge("generate", "execute")
workflow.add_edge("execute", END)

workflow.set_entry_point("fetch")

# Compile and run
app = workflow.compile()

# Process all companies
for symbol in ['AAPL', 'MSFT', 'GOOGL']:
    result = app.invoke({"symbol": symbol})
    print(f"✅ Built graph for {symbol}")
```

**This single pipeline demonstrates**:
1. ✅ LangGraph orchestration
2. ✅ Claude-powered intelligence
3. ✅ Neo4j graph updates
4. ✅ Multi-step workflow

---

## 🔧 Technical Implementation Details

### Docker Compose Configuration (Multi-Pipeline)

```yaml
# axiom/pipelines/docker-compose-langgraph.yml
version: '3.8'

services:
  # Pipeline 1: Prices (already working)
  prices:
    build:
      context: ../..
      dockerfile: axiom/pipelines/prices/Dockerfile
    container_name: axiom-pipeline-prices
    environment:
      - SYMBOLS=AAPL,MSFT,GOOGL,TSLA,NVDA
      - INTERVAL=60
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    networks:
      - axiom_network
      - database_axiom_network
    restart: unless-stopped

  # Pipeline 2: Company Graph Builder (NEW)
  companies:
    build:
      context: ../..
      dockerfile: axiom/pipelines/companies/Dockerfile
    container_name: axiom-pipeline-companies
    environment:
      - SYMBOLS=AAPL,MSFT,GOOGL,TSLA,NVDA,META,AMZN,NFLX
      - INTERVAL=86400  # Daily
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - NEO4J_URI=bolt://neo4j:7687
    networks:
      - axiom_network
      - database_axiom_network
    depends_on:
      - prices
    restart: unless-stopped

  # Pipeline 3: Market Events (NEW)
  events:
    build:
      context: ../..
      dockerfile: axiom/pipelines/events/Dockerfile
    container_name: axiom-pipeline-events
    environment:
      - INTERVAL=300  # 5 minutes
      - NEWS_SOURCES=alpha_vantage,finnhub
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    networks:
      - axiom_network
      - database_axiom_network
    restart: unless-stopped

  # Pipeline 4: Options Chain (Future - Phase 2)
  # options:
  #   ...

  # Pipeline 5: Correlations (Future - Phase 2)
  # correlations:
  #   ...

  # Pipeline 6: Risk Graph (Future - Phase 2)
  # risk:
  #   ...

networks:
  axiom_network:
    external: true
    name: axiom-mcp-network
  database_axiom_network:
    external: true
    name: database_axiom_network
```

---

## 📊 Expected Neo4j Graph After Phase 3

### Nodes:
```
Stock:           10-20 nodes
Company:         10-20 nodes  
Sector:          5-10 nodes
MarketEvent:     50-100 nodes (historical events)
Option:          100-500 nodes (active options)
RiskFactor:      10-20 nodes
Analyst:         20-50 nodes
Pattern:         10-30 nodes (technical patterns)

TOTAL:           ~200-750 nodes
```

### Relationships:
```
IS:                     10-20 (Stock-Company)
BELONGS_TO:            10-20 (Company-Sector)
COMPETES_WITH:         50-100 (Company-Company)
SUPPLIES_TO:           30-50 (Supply chain)
HAS_EVENT:             100-200 (Stock-Event)
AFFECTED_BY:           200-400 (Event impacts)
CORRELATED_WITH:       50-100 (Price correlations)
HAS_OPTION:            100-500 (Stock-Option)
EXPOSED_TO:            50-100 (Company-Risk)

TOTAL:                 ~600-1500 relationships
```

### Graph Density:
- Average degree: 6-8 relationships per node
- Maximum path length: 3-4 (small world network)
- Community detection: 5-7 clusters (sectors)

---

## 🎯 Success Metrics

### Phase 1 (Foundation):
- [x] LangGraph installed and tested
- [x] Neo4j schema defined (8+ node types, 10+ edge types)
- [x] Base agent framework created

### Phase 2 (Multi-Pipeline):
- [ ] 3 pipeline containers running
- [ ] Each uses LangGraph
- [ ] All write to Neo4j
- [ ] Stable operation >1 hour

### Phase 3 (Graph Enrichment):
- [ ] 200+ nodes in graph
- [ ] 600+ relationships
- [ ] <100ms query response time
- [ ] Graph visualizations working

### Phase 4 (Intelligence):
- [ ] Claude generates accurate relationships
- [ ] LangGraph workflows execute successfully
- [ ] Graph queries reveal insights
- [ ] Trading signals generated from graph

---

## 💰 Cost Considerations

### API Usage Estimates:

**Claude API Calls** (per day):
```
Company Graph Builder:
- 20 companies × 3 Claude calls = 60 calls/day
- ~1000 tokens per call = 60k tokens/day
- Cost: ~$0.15/day

Event Tracker:
- ~100 events/day × 2 Claude calls = 200 calls/day
- ~500 tokens per call = 100k tokens/day
- Cost: ~$0.25/day

Options Analyzer:
- 10 symbols × 4 Claude calls/day = 40 calls/day
- ~800 tokens per call = 32k tokens/day
- Cost: ~$0.08/day

TOTAL: ~$0.50/day or $15/month
```

**Data Provider Costs**:
```
Polygon.io: $99/month (5 req/sec)
Finnhub: $0 (60 calls/min free tier)
Alpha Vantage: $0 (500 calls/day free × 6 keys = 3000/day)

TOTAL: ~$100/month for comprehensive data
```

**Combined Monthly Cost**: ~$115/month
**ROI**: Priceless for professional quant insights

---

## 🚀 Implementation Priority

### Start This Week:

**Day 1-2**: 
1. Install LangGraph: `uv pip install langgraph langchain-anthropic`
2. Create base framework in `axiom/pipelines/shared/langgraph_base.py`
3. Define Neo4j schema in `axiom/pipelines/shared/neo4j_schema.py`

**Day 3-4**:
1. Build Company Graph Builder pipeline
2. Containerize it
3. Test LangGraph workflow
4. Verify Neo4j graph creation

**Day 5-7**:
1. Add Events pipeline
2. Add Options pipeline (optional)
3. Deploy all 3 containers
4. Verify graph is growing

### Next Week:
1. Add Correlation pipeline
2. Add Risk pipeline  
3. Implement advanced graph queries
4. Build analytics API on top of graph

---

## 📁 Directory Structure (Target)

```
axiom/pipelines/
├── docker-compose-langgraph.yml    # Multi-pipeline orchestration
│
├── shared/                         # Shared utilities
│   ├── __init__.py
│   ├── langgraph_base.py          # Base LangGraph framework
│   ├── neo4j_schema.py            # Complete graph schema
│   ├── neo4j_client.py            # Graph database client
│   ├── claude_agents.py           # Reusable Claude agents
│   └── graph_utils.py             # Cypher helpers
│
├── prices/                         # Pipeline 1
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pipeline.py
│   └── langgraph_workflow.py
│
├── companies/                      # Pipeline 2 (HIGH PRIORITY)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── company_graph_builder.py
│   └── langgraph_workflow.py
│
├── events/                         # Pipeline 3
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── event_tracker.py
│   └── langgraph_workflow.py
│
├── options/                        # Pipeline 4
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── options_chain.py
│   └── langgraph_workflow.py
│
├── correlations/                   # Pipeline 5
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── correlation_analyzer.py
│   └── langgraph_workflow.py
│
├── risk/                          # Pipeline 6
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── risk_graph_builder.py
│   └── langgraph_workflow.py
│
└── monitoring/                     # Observability
    ├── dashboard.py
    ├── graph_health.py
    └── pipeline_monitor.py
```

---

## 🎬 Getting Started (Next Session)

### Immediate Next Steps:

1. **Install LangGraph**:
```bash
uv pip install langgraph langchain-anthropic langchain-core
```

2. **Create Shared Framework**:
```bash
mkdir -p axiom/pipelines/shared
touch axiom/pipelines/shared/__init__.py
touch axiom/pipelines/shared/langgraph_base.py
touch axiom/pipelines/shared/neo4j_schema.py
```

3. **Implement First LangGraph Pipeline** (Company Graph):
```bash
mkdir -p axiom/pipelines/companies
# Create Dockerfile, requirements.txt, company_graph_builder.py
```

4. **Test Locally** (before containerizing):
```bash
cd axiom/pipelines/companies
python company_graph_builder.py --symbol AAPL --test
```

5. **Containerize**:
```bash
docker compose -f axiom/pipelines/docker-compose-langgraph.yml build companies
docker compose -f axiom/pipelines/docker-compose-langgraph.yml up -d companies
```

6. **Verify Graph**:
```cypher
// Check what was created
MATCH (c:Company {symbol: 'AAPL'})
OPTIONAL MATCH (c)-[r]-(other)
RETURN c, type(r), other
LIMIT 50
```

---

## 🌟 The Big Picture

### Current State:
```
1 simple pipeline → PostgreSQL/Redis/Neo4j
```

### Target State:
```
6 intelligent pipelines
    ↓ (LangGraph workflows)
    ↓ (Claude-powered analysis)
    ↓
Neo4j Knowledge Graph
    ↓ (Graph queries)
    ↓ (Pattern detection)
    ↓
Actionable Trading Intelligence
```

### Ultimate Vision:
Ask questions like:
- "Which tech stocks will be most affected by Fed rate cuts?"
- "Find undervalued stocks with strong supply chain partners"
- "What's the risk propagation if AAPL misses earnings?"

**LangGraph + Neo4j answers these using graph intelligence.**

---

## 📞 Ready to Build?

I can start immediately with:

**Option A**: Install LangGraph and create base framework
**Option B**: Build first LangGraph pipeline (Company Graph Builder)
**Option C**: Design complete multi-pipeline docker-compose
**Option D**: All of the above (comprehensive implementation)

**Which would you like to tackle first?**