# LangGraph Workflow Visualizations
**Created:** November 30, 2025  
**Purpose:** Visual documentation of all LangGraph workflows  
**Status:** All workflows operational or ready to deploy

---

## 🎯 LANGGRAPH ARCHITECTURE OVERVIEW

```
                    AXIOM LANGGRAPH ECOSYSTEM
                              |
         ┌───────────────────┼───────────────────┐
         |                   |                   |
    OPERATIONAL          READY              READY
   (Running Now)      (To Deploy)       (To Deploy)
         |                   |                   |
         ↓                   ↓                   ↓
   M&A Service      Company Intel      Deep Intelligence
   (5-min cycle)    (7 agents)         (37 agents total)
         |                   |                   |
         ↓                   ↓                   ↓
   Analyzes 5        Expands 3→50      SEC Parser (13)
   companies         companies         Earnings (11)
                                      Alt Data (13)
```

---

## 📊 WORKFLOW 1: M&A ACQUISITION ANALYZER (OPERATIONAL ✅)

**Container:** `axiom-langgraph-ma`  
**Status:** Running 4+ hours, 16+ cycles completed  
**Frequency:** Every 5 minutes  
**Companies:** AAPL, MSFT, GOOGL, TSLA, NVDA

### Workflow Diagram

```
                   🚀 START
                      ↓
         ┌────────────────────────┐
         │  Query Neo4j Graph     │
         │  - Company nodes       │
         │  - Relationships       │
         │  - Sector data         │
         └────────────┬───────────┘
                      ↓
         ┌────────────────────────┐
         │  Query PostgreSQL      │
         │  - Fundamentals        │
         │  - Financial metrics   │
         └────────────┬───────────┘
                      ↓
         ┌────────────────────────┐
         │  Claude Sonnet 4       │
         │  M&A Analysis          │
         │  - Valuation assessment│
         │  - Regulatory concerns │
         │  - Acquisition viability│
         └────────────┬───────────┘
                      ↓
         ┌────────────────────────┐
         │  Log Results           │
         │  ✅ Analysis complete  │
         └────────────┬───────────┘
                      ↓
         ┌────────────────────────┐
         │  Sleep 300 seconds     │
         └────────────┬───────────┘
                      ↓
                   🔄 REPEAT
```

### Performance Metrics

```
Cycle Time: ~15 seconds total
├─ Neo4j query: ~1 second
├─ PostgreSQL query: ~1 second
├─ Claude analysis (5 companies): ~10 seconds
└─ Logging: <1 second

Reliability: 100% (16+ cycles, zero failures)
Uptime: 4+ hours continuous
Claude API: All calls successful (200 OK)
```

---

## 📊 WORKFLOW 2: COMPANY INTELLIGENCE (READY TO DEPLOY)

**File:** [`axiom/pipelines/langgraph_company_intelligence.py`](../pipelines/langgraph_company_intelligence.py)  
**Size:** 668 lines  
**Purpose:** Expand 3 → 50 companies with AI-enriched profiles

### 7-Agent Sequential + Parallel Pipeline

```
                    🚀 START: Input company symbol
                              ↓
                 ┌────────────────────────┐
                 │  Agent 1: Fetch Basic  │
                 │  - yfinance API call   │
                 │  - Name, sector, desc  │
                 └───────────┬────────────┘
                             ↓
            ┌────────────────────────────────┐
            │  Agent 2: Claude Profile       │
            │  - Extract business model      │
            │  - Identify target markets     │
            │  - Competitive advantages      │
            └────────────┬───────────────────┘
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │Agent 3: │    │Agent 4: │    │Agent 5: │
   │Compete  │    │Products │    │ Risks   │
   │Analysis │    │Catalog  │    │Assess   │
   └────┬────┘    └────┬────┘    └────┬────┘
        └────────────────┼────────────────┘
                         ↓
            ┌────────────────────────────┐
            │  Agent 6: Quality Check    │
            │  Claude validates profile  │
            │  Score: 0-1                │
            └────────────┬───────────────┘
                         ↓
                    DECISION GATE
                         ↓
                ┌────────┴────────┐
                ↓                 ↓
        Score < 0.7         Score ≥ 0.7
                ↓                 ↓
          🔄 LOOP BACK       ┌─────────┐
          to Agent 2         │Agent 7: │
          (Re-enrich)        │ Store   │
                             │Multi-DB │
                             └────┬────┘
                                  ↓
                      ┌───────────────────┐
                      │  PostgreSQL       │
                      │  company_         │
                      │  fundamentals     │
                      └───────────────────┘
                      ┌───────────────────┐
                      │  Neo4j            │
                      │  Company node +   │
                      │  COMPETES_WITH    │
                      └───────────────────┘
                                  ↓
                            ✅ COMPLETE
```

### Parallel Batch Processing

```
Batch of 5 Companies Simultaneously:
├─ Company 1 → 7-agent pipeline
├─ Company 2 → 7-agent pipeline  
├─ Company 3 → 7-agent pipeline
├─ Company 4 → 7-agent pipeline
└─ Company 5 → 7-agent pipeline

Total: 50 companies in 10 batches
Time: 10-15 minutes total
Cost: ~$2.50 with 70% caching
```

---

## 📊 WORKFLOW 3: INTELLIGENCE SYNTHESIS (READY TO DEPLOY)

**File:** [`axiom/ai_layer/services/langgraph_intelligence_service.py`](../ai_layer/services/langgraph_intelligence_service.py)  
**Size:** 754 lines  
**Purpose:** Real-time market intelligence from live data

### 11-Agent Parallel + Sequential Architecture

```
                    🚀 START: Every 60 seconds
                              ↓
                    DATA GATHERING PHASE
                    (4 agents, parallel)
                              ↓
        ┌─────────┬───────────┼───────────┬─────────┐
        ↓         ↓           ↓           ↓         
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │Prices  │ │Companies│ │Graph   │ │News    │
   │56K rows│ │Profiles │ │4.35M   │ │Events  │
   │        │ │         │ │edges   │ │        │
   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
       └──────────┼──────────┼──────────┘
                  ↓
             ANALYSIS PHASE
             (4 agents, parallel)
                  ↓
        ┌─────────┼─────────┬─────────┐
        ↓         ↓         ↓         ↓
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │Pattern │ │Correla │ │Risk    │ │Opport  │
   │Detect  │ │Analysis│ │Assess  │ │Find    │
   │        │ │        │ │        │ │        │
   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
       └──────────┼──────────┼──────────┘
                  ↓
           SYNTHESIS PHASE
           (2 agents, sequential)
                  ↓
         ┌────────────────────┐
         │  Agent 10:         │
         │  Synthesize        │
         │  - 5-7 key insights│
         │  - Recommendations │
         └────────┬───────────┘
                  ↓
         ┌────────────────────┐
         │  Agent 11:         │
         │  Generate Report   │
         │  - Professional    │
         │  - Investment-grade│
         └────────┬───────────┘
                  ↓
            📄 REPORT OUTPUT
                  ↓
         ┌────────────────────┐
         │  Optional: Stream  │
         │  via WebSocket     │
         │  to dashboard      │
         └────────────────────┘
                  ↓
            ⏳ WAIT 60s
                  ↓
                🔄 REPEAT
```

---

## 📊 WORKFLOW 4: SEC FILING DEEP PARSER (READY TO TEST)

**File:** [`axiom/pipelines/langgraph_sec_deep_parser.py`](../pipelines/langgraph_sec_deep_parser.py)  
**Size:** 476 lines  
**Purpose:** Extract EVERYTHING from 10-K/10-Q filings

### 13-Agent Deep Extraction Pipeline

```
                    🚀 START: Company symbol
                              ↓
                 ┌────────────────────────┐
                 │  Agent 1: SEC Fetcher  │
                 │  - Download 10-K/10-Q  │
                 │  - Extract all text    │
                 └────────────┬───────────┘
                              ↓
               PARALLEL EXTRACTION (12 agents)
                              ↓
    ┌──────┬──────┬──────┬──────┬──────┬──────┐
    ↓      ↓      ↓      ↓      ↓      ↓      ↓
  Risk   MD&A  Footnote Legal  Strategy Compete Geo
  Factors                Proc   Init    Mentions Risk
    
    ↓      ↓      ↓      ↓      ↓      ↓      ↓
  Customer Supply Exec   Related
  Concent  Chain  Comp   Party
  
    └──────┴──────┴──────┴──────┴──────┴──────┘
                              ↓
                 ┌────────────────────────┐
                 │  Agent 13: Synthesis   │
                 │  Claude combines all   │
                 │  - Key insights        │
                 │  - Risk summary        │
                 │  - Strategic analysis  │
                 └────────────┬───────────┘
                              ↓
                    📄 COMPREHENSIVE REPORT
                    (Insights Bloomberg lacks)
```

### Extraction Detail

```
For Apple 10-K (example):
├─ Risk Factors: Extract ALL 50+ risks mentioned
├─ MD&A: Extract forward-looking statements
├─ Footnotes: Find hidden liabilities/commitments
├─ Legal: All lawsuits, settlements, investigations
├─ Strategy: New products, market expansions
├─ Competitive: Who management mentions/fears
├─ Geographic: Country-specific risks
├─ Customers: Concentration risk (top 10 customers)
├─ Suppliers: Critical dependencies
├─ R&D: Breakdown by category
├─ Executive Comp: Alignment with shareholders
└─ Related Party: Potential conflicts

Then: Claude synthesizes → Insights Bloomberg doesn't extract
```

---

## 📊 WORKFLOW 5: EARNINGS CALL ANALYZER (READY TO TEST)

**File:** [`axiom/pipelines/langgraph_earnings_call_analyzer.py`](../pipelines/langgraph_earnings_call_analyzer.py)  
**Size:** 490 lines  
**Purpose:** Sentiment + strategic signals from 40 quarters

### 11-Agent Time-Series Analysis

```
                  🚀 START: Company symbol
                            ↓
               ┌──────────────────────────┐
               │  Agent 1: Call Fetcher   │
               │  - Gather 40 transcripts │
               │  - 10 years history      │
               └──────────────┬───────────┘
                              ↓
            FOR EACH CALL (Sequential):
                              ↓
         ┌─────────────────────────────────┐
         │  Agent 2: Tone Analyzer         │
         │  - Confidence scoring (0-100)   │
         │  - Defensive vs aggressive      │
         └─────────────┬───────────────────┘
                       ↓
         ┌─────────────────────────────────┐
         │  Agent 3: Strategic Focus       │
         │  - Topic modeling               │
         │  - What management emphasizes   │
         └─────────────┬───────────────────┘
                       ↓
         ┌─────────────────────────────────┐
         │  Agent 4: Forward Guidance      │
         │  - Explicit guidance            │
         │  - Hedging language analysis    │
         └─────────────┬───────────────────┘
                       ↓
         ┌─────────────────────────────────┐
         │  Agent 5: Competitive Threats   │
         │  - Who gets mentioned           │
         │  - Frequency of mentions        │
         └─────────────┬───────────────────┘
                       ↓
         ┌─────────────────────────────────┐
         │  Agent 6: Analyst Questions     │
         │  - What analysts worried about  │
         │  - Question themes              │
         └─────────────┬───────────────────┘
                       ↓
         ┌─────────────────────────────────┐
         │  Agent 7: Answer Quality        │
         │  - Direct vs evasive            │
         │  - Transparency score           │
         └─────────────┬───────────────────┘
                       ↓
         ┌─────────────────────────────────┐
         │  Agent 8: Product Emphasis      │
         │  - Time per segment             │
         │  - Revenue driver priorities    │
         └─────────────┬───────────────────┘
                       ↓
         ┌─────────────────────────────────┐
         │  Agent 9: Early Warning         │
         │  - Tone degradation detection   │
         │  - Pre-problem signals          │
         └─────────────┬───────────────────┘
                       ↓
              END LOOP (40 calls)
                       ↓
         ┌─────────────────────────────────┐
         │  Agent 10: Historical Compare   │
         │  - Trend analysis over 40Q      │
         │  - Inflection point detection   │
         └─────────────┬───────────────────┘
                       ↓
         ┌─────────────────────────────────┐
         │  Agent 11: Synthesis            │
         │  - Predict next quarter         │
         │  - Management credibility score │
         │  - Investment recommendation    │
         └─────────────┬───────────────────┘
                       ↓
                 📄 PREDICTION REPORT
                 (Problems BEFORE financials)
```

### Output Example

```json
{
  "company": "AAPL",
  "quarters_analyzed": 40,
  "management_credibility": 0.85,
  "strategic_pivots": [
    {
      "quarter": "Q3 2023",
      "pivot": "AI emphasis increased 400%",
      "impact": "New product line signaled"
    }
  ],
  "early_warnings": [
    {
      "quarter": "Q2 2023",
      "signal": "Tone confidence dropped from 85 to 65",
      "outcome": "Weak Q3 followed (predicted 2 months early!)"
    }
  ],
  "predictions": {
    "next_quarter_sentiment": "positive",
    "confidence": 0.80,
    "key_factors": [...]
  }
}
```

---

## 📊 WORKFLOW 6: ALTERNATIVE DATA SYNTHESIZER (READY TO TEST)

**File:** [`axiom/pipelines/langgraph_alternative_data_synthesizer.py`](../pipelines/langgraph_alternative_data_synthesizer.py)  
**Size:** 493 lines  
**Purpose:** Leading indicators from alternative data

### 13-Agent Multi-Source Pipeline

```
              🚀 START: Company symbol
                        ↓
          PARALLEL DATA GATHERING (12 agents)
                        ↓
    ┌──────┬──────┬──────┬──────┬──────┬──────┐
    ↓      ↓      ↓      ↓      ↓      ↓      ↓
  Job    Patent  App   Social  Web   Employee
  Posts  Filings Store Senti  Traffic Reviews
  
    ↓      ↓      ↓      ↓      ↓      ↓      ↓
  GitHub Supply  Pricing Credit Satellite
  Activity Chain Changes Card  Data
  
    └──────┴──────┴──────┴──────┴──────┴──────┘
                        ↓
           EACH AGENT EXTRACTS SIGNALS:
                        ↓
         ┌────────────────────────────────┐
         │  Job Posts → Hiring Velocity   │
         │  Lead Time: 6-12 months        │
         │  Signal: Growth coming         │
         └────────────┬───────────────────┘
                      ↓
         ┌────────────────────────────────┐
         │  Patents → Innovation Pipeline │
         │  Lead Time: 2-3 years          │
         │  Signal: New products          │
         └────────────┬───────────────────┘
                      ↓
         ┌────────────────────────────────┐
         │  App Store → Engagement        │
         │  Lead Time: 1 quarter          │
         │  Signal: Services revenue      │
         └────────────┬───────────────────┘
                      ↓
         ┌────────────────────────────────┐
         │  Social → Stock Movement       │
         │  Lead Time: 2-3 days           │
         │  Signal: Price direction       │
         └────────────┬───────────────────┘
                      ↓
              (Continue for all 12...)
                      ↓
         ┌────────────────────────────────┐
         │  Agent 13: Synthesis           │
         │  Claude combines ALL signals   │
         │  - Predictions (6mo-3yr lead)  │
         │  - Confidence scoring          │
         │  - Investment thesis           │
         └────────────┬───────────────────┘
                      ↓
              📄 PREDICTIVE REPORT
              (Signals Bloomberg lacks)
```

### Predictive Signal Matrix

```
Signal Type         Lead Time      Predicts
────────────────────────────────────────────
Job postings        6-12 months    Revenue growth
Patent filings      2-3 years      New products
App downloads       1 quarter      Services revenue
Social sentiment    2-3 days       Stock movement
Web traffic         1-2 months     Product interest
Employee reviews    3-6 months     Management problems
GitHub activity     6-12 months    Product launches
Supply chain        1-2 quarters   Production ramp
Pricing changes     1-3 months     Margin pressure
Credit card data    2-4 weeks      Consumer spending
Satellite imagery   1-2 months     Store traffic/factory
```

---

## 🎯 LANGGRAPH FEATURE COMPARISON

### Operational vs Ready to Deploy

```
OPERATIONAL NOW (1 service):
├─ M&A Acquisition Analyzer ✅
│  ├─ Uptime: 4+ hours
│  ├─ Cycles: 16+
│  ├─ Companies: 5 (AAPL, MSFT, GOOGL, TSLA, NVDA)
│  ├─ Frequency: Every 5 minutes
│  └─ Status: Working perfectly

READY TO DEPLOY (5 services):
├─ Company Intelligence (668 lines, 7 agents)
│  └─ Purpose: Expand 3→50 companies
│
├─ Intelligence Synthesis (754 lines, 11 agents)
│  └─ Purpose: Real-time market intelligence
│
├─ SEC Deep Parser (476 lines, 13 agents)
│  └─ Purpose: Extract ALL 10-K insights
│
├─ Earnings Call Analyzer (490 lines, 11 agents)
│  └─ Purpose: 40-quarter sentiment analysis
│
└─ Alternative Data Synthesizer (493 lines, 13 agents)
   └─ Purpose: Leading indicators (6mo-3yr lead)

TOTAL: 2,881 lines of production LangGraph code
AGENTS: 1 operational + 51 ready to deploy = 52 total agents!
```

---

## 🏗️ LANGGRAPH ARCHITECTURE PATTERNS

### Multi-Agent Orchestration

**Sequential Pipeline:**
```
Agent A → Agent B → Agent C → Agent D
(Each depends on previous)

Use When: Complex reasoning requires context
Example: Company Intelligence (fetch → profile → validate → store)
```

**Parallel Fanout:**
```
       Input
         ↓
    ┌────┼────┐
    ↓    ↓    ↓
  Agent Agent Agent
    A    B    C
    └────┼────┘
         ↓
      Combine
```

**Use When:** Independent data gathering
**Example:** Intelligence Synthesis (gather prices || companies || graph || news)

**Conditional Routing:**
```
     Agent A
        ↓
   Decision Gate
        ↓
   ┌────┴────┐
   ↓         ↓
If X    If Y
   ↓         ↓
Agent B  Agent C
```

**Use When:** Quality validation, adaptive workflows
**Example:** Company Intelligence (if quality < 0.7 → re-enrich, else → store)

---

## 🎓 LANGGRAPH BENEFITS DEMONSTRATED

### vs Traditional Pipelines

**Airflow Pipeline:**
```
Pros:
├─ Web UI monitoring
├─ Scheduling built-in
├─ DAG dependencies
└─ Enterprise operators

Cons:
├─ Worker timeouts (3h limit)
├─ Less flexible routing
├─ More overhead
└─ Not AI-native
```

**LangGraph Pipeline:**
```
Pros:
├─ No worker timeouts (native async)
├─ Adaptive routing (quality loops)
├─ AI-native operations
├─ Self-orchestrating
├─ State management built-in
└─ Parallel + sequential elegantly

Cons:
├─ No web UI (yet)
├─ Manual scheduling
└─ Less enterprise tooling
```

**Best Practice:** Use BOTH
- Airflow: Traditional data engineering
- LangGraph: AI-heavy intelligent workflows

---

## 📈 PERFORMANCE CHARACTERISTICS

### Resource Usage

**M&A Service (Operational):**
```
CPU: Low (<5% average)
Memory: ~200 MB
Network: Minimal (API calls only)
Storage: None (stateless)
Claude API: 5 calls per cycle
Cost: ~$0.05 per cycle
```

**Company Intelligence (Projected):**
```
Duration: 10-15 minutes (50 companies)
Parallel: 5 companies at once
Claude calls: ~350 total (70% cached = 105 actual)
Cost: ~$2.50 total
Database writes: PostgreSQL + Neo4j
```

**Intelligence Synthesis (Projected):**
```
Cycle: 60 seconds
Claude calls: ~11 per cycle (parallel + sequential)
Cost: ~$0.05 per cycle = $72/month continuous
Database queries: PostgreSQL + Neo4j
Output: Professional investment report
```

---

## 🚀 DEPLOYMENT STATUS

### What's Running NOW

✅ **axiom-langgraph-ma** (Native M&A Service)
- Container operational
- 4+ hours uptime
- 16+ cycles completed
- Analyzing 5 companies every 5 minutes
- Claude integration working perfectly
- Zero crashes, 100% reliability

### What's Ready to Deploy

⏸️ **Company Intelligence** (3→50 companies)
- File ready: 668 lines
- Dependencies: Already in Airflow container
- Deployment time: 15 minutes
- Can run immediately

⏸️ **Intelligence Synthesis** (Real-time analysis)
- File ready: 754 lines
- Dependencies: Need to add to streaming container
- Deployment time: 30 minutes
- Continuous service

⏸️ **Deep Intelligence Workflows** (Bloomberg differentiation)
- Files ready: 1,459 lines total
- Dependencies: Same as above
- Deployment time: 2-3 hours to test
- Highest strategic value

---

## 🎯 VISUALIZATION RECOMMENDATIONS

### LangGraph Studio (If Available)

**LangGraph Studio** provides visual debugging:
```
Install: pip install langgraph-studio
Run: langgraph studio axiom/pipelines/
Features:
├─ Visual workflow graph
├─ State inspection
├─ Breakpoints
├─ Replay capability
└─ Performance profiling
```

### Custom Dashboard (Recommended)

**Create Grafana Dashboard for LangGraph:**
```
Metrics to Track:
├─ Cycles completed (counter)
├─ Average cycle time (gauge)
├─ Claude API calls (counter)
├─ Cost per cycle (gauge)
├─ Agent success rate (percentage)
├─ Queue depth (if applicable)
└─ Error rate (counter)

Data Source: Prometheus (export metrics from LangGraph)
```

### Workflow Documentation (This File!)

**Already Created:**
- Visual ASCII diagrams ✅
- Agent breakdown ✅
- Flow descriptions ✅
- Performance metrics ✅

---

*LangGraph Visualization Complete*  
*Status: 1 operational + 5 ready = 52 total agents*  
*Next: Deploy Company Intelligence or Deep Intelligence workflows*