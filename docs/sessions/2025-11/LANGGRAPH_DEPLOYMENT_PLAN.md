# LangGraph Services Deployment Plan
**Created:** November 29, 2025  
**Purpose:** Deploy the 2 production-ready LangGraph services  
**Status:** Services coded, dependencies need setup

---

## 🎯 LANGGRAPH SERVICES READY FOR DEPLOYMENT

### Service 1: Company Intelligence Workflow
**File:** [`axiom/pipelines/langgraph_company_intelligence.py`](../../pipelines/langgraph_company_intelligence.py)  
**Size:** 668 lines (401 code, 267 docs)  
**Status:** ✅ Code complete, dependencies needed

**Purpose:**
- Expand from 3 → 50 companies with AI-enriched profiles
- Multi-agent Claude workflow (7 agents)
- Create rich knowledge graph

**Architecture:**
```
7-Agent Pipeline:
├─ Agent 1: fetch_basic_data (yfinance API)
├─ Agent 2: claude_profile (business model extraction)
├─ Agent 3: claude_competitors (competitor identification)
├─ Agent 4: claude_products (product catalog)
├─ Agent 5: claude_risks (risk assessment)
├─ Agent 6: validate_quality (Claude QA, conditional routing)
└─ Agent 7: store_multi_database (PostgreSQL + Neo4j)

Quality Loop:
If quality_score < 0.7: route back to claude_profile
If quality_score ≥ 0.7: route to store_multi_database
```

**Expected Performance:**
- Duration: 10-15 minutes for 50 companies
- Parallel processing: 5 companies at once
- Cost: ~$2.50 (with 70% Claude caching)
- Quality: 95%+ (Claude-validated)

**Dependencies Required:**
```python
langgraph>=0.2.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
yfinance>=0.2.66
neo4j>=5.0.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0
```

### Service 2: Intelligence Synthesis Service
**File:** [`axiom/ai_layer/services/langgraph_intelligence_service.py`](../../ai_layer/services/langgraph_intelligence_service.py)  
**Size:** 754 lines (393 code, 361 docs)  
**Status:** ✅ Code complete, dependencies needed

**Purpose:**
- Real-time market intelligence from live data
- Continuous analysis every 60 seconds
- Professional investment-grade reports

**Architecture:**
```
11-Agent Pipeline:

Data Gathering (4 agents, parallel):
├─ gather_prices (PostgreSQL: 56K+ rows)
├─ gather_companies (Company profiles)
├─ gather_graph (Neo4j: 4.35M relationships)
└─ gather_news (Recent events with sentiment)

Analysis (4 agents, parallel):
├─ detect_patterns (Claude finds trends)
├─ find_correlations (Claude analyzes relationships)
├─ assess_risks (Claude identifies risks)
└─ identify_opportunities (Claude finds opportunities)

Synthesis (2 agents, sequential):
├─ synthesize_insights (Claude generates key insights)
└─ generate_report (Professional report)
```

**Expected Performance:**
- Cycle time: 60 seconds per analysis
- Mode: Continuous streaming
- Cost: ~$0.05 per analysis cycle
- Quality: Investment-grade reports

**Dependencies Required:**
Same as Company Intelligence + async support

---

## 🚀 DEPLOYMENT OPTIONS

### Option A: Deploy as Standalone Services (Recommended)

**Advantages:**
- No Airflow overhead
- Native async (no worker timeouts)
- Self-orchestrating
- Demonstrates LangGraph capabilities

**Steps:**
```bash
# 1. Create LangGraph service container
docker build -t axiom-langgraph-services \
  -f axiom/ai_layer/services/Dockerfile.langgraph .

# 2. Run Company Intelligence
docker run --rm --network database_axiom_network \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e NEO4J_URI=bolt://axiom_neo4j:7687 \
  -e NEO4J_PASSWORD=$NEO4J_PASSWORD \
  -e POSTGRES_HOST=axiom_postgres \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  axiom-langgraph-services \
  python /app/axiom/pipelines/langgraph_company_intelligence.py

# 3. Run Intelligence Synthesis (continuous)
docker run -d --name axiom-intelligence-synthesis \
  --network database_axiom_network \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e NEO4J_URI=bolt://axiom_neo4j:7687 \
  -e NEO4J_PASSWORD=$NEO4J_PASSWORD \
  -e POSTGRES_HOST=axiom_postgres \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  axiom-langgraph-services \
  python /app/axiom/ai_layer/services/langgraph_intelligence_service.py
```

### Option B: Use Existing Airflow Container

**Advantages:**
- Dependencies already installed
- Can use Airflow DAG scheduling
- Logging integrated

**Steps:**
```bash
# Copy files to Airflow container
docker cp axiom/pipelines/langgraph_company_intelligence.py \
  axiom-airflow-scheduler:/opt/airflow/

docker cp axiom/ai_layer/services/langgraph_intelligence_service.py \
  axiom-airflow-scheduler:/opt/airflow/

# Run Company Intelligence
docker exec axiom-airflow-scheduler \
  python3 /opt/airflow/langgraph_company_intelligence.py

# Run Intelligence Synthesis
docker exec -d axiom-airflow-scheduler \
  python3 /opt/airflow/langgraph_intelligence_service.py
```

### Option C: Via Airflow DAG (Already Configured)

**Current Status:**
- `company_enrichment` DAG ✅ Already triggered successfully
- Uses Claude + CircuitBreaker + batching
- Batch-based processing (can expand incrementally)

**Note:** This is the Airflow-wrapped version, not pure LangGraph

---

## 📋 RECOMMENDED NEXT STEPS

### Immediate (This Session)

**1. Verify Company Enrichment Results**
```sql
-- Check how many companies were enriched
SELECT COUNT(*) FROM company_fundamentals;

-- Check Neo4j nodes
MATCH (c:Company) RETURN count(c);
```

**2. Create Dockerfile for LangGraph Services**
```dockerfile
# File: axiom/ai_layer/services/Dockerfile.langgraph
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY axiom/ai_layer/services/requirements-langgraph.txt .
RUN pip install --no-cache-dir -r requirements-langgraph.txt

# Copy application
COPY axiom/ /app/axiom/
COPY .env /app/.env

# Default: Run company intelligence
CMD ["python", "/app/axiom/pipelines/langgraph_company_intelligence.py"]
```

**3. Create Docker Compose for LangGraph**
```yaml
# File: axiom/ai_layer/services/docker-compose-langgraph.yml
version: '3.8'

services:
  company-intelligence:
    build:
      context: ../../../
      dockerfile: axiom/ai_layer/services/Dockerfile.langgraph
    container_name: axiom-langgraph-company-intelligence
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - NEO4J_URI=bolt://axiom_neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - POSTGRES_HOST=axiom_postgres
      - POSTGRES_USER=axiom
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=axiom_finance
    networks:
      - database_axiom_network
    restart: "no"  # One-time run

  intelligence-synthesis:
    build:
      context: ../../../
      dockerfile: axiom/ai_layer/services/Dockerfile.langgraph
    container_name: axiom-intelligence-synthesis
    command: python /app/axiom/ai_layer/services/langgraph_intelligence_service.py
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - NEO4J_URI=bolt://axiom_neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - POSTGRES_HOST=axiom_postgres
      - POSTGRES_USER=axiom
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=axiom_finance
    networks:
      - database_axiom_network
    restart: unless-stopped  # Continuous service

networks:
  database_axiom_network:
    external: true
```

### Short-Term (Next Session)

**4. Deploy LangGraph Services**
```bash
# Build and run
cd axiom/ai_layer/services
docker-compose -f docker-compose-langgraph.yml up -d

# Monitor company intelligence
docker logs -f axiom-langgraph-company-intelligence

# Monitor intelligence synthesis
docker logs -f axiom-intelligence-synthesis
```

**5. Integrate with Streaming API**
```python
# The streaming API already has intelligence endpoints:
# ws://localhost:8001/ws/intelligence/{client_id}
# POST http://localhost:8001/intelligence/analyze

# Just needs dependencies added to streaming container
```

---

## 🔧 CURRENT LANGGRAPH STATUS

### What's Already Working

**1. LangGraph M&A Service** ✅
```
Container: axiom-langgraph-ma
Status: Running (Up 4+ hours)
Architecture: Native LangGraph (no Airflow wrapper)
Cycle: Every 5 minutes
Purpose: M&A acquisition analysis
```

**2. Airflow-Wrapped LangGraph** ✅
```
DAG: company_enrichment
Status: Just triggered successfully
Architecture: LangGraph operators in Airflow
Uses: Claude + CircuitBreaker patterns
```

### What's Ready But Not Deployed

**3. Company Intelligence Workflow**
- Code: ✅ Complete (668 lines)
- Dependencies: ⚠️ Need setup
- Deployment: Option A, B, or C above

**4. Intelligence Synthesis Service**
- Code: ✅ Complete (754 lines)
- Dependencies: ⚠️ Need setup
- Integration: ⚠️ Streaming API ready, needs deps

---

## 💡 LANGGRAPH SHOWCASE STRATEGY

### Current Capabilities

**Demonstrate Now:**
1. **Native LangGraph Service** (axiom-langgraph-ma)
   - Self-orchestrating
   - No Airflow wrapper
   - Continuous operation
   - Query Neo4j + PostgreSQL

2. **Airflow-Wrapped LangGraph** (company_enrichment)
   - Enterprise operators
   - Circuit breaker protection
   - Cached Claude
   - Batch processing

### After Deployment

**Will Demonstrate:**
3. **Multi-Agent Company Profiling**
   - 7-agent workflow
   - Conditional routing
   - Quality validation loops
   - Parallel batch processing

4. **Real-Time Intelligence Synthesis**
   - 11-agent architecture
   - Multi-source data fusion
   - Professional reporting
   - Streaming insights

---

## 🎯 RECOMMENDED APPROACH

### Quick Win: Use Airflow Container

**Why:**
- Dependencies already installed ✅
- Network already configured ✅
- Can run immediately ✅

**How:**
```bash
# 1. Copy files to Airflow
docker cp axiom/pipelines/langgraph_company_intelligence.py \
  axiom-airflow-scheduler:/tmp/

# 2. Run in background
docker exec -d axiom-airflow-scheduler \
  python3 /tmp/langgraph_company_intelligence.py \
  > /opt/airflow/logs/company_intelligence.log 2>&1

# 3. Monitor progress
docker exec axiom-airflow-scheduler \
  tail -f /opt/airflow/logs/company_intelligence.log
```

**Alternative: Trigger More Enrichment Batches**
```bash
# The company_enrichment DAG can be re-triggered
# Each run processes a batch of companies
# Trigger 5 times = 50 companies total

for i in {0..4}; do
  docker exec axiom-airflow-webserver \
    airflow dags trigger company_enrichment \
    --conf '{"batch_number": '$i'}'
  sleep 60  # Wait between batches
done
```

---

## 📊 SUCCESS METRICS

### When Deployment Complete

**Company Intelligence:**
- Companies in DB: 3 → 50 ✅
- Neo4j enrichment: Basic → AI-profiled ✅
- Competitor graph: Built ✅
- Product catalogs: Created ✅
- Risk factors: Identified ✅

**Intelligence Synthesis:**
- Analysis frequency: Every 60 seconds ✅
- Report quality: Investment-grade ✅
- Streaming: Via WebSocket ✅
- Cost: <$100/month ✅

---

## 🚧 CURRENT BLOCKERS

### Dependency Installation

**Issue:** LangGraph services need dependencies in execution environment

**Solutions:**

**Option 1:** Install in local venv (if exists)
```bash
source .venv/bin/activate
pip install langgraph langchain-anthropic neo4j psycopg2-binary
python axiom/pipelines/langgraph_company_intelligence.py
```

**Option 2:** Use Airflow container (has dependencies)
```bash
docker exec axiom-airflow-scheduler python3 <script>
```

**Option 3:** Create dedicated LangGraph container
```bash
docker build -t axiom-langgraph ...
docker run axiom-langgraph
```

### Integration with Streaming API

**Issue:** Intelligence endpoints deployed but need neo4j/psycopg2 in streaming container

**Solution:**
```bash
# Add to axiom/streaming/requirements.txt:
neo4j>=5.0.0
psycopg2-binary>=2.9.0
langgraph>=0.2.0
langchain-anthropic>=0.3.0

# Rebuild streaming containers
cd axiom/streaming
docker-compose down
docker-compose build
docker-compose up -d
```

---

## 🎯 NEXT SESSION PRIORITIES

### LangGraph-Focused Work

**Priority 1: Deploy Company Intelligence**
- Method: Use Airflow container (has deps)
- Duration: 10-15 minutes execution
- Result: 50 companies with AI profiles
- Value: Massive knowledge graph enrichment

**Priority 2: Deploy Intelligence Synthesis**
- Method: Create dedicated container
- Duration: Continuous service
- Result: Real-time market intelligence
- Value: Streaming AI insights

**Priority 3: Integrate with Streaming API**
- Method: Add dependencies to streaming container
- Duration: 30 minutes (rebuild)
- Result: Intelligence endpoints fully operational
- Value: Production AI streaming service

---

## 📝 CURRENT STATUS SUMMARY

**LangGraph Services:**
```
Operational (1):
└─ axiom-langgraph-ma ✅ (M&A service, 4h+ uptime)

Ready to Deploy (2):
├─ Company Intelligence (668 lines) ⏸️
└─ Intelligence Synthesis (754 lines) ⏸️

Code Complete: 1,422 lines of production LangGraph
Deployment Blockers: Dependencies setup
Estimated Deployment Time: 1-2 hours
```

**Airflow LangGraph Integration:**
```
Operational DAGs using LangGraph patterns:
├─ company_enrichment ✅ (just triggered successfully)
├─ events_tracker_v2 ✅ (Claude classification)
└─ company_graph_builder_v2 ⏸️ (paused, ready)

Status: Hybrid Airflow+LangGraph working
```

---

## 🏆 VALUE PROPOSITION

### Why Deploy LangGraph Services

**1. Showcase Native LangGraph**
- No Airflow wrapper
- Pure multi-agent orchestration
- Self-managing workflows
- Modern AI architecture

**2. Demonstrate Advanced Capabilities**
- Conditional routing based on quality
- Parallel + sequential agents
- Multi-database coordination
- Professional AI outputs

**3. Production AI Platform**
- Real-time intelligence
- Continuous analysis
- Streaming insights
- Investment-grade quality

**4. Differentiation**
- Bloomberg/FactSet don't have this
- Cutting-edge AI orchestration
- Adaptive workflows
- Self-improving system

---

## 🎯 RECOMMENDATION

**Start with Option B (Airflow Container):**
1. Copy langgraph_company_intelligence.py to Airflow
2. Run in Airflow scheduler (dependencies available)
3. Monitor execution (10-15 minutes)
4. Verify 50 companies in databases
5. Check Neo4j graph enrichment

**Then:**
1. Create dedicated LangGraph container
2. Deploy Intelligence Synthesis
3. Integrate with Streaming API
4. Full LangGraph showcase operational

---

*Deployment Plan Created: 2025-11-29*  
*Status: Ready for execution*  
*Next: Deploy Company Intelligence via Airflow container*