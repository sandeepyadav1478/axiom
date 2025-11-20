# 🎯 Graph Enrichment Strategy - Building Dense Knowledge Networks

## The Problem: Sparse Graphs Don't Reveal Insights

**Current State**:
```
3 Company nodes
0-5 relationships
Minimal connectivity
```

**This won't work for**:
- Network analysis
- Influence propagation  
- Community detection
- Pattern discovery
- Predictive modeling

**We need**: **DENSE, RICHLY CONNECTED GRAPHS**

---

## 🎯 Target: Build a Rich Knowledge Graph

### Target Metrics (30 Days):

```
NODES:
├─ Companies: 100+ nodes
├─ Sectors: 15-20 nodes
├─ People: 50+ nodes (CEOs, board members)
├─ Locations: 30+ nodes (HQ cities, markets)
├─ Products: 50+ nodes (major products/services)
├─ Events: 500+ nodes (earnings, M&A, regulatory)
├─ Technologies: 30+ nodes (AI, semiconductors, cloud)
├─ Risk Factors: 20+ nodes (geopolitical, regulatory)
└─ TOTAL: 800-1,000 nodes

RELATIONSHIPS (10+ types):
├─ COMPETES_WITH: 200+ edges
├─ SUPPLIES_TO: 150+ edges  
├─ CUSTOMER_OF: 150+ edges
├─ ACQUIRED: 50+ edges
├─ PARTNERED_WITH: 100+ edges
├─ EMPLOYS: 100+ edges (C-suite)
├─ OPERATES_IN: 200+ edges (geographic)
├─ OWNS_PRODUCT: 150+ edges
├─ USES_TECHNOLOGY: 100+ edges
├─ EXPOSED_TO_RISK: 200+ edges
├─ AFFECTED_BY_EVENT: 500+ edges
├─ CORRELATED_WITH: 500+ edges
└─ TOTAL: 2,400+ edges

GRAPH DENSITY: Average 5-6 relationships per node
NETWORK EFFECTS: Strong (many paths between entities)
```

---

## 🚀 Phase 1: Rapid Graph Building (Week 1)

### Strategy 1: Expand Company Universe

**Current**: 5-8 companies
**Target**: 100 companies

**Implementation**:
```python
# Update docker-compose-langgraph.yml
company-graph:
  environment:
    - SYMBOLS=AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,NFLX,
               GOOG,UBER,LYFT,SNAP,TWTR,ORCL,CRM,ADBE,
               INTC,AMD,QCOM,AVGO,TXN,MU,LRCX,KLAC,
               JPM,BAC,GS,MS,C,WFC,USB,PNC,BK,STT,
               JNJ,PFE,ABBV,MRK,LLY,BMY,AMGN,GILD,
               XOM,CVX,COP,SLB,HAL,MPC,PSX,VLO,
               WMT,TGT,COST,HD,LOW,DG,DLTR,ROST,
               DIS,CMCSA,NFLX,PARA,WBD,FOXA,
               BA,LMT,RTX,NOC,GD,
               ... (100 total)
```

**Run once to backfill**:
```bash
# Temporarily set interval to 60s instead of 3600s
# Process all 100 companies in ~2 hours
docker compose -f axiom/pipelines/docker-compose-langgraph.yml \
  exec company-graph \
  sh -c "PIPELINE_INTERVAL=60 python /app/pipeline.py"
```

**Result**: 100 Company nodes in 2 hours

---

### Strategy 2: Multi-Dimensional Relationships

**Add 10+ Relationship Types Using Claude**:

```python
# Enhanced Company Graph Builder
class EnhancedCompanyGraph(CompanyGraphBuilderPipeline):
    
    async def extract_all_relationships(self, symbol, company_data):
        """Use Claude to extract multiple relationship types."""
        
        # 1. Competitors
        competitors = await self.claude_extract_competitors(symbol)
        
        # 2. Supply Chain (NEW)
        suppliers = await self.claude_extract_suppliers(symbol, company_data['description'])
        customers = await self.claude_extract_customers(symbol, company_data['description'])
        
        # 3. M&A History (NEW)
        acquisitions = await self.claude_extract_acquisitions(symbol)
        
        # 4. Partnerships (NEW)
        partners = await self.claude_extract_partners(symbol, company_data['description'])
        
        # 5. Key People (NEW)
        executives = await self.claude_extract_executives(symbol, company_data)
        
        # 6. Product Portfolio (NEW)
        products = await self.claude_extract_products(symbol, company_data['description'])
        
        # 7. Technology Stack (NEW)
        technologies = await self.claude_extract_technologies(symbol)
        
        # 8. Geographic Presence (NEW)
        locations = await self.claude_extract_locations(symbol)
        
        # 9. Risk Exposures (NEW)
        risks = await self.claude_extract_risks(symbol, company_data)
        
        return {
            'competitors': competitors,
            'suppliers': suppliers,
            'customers': customers,
            'acquisitions': acquisitions,
            'partners': partners,
            'executives': executives,
            'products': products,
            'technologies': technologies,
            'locations': locations,
            'risks': risks
        }
```

**Example Claude Prompts**:

```python
# Supply Chain Extraction
prompt_suppliers = f"""
Company: {symbol} - {company_data['name']}
Description: {company_data['description']}

Identify the top 5 supplier companies that {symbol} depends on.
Consider: Raw materials, components, services, technology

Return ONLY stock symbols, comma-separated.
Example: TSM,SSNLF,QCOM,INTC,AAPL
"""

# Customer Extraction  
prompt_customers = f"""
Company: {symbol}
Revenue breakdown: {company_data.get('revenue_by_segment')}

Who are {symbol}'s largest customers (other public companies)?
Return stock symbols of B2B customers only.
"""

# M&A History
prompt_acquisitions = f"""
Company: {symbol} - {company_data['name']}

List all significant acquisitions made by {symbol} in the last 5 years.
Format: CompanyName (if public, stock symbol)
Example: LinkedIn (LNKD), GitHub (private)
"""

# Technology Stack
prompt_tech = f"""
Based on {symbol}'s business description:
{company_data['description']}

What core technologies does {symbol} use or develop?
Categories: AI, Cloud, Semiconductors, 5G, Blockchain, IoT, AR/VR, Quantum

Return comma-separated: AI,Cloud Computing,Semiconductors
```

---

### Strategy 3: Historical Graph Building

**Backfill Historical Relationships**:

```python
# New pipeline: Historical Graph Builder
# axiom/pipelines/historical/graph_backfill.py

class HistoricalGraphBuilder:
    """One-time backfill to create rich historical context."""
    
    async def build_complete_graph(self, symbols):
        """Build comprehensive graph for all symbols."""
        
        for symbol in symbols:
            # Fetch comprehensive data
            data = await self.fetch_complete_company_data(symbol)
            
            # Extract ALL relationship types
            relationships = await self.extract_multi_dimensional_relationships(symbol, data)
            
            # Build in Neo4j
            await self.create_rich_company_graph(symbol, relationships)
        
        # Build cross-company relationships
        await self.discover_implicit_relationships(symbols)
    
    async def discover_implicit_relationships(self, symbols):
        """Use Claude to find non-obvious connections."""
        
        # Example: Find companies in same supply chain
        prompt = f"""
        Given these companies: {', '.join(symbols[:20])}
        
        Identify supply chain connections:
        - Which companies supply to which?
        - Which are in the same value chain?
        - Which share common suppliers?
        
        Return as JSON:
        [
          {{"supplier": "TSM", "customer": "AAPL", "product": "chips"}},
          {{"supplier": "MSFT", "customer": "AAPL", "product": "cloud services"}}
        ]
        """
        
        connections = await self.claude.invoke(prompt)
        
        # Create SUPPLIES_TO relationships
        for conn in parse_json(connections):
            self.neo4j.create_supply_chain(
                conn['supplier'],
                conn['customer'],
                product=conn['product']
            )
```

---

## 🔄 Enhanced Pipeline Workflows

### Pipeline 2: Enhanced Company Graph Builder

**Add More Relationship Extraction**:

```python
class SuperEnhancedCompanyGraph(CompanyGraphBuilderPipeline):
    
    def build_workflow(self) -> StateGraph:
        workflow = StateGraph(CompanyGraphState)
        
        # Original agents
        workflow.add_node("fetch_data", self.fetch_company_data)
        workflow.add_node("extract_competitors", self.extract_competitors_with_claude)
        
        # NEW AGENTS - Extract more relationships
        workflow.add_node("extract_supply_chain", self.extract_supply_chain)
        workflow.add_node("extract_ma_history", self.extract_ma_history)
        workflow.add_node("extract_partnerships", self.extract_partnerships)
        workflow.add_node("extract_key_people", self.extract_key_people)
        workflow.add_node("extract_products", self.extract_products)
        workflow.add_node("extract_tech_stack", self.extract_tech_stack)
        workflow.add_node("extract_geo_presence", self.extract_geo_presence)
        workflow.add_node("extract_risk_factors", self.extract_risk_factors)
        
        # Continue with original
        workflow.add_node("generate_cypher", self.generate_comprehensive_cypher)
        workflow.add_node("execute_neo4j", self.execute_neo4j_updates)
        workflow.add_node("validate", self.validate_graph)
        
        # Connect all agents
        workflow.add_edge("fetch_data", "extract_competitors")
        workflow.add_edge("extract_competitors", "extract_supply_chain")
        workflow.add_edge("extract_supply_chain", "extract_ma_history")
        workflow.add_edge("extract_ma_history", "extract_partnerships")
        workflow.add_edge("extract_partnerships", "extract_key_people")
        workflow.add_edge("extract_key_people", "extract_products")
        workflow.add_edge("extract_products", "extract_tech_stack")
        workflow.add_edge("extract_tech_stack", "extract_geo_presence")
        workflow.add_edge("extract_geo_presence", "extract_risk_factors")
        workflow.add_edge("extract_risk_factors", "generate_cypher")
        workflow.add_edge("generate_cypher", "execute_neo4j")
        workflow.add_edge("execute_neo4j", "validate")
        workflow.add_edge("validate", END)
        
        return workflow
```

**Result**: Each company gets 50-100 relationships instead of 3-5!

---

## 📈 Quick Wins to Increase Graph Density

### Quick Win #1: Expand Symbol List (Today)

```yaml
# axiom/pipelines/docker-compose-langgraph.yml
company-graph:
  environment:
    # From 8 symbols to 50 symbols
    - SYMBOLS=AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,NFLX,GOOG,UBER,
               LYFT,SNAP,TWTR,ORCL,CRM,ADBE,INTC,AMD,QCOM,AVGO,
               JPM,BAC,GS,MS,C,WFC,USB,PNC,BK,STT,
               JNJ,PFE,ABBV,MRK,LLY,BMY,AMGN,GILD,REGN,VRTX,
               XOM,CVX,COP,SLB,HAL,MPC,PSX,VLO,OXY,DVN
```

**Impact**: 50 companies × 50 potential relationships = 2,500 edges

---

### Quick Win #2: Run Correlation on All Pairs

```python
# In correlation_analyzer.py
# Calculate ALL pairwise correlations, not just significant ones

# Current: Only stores correlations > 0.7
# Enhanced: Store ALL correlations with explanations

for (sym1, sym2), coef in all_correlations.items():
    if abs(coef) > 0.3:  # Lower threshold
        # Get Claude explanation
        explanation = claude.explain_correlation(sym1, sym2, coef)
        
        # Create edge with rich properties
        neo4j.create_correlation(
            sym1, sym2, coef,
            explanation=explanation,
            strength='strong' if abs(coef) > 0.7 else 'moderate' if abs(coef) > 0.5 else 'weak',
            confidence=calculate_confidence(coef),
            sample_size=30,  # days
            created_at=datetime.now()
        )
```

**Impact**: 50 symbols × 49 pairs / 2 = 1,225 correlation edges

---

### Quick Win #3: Add Sector Relationships

```cypher
// Create sector hierarchy
CREATE (tech:Sector {name: 'Technology'})
CREATE (software:Sector {name: 'Software', parent: 'Technology'})
CREATE (hardware:Sector {name: 'Hardware', parent: 'Technology'})
CREATE (semiconductors:Sector {name: 'Semiconductors', parent: 'Hardware'})

// Link sectors
CREATE (software)-[:PART_OF]->(tech)
CREATE (hardware)-[:PART_OF]->(tech)
CREATE (semiconductors)-[:PART_OF]->(hardware)

// Inter-sector dependencies
CREATE (software)-[:DEPENDS_ON {reason: 'runs on'}]->(hardware)
CREATE (hardware)-[:DEPENDS_ON {reason: 'requires'}]->(semiconductors)
```

**Impact**: +100 sector relationships

---

### Quick Win #4: Add Market Events Aggressively

```python
# Change events pipeline interval
events-tracker:
  environment:
    - PIPELINE_INTERVAL=60  # Every minute instead of 5 min
    - MAX_NEWS_PER_SYMBOL=20  # More news items
```

**Impact**: 100 companies × 20 news/day = 2,000 event nodes/day

---

## 🎨 New Relationship Types to Add

### 1. Supply Chain Network

```cypher
// Example: Apple's supply chain
(:Company {symbol: 'AAPL'})
  -[:SUPPLIES_TO {product: 'chips', revenue_pct: 0.4}]->
  (:Company {symbol: 'TSM'})

(:Company {symbol: 'AAPL'})
  -[:CUSTOMER_OF {spend_pct: 0.15}]->
  (:Company {symbol: 'GOOG'})  // For cloud services
```

### 2. M&A Network

```cypher
// Microsoft's acquisitions
(:Company {symbol: 'MSFT'})
  -[:ACQUIRED {date: date('2016-06-13'), price: 26200000000}]->
  (:Company {symbol: 'LNKD', status: 'private'})

(:Company {symbol: 'META'})
  -[:ACQUIRED {date: date('2012-04-09'), price: 1000000000}]->
  (:Company {name: 'Instagram', status: 'subsidiary'})
```

### 3. People Network

```cypher
// Key executives
(:Person {name: 'Tim Cook', role: 'CEO'})
  -[:LEADS]->
  (:Company {symbol: 'AAPL'})

(:Person {name: 'Tim Cook'})
  -[:FORMERLY_AT]->
  (:Company {symbol: 'COMPAQ'})

// Board connections
(:Person {name: 'Al Gore'})
  -[:BOARD_MEMBER]->
  (:Company {symbol: 'AAPL'})

(:Person {name: 'Al Gore'})
  -[:BOARD_MEMBER]->
  (:Company {symbol: 'GOOG'})  // Board interlock!
```

### 4. Product Network

```cypher
(:Company {symbol: 'AAPL'})
  -[:OWNS_PRODUCT]->
  (:Product {name: 'iPhone', category: 'Consumer Electronics'})

(:Product {name: 'iPhone'})
  -[:COMPETES_WITH]->
  (:Product {name: 'Galaxy', owner: 'Samsung'})

(:Product {name: 'iPhone'})
  -[:USES_COMPONENT]->
  (:Product {name: 'A-series chip', manufacturer: 'TSM'})
```

### 5. Technology Stack

```cypher
(:Company {symbol: 'NFLX'})
  -[:USES_TECHNOLOGY {since: 2015}]->
  (:Technology {name: 'AWS Cloud'})

(:Technology {name: 'AWS Cloud'})
  -[:OWNED_BY]->
  (:Company {symbol: 'AMZN'})

// Creates transitive relationship: NFLX depends on AMZN
```

### 6. Geographic Network

```cypher
(:Company {symbol: 'AAPL'})
  -[:HAS_HEADQUARTERS_IN]->
  (:Location {city: 'Cupertino', state: 'CA', country: 'USA'})

(:Company {symbol: 'AAPL'})
  -[:OPERATES_IN {revenue_pct: 0.35}]->
  (:Location {country: 'China'})

// Find China exposure
MATCH (c:Company)-[r:OPERATES_IN {country: 'China'}]
WHERE r.revenue_pct > 0.2
RETURN c.symbol, r.revenue_pct ORDER BY r.revenue_pct DESC
```

### 7. Risk Exposure Network

```cypher
(:RiskFactor {
  type: 'geopolitical',
  region: 'China',
  severity: 0.8,
  description: 'US-China trade tensions'
})
  <-[:EXPOSED_TO {exposure: 0.35, channel: 'supply_chain'}]-
  (:Company {symbol: 'AAPL'})

// Risk propagation
MATCH path = (risk:RiskFactor)-[:PROPAGATES_THROUGH*1..3]-(company)
RETURN path, company.symbol, length(path) as hops
```

---

## 🧠 Claude-Powered Relationship Discovery

### Technique 1: Batch Relationship Extraction

```python
async def claude_discover_all_relationships(companies: List[str]):
    """Ask Claude to find ALL relationships at once."""
    
    prompt = f"""
    Given these 50 companies:
    {', '.join(companies)}
    
    Identify ALL significant relationships:
    1. Supply chain (who supplies to whom)
    2. Partnerships (strategic alliances)
    3. Competition (direct competitors)
    4. M&A (who acquired whom)
    5. Technology dependencies
    6. Common customers/markets
    
    Return comprehensive JSON mapping ALL connections.
    Be exhaustive - we want a DENSE graph.
    """
    
    response = await claude.invoke(prompt)
    relationships = parse_json(response)
    
    # Create 100s of relationships from one Claude call
    for rel in relationships:
        create_relationship_in_neo4j(rel)
```

### Technique 2: Similarity-Based Relationship Creation

```python
async def find_similar_companies_create_edges(symbol):
    """Use Claude to find similar companies, create SIMILAR_TO edges."""
    
    prompt = f"""
    Find 10 companies most similar to {symbol}.
    Consider: Business model, revenue streams, customer base, geography
    
    For each, explain WHY similar (1 sentence).
    
    Return JSON:
    [
      {{"symbol": "MSFT", "similarity": 0.85, "reason": "Both cloud + enterprise software"}},
      ...
    ]
    """
    
    similar = await claude.invoke(prompt)
    
    for comp in parse_json(similar):
        neo4j.run("""
            MATCH (a:Company {symbol: $symbol})
            MATCH (b:Company {symbol: $comp_symbol})
            CREATE (a)-[:SIMILAR_TO {
                similarity_score: $similarity,
                reason: $reason,
                discovered_by: 'claude',
                discovered_at: datetime()
            }]-(b)
        """, symbol=symbol, comp_symbol=comp['symbol'], 
             similarity=comp['similarity'], reason=comp['reason'])
```

---

## 📊 Graph Density Targets

### Current (Sparse):
```
3 nodes
0-5 relationships
Density: ~1.7 edges/node
Network effects: NONE
```

### Target (Dense):
```
100 nodes
2,400 relationships  
Density: 24 edges/node
Network effects: STRONG

Graph properties:
├─ Average path length: 2-3 hops
├─ Clustering coefficient: >0.6
├─ Communities detected: 10-15 (sectors/industries)
├─ Hub nodes: 5-10 (Apple, Microsoft, Amazon, etc.)
└─ Bridge nodes: Identified (companies connecting sectors)
```

---

## 🚀 Rapid Enrichment Plan (Next 48 Hours)

### Day 1 (Today):

**Hour 1-2**: Expand to 50 companies
```bash
# Update docker-compose with 50 symbols
# Run backfill script
```

**Hour 3-4**: Add supply chain relationships
```python
# Update company_graph_builder.py with supply chain extraction
# Redeploy
```

**Hour 5-6**: Add M&A relationships
```python
# Add acquisition history extraction
```

**Result**: 50 nodes, 300-500 relationships

### Day 2 (Tomorrow):

**Morning**: Add people & products
```python
# Extract CEOs, board members
# Extract product portfolios
```

**Afternoon**: Add geographic & technology relationships
```python
# Extract operating regions
# Extract technology dependencies
```

**Evening**: Run comprehensive backfill
```bash
# Historical graph builder
# Process all 100 companies with all relationship types
```

**Result**: 800+ nodes, 2,000+ relationships

---

## 💡 Example: Dense Graph for AAPL

### Target Relationship Count for Single Company:

```cypher
MATCH (aapl:Company {symbol: 'AAPL'})-[r]-(other)
RETURN type(r), count(r)

Expected output:
├─ COMPETES_WITH: 10 edges (Microsoft, Google, Samsung, etc.)
├─ SUPPLIES_TO: 0 (Apple doesn't supply, it's a customer)
├─ CUSTOMER_OF: 15 edges (TSM, SSNLF, QCOM for chips, etc.)
├─ PARTNERED_WITH: 5 edges (IBM, Cisco, etc.)
├─ ACQUIRED: 20 edges (Beats, Shazam, etc.)
├─ EMPLOYS: 10 edges (Tim Cook, other C-suite)
├─ OWNS_PRODUCT: 15 edges (iPhone, iPad, Mac, etc.)
├─ OPERATES_IN: 20 edges (USA, China, Europe countries)
├─ USES_TECHNOLOGY: 5 edges (ARM, Swift, etc.)
├─ EXPOSED_TO_RISK: 10 edges (China risk, semiconductor shortage)
├─ AFFECTED_BY_EVENT: 50 edges (earnings, product launches)
├─ CORRELATED_WITH: 30 edges (tech stocks)
├─ SIMILAR_TO: 10 edges (based on Claude similarity)
└─ TOTAL: ~200 relationships for ONE company

For 100 companies: ~20,000 total relationships!
```

---

## 🎯 Implementation Priority

### Immediate (This Week):
1. ✅ Expand to 50-100 companies
2. ✅ Lower correlation threshold (0.3 instead of 0.7)
3. ✅ Add supply chain extraction
4. ✅ Add M&A history extraction

### Next Week:
1. Add people network
2. Add product network
3. Add technology dependencies
4. Add geographic relationships

### Month 1:
1. Complete 100-company rich graph
2. 10+ relationship types
3. 2,000+ total relationships
4. Enable graph algorithms (PageRank, community detection)

---

## 📊 Success Metrics

**Sparse Graph** (Current - Bad):
```
3 nodes, 5 edges → Can't do meaningful analysis
```

**Medium Density** (Next Week - Good):
```
50 nodes, 500 edges → Basic network analysis possible
Can find: Direct competitors, simple correlations
```

**High Density** (Month 1 - Excellent):
```
100 nodes, 2,000+ edges → Rich network effects
Can find: Hidden connections, influence paths, communities
Enables: Predictive models, risk propagation, opportunity discovery
```

**Ultimate** (Month 3 - Production):
```
500 nodes, 10,000+ edges → Production intelligence
Graph algorithms: PageRank, centrality, path finding
AI queries: "Find undervalued companies with strong network position"
```

---

## 🚀 Start Building Dense Graph NOW

**Command to expand immediately**:
```bash
# 1. Update symbols in docker-compose-langgraph.yml (50-100 companies)
# 2. Temporarily run hourly pipeline every 5 minutes to backfill
docker compose -f axiom/pipelines/docker-compose-langgraph.yml down
# Edit file, change PIPELINE_INTERVAL=300 for faster backfill
docker compose -f axiom/pipelines/docker-compose-langgraph.yml up -d

# 3. Watch graph grow
watch -n 10 'docker exec axiom_neo4j cypher-shell -u neo4j -p axiom_neo4j "MATCH (n) RETURN labels(n)[0], count(n);"'
```

**You're right - we need WAY more edges to unlock the true power of the knowledge graph!**

Let's build it.