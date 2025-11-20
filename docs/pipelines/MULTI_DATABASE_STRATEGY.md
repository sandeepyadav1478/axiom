# 🎯 Multi-Database Pipeline Strategy
## Leveraging All 4 Databases for Optimal Performance

You're absolutely right - we have 4 databases, each with unique strengths. **Every pipeline should leverage the right database for the right task.**

---

## 🗄️ Database Strengths

### PostgreSQL
**Best For**:
- Time-series data (OHLCV prices)
- Structured fundamentals
- ACID transactions
- Complex SQL queries
- Historical analysis

### Redis
**Best For**:
- Ultra-fast caching (<1ms)
- Latest prices
- Real-time updates
- Pub/sub messaging between pipelines
- Session data

### Neo4j
**Best For**:
- Relationships & networks
- Graph traversal
- Influence propagation
- Community detection
- Path finding

### ChromaDB
**Best For**:
- Vector embeddings
- Semantic search
- Similarity queries
- Document retrieval
- AI-powered search

---

## 🔄 Enhanced Pipeline Architecture (Using All 4 DBs)

### Pipeline 1: Data Ingestion
```
Input: Market data from APIs
    ↓
PostgreSQL: Store complete OHLCV history (time-series)
    ↓
Redis: Cache latest prices (sub-millisecond access)
    ↓
Neo4j: Update Stock nodes with current prices
    ↓
ChromaDB: Store price data as embeddings for pattern matching
```

**Implementation**:
```python
async def ingest_price(symbol, ohlcv_data):
    # 1. PostgreSQL: Permanent storage
    pg.execute("""
        INSERT INTO price_data (symbol, timestamp, open, high, low, close, volume)
        VALUES (...)
    """)
    
    # 2. Redis: Cache for ultra-fast access
    redis.set(f"price:{symbol}:latest", json.dumps({
        'close': ohlcv_data['close'],
        'timestamp': now()
    }), ex=60)
    
    # 3. Redis Pub/Sub: Notify other pipelines of new data
    redis.publish('price_updates', f"{symbol}:{ohlcv_data['close']}")
    
    # 4. Neo4j: Update graph node
    neo4j.run("""
        MERGE (s:Stock {symbol: $symbol})
        SET s.last_price = $price, s.last_updated = datetime()
    """)
    
    # 5. ChromaDB: Store as vector for pattern matching
    chroma.add(
        documents=[f"{symbol} price {ohlcv_data['close']} on {now()}"],
        embeddings=[create_price_embedding(ohlcv_data)],
        metadatas=[{'symbol': symbol, 'type': 'price'}]
    )
```

---

### Pipeline 2: Company Graph Builder
```
Input: Company symbol
    ↓
yfinance: Fetch fundamentals
    ↓
PostgreSQL: Store complete fundamental data (structured)
    ↓
Claude: Analyze and extract relationships
    ↓
Neo4j: Build company relationship graph
    ↓
ChromaDB: Store company descriptions as vectors
    ↓
Redis: Cache company metadata for fast API access
```

**Implementation**:
```python
async def build_company_graph(symbol):
    # 1. Fetch and store in PostgreSQL
    fundamentals = fetch_fundamentals(symbol)
    pg.execute("""
        INSERT INTO company_fundamentals (symbol, name, sector, market_cap, ...)
        VALUES (...)
    """)
    
    # 2. Use Claude to identify relationships
    competitors = claude.invoke("Identify competitors of {symbol}")
    
    # 3. Neo4j: Build relationship graph
    neo4j.run("""
        MERGE (c:Company {symbol: $symbol})
        SET c.name = $name, c.sector = $sector
    """)
    for competitor in competitors:
        neo4j.run("""
            MATCH (c1:Company {symbol: $symbol})
            MATCH (c2:Company {symbol: $competitor})
            MERGE (c1)-[:COMPETES_WITH {intensity: 0.8}]-(c2)
        """)
    
    # 4. ChromaDB: Store company description as vector
    description = fundamentals['longBusinessSummary']
    chroma.add(
        documents=[description],
        metadatas=[{
            'symbol': symbol,
            'type': 'company_description',
            'sector': fundamentals['sector']
        }],
        ids=[f"company_{symbol}"]
    )
    
    # 5. Redis: Cache company metadata
    redis.setex(
        f"company:{symbol}:metadata",
        3600,  # 1 hour TTL
        json.dumps({
            'name': fundamentals['name'],
            'sector': fundamentals['sector'],
            'market_cap': fundamentals['marketCap']
        })
    )
    
    # 6. Redis Pub/Sub: Notify that company graph was updated
    redis.publish('graph_updates', f"company:{symbol}:updated")
```

---

### Pipeline 3: Market Events Tracker
```
Input: Company news/events
    ↓
PostgreSQL: Store raw events (structured, queryable)
    ↓
Claude: Classify event types + analyze impact
    ↓
Neo4j: Create MarketEvent nodes + Company relationships
    ↓
ChromaDB: Store event descriptions as vectors
    ↓
Redis: Cache recent events + Pub/Sub alerts
```

**Implementation**:
```python
async def track_event(symbol, news_item):
    # 1. PostgreSQL: Store raw event
    event_id = pg.execute("""
        INSERT INTO market_events (symbol, title, date, source, raw_data)
        VALUES (...)
        RETURNING id
    """).fetchone()[0]
    
    # 2. Claude: Classify event
    classification = claude.invoke(f"""
        Classify this event:
        Title: {news_item['title']}
        
        Return JSON: {{"type": "earnings|merger|fed|other", "impact": 0-1}}
    """)
    
    # 3. Update PostgreSQL with classification
    pg.execute("""
        UPDATE market_events
        SET event_type = $type, impact_score = $impact
        WHERE id = $id
    """)
    
    # 4. Neo4j: Create event node and relationships
    neo4j.run("""
        CREATE (e:MarketEvent {
            id: $event_id,
            type: $type,
            date: date($date),
            title: $title
        })
    """)
    
    neo4j.run("""
        MATCH (e:MarketEvent {id: $event_id})
        MATCH (c:Company {symbol: $symbol})
        CREATE (c)-[:AFFECTED_BY {impact_score: $impact}]->(e)
    """)
    
    # 5. ChromaDB: Store event description as vector
    chroma.add(
        documents=[news_item['title'] + " " + news_item.get('summary', '')],
        metadatas=[{
            'event_id': event_id,
            'symbol': symbol,
            'type': classification['type'],
            'date': news_item['date']
        }],
        ids=[f"event_{event_id}"]
    )
    
    # 6. Redis: Cache recent events
    redis.zadd(
        f"events:{symbol}:recent",
        {event_id: news_item['timestamp']}
    )
    
    # 7. Redis Pub/Sub: Alert subscribers to new event
    redis.publish('market_events', json.dumps({
        'symbol': symbol,
        'type': classification['type'],
        'impact': classification['impact'],
        'event_id': event_id
    }))
```

---

### Pipeline 4: Correlation Analyzer
```
Input: List of symbols
    ↓
PostgreSQL: Query 30-day price history
    ↓
Calculate correlation matrix (numpy)
    ↓
Claude: Explain WHY correlations exist
    ↓
PostgreSQL: Store correlation coefficients (time-series)
    ↓
Neo4j: Create CORRELATED_WITH relationships
    ↓
Redis: Cache current correlations
    ↓
ChromaDB: Store correlation patterns as vectors
```

**Implementation**:
```python
async def analyze_correlations(symbols):
    # 1. PostgreSQL: Fetch price history
    prices = pg.execute("""
        SELECT symbol, timestamp, close
        FROM price_data
        WHERE symbol = ANY($symbols)
          AND timestamp >= NOW() - INTERVAL '30 days'
    """).fetchall()
    
    # 2. Calculate correlation matrix
    import pandas as pd
    df = pd.DataFrame(prices).pivot(index='timestamp', columns='symbol', values='close')
    corr_matrix = df.corr()
    
    # 3. PostgreSQL: Store correlation time-series
    for (sym1, sym2), coef in corr_matrix.items():
        if abs(coef) > 0.7:
            pg.execute("""
                INSERT INTO correlations (symbol1, symbol2, coefficient, period_days, calculated_at)
                VALUES ($sym1, $sym2, $coef, 30, NOW())
            """)
    
    # 4. Claude: Explain significant correlations
    for (sym1, sym2), coef in significant_correlations:
        explanation = claude.invoke(f"""
            Why are {sym1} and {sym2} correlated at {coef:.2f}?
            Return 1-2 sentence explanation.
        """)
        
        # 5. Neo4j: Create relationship with explanation
        neo4j.run("""
            MATCH (s1:Stock {symbol: $sym1})
            MATCH (s2:Stock {symbol: $sym2})
            MERGE (s1)-[r:CORRELATED_WITH]-(s2)
            SET r.coefficient = $coef,
                r.period_days = 30,
                r.explanation = $explanation,
                r.calculated_at = datetime()
        """)
    
    # 6. Redis: Cache current correlation matrix
    redis.setex(
        "correlations:matrix:latest",
        3600,  # 1 hour
        json.dumps(corr_matrix.to_dict())
    )
    
    # 7. ChromaDB: Store correlation patterns as vectors
    for (sym1, sym2), coef in significant_correlations:
        chroma.add(
            documents=[f"{sym1} and {sym2} are correlated at {coef:.2f}. {explanation}"],
            metadatas=[{
                'symbol1': sym1,
                'symbol2': sym2,
                'coefficient': coef,
                'type': 'correlation'
            }],
            ids=[f"corr_{sym1}_{sym2}"]
        )
```

---

## 💎 NEW Pipeline: Semantic Search (ChromaDB-Focused)

Let's add a **5th pipeline** that leverages ChromaDB for AI-powered search:

```
Pipeline 5: Semantic Search Indexer
    ↓
Input: All data from other pipelines
    ↓
ChromaDB: Create searchable vector store
    ↓
Enables: "Find companies similar to AAPL"
         "Search for earnings surprises"
         "Find correlated pairs"
```

**Use Cases**:
```python
# Question: "Find tech companies with strong earnings"
results = chroma.query(
    query_texts=["technology company strong earnings positive surprise"],
    n_results=10
)

# Question: "Which stocks behave like AAPL?"
aapl_embedding = get_stock_embedding('AAPL')
similar = chroma.query(
    query_embeddings=[aapl_embedding],
    n_results=5
)

# Question: "Find news about AI and semiconductors"
ai_news = chroma.query(
    query_texts=["artificial intelligence semiconductor chips"],
    where={"type": "market_event"},
    n_results=20
)
```

---

## 🎯 Optimal Database Usage Matrix

| Pipeline | PostgreSQL | Redis | Neo4j | ChromaDB |
|----------|------------|-------|-------|----------|
| **Data Ingestion** | ✅ OHLCV storage | ✅ Latest prices cache | ✅ Stock nodes | ✅ Price vectors |
| **Company Graph** | ✅ Fundamentals | ✅ Metadata cache | ✅ Relationships | ✅ Description vectors |
| **Events Tracker** | ✅ Event log | ✅ Recent events + Alerts | ✅ Event → Company links | ✅ News vectors |
| **Correlations** | ✅ Corr time-series | ✅ Matrix cache | ✅ CORRELATED_WITH edges | ✅ Pattern vectors |
| **Semantic Search** | ❌ Not used | ❌ Not used | ❌ Not used | ✅ PRIMARY |

---

## 🔥 Enhanced Multi-Database Workflows

### Workflow 1: Real-Time Trading Signal

```
User Query: "Is AAPL a buy right now?"

Step 1: Redis - Get latest price (0.5ms)
├─ Current price: $150.25

Step 2: PostgreSQL - Get price trend (10ms)
├─ SQL: SELECT AVG(close) FROM price_data WHERE symbol='AAPL' AND timestamp >= NOW() - INTERVAL '5 days'
└─ 5-day avg: $148.50 (trending up)

Step 3: Neo4j - Check competitive position (50ms)
├─ Cypher: MATCH (aapl:Company {symbol:'AAPL'})-[:COMPETES_WITH]-(comp) RETURN comp
└─ Has strong position vs MSFT, GOOGL

Step 4: Neo4j - Recent events impact (50ms)
├─ Cypher: MATCH (aapl)-[:AFFECTED_BY {impact_score: >0.7}]->(e:MarketEvent) WHERE e.date >= date()-duration('P7D')
└─ Positive earnings surprise 2 days ago

Step 5: ChromaDB - Find similar past scenarios (100ms)
├─ Query: "AAPL trending up, positive earnings, strong competitive position"
└─ 3 similar scenarios found - avg return: +5.2% over next week

LangGraph Decision:
├─ Claude analyzes all data
└─ Output: "BUY - Strong technicals + positive catalyst + competitive strength"

TOTAL LATENCY: ~210ms (combining 4 databases)
```

---

### Workflow 2: Sector Rotation Strategy

```
User Query: "Which sectors are rotating into strength?"

Step 1: PostgreSQL - Get all sectors performance (50ms)
├─ SQL: Calculate sector returns over 1d, 5d, 20d, 60d

Step 2: Redis - Get real-time sector momentum (5ms)
├─ Retrieve cached sector scores

Step 3: Neo4j - Analyze sector relationships (100ms)
├─ Cypher: Find sectors with most inflows
├─ Detect rotation patterns (Energy → Tech, Finance → Healthcare)

Step 4: ChromaDB - Find similar rotation patterns historically (150ms)
├─ Query: "sector rotation Energy to Technology"
├─ Find: 5 similar past rotations
└─ Avg subsequent performance: Tech +8% in 30 days

Step 5: Claude - Synthesize recommendation
├─ "Technology sector showing rotation strength..."

RESULT: Actionable sector rotation trade
```

---

### Workflow 3: Risk Cascade Analysis

```
Scenario: "Fed announces rate hike"

Step 1: PostgreSQL - Log the event
├─ INSERT INTO market_events (type='fed_decision', ...)

Step 2: Redis Pub/Sub - Alert all pipelines immediately
├─ PUBLISH 'fed_events' '{"decision":"rate_hike","magnitude":0.25}'

Step 3: Neo4j - Trace risk propagation through graph
├─ Cypher: 
    MATCH path = (fed:MarketEvent {type:'fed_decision'})
                -[:AFFECTS*1..3]-(company:Company)
    RETURN path, company.symbol, company.debt_to_equity
├─ Find: High-debt companies most affected

Step 4: PostgreSQL - Historical analysis
├─ SQL: SELECT symbol, AVG(return) FROM price_data
         JOIN market_events ON ...
         WHERE event_type = 'fed_hike'
├─ Find: Which stocks historically performed well during hikes

Step 5: ChromaDB - Semantic search for similar scenarios
├─ Query: "federal reserve interest rate increase high debt companies"
├─ Find: Past research, analyst reports, similar events

Step 6: Claude - Comprehensive risk analysis
├─ Synthesizes: Graph data + Historical data + Semantic search
└─ Output: "High-debt tech companies (TSLA, SNAP) face -15% downside risk.
            Defensive sectors (Utilities, Consumer Staples) likely +5%."

RESULT: Complete risk cascade with recommendations
```

---

## 🎨 Enhanced Pipeline Implementations

### Updated Data Ingestion Pipeline

```python
class EnhancedDataIngestion:
    def __init__(self):
        self.pg = PostgreSQLClient()
        self.redis = RedisClient()
        self.neo4j = Neo4jGraphClient()
        self.chroma = ChromaDBClient()
    
    async def ingest_symbol(self, symbol):
        # Fetch data
        ohlcv = await fetch_ohlcv(symbol)
        
        # 1. PostgreSQL: Time-series storage
        self.pg.store_ohlcv(symbol, ohlcv)
        
        # 2. Redis: Ultra-fast cache
        self.redis.cache_latest_price(symbol, ohlcv['close'])
        
        # 3. Redis Pub/Sub: Real-time alerts
        self.redis.publish_price_update(symbol, ohlcv)
        
        # 4. Neo4j: Update stock nodes
        self.neo4j.update_stock_price(symbol, ohlcv['close'])
        
        # 5. ChromaDB: Price pattern embeddings
        embedding = create_price_embedding(ohlcv)
        self.chroma.store_price_pattern(symbol, embedding, ohlcv)
```

### Updated Company Graph Builder

```python
class EnhancedCompanyGraph:
    async def build_graph(self, symbol):
        # Fetch fundamentals
        fundamentals = await fetch_fundamentals(symbol)
        
        # 1. PostgreSQL: Structured storage
        self.pg.store_fundamentals(symbol, fundamentals)
        
        # 2. Claude: Extract relationships
        competitors = await self.claude_identify_competitors(symbol, fundamentals)
        
        # 3. Neo4j: Build relationship graph
        self.neo4j.create_company_node(symbol, fundamentals)
        for comp in competitors:
            self.neo4j.create_competitor_edge(symbol, comp)
        
        # 4. ChromaDB: Searchable company descriptions
        self.chroma.add(
            documents=[fundamentals['description']],
            metadatas={'symbol': symbol, 'sector': fundamentals['sector']}
        )
        
        # 5. Redis: Cache for API responses
        self.redis.cache_company_data(symbol, fundamentals, ttl=3600)
        
        # 6. Redis Pub/Sub: Notify graph update
        self.redis.publish('graph_updates', {'symbol': symbol, 'type': 'company'})
```

---

## 📊 Data Flow Across All 4 Databases

### Example: AAPL Data Journey

```
AAPL Market Data Arrives
    │
    ├─→ PostgreSQL
    │   ├─ price_data table: Complete OHLCV history
    │   ├─ company_fundamentals: PE, Market Cap, Revenue
    │   ├─ market_events: Earnings, news, announcements
    │   └─ correlations: Historical correlation coefficients
    │
    ├─→ Redis
    │   ├─ price:AAPL:latest → {close: 150.25, timestamp: ...}
    │   ├─ company:AAPL:metadata → {name: "Apple Inc.", sector: "Tech"}
    │   ├─ events:AAPL:recent → [event_id1, event_id2, ...]
    │   ├─ correlations:AAPL → [MSFT:0.85, GOOGL:0.78, ...]
    │   └─ Pub/Sub channels: price_updates, graph_updates, market_events
    │
    ├─→ Neo4j
    │   ├─ (:Stock {symbol: 'AAPL'})
    │   ├─ (:Company {symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology'})
    │   ├─ (AAPL)-[:BELONGS_TO]->(Technology)
    │   ├─ (AAPL)-[:COMPETES_WITH {intensity: 0.85}]-(MSFT)
    │   ├─ (AAPL)-[:AFFECTED_BY {impact: 0.9}]->(EarningsEvent)
    │   └─ (AAPL)-[:CORRELATED_WITH {coef: 0.85, explanation: "..."}]-(MSFT)
    │
    └─→ ChromaDB
        ├─ Price patterns embedding (for similarity search)
        ├─ Company description embedding (for "find similar companies")
        ├─ Event news embedding (for semantic event search)
        └─ Correlation pattern embedding (for pattern matching)
```

---

## 🚀 Enhanced Multi-Database docker-compose

### Add ChromaDB Integration to All Pipelines:

```yaml
services:
  data-ingestion:
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
    # ... rest of config

  company-graph:
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
    # ... rest of config

  events-tracker:
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
    # ... rest of config

  correlations:
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
    # ... rest of config
```

---

## 💡 Use Case Examples

### Use Case 1: "Find me another stock like AAPL"

```python
# Step 1: ChromaDB - Semantic similarity
similar_descriptions = chroma.query(
    query_texts=["Large tech company, consumer electronics, ecosystem"],
    n_results=10
)

# Step 2: Neo4j - Graph similarity
graph_similar = neo4j.run("""
    MATCH (aapl:Company {symbol: 'AAPL'})-[:COMPETES_WITH]-(comp)
    MATCH (aapl)-[:BELONGS_TO]->(sector)<-[:BELONGS_TO]-(comp)
    RETURN comp.symbol, comp.market_cap
    ORDER BY comp.market_cap DESC
""")

# Step 3: Redis - Get real-time prices for candidates
for symbol in candidates:
    price = redis.get(f"price:{symbol}:latest")

# Step 4: Claude - Final recommendation
recommendation = claude.invoke(f"""
    Based on:
    - Semantic similarity: {similar_descriptions}
    - Graph relationships: {graph_similar}
    - Current prices: {prices}
    
    Which stock is most similar to AAPL? Why?
""")
```

### Use Case 2: "Alert me when correlated stocks diverge"

```python
# Redis Pub/Sub listener
def on_price_update(message):
    symbol = message['symbol']
    price = message['price']
    
    # Get correlated stocks from Neo4j
    correlated = neo4j.run("""
        MATCH (s:Stock {symbol: $symbol})-[r:CORRELATED_WITH {coefficient: >0.8}]-(other)
        RETURN other.symbol, r.coefficient
    """)
    
    # Check if prices diverging
    for corr_symbol, expected_corr in correlated:
        other_price = redis.get(f"price:{corr_symbol}:latest")
        
        # Calculate if divergence exceeds expected
        if is_diverging(price, other_price, expected_corr):
            # Alert: Trading opportunity!
            redis.publish('trading_signals', {
                'type': 'correlation_break',
                'symbol1': symbol,
                'symbol2': corr_symbol,
                'opportunity': 'mean_reversion_trade'
            })
```

---

## 📈 Benefits of Multi-Database Architecture

### Speed
- **Redis**: <1ms for latest prices, cached data
- **PostgreSQL**: 10-50ms for complex time-series queries
- **Neo4j**: 50-200ms for graph traversals
- **ChromaDB**: 100-300ms for semantic search

### Scalability
- **PostgreSQL**: Billions of price rows
- **Redis**: Millions of cache entries
- **Neo4j**: Millions of nodes/relationships
- **ChromaDB**: Millions of embeddings

### Capabilities
- **PostgreSQL**: SQL analytics, aggregations
- **Redis**: Real-time, pub/sub, distributed caching
- **Neo4j**: Graph algorithms, path finding
- **ChromaDB**: AI-powered search, similarity

---

## 🎬 Next Steps

### 1. Add ChromaDB to All Pipelines
Update each pipeline to also write to ChromaDB:
```python
# Add to each pipeline
self.chroma = ChromaDBClient(host='chromadb', port=8000)
```

### 2. Add Redis Pub/Sub
Enable real-time communication between pipelines:
```python
# In data ingestion
redis.publish('price_updates', {'symbol': 'AAPL', 'price': 150.25})

# In other pipelines (subscribers)
def on_price_update(message):
    # React to new prices immediately
    trigger_analysis(message['symbol'])
```

### 3. Create Semantic Search Pipeline
Build the 5th pipeline focused on ChromaDB:
```bash
mkdir axiom/pipelines/semantic
# Create semantic_search_indexer.py
```

---

## ✅ Summary

**Current**: Pipelines use Neo4j (graph focus)  
**Enhanced**: Pipelines use ALL 4 databases optimally

**Benefits**:
- ⚡ Faster queries (right DB for right task)
- 🧠 Richer data (multi-dimensional storage)
- 🔗 Real-time coordination (Redis Pub/Sub)
- 🔍 AI-powered search (ChromaDB vectors)
- 📊 Complete analytics (PostgreSQL + Neo4j + Claude)

**The multi-database architecture transforms Axiom into a true quantitative intelligence platform!**