# MCP to FastMCP Conversion Plan
**Created:** December 1, 2025  
**Purpose:** Strategic plan to convert select MCP servers to FastMCP  
**Goal:** 75% code reduction, 10x faster development, maintain production stability

---

## 📊 COMPLETE MCP SERVER INVENTORY

### Current MCP Servers (18 Total)

**Storage Category (4 servers):**
1. [`redis_server.py`](../integrations/mcp_servers/storage/redis_server.py) - 760 lines
2. [`postgres_server.py`](../integrations/mcp_servers/storage/postgres_server.py) - Unknown size
3. [`vector_db_server.py`](../integrations/mcp_servers/storage/vector_db_server.py) - Unknown size

**Analytics Category (1 server):**
4. [`sql_server.py`](../integrations/mcp_servers/analytics/sql_server.py) - 1,210 lines ⭐

**Documents Category (2 servers):**
5. [`pdf_server.py`](../integrations/mcp_servers/documents/pdf_server.py) - 924 lines
6. [`excel_server.py`](../integrations/mcp_servers/documents/excel_server.py) - Unknown size

**Research Category (1 server):**
7. [`arxiv_server.py`](../integrations/mcp_servers/research/arxiv_server.py) - 724 lines ⭐

**Cloud Category (2 servers):**
8. [`aws_server.py`](../integrations/mcp_servers/cloud/aws_server.py) - Unknown size
9. [`gcp_server.py`](../integrations/mcp_servers/cloud/gcp_server.py) - Unknown size

**DevOps Category (3 servers):**
10. [`docker_server.py`](../integrations/mcp_servers/devops/docker_server.py) - Unknown size
11. [`git_server.py`](../integrations/mcp_servers/devops/git_server.py) - Unknown size
12. [`kubernetes_server.py`](../integrations/mcp_servers/devops/kubernetes_server.py) - Unknown size

**Communication Category (2 servers):**
13. [`notification_server.py`](../integrations/mcp_servers/communication/notification_server.py) - Unknown size
14. [`slack_server.py`](../integrations/mcp_servers/communication/slack_server.py) - Unknown size

**Monitoring Category (1 server):**
15. [`prometheus_server.py`](../integrations/mcp_servers/monitoring/prometheus_server.py) - Unknown size

**Code Quality Category (1 server):**
16. [`linting_server.py`](../integrations/mcp_servers/code_quality/linting_server.py) - Unknown size

**ML Ops Category (1 server):**
17. [`model_serving_server.py`](../integrations/mcp_servers/mlops/model_serving_server.py) - Unknown size

**Filesystem Category (1 server):**
18. [`server.py`](../integrations/mcp_servers/filesystem/server.py) - Unknown size

**Total Lines Analyzed:** ~3,618 lines across 4 examined servers

---

## 🎯 FASTMCP CONVERSION CANDIDATES (Ranked)

### Tier 1: BEST Candidates (High Value, Easy Migration)

#### 1. arXiv Research Server ⭐⭐⭐⭐⭐ (HIGHEST PRIORITY)

**Current:** 724 lines with official MCP SDK  
**After FastMCP:** ~150 lines (79% reduction!)

**Why Perfect Candidate:**
```
✅ Simple, clean tools (8 tools)
✅ No complex state management
✅ Straightforward async operations
✅ Type-safe benefit (Pydantic models)
✅ High development value (research integration)
✅ Easy to test
✅ Clear inputs/outputs
✅ Frequently used in research workflows
```

**Tools:**
1. search_papers
2. get_paper
3. download_pdf
4. get_latest
5. search_by_author
6. get_citations
7. extract_formulas
8. summarize_paper

**FastMCP Benefit:**
```python
# Current (Official MCP):
@self.server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="search_papers",
            description="Search arXiv papers...",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "..."},
                    "category": {"type": "string", "description": "..."},
                    # ... 10 more lines
                }
            }
        ),
        # ... repeat for 7 more tools = 200+ lines
    ]

# With FastMCP:
@mcp.tool()
async def search_papers(
    query: str,
    category: Optional[str] = None,
    max_results: int = 10,
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance"
) -> Dict[str, Any]:
    """Search arXiv papers by keywords, categories, or date ranges."""
    # 8 tools = ~40 lines total!
```

**Conversion Effort:** 2-3 hours  
**Code Reduction:** 574 lines (79%)  
**Risk:** Very Low (isolated, well-defined)

---

#### 2. Redis Server ⭐⭐⭐⭐⭐ (PRODUCTION CRITICAL)

**Current:** 760 lines  
**After FastMCP:** ~180 lines (76% reduction!)

**Why Excellent Candidate:**
```
✅ Production-critical (caching, pub/sub)
✅ Simple CRUD operations (8 tools)
✅ Type-safe benefit huge (Redis keys/values)
✅ Pydantic validation perfect fit
✅ High usage (streaming API uses this!)
✅ Easy testing
✅ Performance metrics tracking
```

**Tools:**
1. get_value
2. set_value
3. delete_key
4. publish_message
5. subscribe_channel
6. zadd (sorted sets)
7. zrange (sorted sets)
8. get_stats

**FastMCP Benefit:**
```python
# Current: ~100 lines per tool (definition + handler + error handling)
# FastMCP: ~10 lines per tool

from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("redis-cache")

class SetValueRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = None
    serialize: bool = True

@mcp.tool()
async def set_value(request: SetValueRequest) -> Dict:
    """Set value in Redis with optional TTL."""
    # Just the logic, no boilerplate!
    redis = await ensure_connection()
    if request.serialize:
        value = json.dumps(request.value)
    else:
        value = request.value
    
    if request.ttl:
        await redis.setex(request.key, request.ttl, value)
    else:
        await redis.set(request.key, value)
    
    return {"success": True, "key": request.key}

# 15 lines vs 100 lines! 85% reduction
```

**Conversion Effort:** 3-4 hours  
**Code Reduction:** 580 lines (76%)  
**Risk:** Medium (production-critical, needs thorough testing)

---

#### 3. SQL Analytics Server ⭐⭐⭐⭐ (HIGHEST VALUE)

**Current:** 1,210 lines (LARGEST!)  
**After FastMCP:** ~280 lines (77% reduction!)

**Why Great Candidate:**
```
✅ Huge code reduction potential (930 lines saved!)
✅ Complex analytics logic benefits from clean API
✅ Type-safe models for financial queries
✅ Pydantic perfect for SQL params
✅ High business value (analytics platform)
✅ Frequently extended (new analytics)
✅ 11 sophisticated tools
```

**Tools:**
1. generate_sql (NL to SQL)
2. execute_query
3. create_view
4. aggregate_data
5. pivot_table
6. time_series_agg
7. cohort_analysis
8. funnel_analysis
9. trend_analysis
10. anomaly_detection
11. forecast_timeseries

**FastMCP Benefit:**
```python
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Literal

mcp = FastMCP("sql-analytics")

class QueryRequest(BaseModel):
    query: str = Field(..., description="SQL query to execute")
    params: Optional[List[Any]] = None
    limit: int = Field(1000, description="Max rows")
    format: Literal["json", "csv", "markdown"] = "json"

@mcp.tool()
async def execute_query(request: QueryRequest) -> Dict:
    """Execute SQL and return structured data."""
    # Clean implementation, no JSON schema boilerplate
    result = await conn.execute(request.query, request.params).fetchdf()
    
    if request.format == "csv":
        return {"data": result.to_csv()}
    return {"data": result.to_dict(orient="records")}

# 12 lines vs 80+ lines! Type-safe with Pydantic
```

**Conversion Effort:** 4-5 hours  
**Code Reduction:** 930 lines (77%)  
**Risk:** Low (isolated analytics, easy to test)

---

### Tier 2: GOOD Candidates (Medium Value)

#### 4. PDF Processing Server ⭐⭐⭐⭐

**Current:** 924 lines  
**After FastMCP:** ~220 lines (76% reduction!)

**Why Good Candidate:**
```
✅ Document processing critical for SEC filings
✅ 9 tools (text, tables, 10-K, OCR)
✅ Type-safe benefit for financial docs
✅ Used in deep intelligence workflows
✅ Clean separation of concerns
```

**Conversion Effort:** 3-4 hours  
**Code Reduction:** 704 lines (76%)  
**Risk:** Low-Medium (depends on PDF libraries)

#### 5. Notification Server ⭐⭐⭐

**Current:** Unknown (estimate ~500 lines)  
**After FastMCP:** ~100 lines (80% reduction!)

**Why Good:**
```
✅ Unified communications (email, SMS, alerts)
✅ Type-safe recipients/messages
✅ Pydantic models for notifications
✅ Production alerts use this
```

**Conversion Effort:** 2-3 hours  
**Code Reduction:** ~400 lines (80%)  
**Risk:** Low (simple message passing)

### Tier 3: SKIP (Low Value or High Risk)

**Don't Convert:**
- AWS Server (complex, cloud-specific)
- GCP Server (complex, cloud-specific)
- Kubernetes Server (complex orchestration)
- Docker Server (system-level operations)
- Postgres Server (stable, working)
- Vector DB Server (complex multi-provider)

**Reason:** Complex state management, cloud SDKs, or stable production use

---

## 🚀 RECOMMENDED CONVERSION SEQUENCE

### Phase 1: Proof of Concept (1 server, 2-3 hours)

**Convert:** arXiv Research Server (EASIEST)

**Steps:**
1. Install FastMCP: `uv add fastmcp`
2. Create new file: `research/arxiv_server_fastmcp.py`
3. Implement 8 tools with FastMCP decorators
4. Test thoroughly (compare outputs)
5. If successful → proceed to Phase 2

**Success Criteria:**
- Code reduced by 75%+
- All tools work identically
- Tests pass
- Team prefers new code

**Expected Result:**
- 724 lines → ~150 lines
- 574 lines saved (79% reduction!)
- Proves FastMCP viability

---

### Phase 2: Production Deployment (2 servers, 6-8 hours)

**Convert:**
1. **SQL Analytics Server** (highest value)
2. **Redis Server** (production-critical)

**Why These Two:**
- SQL: Biggest code reduction (930 lines!)
- Redis: Production validation (streaming API uses it)
- Together: Proves FastMCP production-ready

**Success Criteria:**
- Code reduction 75%+
- Production stability maintained
- Performance same or better
- Team productivity improved

**Expected Result:**
- 1,970 lines → ~460 lines
- 1,510 lines saved (77% reduction!)
- Production FastMCP validated

---

### Phase 3: Expand (Optional, 4-6 more servers)

**Convert If Phase 1-2 Successful:**
1. PDF Processing Server
2. Notification Server
3. Excel Server
4. Prometheus Server

**Total Potential Savings:**
- Phase 1: 574 lines (1 server)
- Phase 2: 1,510 lines (2 servers)
- Phase 3: ~2,000 lines (4 servers)
- **Total: ~4,084 lines saved (estimated)**

---

## 💻 CONVERSION EXAMPLE: arXiv Server

### Current Implementation (Official MCP)

```python
# File: research/arxiv_server.py (724 lines)

from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

class ArxivMCPServer:
    def __init__(self):
        self.server = Server("arxiv-research")
        self._register_handlers()
    
    def _register_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="search_papers",
                    description="Search arXiv papers...",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "category": {
                                "type": "string",
                                "description": "arXiv category"
                            },
                            # ... 20 more lines per tool
                        },
                        "required": ["query"]
                    }
                ),
                # ... 7 more tools, each 20-30 lines
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            if name == "search_papers":
                result = await self._search_papers(**arguments)
            elif name == "get_paper":
                result = await self._get_paper(**arguments)
            # ... manual routing for 8 tools
            
            return [TextContent(type="text", text=json.dumps(result))]
    
    async def _search_papers(self, query, category=None, max_results=10, ...):
        # Implementation (60 lines)
        pass
    
    # ... 7 more tool implementations
    # Total: 724 lines
```

### FastMCP Implementation

```python
# File: research/arxiv_server_fastmcp.py (150 lines!)

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import arxiv

mcp = FastMCP("arxiv-research")

# Type-safe models (Pydantic)
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search keywords")
    category: Optional[str] = Field(None, description="arXiv category (e.g., 'q-fin.PM')")
    max_results: int = Field(10, ge=1, le=100)
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance"
    date_from: Optional[str] = Field(None, description="YYYY-MM-DD")

class PaperResponse(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    pdf_url: str

# Tools with decorators (CLEAN!)
@mcp.tool()
async def search_papers(request: SearchRequest) -> Dict[str, Any]:
    """
    Search arXiv papers by keywords, categories, or dates.
    Returns paper metadata with abstracts and PDF links.
    """
    search_query = request.query
    if request.category:
        search_query = f"cat:{request.category} AND {request.query}"
    
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate
    }
    
    search = arxiv.Search(
        query=search_query,
        max_results=request.max_results,
        sort_by=sort_map[request.sort_by]
    )
    
    papers = []
    for result in search.results():
        paper = PaperResponse(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            categories=result.categories,
            published=result.published.isoformat(),
            pdf_url=result.pdf_url
        )
        papers.append(paper.dict())
    
    return {
        "query": request.query,
        "total_results": len(papers),
        "papers": papers
    }

@mcp.tool()
async def get_paper(arxiv_id: str) -> PaperResponse:
    """Get detailed metadata for a specific paper."""
    arxiv_id = arxiv_id.replace("arXiv:", "").strip()
    
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(search.results())
    
    return PaperResponse(
        arxiv_id=result.entry_id.split("/")[-1],
        title=result.title,
        authors=[author.name for author in result.authors],
        abstract=result.summary,
        categories=result.categories,
        published=result.published.isoformat(),
        pdf_url=result.pdf_url
    )

@mcp.tool()
async def download_pdf(arxiv_id: str, filename: Optional[str] = None) -> Dict:
    """Download paper PDF."""
    # Clean implementation
    pass

# ... 5 more tools, each 10-20 lines
# Total: ~150 lines vs 724 lines!
```

**Benefits:**
- 574 lines eliminated (79% reduction!)
- Type-safe with Pydantic
- Auto-validation
- Clearer code
- Easier testing
- FastAPI-like (team familiar)

**Conversion Time:** 2-3 hours  
**Risk:** Very Low  
**Value:** High (research workflows use this)

---

#### 2. Redis Server ⭐⭐⭐⭐⭐ (PRODUCTION VALIDATION)

**Current:** 760 lines  
**After FastMCP:** ~180 lines

**FastMCP Implementation:**

```python
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Any, Optional, List
import redis.asyncio as aioredis

mcp = FastMCP("redis-cache")

# Type-safe request models
class SetValueRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = Field(None, description="TTL in seconds")
    serialize: bool = True

class SortedSetAdd(BaseModel):
    key: str
    score: float = Field(..., description="Score for ordering (timestamp)")
    member: Any
    serialize: bool = True

# Connection management (shared)
redis_client: Optional[aioredis.Redis] = None

async def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = aioredis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )
    return redis_client

# Tools (CONCISE!)
@mcp.tool()
async def get_value(key: str, deserialize: bool = True) -> Dict:
    """Get value from Redis cache."""
    redis = await get_redis()
    value = await redis.get(key)
    
    if value and deserialize:
        try:
            value = json.loads(value)
        except:
            pass
    
    return {
        "key": key,
        "value": value,
        "found": value is not None
    }

@mcp.tool()
async def set_value(request: SetValueRequest) -> Dict:
    """Set value in Redis with optional TTL."""
    redis = await get_redis()
    
    value = json.dumps(request.value) if request.serialize else request.value
    
    if request.ttl:
        await redis.setex(request.key, request.ttl, value)
    else:
        await redis.set(request.key, value)
    
    return {"success": True, "key": request.key}

@mcp.tool()
async def publish_message(channel: str, message: Any, serialize: bool = True) -> Dict:
    """Publish to Redis pub/sub channel."""
    redis = await get_redis()
    
    msg = json.dumps(message) if serialize else message
    subscribers = await redis.publish(channel, msg)
    
    return {
        "channel": channel,
        "subscribers": subscribers,
        "success": True
    }

# ... 5 more tools, each 10-15 lines
# Total: ~180 lines vs 760 lines!
```

**Benefits:**
- 580 lines saved (76% reduction!)
- Type-safe Redis operations
- Production-tested pattern
- Streaming API uses this (can validate)
- Pydantic validation built-in

**Conversion Time:** 3-4 hours  
**Risk:** Medium (production-critical)  
**Value:** Very High (validates FastMCP for production)

---

#### 3. SQL Analytics Server ⭐⭐⭐⭐ (MOST CODE SAVED)

**Current:** 1,210 lines (LARGEST SERVER!)  
**After FastMCP:** ~280 lines

**Why Valuable:**
```
✅ 930 lines saved (MOST of any server!)
✅ Complex analytics benefits from clean API
✅ 11 tools (most tools of any server)
✅ Type-safe SQL parameters
✅ Financial analytics platform core
✅ Frequently extended with new analytics
```

**FastMCP Implementation:**

```python
from fastmcp import FastMCP
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
import duckdb

mcp = FastMCP("sql-analytics")

# DuckDB connection (shared)
conn = duckdb.connect(":memory:")

class AggregationRequest(BaseModel):
    table: str
    aggregations: Dict[str, List[Literal["sum", "avg", "count", "min", "max"]]]
    group_by: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    order_by: Optional[List[str]] = None

@mcp.tool()
async def execute_query(
    query: str,
    params: Optional[List] = None,
    limit: int = 1000,
    format: Literal["json", "csv", "markdown"] = "json"
) -> Dict:
    """Execute SQL query and return structured data."""
    if "limit" not in query.lower():
        query = f"{query.rstrip(';')} LIMIT {limit}"
    
    result = conn.execute(query, params).fetchdf() if params else conn.execute(query).fetchdf()
    
    if format == "csv":
        return {"data": result.to_csv()}
    return {"data": result.to_dict(orient="records"), "rows": len(result)}

@mcp.tool()
async def aggregate_data(request: AggregationRequest) -> Dict:
    """Perform data aggregation with grouping and functions."""
    # Build SELECT
    select_parts = request.group_by or []
    for col, funcs in request.aggregations.items():
        for func in funcs:
            select_parts.append(f"{func.upper()}({col}) as {col}_{func}")
    
    query = f"SELECT {', '.join(select_parts)} FROM {request.table}"
    
    if request.filters:
        where = " AND ".join(f"{k} = ?" for k in request.filters.keys())
        query += f" WHERE {where}"
    
    if request.group_by:
        query += f" GROUP BY {', '.join(request.group_by)}"
    
    # Execute
    params = list(request.filters.values()) if request.filters else None
    result = conn.execute(query, params).fetchdf() if params else conn.execute(query).fetchdf()
    
    return {"data": result.to_dict(orient="records")}

@mcp.tool()
async def time_series_agg(
    table: str,
    timestamp_column: str,
    value_column: str,
    bucket: Literal["hour", "day", "week", "month"] = "day",
    aggfunc: Literal["sum", "avg", "min", "max"] = "avg"
) -> Dict:
    """Time-series aggregation with buckets."""
    query = f"""
        SELECT
            date_trunc('{bucket}', {timestamp_column}) as period,
            {aggfunc.upper()}({value_column}) as value
        FROM {table}
        GROUP BY period
        ORDER BY period
    """
    
    result = conn.execute(query).fetchdf()
    return {"data": result.to_dict(orient="records"), "periods": len(result)}

# ... 8 more tools, each 15-30 lines
# Total: ~280 lines vs 1,210 lines!
```

**Benefits:**
- 930 lines eliminated (77% reduction!)
- Type-safe analytics
- Pydantic models for complex queries
- Cleaner codebase
- Easier to extend with new analytics

**Conversion Time:** 4-5 hours  
**Risk:** Low (isolated)  
**Value:** Highest (most code saved + business value)

---

## 📈 EXPECTED OUTCOMES

### Code Reduction

**Phase 1 (arXiv):**
- Before: 724 lines
- After: 150 lines
- Saved: 574 lines (79%)

**Phase 2 (SQL + Redis):**
- Before: 1,970 lines
- After: 460 lines
- Saved: 1,510 lines (77%)

**Phase 3 (4 more servers):**
- Before: ~2,500 lines (estimated)
- After: ~600 lines
- Saved: ~1,900 lines (76%)

**TOTAL:**
- Before: 5,194 lines
- After: 1,210 lines
- **Saved: 3,984 lines (77% reduction!)**

### Development Velocity

**New MCP Server:**
- Official MCP: 2-3 days (learning + boilerplate)
- FastMCP: 2-3 hours (if know FastAPI)

**Speedup: 10x faster development**

### Code Quality

**Official MCP:**
- Verbose JSON schemas
- Manual tool routing
- Repetitive error handling
- Type safety via JSON validation

**FastMCP:**
- Clean Pydantic models
- Automatic tool routing
- Decorator-based composition
- Native Python type hints

**Result: 5x more readable code**

---

## 🎯 MIGRATION STRATEGY

### Conservative Approach (Recommended)

**1. New File Pattern:**
```
Keep:    research/arxiv_server.py (original)
Create:  research/arxiv_server_fastmcp.py (new)
Test:    Both in parallel
Compare: Performance, code clarity, team preference
Decide:  Switch when confident
```

**2. No Breaking Changes:**
- Original servers stay operational
- New FastMCP servers run alongside
- UnifiedMCPManager supports both
- Gradual migration, zero risk

**3. Validation:**
```
For each converted server:
1. Unit tests pass (same behavior)
2. Integration tests pass
3. Performance same or better
4. Team review positive
5. Then: Deprecate old, activate new
```

---

## 🔧 IMPLEMENTATION STEPS

### Step 1: Install FastMCP

```bash
# Add to requirements
uv add fastmcp

# Or manual
echo "fastmcp>=1.0.0" >> requirements.txt
pip install fastmcp
```

### Step 2: Convert arXiv Server (Proof of Concept)

```bash
# Create new file
touch axiom/integrations/mcp_servers/research/arxiv_server_fastmcp.py

# Implement with FastMCP (150 lines vs 724)
# Test thoroughly
# Compare results
```

### Step 3: Validate in Production

```python
# Register both versions temporarily
mcp_manager.register_server(arxiv_server_original)  # Keep as backup
mcp_manager.register_server(arxiv_server_fastmcp)   # Test new version

# Route traffic to new version
# Monitor performance
# Verify outputs identical
# If stable after 1 week → remove original
```

### Step 4: Expand to SQL + Redis

```bash
# Only if Phase 1 successful
# Convert SQL Analytics (930 lines saved!)
# Convert Redis (580 lines saved!)
# Validate production usage (streaming API)
```

---

## ⚖️ DECISION MATRIX

### Convert to FastMCP If:
- ✅ Server has 5+ simple tools
- ✅ Type safety would help (financial data)
- ✅ Code is verbose (>500 lines)
- ✅ Frequently modified (new features)
- ✅ Team prefers FastAPI style
- ✅ Not production-critical OR can test safely

### Keep Official MCP If:
- ⚠️ Server is complex (>1500 lines, complex state)
- ⚠️ Production-critical with no test environment
- ⚠️ Heavy cloud SDK integration (AWS/GCP wrappers)
- ⚠️ Stable, rarely modified
- ⚠️ Migration effort > benefit

---

## 🏆 RECOMMENDED PLAN

### This Week: Phase 1 (Proof of Concept)

**Convert:** arXiv Research Server

**Timeline:**
- Day 1: Install FastMCP, create new file (1 hour)
- Day 2: Implement 8 tools with FastMCP (2-3 hours)
- Day 3: Test, validate, compare (1-2 hours)
- Day 4: Team review, decision

**Success = Proceed to Phase 2**

### Next Week: Phase 2 (Production Validation)

**Convert:** SQL Analytics + Redis

**Timeline:**
- Days 1-2: SQL Analytics (4-5 hours)
- Days 3-4: Redis Server (3-4 hours)
- Day 5: Production testing, monitoring
- Result: 1,510 lines saved, FastMCP proven

### Month 2: Phase 3 (Expansion)

**Convert:** PDF, Notification, Excel, Prometheus (if Phase 1-2 successful)

**Total Impact:** ~4,000 lines saved, 10x faster new server development

---

## 📝 RISKS & MITIGATION

### Risk 1: FastMCP Spec Lag

**Risk:** FastMCP may lag behind official MCP spec updates

**Mitigation:**
- Keep official servers as reference
- Monitor FastMCP GitHub for updates
- Community is active (growing adoption)
- Can contribute fixes if needed

### Risk 2: Production Stability

**Risk:** New code might have bugs

**Mitigation:**
- Parallel deployment (both versions)
- Thorough testing before switch
- Easy rollback (keep original)
- Gradual traffic migration

### Risk 3: Team Learning Curve

**Risk:** Team needs to learn FastMCP

**Mitigation:**
- Very similar to FastAPI (already know)
- Better docs than official MCP
- Simpler API (easier to learn)
- Start with 1 server (low risk)

---

## 🎯 SUCCESS METRICS

### Code Quality
- ✅ 75%+ code reduction
- ✅ Type safety with Pydantic
- ✅ Cleaner, more readable
- ✅ Easier to maintain

### Development Velocity
- ✅ 10x faster new server creation
- ✅ Easier to add new tools
- ✅ Faster debugging
- ✅ Better testing

### Production Stability
- ✅ Same or better performance
- ✅ Zero regressions
- ✅ All tests passing
- ✅ Team approval

---

## 🏁 RECOMMENDATION

### **START WITH:** arXiv Research Server (Phase 1)

**Why:**
1. Easiest conversion (574 lines → 150 lines)
2. Lowest risk (isolated, non-critical)
3. Highest learning value (proves FastMCP)
4. Quick win (2-3 hours)
5. Clear success criteria

**Action:**
1. Install: `uv add fastmcp`
2. Create: `research/arxiv_server_fastmcp.py`
3. Implement: 8 tools with decorators (~150 lines)
4. Test: Unit + integration tests
5. Compare: Side-by-side with original
6. Decide: Proceed to Phase 2 if successful

**Expected Result:**
- Proof that FastMCP works for Axiom
- 79% code reduction demonstrated
- Team preference validated
- Green light for SQL + Redis conversion

---

**Total Potential:** 3,984 lines saved across 7 servers (77% reduction)  
**Recommended Start:** arXiv server (2-3 hours, 574 lines saved, low risk)  
**Next:** SQL + Redis (6-8 hours, 1,510 lines saved, production validation)  
**Future:** 4 more servers (if Phases 1-2 successful)