"""
Company Enrichment DAG
Expand company universe with RICH metadata profiles

Purpose: Create deep company profiles for AI showcase (LangGraph/DSPy/Neo4j)
Target: 50 companies with detailed text descriptions, relationships, risk factors
Focus: TEXT data for AI, not just numbers

This showcases:
- LangGraph: Multi-step enrichment workflow
- DSPy: Structured extraction from text
- Claude: Intelligent analysis
- Neo4j: Rich knowledge graph
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import sys
import os

sys.path.insert(0, '/opt/airflow')

from dotenv import load_dotenv
load_dotenv('/opt/airflow/.env', override=True)

# Import operators
operators_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, operators_path)
from operators.claude_operator import CachedClaudeOperator
from operators.resilient_operator import CircuitBreakerOperator

# Import utilities
utils_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, utils_path)
from utils.config_loader import dag_config, get_symbols_for_dag

# ================================================================
# Configuration
# ================================================================
DAG_NAME = 'company_enrichment'
config = dag_config.get_dag_config(DAG_NAME)
default_args = dag_config.get_default_args(DAG_NAME)
SYMBOLS = get_symbols_for_dag(DAG_NAME)  # Will use extended list (50 companies)
enrichment_config = config.get('enrichment', {})
claude_config = dag_config.get_claude_config(DAG_NAME)

# ================================================================
# Helper Functions
# ================================================================

def fetch_company_metadata(context):
    """
    Fetch rich company metadata from yfinance.
    
    Collects TEXT descriptions for AI processing:
    - Business summary (paragraph)
    - Sector, industry
    - Products/services
    - Geographic presence
    - Key people
    """
    import yfinance as yf
    
    symbols = context['params'].get('symbols', SYMBOLS)
    batch_size = context['params'].get('batch_size', 10)
    
    # Process in batches
    current_batch = context['params'].get('batch_number', 0)
    start_idx = current_batch * batch_size
    end_idx = min(start_idx + batch_size, len(symbols))
    batch_symbols = symbols[start_idx:end_idx]
    
    companies = []
    failed = []
    
    for symbol in batch_symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract rich TEXT data
            company = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                
                # TEXT FIELDS (perfect for DSPy extraction)
                'business_summary': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'city': info.get('city', ''),
                'state': info.get('state', ''),
                'country': info.get('country', ''),
                
                # Company size
                'market_cap': info.get('marketCap', 0),
                'employees': info.get('fullTimeEmployees', 0),
                
                # Financial highlights
                'revenue': info.get('totalRevenue', 0),
                'profit_margin': info.get('profitMargins', 0),
                'pe_ratio': info.get('trailingPE', 0),
                
                # Metadata
                'fetched_at': datetime.now().isoformat()
            }
            
            companies.append(company)
            
            context['task'].log.info(
                f"✅ {symbol}: {company['name']} - {len(company['business_summary'])} char description"
            )
            
        except Exception as e:
            context['task'].log.error(f"Failed to fetch {symbol}: {e}")
            failed.append(symbol)
    
    context['ti'].xcom_push(key='company_metadata', value=companies)
    context['ti'].xcom_push(key='failed_symbols', value=failed)
    
    result = {
        'batch_number': current_batch,
        'symbols_processed': len(companies),
        'symbols_failed': len(failed),
        'total_symbols_in_batch': len(batch_symbols)
    }
    
    context['task'].log.info(
        f"Batch {current_batch}: {len(companies)}/{len(batch_symbols)} companies fetched"
    )
    
    return result


def create_company_nodes(context):
    """
    Create rich Company nodes in Neo4j.
    
    Stores TEXT descriptions for future Claude analysis.
    """
    from neo4j import GraphDatabase
    
    companies = context['ti'].xcom_pull(task_ids='fetch_metadata', key='company_metadata')
    
    if not companies:
        context['task'].log.warning("No companies to create")
        return {'created': 0}
    
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    created_count = 0
    
    with driver.session() as session:
        for company in companies:
            try:
                session.run("""
                    MERGE (c:Company {symbol: $symbol})
                    SET c.name = $name,
                        c.sector = $sector,
                        c.industry = $industry,
                        c.business_summary = $business_summary,
                        c.website = $website,
                        c.city = $city,
                        c.country = $country,
                        c.market_cap = $market_cap,
                        c.employees = $employees,
                        c.revenue = $revenue,
                        c.profit_margin = $profit_margin,
                        c.pe_ratio = $pe_ratio,
                        c.updated_at = datetime($updated_at)
                """, **company, updated_at=datetime.now().isoformat())
                
                created_count += 1
                
            except Exception as e:
                context['task'].log.error(f"Failed to create node for {company['symbol']}: {e}")
    
    driver.close()
    
    context['task'].log.info(f"✅ Created/updated {created_count} Company nodes in Neo4j")
    
    return {'created': created_count}


def store_in_postgresql(context):
    """
    Store company metadata in PostgreSQL company_fundamentals table.
    
    Provides SQL-queryable company data.
    """
    import psycopg2
    from psycopg2.extras import execute_batch
    
    companies = context['ti'].xcom_pull(task_ids='fetch_metadata', key='company_metadata')
    
    if not companies:
        return {'stored': 0}
    
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        user=os.getenv('POSTGRES_USER', 'axiom'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB', 'axiom_finance')
    )
    
    cur = conn.cursor()
    
    # Prepare records
    records = []
    for company in companies:
        record = {
            'symbol': company['symbol'],
            'report_date': datetime.now(),
            'fiscal_period': 'CURRENT',
            'company_name': company['name'],
            'sector': company['sector'],
            'industry': company['industry'],
            'market_cap': company['market_cap'],
            'revenue': company['revenue']
        }
        records.append(record)
    
    # Batch insert
    insert_query = """
        INSERT INTO company_fundamentals 
        (symbol, report_date, fiscal_period, company_name, sector, industry, market_cap, revenue)
        VALUES (%(symbol)s, %(report_date)s, %(fiscal_period)s, %(company_name)s, 
                %(sector)s, %(industry)s, %(market_cap)s, %(revenue)s)
        ON CONFLICT (symbol, report_date, fiscal_period) DO UPDATE
        SET company_name = EXCLUDED.company_name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            market_cap = EXCLUDED.market_cap,
            revenue = EXCLUDED.revenue
    """
    
    execute_batch(cur, insert_query, records, page_size=50)
    conn.commit()
    
    rows_affected = cur.rowcount
    
    cur.close()
    conn.close()
    
    context['task'].log.info(f"✅ Stored {rows_affected} company records in PostgreSQL")
    
    return {'stored': rows_affected}


# ================================================================
# Define DAG
# ================================================================
with DAG(
    dag_id=config.get('dag_id', 'company_enrichment'),
    default_args=default_args,
    description=config.get('description', 'Enrich company universe with deep profiles'),
    schedule_interval=config.get('schedule_interval', None),  # Manual trigger
    start_date=days_ago(1),
    catchup=False,
    tags=config.get('tags', ['enrichment', 'companies', 'ai-showcase']),
    max_active_runs=1,
) as dag:
    
    # Task 1: Fetch company metadata (10 at a time)
    fetch_metadata = CircuitBreakerOperator(
        task_id='fetch_metadata',
        callable_func=fetch_company_metadata,
        failure_threshold=5,
        recovery_timeout_seconds=120,
        xcom_key='fetch_result',
        params={
            'symbols': SYMBOLS,
            'batch_size': enrichment_config.get('batch_size', 10),
            'batch_number': 0  # Can parameterize for multiple runs
        }
    )
    
    # Task 2: Extract competitors with Claude (DSPy-style)
    extract_competitors = CachedClaudeOperator(
        task_id='extract_competitors',
        prompt="""Analyze these companies and identify their top 3-5 direct competitors:

{{ ti.xcom_pull(task_ids='fetch_metadata', key='company_metadata') }}

For each company, identify competitors based on:
- Same industry/sector
- Similar products/services  
- Overlapping markets

Return ONLY a JSON object (no markdown):
{
  "SYMBOL": ["COMP1", "COMP2", "COMP3"],
  ...
}""",
        system_message='You are a market analyst. Return ONLY valid JSON, no other text.',
        max_tokens=claude_config.get('max_tokens', 2048),
        cache_ttl_hours=claude_config.get('cache_ttl_hours', 168),  # 7 days
        track_cost=claude_config.get('track_cost', True),
        xcom_key='competitors'
    )
    
    # Task 3: Extract key products/services with Claude
    extract_products = CachedClaudeOperator(
        task_id='extract_products',
        prompt="""From these business summaries, extract the key products and services:

{{ ti.xcom_pull(task_ids='fetch_metadata', key='company_metadata') }}

For each company, list their 3-5 main products/services.

Return ONLY a JSON object:
{
  "SYMBOL": ["Product1", "Product2", "Product3"],
  ...
}""",
        system_message='You are a business analyst. Return ONLY valid JSON.',
        max_tokens=claude_config.get('max_tokens', 2048),
        cache_ttl_hours=claude_config.get('cache_ttl_hours', 168),
        track_cost=True,
        xcom_key='products'
    )
    
    # Task 4: Create Neo4j nodes with metadata
    create_nodes = PythonOperator(
        task_id='create_nodes',
        python_callable=create_company_nodes
    )
    
    # Task 5: Store in PostgreSQL
    store_postgres = PythonOperator(
        task_id='store_postgres',
        python_callable=store_in_postgresql
    )
    
    # Task dependencies
    fetch_metadata >> [extract_competitors, extract_products, create_nodes, store_postgres]


# ================================================================
# DAG Documentation
# ================================================================
dag.doc_md = """
# Company Enrichment DAG

## Purpose

Expand company universe from 5 to 50 companies with RICH TEXT profiles for AI showcase.

**Focus:** Deep qualitative data (descriptions, relationships) NOT just numbers  
**Showcases:** LangGraph workflows + DSPy extraction + Claude analysis + Neo4j graphs

## What It Does

### 1. Fetches Rich Metadata (yfinance)
```
For each company, collects:
├─ Business summary (200-1000 char TEXT) ← Perfect for Claude!
├─ Sector & industry classification
├─ Geographic presence (city, state, country)
├─ Website, employee count
├─ Financial highlights (revenue, margins)
└─ Market cap, P/E ratio

Data Type: 80% TEXT, 20% numbers
Perfect for: DSPy extraction, Claude analysis
```

### 2. Extracts Competitors (Claude + DSPy Pattern)
```
Claude analyzes business summaries:
├─ Input: "Apple designs smartphones, tablets, computers..."
├─ DSPy-style extraction: Identify competitors from text
├─ Output: ["Samsung", "Google", "Microsoft"]
└─ Caches for 7 days (competitors don't change fast)

Showcase: Intelligent structured extraction from unstructured text
```

### 3. Extracts Products (Claude + NLP)
```
Claude extracts key offerings:
├─ Input: Business summary text
├─ Claude identifies: Main products/services
├─ Output: ["iPhone", "iPad", "Mac", "Services"]
└─ Structured from unstructured

Showcase: NLP and entity extraction
```

### 4. Creates Rich Neo4j Nodes
```cypher
CREATE (c:Company {
  symbol: 'AAPL',
  name: 'Apple Inc.',
  business_summary: '500 char description...',  ← TEXT for AI!
  sector: 'Technology',
  industry: 'Consumer Electronics',
  website: 'https://apple.com',
  country: 'United States',
  market_cap: 2800000000000,
  employees: 164000,
  products: ['iPhone', 'iPad', 'Mac'],
  competitors: ['Samsung', 'Google', 'Microsoft']
})

Then: CREATE relationships
MATCH (aapl:Company {symbol: 'AAPL'})
MATCH (samsung:Company {symbol: 'SSNLF'})
CREATE (aapl)-[:COMPETES_WITH {confidence: 0.9}]->(samsung)
```

### 5. Stores in PostgreSQL
```
Table: company_fundamentals
Purpose: SQL-queryable company data
Enables: Fast filtering, screening queries
```

## Target Company List (50 Total)

### Batch 1 (Top 10 Mega-Cap):
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK.B, UNH, JPM

### Batch 2 (Next 10 Large-Cap Tech):
V, MA, HD, PG, DIS, NFLX, ADBE, CRM, ORCL, INTC

### Batch 3 (Next 10 Large-Cap Diversified):
PFE, MRK, ABBV, KO, PEP, WMT, COST, CVX, XOM, BA

### Batch 4-5 (Next 20 Companies):
Mid-large cap across sectors for diversity

## Processing Strategy

### Batched Execution
```
Batch Size: 10 companies per run
Total Batches: 5 runs
Reason: Avoid API rate limits, manageable processing
```

### Parameterized Runs
```bash
# Run batch 0 (companies 0-9)
airflow dags trigger company_enrichment --conf '{"batch_number": 0}'

# Run batch 1 (companies 10-19)
airflow dags trigger company_enrichment --conf '{"batch_number": 1}'

# ... repeat for batches 2-4
```

### Or: Sequential Automation
```
Create wrapper DAG that triggers this 5 times with different batch_number
OR: Run manually 5 times over a week
```

## What This Enables (AI Showcase)

### LangGraph Demonstrations
```python
# Multi-step company analysis workflow
Workflow:
├─ Step 1: Fetch metadata (structured API call)
├─ Step 2: Claude extracts competitors (unstructured → structured)
├─ Step 3: Claude extracts products (text → entities)
├─ Step 4: Create Neo4j nodes (data persistence)
└─ Step 5: Build relationships (knowledge graph)

Shows: Complex multi-agent orchestration
```

### DSPy Showcase
```python
# Structured extraction from text
Input: Business summary (unstructured text)
DSPy/Claude: Extract facts
Output: {
  "competitors": [...],
  "products": [...],
  "markets": [...],
  "strengths": [...]
}

Shows: Prompt optimization, structured outputs
```

### Neo4j Graph Growth
```
Before: 5 companies, basic data
After:  50 companies, RICH profiles
├─ Business summaries for NLP
├─ Competitor relationships
├─ Product catalogs
├─ Geographic presence
└─ Ready for graph ML

Shows: Knowledge graph construction
```

## Configuration

Edit [`dag_config.yaml`](../dag_configs/dag_config.yaml):

```yaml
company_enrichment:
  dag_id: company_enrichment
  schedule_interval: null  # Manual trigger
  
  enrichment:
    batch_size: 10
    cache_ttl_hours: 168  # 7 days
  
  claude:
    max_tokens: 2048
    cache_ttl_hours: 168
    track_cost: true
  
  symbols_list: top_50  # Defines in YAML
```

## Expected Results

After 5 batch runs:
```
Neo4j:
├─ 50 Company nodes with rich TEXT descriptions
├─ 100+ COMPETES_WITH relationships
├─ 50+ BELONGS_TO_SECTOR relationships
└─ Foundation for M&A analysis graph

PostgreSQL:
├─ 50 records in company_fundamentals
└─ SQL-queryable company data

Knowledge:
├─ 50 detailed business summaries (TEXT)
├─ Competitor networks
├─ Product portfolios
└─ Ready for AI demonstrations
```

## Next Steps After Enrichment

1. **M&A Target Screening Demo:**
   - Query: "Find tech companies for acquisition"
   - LangGraph: Multi-agent analysis using rich text
   - Result: Ranked targets with strategic fit

2. **Competitive Intelligence:**
   - Query Neo4j for competitor networks
   - Claude analyzes competitive positioning
   - Generate market landscape reports

3. **Investment Research:**
   - Use business summaries for research
   - Extract insights with Claude
   - Build investment theses

---

**This DAG transforms the platform from "5 basic stocks" to "50 rich company profiles" - perfect foundation for AI showcase.**

Focuses on TEXT data for LangGraph/DSPy, not numerical data for traditional quant.
"""