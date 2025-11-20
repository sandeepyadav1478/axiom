"""
Apache Airflow DAG: Company Graph Builder
Orchestrates the LangGraph-powered company relationship analysis

This DAG:
1. Fetches company data from Yahoo Finance
2. Uses Claude to identify competitors and sector peers
3. Generates Cypher queries for Neo4j relationships
4. Updates the knowledge graph
5. Validates the results

Schedule: Hourly
Retries: 3 attempts with 5-minute delays
Timeout: 30 minutes per run
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os

# Add axiom to Python path
sys.path.insert(0, '/opt/airflow/axiom')

from dotenv import load_dotenv
load_dotenv('/opt/airflow/.env')

# ================================================================
# DAG Configuration
# ================================================================
default_args = {
    'owner': 'axiom',
    'depends_on_past': False,
    'email': ['admin@axiom.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
    'sla': timedelta(minutes=45)  # Alert if takes longer than 45min
}

# Define symbols to process (can be parameterized)
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
    'META', 'AMZN', 'NFLX', 'GOOG', 'CRM',
    'ORCL', 'ADBE', 'INTC', 'AMD', 'QCOM',
    'AVGO', 'CSCO', 'TXN', 'UBER', 'LYFT',
    'SNAP', 'TWTR', 'JPM', 'BAC', 'GS',
    'MS', 'C', 'WFC', 'USB', 'PNC'
]

# ================================================================
# Task Functions
# ================================================================

def initialize_pipeline(**context):
    """Initialize connections and verify setup"""
    try:
        # Verify Neo4j connection
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
        driver.verify_connectivity()
        driver.close()
        
        context['ti'].xcom_push(key='pipeline_initialized', value=True)
        return {'status': 'initialized', 'timestamp': datetime.now().isoformat()}
    except Exception as e:
        raise Exception(f"Pipeline initialization failed: {e}")


def fetch_company_data(**context):
    """Task 1: Fetch company data from Yahoo Finance"""
    import yfinance as yf
    
    companies = {}
    failed = []
    
    for symbol in SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            companies[symbol] = {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0)
            }
        except Exception as e:
            failed.append(symbol)
            print(f"Failed to fetch {symbol}: {e}")
    
    # Store in XCom for next task
    context['ti'].xcom_push(key='company_data', value=companies)
    context['ti'].xcom_push(key='failed_symbols', value=failed)
    
    return {
        'fetched': len(companies),
        'failed': len(failed),
        'symbols': list(companies.keys())
    }


def identify_relationships(**context):
    """Task 2: Use Claude to identify company relationships"""
    import asyncio
    from neo4j import GraphDatabase
    from langchain_anthropic import ChatAnthropic
    
    # Get company data from previous task
    companies = context['ti'].xcom_pull(key='company_data', task_ids='fetch_companies')
    
    claude = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        max_tokens=4096
    )
    
    relationships = {}
    
    for symbol, info in companies.items():
        try:
            # Ask Claude to identify competitors
            prompt = f"""Identify the top 5 direct competitors for {info['name']} ({symbol}).
            Sector: {info['sector']}
            Industry: {info['industry']}
            
            Return only ticker symbols, comma-separated."""
            
            response = claude.invoke(prompt)
            competitors = response.content.strip().split(',')
            competitors = [c.strip() for c in competitors]
            
            relationships[symbol] = {
                'competitors': competitors,
                'sector': info['sector']
            }
            
        except Exception as e:
            print(f"Failed to analyze {symbol}: {e}")
    
    # Store for next task
    context['ti'].xcom_push(key='relationships', value=relationships)
    
    return {
        'analyzed': len(relationships),
        'total_relationships': sum(len(r['competitors']) for r in relationships.values())
    }


def generate_cypher_queries(**context):
    """Task 3: Generate Cypher queries for Neo4j"""
    relationships = context['ti'].xcom_pull(key='relationships', task_ids='identify_relationships')
    companies = context['ti'].xcom_pull(key='company_data', task_ids='fetch_companies')
    
    queries = []
    
    for symbol, rels in relationships.items():
        company_info = companies[symbol]
        
        # Create company node
        queries.append({
            'query': f"""
            MERGE (c:Company {{symbol: $symbol}})
            SET c.name = $name,
                c.sector = $sector,
                c.industry = $industry,
                c.market_cap = $market_cap,
                c.updated_at = datetime()
            """,
            'params': {
                'symbol': symbol,
                'name': company_info['name'],
                'sector': company_info['sector'],
                'industry': company_info['industry'],
                'market_cap': company_info['market_cap']
            }
        })
        
        # Create competitor relationships
        for competitor in rels['competitors']:
            queries.append({
                'query': f"""
                MATCH (c1:Company {{symbol: $symbol}})
                MERGE (c2:Company {{symbol: $competitor}})
                MERGE (c1)-[r:COMPETES_WITH]->(c2)
                SET r.identified_at = datetime()
                """,
                'params': {
                    'symbol': symbol,
                    'competitor': competitor
                }
            })
    
    context['ti'].xcom_push(key='cypher_queries', value=queries)
    
    return {
        'queries_generated': len(queries),
        'companies': len(relationships)
    }


def execute_neo4j_updates(**context):
    """Task 4: Execute Cypher queries in Neo4j"""
    from neo4j import GraphDatabase
    
    queries = context['ti'].xcom_pull(key='cypher_queries', task_ids='generate_cypher')
    
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    
    successful = 0
    failed = 0
    
    for query_info in queries:
        try:
            driver.execute_query(
                query_info['query'],
                query_info['params']
            )
            successful += 1
        except Exception as e:
            print(f"Query failed: {e}")
            failed += 1
    
    driver.close()
    
    return {
        'successful': successful,
        'failed': failed,
        'total': len(queries)
    }


def validate_graph(**context):
    """Task 5: Validate Neo4j graph was updated correctly"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    
    # Count nodes
    node_result = driver.execute_query(
        "MATCH (c:Company) RETURN count(c) as count"
    )
    node_count = node_result.records[0]['count'] if node_result.records else 0
    
    # Count relationships
    rel_result = driver.execute_query(
        "MATCH ()-[r:COMPETES_WITH]->() RETURN count(r) as count"
    )
    rel_count = rel_result.records[0]['count'] if rel_result.records else 0
    
    # Get sample
    sample_result = driver.execute_query(
        "MATCH (c:Company {symbol: 'AAPL'})-[:COMPETES_WITH]->(comp) RETURN comp.symbol as symbol LIMIT 5"
    )
    sample = [rec['symbol'] for rec in sample_result.records] if sample_result.records else []
    
    driver.close()
    
    validation_result = {
        'total_companies': node_count,
        'total_relationships': rel_count,
        'sample_aapl_competitors': sample,
        'validation_passed': node_count > 0 and rel_count > 0
    }
    
    if not validation_result['validation_passed']:
        raise Exception("Graph validation failed: No data found in Neo4j")
    
    return validation_result


# ================================================================
# Define the DAG
# ================================================================
with DAG(
    dag_id='company_graph_builder',
    default_args=default_args,
    description='Build company relationship graph using LangGraph + Claude + Neo4j',
    schedule_interval='@hourly',  # Run every hour
    start_date=days_ago(1),
    catchup=False,  # Don't backfill
    tags=['langgraph', 'claude', 'neo4j', 'knowledge-graph'],
    max_active_runs=1,  # Only one instance at a time
) as dag:
    
    # Task 0: Initialize
    init_task = PythonOperator(
        task_id='initialize_pipeline',
        python_callable=initialize_pipeline,
        provide_context=True
    )
    
    # Task 1: Fetch company data
    fetch_task = PythonOperator(
        task_id='fetch_companies',
        python_callable=fetch_company_data,
        provide_context=True
    )
    
    # Task 2: Identify relationships with Claude
    identify_task = PythonOperator(
        task_id='identify_relationships',
        python_callable=identify_relationships,
        provide_context=True
    )
    
    # Task 3: Generate Cypher queries
    cypher_task = PythonOperator(
        task_id='generate_cypher',
        python_callable=generate_cypher_queries,
        provide_context=True
    )
    
    # Task 4: Execute Neo4j updates
    execute_task = PythonOperator(
        task_id='execute_neo4j',
        python_callable=execute_neo4j_updates,
        provide_context=True
    )
    
    # Task 5: Validate results
    validate_task = PythonOperator(
        task_id='validate_graph',
        python_callable=validate_graph,
        provide_context=True
    )
    
    # Define task dependencies (linear workflow)
    init_task >> fetch_task >> identify_task >> cypher_task >> execute_task >> validate_task

# ================================================================
# DAG Documentation
# ================================================================
dag.doc_md = """
# Company Graph Builder DAG

## Purpose
Build a comprehensive company relationship knowledge graph using AI-powered analysis.

## Workflow
1. **Initialize**: Setup pipeline and connections
2. **Fetch**: Get company data from Yahoo Finance
3. **Identify**: Use Claude AI to identify competitors and sector relationships
4. **Generate**: Create Cypher queries for Neo4j
5. **Execute**: Update the knowledge graph
6. **Validate**: Ensure data was written correctly

## Technologies
- **LangGraph**: Workflow orchestration
- **Claude Sonnet 4**: Relationship identification
- **Neo4j**: Knowledge graph storage
- **Yahoo Finance**: Company data source

## Monitoring
- Check Airflow UI for task status
- SLA: 45 minutes (alert if exceeded)
- Retries: 3 attempts with 5-minute delays
- Success rate tracked automatically

## Dependencies
- Neo4j must be running (bolt://localhost:7687)
- Claude API key must be configured
- PostgreSQL must be accessible for Airflow metadata
"""