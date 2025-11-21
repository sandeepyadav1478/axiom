"""
Apache Airflow DAG: Correlation Analyzer
Analyzes stock price correlations and explains them with Claude AI

This DAG:
1. Fetches 30-day price history from PostgreSQL
2. Calculates correlation matrix
3. Uses Claude to explain significant correlations
4. Creates CORRELATED_WITH relationships in Neo4j
5. Stores insights for portfolio optimization

Schedule: Hourly
Retries: 3 attempts
Timeout: 20 minutes per run
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os

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
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=20)
}

SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'CRM', 'ORCL']

# ================================================================
# Task Functions
# ================================================================

def fetch_price_history(**context):
    """Fetch 30-day price history from PostgreSQL"""
    import psycopg2
    from datetime import datetime, timedelta
    import json
    
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )
    
    cur = conn.cursor()
    
    # Get prices for last 30 days
    price_data = {}
    
    for symbol in SYMBOLS:
        cur.execute("""
            SELECT timestamp, close 
            FROM stock_prices 
            WHERE symbol = %s 
            AND timestamp > NOW() - INTERVAL '30 days'
            ORDER BY timestamp
        """, (symbol,))
        
        rows = cur.fetchall()
        price_data[symbol] = [
            {'timestamp': row[0].isoformat(), 'price': float(row[1])}
            for row in rows
        ]
    
    cur.close()
    conn.close()
    
    context['ti'].xcom_push(key='price_history', value=price_data)
    
    return {
        'symbols': len(price_data),
        'data_points': sum(len(prices) for prices in price_data.values())
    }


def calculate_correlations(**context):
    """Calculate correlation matrix"""
    import numpy as np
    import pandas as pd
    
    price_history = context['ti'].xcom_pull(key='price_history', task_ids='fetch_prices')
    
    # Convert to DataFrame
    df_dict = {}
    for symbol, prices in price_history.items():
        if len(prices) > 0:
            df_dict[symbol] = [p['price'] for p in prices]
    
    df = pd.DataFrame(df_dict)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Find significant correlations (> 0.7 or < -0.3)
    significant_corrs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) > 0.7 or corr_value < -0.3:
                significant_corrs.append({
                    'symbol1': corr_matrix.columns[i],
                    'symbol2': corr_matrix.columns[j],
                    'correlation': float(corr_value)
                })
    
    context['ti'].xcom_push(key='correlations', value=significant_corrs)
    
    return {
        'total_pairs': len(significant_corrs),
        'positive': len([c for c in significant_corrs if c['correlation'] > 0]),
        'negative': len([c for c in significant_corrs if c['correlation'] < 0])
    }


def explain_correlations_with_claude(**context):
    """Use Claude to explain why stocks are correlated"""
    from langchain_anthropic import ChatAnthropic
    
    correlations = context['ti'].xcom_pull(key='correlations', task_ids='calculate_correlations')
    
    claude = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        max_tokens=512
    )
    
    explained_corrs = []
    
    for corr in correlations[:20]:  # Explain top 20 correlations
        try:
            prompt = f"""Explain in one sentence why {corr['symbol1']} and {corr['symbol2']} have a correlation of {corr['correlation']:.2f}.
            
Consider:
- Shared sector/industry
- Business relationships
- Market dynamics
- Common risk factors

Format: Single sentence explanation."""

            response = claude.invoke(prompt)
            
            explained_corrs.append({
                **corr,
                'explanation': response.content.strip()
            })
            
        except Exception as e:
            print(f"Failed to explain correlation: {e}")
            explained_corrs.append({
                **corr,
                'explanation': 'Analysis failed'
            })
    
    context['ti'].xcom_push(key='explained_correlations', value=explained_corrs)
    
    return {
        'explained': len(explained_corrs)
    }


def create_correlation_relationships(**context):
    """Create CORRELATED_WITH relationships in Neo4j"""
    from pipelines.shared.neo4j_client import Neo4jGraphClient
    
    correlations = context['ti'].xcom_pull(
        key='explained_correlations',
        task_ids='explain_correlations'
    )
    
    neo4j = Neo4jGraphClient(
        uri=os.getenv('NEO4J_URI'),
        user=os.getenv('NEO4J_USER'),
        password=os.getenv('NEO4J_PASSWORD')
    )
    
    created = 0
    
    for corr in correlations:
        try:
            query = """
            MATCH (c1:Company {symbol: $symbol1})
            MATCH (c2:Company {symbol: $symbol2})
            MERGE (c1)-[r:CORRELATED_WITH]-(c2)
            SET r.correlation = $correlation,
                r.explanation = $explanation,
                r.updated_at = datetime()
            """
            
            neo4j.driver.execute_query(query, corr)
            created += 1
        except Exception as e:
            print(f"Failed to create correlation: {e}")
    
    neo4j.close()
    
    return {
        'correlations_created': created,
        'total': len(correlations)
    }


# ================================================================
# Define the DAG
# ================================================================
with DAG(
    dag_id='correlation_analyzer',
    default_args=default_args,
    description='Analyze and explain stock correlations using Claude AI',
    schedule_interval='@hourly',  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['langgraph', 'claude', 'neo4j', 'correlations', 'quant'],
    max_active_runs=1,
) as dag:
    
    fetch_task = PythonOperator(
        task_id='fetch_prices',
        python_callable=fetch_price_history,
        provide_context=True
    )
    
    calc_task = PythonOperator(
        task_id='calculate_correlations',
        python_callable=calculate_correlations,
        provide_context=True
    )
    
    explain_task = PythonOperator(
        task_id='explain_correlations',
        python_callable=explain_correlations_with_claude,
        provide_context=True
    )
    
    create_task = PythonOperator(
        task_id='create_relationships',
        python_callable=create_correlation_relationships,
        provide_context=True
    )
    
    # Linear workflow
    fetch_task >> calc_task >> explain_task >> create_task

dag.doc_md = """
# Correlation Analyzer DAG

## Purpose
Quantitatively analyze stock correlations and use AI to explain the relationships.

## Workflow
1. **Fetch**: Get 30-day price history from PostgreSQL
2. **Calculate**: Compute correlation matrix
3. **Explain**: Use Claude to explain significant correlations
4. **Store**: Create CORRELATED_WITH relationships in Neo4j

## Thresholds
- Strong positive: correlation > 0.7
- Moderate negative: correlation < -0.3

## Use Cases
- Portfolio diversification
- Risk management
- Pair trading strategies
- Sector rotation analysis
"""