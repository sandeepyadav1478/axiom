"""
Apache Airflow DAG v2: Correlation Analyzer
Analyze stock correlations with centralized YAML configuration

IMPROVEMENTS OVER V1:
- ✅ CachedClaudeOperator (cost savings configurable)
- ✅ Batch correlation calculations (faster)
- ✅ Data quality validation on price data
- ✅ Circuit breaker for database operations
- ✅ Centralized YAML configuration

Schedule: Configurable via YAML
Cost: Configurable cache TTL and settings
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/opt/airflow')

from dotenv import load_dotenv
load_dotenv('/opt/airflow/.env')

# Import operators from local path
operators_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, operators_path)

from operators.claude_operator import CachedClaudeOperator
from operators.resilient_operator import CircuitBreakerOperator
from operators.quality_check_operator import DataQualityOperator

# Import centralized configuration
utils_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, utils_path)
from utils.config_loader import (
    dag_config,
    get_symbols_for_dag,
    build_postgres_conn_params,
    build_neo4j_conn_params
)

# ================================================================
# Load Configuration from YAML
# ================================================================
DAG_NAME = 'correlation_analyzer'
config = dag_config.get_dag_config(DAG_NAME)
default_args = dag_config.get_default_args(DAG_NAME)
SYMBOLS = get_symbols_for_dag(DAG_NAME)
circuit_breaker_config = dag_config.get_circuit_breaker_config(DAG_NAME)
claude_config = dag_config.get_claude_config(DAG_NAME)
corr_config = dag_config.get_correlation_config()
quality_config = config.get('quality', {})

# ================================================================
# Helper Functions
# ================================================================

def fetch_and_validate_prices(context):
    """Fetch price history with validation"""
    import psycopg2
    import pandas as pd
    
    # Get config from context
    symbols = context['params'].get('symbols', SYMBOLS)
    lookback_days = context['params'].get('lookback_days', 30)
    
    # Use centralized config for DB connection
    db_params = build_postgres_conn_params()
    conn = psycopg2.connect(**db_params)
    
    cur = conn.cursor()
    price_data = {}
    
    for symbol in symbols:
        cur.execute(f"""
            SELECT timestamp, close
            FROM stock_prices
            WHERE symbol = %s
            AND timestamp > NOW() - INTERVAL '{lookback_days} days'
            ORDER BY timestamp
        """, (symbol,))
        
        rows = cur.fetchall()
        price_data[symbol] = [
            {'timestamp': row[0].isoformat(), 'price': float(row[1])}
            for row in rows
        ]
    
    cur.close()
    conn.close()
    
    # Validate we have enough data (configurable)
    min_data_points = context['params'].get('min_data_points', 20)
    insufficient = [s for s, prices in price_data.items() if len(prices) < min_data_points]
    
    if insufficient:
        context['task'].log.warning(
            f"⚠️ Insufficient data for: {insufficient} (need {min_data_points} points)"
        )
    
    context['ti'].xcom_push(key='price_history', value=price_data)
    
    return {
        'symbols': len(price_data),
        'data_points': sum(len(p) for p in price_data.values()),
        'insufficient_data': insufficient
    }


def calculate_correlations_batch(context):
    """Calculate correlations in batch"""
    import numpy as np
    import pandas as pd
    
    # Get config from context
    min_data_points = context['params'].get('min_data_points', 20)
    significance_threshold = context['params'].get('significance_threshold', 0.5)
    top_n = context['params'].get('top_n_correlations', 20)
    
    price_history = context['ti'].xcom_pull(
        task_ids='fetch_prices_validated',
        key='price_history'
    )
    
    # Convert to DataFrame
    df_dict = {}
    for symbol, prices in price_history.items():
        if len(prices) >= min_data_points:
            df_dict[symbol] = [p['price'] for p in prices]
    
    df = pd.DataFrame(df_dict)
    corr_matrix = df.corr()
    
    # Find significant correlations
    significant_corrs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            
            # Use configurable threshold
            if abs(corr_value) > significance_threshold:
                significant_corrs.append({
                    'symbol1': corr_matrix.columns[i],
                    'symbol2': corr_matrix.columns[j],
                    'correlation': float(corr_value),
                    'abs_correlation': abs(float(corr_value))
                })
    
    # Sort by absolute correlation
    significant_corrs.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    context['ti'].xcom_push(key='correlations', value=significant_corrs)
    
    return {
        'total_pairs': len(significant_corrs),
        'positive': len([c for c in significant_corrs if c['correlation'] > 0]),
        'negative': len([c for c in significant_corrs if c['correlation'] < 0]),
        'top_correlation': max([c['correlation'] for c in significant_corrs]) if significant_corrs else 0
    }


def create_correlation_relationships_batch(context):
    """Create CORRELATED_WITH relationships in batch"""
    from neo4j import GraphDatabase
    
    explained_corrs = context['ti'].xcom_pull(
        task_ids='explain_correlations_cached',
        key='claude_response'
    )
    
    if not explained_corrs or not explained_corrs.get('success'):
        return {'created': 0}
    
    # Parse Claude's explanations
    import json
    try:
        correlations = json.loads(explained_corrs['content'])
    except:
        correlations = []
    
    # Use centralized config for Neo4j connection
    neo4j_params = build_neo4j_conn_params()
    driver = GraphDatabase.driver(
        neo4j_params['uri'],
        auth=(neo4j_params['user'], neo4j_params['password'])
    )
    
    # Batch create relationships
    query = """
    UNWIND $correlations AS corr
    MATCH (c1:Company {symbol: corr.symbol1})
    MATCH (c2:Company {symbol: corr.symbol2})
    MERGE (c1)-[r:CORRELATED_WITH]-(c2)
    SET r.correlation = corr.correlation,
        r.explanation = corr.explanation,
        r.updated_at = datetime()
    """
    
    records, summary, keys = driver.execute_query(query, {'correlations': correlations})
    driver.close()
    
    return {
        'relationships_created': summary.counters.relationships_created
    }


# ================================================================
# Define the Enhanced DAG (Config-Driven)
# ================================================================
with DAG(
    dag_id=config.get('dag_id', 'correlation_analyzer_v2'),
    default_args=default_args,
    description=config.get('description', 'Correlation analysis with cached explanations'),
    schedule_interval=config.get('schedule_interval', '@hourly'),
    start_date=days_ago(1),
    catchup=dag_config.get_global('catchup', False),
    tags=config.get('tags', ['v2', 'enterprise']),
    max_active_runs=dag_config.get_global('max_active_runs', 1),
) as dag:
    
    # Task 1: Fetch and validate price data (config-driven parameters)
    fetch_prices = CircuitBreakerOperator(
        task_id='fetch_prices_validated',
        callable_func=fetch_and_validate_prices,
        failure_threshold=circuit_breaker_config.get('failure_threshold', 5),
        recovery_timeout_seconds=circuit_breaker_config.get('recovery_timeout_seconds', 120),
        xcom_key='fetch_result',
        params={
            'symbols': SYMBOLS,
            'lookback_days': corr_config.get('lookback_days', 30),
            'min_data_points': corr_config.get('min_data_points', 20)
        }
    )
    
    # Task 2: Validate price data quality (config-driven)
    min_recent_symbols = quality_config.get('min_recent_symbols', 10)
    max_null_percent = quality_config.get('max_null_percent', 0)
    
    validate_prices = DataQualityOperator(
        task_id='validate_price_data',
        table_name='stock_prices',
        checks=[
            {
                'name': 'recent_prices',
                'type': 'custom_sql',
                'sql': f"""
                    SELECT COUNT(DISTINCT symbol) >= {min_recent_symbols}
                    FROM stock_prices
                    WHERE timestamp > NOW() - INTERVAL '1 day'
                """,
                'expected': True
            },
            {
                'name': 'no_null_prices',
                'type': 'null_count',
                'column': 'close',
                'max_null_percent': max_null_percent
            }
        ],
        fail_on_error=False
    )
    
    # Task 3: Calculate correlations (config-driven thresholds)
    calc_correlations = CircuitBreakerOperator(
        task_id='calculate_correlations',
        callable_func=calculate_correlations_batch,
        failure_threshold=circuit_breaker_config.get('failure_threshold', 3),
        recovery_timeout_seconds=circuit_breaker_config.get('recovery_timeout_seconds', 60),
        xcom_key='calc_result',
        params={
            'min_data_points': corr_config.get('min_data_points', 20),
            'significance_threshold': corr_config.get('significance_threshold', 0.5),
            'top_n_correlations': corr_config.get('top_n_correlations', 20)
        }
    )
    
    # Task 4: Explain correlations with CACHED Claude (config-driven)
    explain_correlations = CachedClaudeOperator(
        task_id='explain_correlations_cached',
        prompt="""Explain these stock correlations:
{{ ti.xcom_pull(task_ids='calculate_correlations', key='correlations')[:""" + str(corr_config.get('top_n_correlations', 20)) + """] }}

For each correlation pair, provide a one-sentence explanation of why they're correlated.

Return as JSON:
[
  {
    "symbol1": "AAPL",
    "symbol2": "MSFT",
    "correlation": 0.85,
    "explanation": "Both are large-cap tech companies..."
  },
  ...
]""",
        system_message='You are a quantitative analyst specializing in statistical relationships.',
        max_tokens=claude_config.get('max_tokens', 2048),
        cache_ttl_hours=claude_config.get('cache_ttl_hours', 48),
        track_cost=claude_config.get('track_cost', True),
        xcom_key='claude_response'
    )
    
    # Task 5: Create relationships in batch (config-driven)
    create_relationships = CircuitBreakerOperator(
        task_id='create_relationships_batch',
        callable_func=create_correlation_relationships_batch,
        failure_threshold=circuit_breaker_config.get('failure_threshold', 5),
        recovery_timeout_seconds=circuit_breaker_config.get('recovery_timeout_seconds', 60),
        xcom_key='create_result'
    )
    
    # Dependencies
    fetch_prices >> validate_prices >> calc_correlations >> explain_correlations >> create_relationships


dag.doc_md = """
# Enhanced Correlation Analyzer DAG (Config-Driven)

## 🚀 Centralized Configuration

### All Settings in YAML
- **Lookback period**: Configurable (default: 30 days)
- **Min data points**: Configurable (default: 20)
- **Significance threshold**: Configurable (default: 0.5)
- **Top N correlations**: Configurable (default: 20)
- **Cache TTL**: Configurable (default: 48 hours)
- **Quality thresholds**: All configurable

### Cost Optimization
- **Cache TTL**: Configurable for cost/freshness balance
- **Correlation changes slowly**: Long cache periods save money
- **Track costs**: Configurable cost tracking

### Performance
- **Correlation calculation**: Fast batch processing
- **Claude explanations**: Cache TTL configurable
- **Neo4j operations**: Circuit breaker configurable
- **All thresholds**: Tunable via YAML

## 🎯 Data Quality (Configurable)

Validates:
- Minimum symbols with recent data (configurable)
- No null prices (configurable tolerance)
- Sufficient historical data (configurable min points)
- Correlation significance (configurable threshold)

## 💡 Why Configuration-Driven?

1. **Easy Tuning**: Adjust thresholds without code changes
2. **Cost Control**: Configure cache duration to balance cost/freshness
3. **Performance Optimization**: Tune batch sizes and timeouts
4. **Quality Standards**: Set validation thresholds per environment
5. **Flexibility**: Different settings for dev/staging/prod

All parameters centralized in [`dag_config.yaml`](../dag_configs/dag_config.yaml)!
"""