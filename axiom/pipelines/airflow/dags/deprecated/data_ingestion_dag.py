"""
Apache Airflow DAG: Data Ingestion
Fetches real-time stock prices and stores in multi-database architecture

This DAG:
1. Fetches OHLCV data from Yahoo Finance
2. Stores in PostgreSQL (time-series data)
3. Caches in Redis (real-time access)
4. Updates Neo4j (graph context)

Schedule: Every minute
Retries: 2 attempts
Timeout: 5 minutes per run
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
    'email_on_failure': False,  # Too frequent for alerts
    'retries': 2,
    'retry_delay': timedelta(seconds=30),
    'execution_timeout': timedelta(minutes=5)
}

SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX',
    'GOOG', 'CRM', 'ORCL', 'ADBE', 'INTC', 'AMD', 'QCOM', 'AVGO',
    'CSCO', 'TXN', 'UBER', 'LYFT', 'SNAP', 'JPM', 'BAC', 'GS', 'MS'
]

# ================================================================
# Task Functions
# ================================================================

def fetch_market_data(**context):
    """Fetch current market data"""
    import yfinance as yf
    from datetime import datetime
    
    market_data = []
    
    for symbol in SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='1m')
            
            if not hist.empty:
                latest = hist.iloc[-1]
                market_data.append({
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'open': float(latest['Open']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'close': float(latest['Close']),
                    'volume': int(latest['Volume'])
                })
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
    
    context['ti'].xcom_push(key='market_data', value=market_data)
    
    return {
        'symbols_fetched': len(market_data),
        'total_symbols': len(SYMBOLS)
    }


def store_in_postgresql(**context):
    """Store data in PostgreSQL"""
    import psycopg2
    from psycopg2 import sql
    
    market_data = context['ti'].xcom_pull(key='market_data', task_ids='fetch_data')
    
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )
    
    cur = conn.cursor()
    stored = 0
    
    for data in market_data:
        try:
            cur.execute(sql.SQL("""
                INSERT INTO stock_prices (symbol, timestamp, open, high, low, close, volume, timeframe)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE
                SET open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """), (
                data['symbol'],
                data['timestamp'],
                data['open'],
                data['high'],
                data['low'],
                data['close'],
                data['volume'],
                'MINUTE_1'
            ))
            stored += 1
        except Exception as e:
            print(f"Failed to store {data['symbol']}: {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    return {'stored': stored, 'total': len(market_data)}


def cache_in_redis(**context):
    """Cache latest prices in Redis"""
    import redis
    import json
    
    market_data = context['ti'].xcom_pull(key='market_data', task_ids='fetch_data')
    
    r = redis.Redis(
        host=os.getenv('REDIS_HOST'),
        password=os.getenv('REDIS_PASSWORD'),
        decode_responses=True
    )
    
    cached = 0
    
    for data in market_data:
        try:
            key = f"price:{data['symbol']}:latest"
            r.set(key, json.dumps(data), ex=300)  # 5 minute TTL
            cached += 1
        except Exception as e:
            print(f"Failed to cache {data['symbol']}: {e}")
    
    return {'cached': cached, 'total': len(market_data)}


def update_neo4j_prices(**context):
    """Update Neo4j company nodes with latest prices"""
    from neo4j import GraphDatabase
    
    market_data = context['ti'].xcom_pull(key='market_data', task_ids='fetch_data')
    
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    
    updated = 0
    
    for data in market_data:
        try:
            query = """
            MERGE (c:Company {symbol: $symbol})
            SET c.last_price = $price,
                c.price_updated_at = datetime($timestamp)
            """
            
            driver.execute_query(query, {
                'symbol': data['symbol'],
                'price': data['close'],
                'timestamp': data['timestamp']
            })
            updated += 1
        except Exception as e:
            print(f"Failed to update {data['symbol']}: {e}")
    
    driver.close()
    
    return {'updated': updated, 'total': len(market_data)}


# ================================================================
# Define the DAG
# ================================================================
with DAG(
    dag_id='data_ingestion',
    default_args=default_args,
    description='Real-time stock price ingestion to multi-database architecture',
    schedule_interval='*/1 * * * *',  # Every minute
    start_date=days_ago(1),
    catchup=False,
    tags=['data', 'real-time', 'market-data'],
    max_active_runs=1,
) as dag:
    
    fetch_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_market_data,
        provide_context=True
    )
    
    # Parallel storage in different databases
    postgres_task = PythonOperator(
        task_id='store_postgresql',
        python_callable=store_in_postgresql,
        provide_context=True
    )
    
    redis_task = PythonOperator(
        task_id='cache_redis',
        python_callable=cache_in_redis,
        provide_context=True
    )
    
    neo4j_task = PythonOperator(
        task_id='update_neo4j',
        python_callable=update_neo4j_prices,
        provide_context=True
    )
    
    # Fetch first, then parallel storage
    fetch_task >> [postgres_task, redis_task, neo4j_task]

dag.doc_md = """
# Data Ingestion DAG

## Purpose
Continuously fetch and store real-time stock market data across multiple databases.

## Architecture
```
Yahoo Finance → Fetch → [PostgreSQL, Redis, Neo4j]
```

## Storage Strategy
- **PostgreSQL**: Time-series data for analysis
- **Redis**: Real-time cache (5-minute TTL)
- **Neo4j**: Company node price updates

## Frequency
Runs every minute to maintain real-time data freshness.

## Parallel Execution
After fetch, all three database writes happen in parallel for maximum throughput.
"""