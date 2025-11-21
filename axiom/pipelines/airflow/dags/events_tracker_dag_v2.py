"""
Apache Airflow DAG v2: Market Events Tracker
Track and classify market events with centralized YAML configuration

IMPROVEMENTS OVER V1:
- ✅ CachedClaudeOperator (cost savings configurable)
- ✅ Circuit breaker for news API calls
- ✅ Batch processing of events
- ✅ Cost tracking per event classification
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

# Load environment early
from dotenv import load_dotenv
load_dotenv('/opt/airflow/.env')
os.environ.setdefault('ANTHROPIC_API_KEY', os.getenv('CLAUDE_API_KEY', ''))

# Import operators from local path
operators_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, operators_path)

from operators.claude_operator import CachedClaudeOperator
from operators.resilient_operator import CircuitBreakerOperator

# Import centralized configuration
utils_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, utils_path)
from utils.config_loader import (
    dag_config,
    get_symbols_for_dag,
    build_neo4j_conn_params
)

# ================================================================
# Load Configuration from YAML
# ================================================================
DAG_NAME = 'events_tracker'
config = dag_config.get_dag_config(DAG_NAME)
default_args = dag_config.get_default_args(DAG_NAME)
SYMBOLS = get_symbols_for_dag(DAG_NAME)
circuit_breaker_config = dag_config.get_circuit_breaker_config(DAG_NAME)
claude_config = dag_config.get_claude_config(DAG_NAME)
news_config = dag_config.get_news_config()

# ================================================================
# Helper Functions
# ================================================================

def fetch_news_safe(context):
    """Fetch news with error handling"""
    import yfinance as yf
    
    # Get config from context
    symbols = context['params'].get('symbols', SYMBOLS)
    max_items_per_symbol = context['params'].get('max_items_per_symbol', 5)
    
    news_items = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for item in news[:max_items_per_symbol]:
                news_items.append({
                    'symbol': symbol,
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'published': item.get('providerPublishTime', int(datetime.now().timestamp())),
                    'type': item.get('type', 'news')
                })
        except Exception as e:
            print(f"Failed to fetch news for {symbol}: {e}")
    
    context['ti'].xcom_push(key='news_items', value=news_items)
    
    return {'news_fetched': len(news_items)}


def create_event_nodes_batch(context):
    """Create event nodes in batch (faster)"""
    from neo4j import GraphDatabase
    
    events = context['ti'].xcom_pull(
        task_ids='classify_events_cached',
        key='claude_response'
    )
    
    if not events or not events.get('success'):
        return {'created': 0}
    
    # Use centralized config for Neo4j connection
    neo4j_params = build_neo4j_conn_params()
    driver = GraphDatabase.driver(
        neo4j_params['uri'],
        auth=(neo4j_params['user'], neo4j_params['password'])
    )
    
    # Parse Claude's response (should be JSON with events)
    import json
    try:
        classified_events = json.loads(events['content'])
    except:
        # If not JSON, skip
        return {'created': 0}
    
    # Batch create
    query = """
    UNWIND $events AS event
    MATCH (c:Company {symbol: event.symbol})
    CREATE (e:MarketEvent {
        id: randomUUID(),
        type: event.event_type,
        title: event.title,
        sentiment: event.sentiment,
        impact: event.impact,
        published_at: datetime({epochSeconds: event.published}),
        created_at: datetime()
    })
    CREATE (c)-[:AFFECTED_BY]->(e)
    """
    
    records, summary, keys = driver.execute_query(query, {'events': classified_events})
    driver.close()
    
    return {
        'events_created': summary.counters.nodes_created,
        'relationships_created': summary.counters.relationships_created
    }


# ================================================================
# Define the Enhanced DAG (Config-Driven)
# ================================================================
with DAG(
    dag_id=config.get('dag_id', 'events_tracker_v2'),
    default_args=default_args,
    description=config.get('description', 'Track market events with cached Claude classification'),
    schedule_interval=config.get('schedule_interval', '*/15 * * * *'),
    start_date=days_ago(1),
    catchup=dag_config.get_global('catchup', False),
    tags=config.get('tags', ['v2', 'enterprise']),
    max_active_runs=dag_config.get_global('max_active_runs', 1),
) as dag:
    
    # Task 1: Fetch news with circuit breaker (config-driven)
    fetch_news = CircuitBreakerOperator(
        task_id='fetch_news_protected',
        callable_func=fetch_news_safe,
        failure_threshold=circuit_breaker_config.get('failure_threshold', 5),
        recovery_timeout_seconds=circuit_breaker_config.get('recovery_timeout_seconds', 120),
        xcom_key='fetch_result',
        params={
            'symbols': SYMBOLS,
            'max_items_per_symbol': news_config.get('max_items_per_symbol', 5)
        }
    )
    
    # Get event classification config
    event_types = config.get('event_types', ['earnings', 'product_launch', 'regulatory', 'acquisition', 'leadership_change', 'other'])
    sentiments = config.get('sentiments', ['positive', 'negative', 'neutral'])
    impact_levels = config.get('impact_levels', ['high', 'medium', 'low'])
    
    # Task 2: Classify events with CACHED Claude (config-driven)
    classify_events = CachedClaudeOperator(
        task_id='classify_events_cached',
        prompt=f"""Analyze these market events and classify each one:
{{{{ ti.xcom_pull(task_ids='fetch_news_protected', key='news_items') }}}}

For each event, provide:
1. Event Type ({', '.join(event_types)})
2. Sentiment ({', '.join(sentiments)})
3. Impact Level ({', '.join(impact_levels)})

Return as JSON array:
[
  {{
    "symbol": "AAPL",
    "title": "...",
    "event_type": "earnings",
    "sentiment": "positive",
    "impact": "high",
    "published": 1234567890
  }},
  ...
]""",
        system_message='You are a financial news analyst specializing in event classification.',
        max_tokens=claude_config.get('max_tokens', 4096),
        cache_ttl_hours=claude_config.get('cache_ttl_hours', 6),
        track_cost=claude_config.get('track_cost', True),
        xcom_key='claude_response'
    )
    
    # Task 3: Create event nodes in batch (config-driven)
    create_events = CircuitBreakerOperator(
        task_id='create_event_nodes_batch',
        callable_func=create_event_nodes_batch,
        failure_threshold=circuit_breaker_config.get('failure_threshold', 5),
        recovery_timeout_seconds=circuit_breaker_config.get('recovery_timeout_seconds', 60),
        xcom_key='create_result'
    )
    
    # Dependencies
    fetch_news >> classify_events >> create_events


dag.doc_md = """
# Enhanced Events Tracker DAG (Config-Driven)

## 🚀 Centralized Configuration

### All Settings in YAML
- **Schedule**: Configurable (default: every 15 minutes)
- **Max items per symbol**: Configurable (default: 5)
- **Event types**: Configurable list
- **Sentiments**: Configurable list
- **Impact levels**: Configurable list
- **Cache TTL**: Configurable (default: 6 hours)

### Cost Optimization
- **Cache duration**: Configurable for cost/freshness balance
- **News often repeated**: Cache saves significant costs
- **Track costs**: Configurable cost tracking

### Performance
- **News fetching**: Circuit breaker configurable
- **Event classification**: Cache TTL configurable
- **Neo4j operations**: Circuit breaker configurable
- **All thresholds**: Tunable via YAML

## 🎯 Event Classification (Configurable)

### Event Types
Configurable list (default includes):
- earnings, product_launch, regulatory, acquisition, leadership_change, other

### Sentiments
Configurable list (default includes):
- positive, negative, neutral

### Impact Levels
Configurable list (default includes):
- high, medium, low

## 💡 Why Configuration-Driven?

1. **Easy Tuning**: Adjust classification categories without code changes
2. **Cost Control**: Configure cache duration to balance cost/freshness
3. **Performance Optimization**: Tune circuit breaker thresholds
4. **Flexibility**: Different settings for dev/staging/prod
5. **Extensibility**: Add new event types/sentiments easily

All parameters centralized in [`dag_config.yaml`](../dag_configs/dag_config.yaml)!
"""