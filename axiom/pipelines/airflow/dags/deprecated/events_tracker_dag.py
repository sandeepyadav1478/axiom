"""
Apache Airflow DAG: Market Events Tracker
Monitors and classifies market-moving events using Claude AI

This DAG:
1. Fetches latest news for tracked companies
2. Uses Claude to classify event types and impact
3. Creates MarketEvent nodes in Neo4j
4. Links events to affected companies
5. Tracks sentiment and market impact

Schedule: Every 5 minutes
Retries: 2 attempts
Timeout: 10 minutes per run
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
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=10)
}

SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']

# ================================================================
# Task Functions
# ================================================================

def fetch_news(**context):
    """Fetch latest news for companies"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    news_items = []
    
    for symbol in SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            # Get news from last 24 hours
            news = ticker.news
            
            for item in news[:5]:  # Top 5 news per company
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
    
    return {
        'news_fetched': len(news_items),
        'companies': len(SYMBOLS)
    }


def classify_events_with_claude(**context):
    """Use Claude to classify and analyze market events"""
    from langchain_anthropic import ChatAnthropic
    
    news_items = context['ti'].xcom_pull(key='news_items', task_ids='fetch_news')
    
    claude = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        max_tokens=2048
    )
    
    classified_events = []
    
    for item in news_items:
        try:
            prompt = f"""Analyze this market event:
Title: {item['title']}
Company: {item['symbol']}

Classify this event and provide:
1. Event Type (earnings, product_launch, regulatory, acquisition, leadership_change, other)
2. Sentiment (positive, negative, neutral)
3. Impact Level (high, medium, low)
4. One-sentence summary

Format: type|sentiment|impact|summary
Example: earnings|positive|high|Strong Q4 results beat expectations"""

            response = claude.invoke(prompt)
            parts = response.content.strip().split('|')
            
            if len(parts) >= 4:
                classified_events.append({
                    'symbol': item['symbol'],
                    'title': item['title'],
                    'event_type': parts[0].strip(),
                    'sentiment': parts[1].strip(),
                    'impact': parts[2].strip(),
                    'summary': parts[3].strip(),
                    'published_at': item['published'],
                    'source_url': item['link']
                })
        except Exception as e:
            print(f"Failed to classify event: {e}")
    
    context['ti'].xcom_push(key='classified_events', value=classified_events)
    
    return {
        'events_classified': len(classified_events),
        'sentiment_distribution': {
            'positive': len([e for e in classified_events if e['sentiment'] == 'positive']),
            'negative': len([e for e in classified_events if e['sentiment'] == 'negative']),
            'neutral': len([e for e in classified_events if e['sentiment'] == 'neutral'])
        }
    }


def create_event_nodes(**context):
    """Create MarketEvent nodes in Neo4j"""
    from pipelines.shared.neo4j_client import Neo4jGraphClient
    
    events = context['ti'].xcom_pull(key='classified_events', task_ids='classify_events')
    
    neo4j = Neo4jGraphClient(
        uri=os.getenv('NEO4J_URI'),
        user=os.getenv('NEO4J_USER'),
        password=os.getenv('NEO4J_PASSWORD')
    )
    
    created = 0
    
    for event in events:
        try:
            query = """
            MATCH (c:Company {symbol: $symbol})
            CREATE (e:MarketEvent {
                id: randomUUID(),
                type: $event_type,
                title: $title,
                summary: $summary,
                sentiment: $sentiment,
                impact: $impact,
                published_at: datetime({epochSeconds: $published_at}),
                source_url: $source_url,
                created_at: datetime()
            })
            CREATE (c)-[:AFFECTED_BY]->(e)
            """
            
            neo4j.driver.execute_query(query, event)
            created += 1
        except Exception as e:
            print(f"Failed to create event node: {e}")
    
    neo4j.close()
    
    return {
        'events_created': created,
        'total': len(events)
    }


# ================================================================
# Define the DAG
# ================================================================
with DAG(
    dag_id='events_tracker',
    default_args=default_args,
    description='Track and classify market events using Claude AI',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['langgraph', 'claude', 'neo4j', 'market-events'],
    max_active_runs=1,
) as dag:
    
    fetch_task = PythonOperator(
        task_id='fetch_news',
        python_callable=fetch_news,
        provide_context=True
    )
    
    classify_task = PythonOperator(
        task_id='classify_events',
        python_callable=classify_events_with_claude,
        provide_context=True
    )
    
    create_task = PythonOperator(
        task_id='create_event_nodes',
        python_callable=create_event_nodes,
        provide_context=True
    )
    
    # Linear workflow
    fetch_task >> classify_task >> create_task

dag.doc_md = """
# Market Events Tracker DAG

## Purpose
Continuously monitor and classify market-moving events using AI.

## Workflow
1. **Fetch News**: Get latest news from Yahoo Finance
2. **Classify**: Use Claude to categorize and analyze events
3. **Store**: Create MarketEvent nodes linked to companies

## Event Types
- Earnings reports
- Product launches
- Regulatory changes
- M&A activity
- Leadership changes
- Market commentary

## Frequency
Runs every 5 minutes to catch breaking news in real-time.
"""