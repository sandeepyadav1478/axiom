# 🏗️ Enterprise-Grade Pipeline Architecture

## Current vs Enterprise-Ready

### Current (Basic):
```
Python asyncio scripts
Manual container management
Basic logging
No monitoring
No fault tolerance
```

### Enterprise Target:
```
Apache Airflow orchestration
Kafka streaming
Ray distributed compute
Prometheus monitoring  
Auto-scaling
Fault-tolerant
```

---

## 🎯 Enterprise Technology Stack

### Layer 1: Orchestration - Apache Airflow

**Why Airflow**:
- Industry standard for data pipelines
- Visual DAG monitoring
- Built-in retry logic
- Dependency management
- Scalable workers
- Rich integrations

**Current Problem**:
```python
# Basic while loop - fragile
while True:
    process_data()
    await asyncio.sleep(60)
```

**Airflow Solution**:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Define DAG
dag = DAG(
    'company_graph_builder',
    schedule_interval='@hourly',
    default_args={
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'execution_timeout': timedelta(minutes=30)
    }
)

# Tasks
fetch_task = PythonOperator(
    task_id='fetch_company_data',
    python_callable=fetch_company_data,
    dag=dag
)

extract_task = PythonOperator(
    task_id='extract_relationships',
    python_callable=extract_with_claude,
    dag=dag
)

neo4j_task = PythonOperator(
    task_id='update_neo4j',
    python_callable=update_graph,
    dag=dag
)

# Dependencies
fetch_task >> extract_task >> neo4j_task
```

**Benefits**:
- ✅ Web UI for monitoring
- ✅ Automatic retries
- ✅ Task dependencies
- ✅ Parallel execution
- ✅ Backfill support

---

### Layer 2: Streaming - Apache Kafka

**Why Kafka**:
- Real-time event streaming
- Durable message queue
- Exactly-once semantics
- Horizontal scaling
- Multiple consumers

**Architecture**:
```
Market Data → Kafka Topics → Multiple Consumers

Topics:
├─ price-updates (real-time OHLCV)
├─ company-fundamentals
├─ market-events
├─ correlation-updates
└─ graph-mutations

Consumers:
├─ PostgreSQL writer
├─ Redis cache updater
├─ Neo4j graph builder
├─ ChromaDB indexer
└─ Analytics engine
```

**Implementation**:
```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer (in data ingestion)
producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Publish price update
producer.send('price-updates', {
    'symbol': 'AAPL',
    'price': 150.25,
    'timestamp': datetime.now().isoformat()
})

# Consumer (in Neo4j updater)
consumer = KafkaConsumer(
    'price-updates',
    bootstrap_servers=['kafka:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='neo4j-updater'
)

for message in consumer:
    data = message.value
    update_neo4j_stock_price(data['symbol'], data['price'])
```

**Benefits**:
- ✅ Decouples producers and consumers
- ✅ Replay capability (reprocess historical data)
- ✅ Multiple pipelines consume same data
- ✅ Guaranteed delivery

---

### Layer 3: Distributed Computing - Ray

**Why Ray**:
- Parallel processing at scale
- GPU support (for ML models)
- Distributed data structures
- Actor model for stateful computation

**Use Case**: Process 1,000 symbols in parallel

**Current (Sequential)**:
```python
# Slow - processes one at a time
for symbol in symbols:  # 1000 symbols
    process_company(symbol)  # 30 seconds each
# Total: 8.3 hours
```

**Ray (Parallel)**:
```python
import ray

ray.init()

@ray.remote
def process_company_remote(symbol):
    return process_company(symbol)

# Process 1000 symbols in parallel (10 workers)
futures = [process_company_remote.remote(s) for s in symbols]
results = ray.get(futures)
# Total: 50 minutes (10x speedup with 10 workers)
```

**For LangGraph + Claude**:
```python
@ray.remote
class DistributedCompanyGraphBuilder:
    def __init__(self):
        self.claude = ChatAnthropic(...)
        self.neo4j = Neo4jGraphClient()
    
    def process(self, symbol):
        # Each actor has its own Claude + Neo4j connection
        # Process in parallel across multiple workers
        return self.run_langgraph_workflow(symbol)

# Deploy 10 actors
actors = [DistributedCompanyGraphBuilder.remote() for _ in range(10)]

# Process 100 companies in parallel
futures = []
for i, symbol in enumerate(symbols):
    actor = actors[i % 10]  # Round-robin
    future = actor.process.remote(symbol)
    futures.append(future)

results = ray.get(futures)
```

---

### Layer 4: API Layer - FastAPI

**Why FastAPI**:
- Modern async Python framework
- Automatic API docs (Swagger)
- Type validation (Pydantic)
- WebSocket support for real-time

**Pipeline Control API**:
```python
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

app = FastAPI(title="Axiom Pipeline Control API")

class PipelineStatus(BaseModel):
    name: str
    status: str
    last_run: datetime
    records_processed: int
    errors: List[str]

@app.get("/pipelines/status")
async def get_all_pipeline_status():
    """Get status of all 4 pipelines."""
    return {
        'ingestion': check_pipeline_health('ingestion'),
        'company_graph': check_pipeline_health('company-graph'),
        'events': check_pipeline_health('events'),
        'correlations': check_pipeline_health('correlations')
    }

@app.post("/pipelines/{name}/trigger")
async def trigger_pipeline(name: str):
    """Manually trigger a pipeline run."""
    trigger_airflow_dag(name)
    return {"status": "triggered", "pipeline": name}

@app.get("/graph/stats")
async def get_graph_stats():
    """Get Neo4j graph statistics."""
    neo4j = Neo4jGraphClient()
    return neo4j.get_graph_stats()

@app.websocket("/ws/realtime-updates")
async def websocket_endpoint(websocket: WebSocket):
    """Stream real-time updates to clients."""
    await websocket.accept()
    
    # Subscribe to Kafka/Redis
    consumer = KafkaConsumer('price-updates')
    
    for message in consumer:
        await websocket.send_json(message.value)
```

---

### Layer 5: Monitoring - Prometheus + Grafana

**Metrics Collection**:
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
records_processed = Counter('pipeline_records_processed', 'Records processed', ['pipeline'])
processing_time = Histogram('pipeline_processing_seconds', 'Processing time', ['pipeline'])
graph_size = Gauge('neo4j_graph_size', 'Number of nodes/relationships', ['type'])
api_calls = Counter('claude_api_calls', 'Claude API calls', ['pipeline'])

# Instrument code
with processing_time.labels(pipeline='company-graph').time():
    result = process_company(symbol)
    records_processed.labels(pipeline='company-graph').inc()

# Update graph metrics
graph_size.labels(type='nodes').set(count_neo4j_nodes())
graph_size.labels(type='edges').set(count_neo4j_relationships())
```

**Grafana Dashboards**:
- Pipeline throughput (records/second)
- Error rates by pipeline
- Neo4j graph growth over time
- Claude API usage & costs
- Database sizes

---

## 🏗️ Proposed Enterprise Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   APACHE AIRFLOW                                │
│              (Orchestration & Scheduling)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Company   │  │Events    │  │Correlation│ │Risk      │       │
│  │Graph DAG │  │Track DAG │  │Analysis DAG│ │Propagation│      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  KAFKA   │  │   RAY    │  │ FastAPI  │
    │Streaming │  │Parallel  │  │  API     │
    └──────────┘  └──────────┘  └──────────┘
            │             │             │
            └─────────────┼─────────────┘
                          ▼
            ┌──────────────────────────┐
            │    LANGGRAPH AGENTS      │
            │  (Claude-powered logic)  │
            └──────────────────────────┘
                          │
            ┌─────────────┼─────────────┬─────────────┐
            ▼             ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │PostgreSQL│  │  Redis   │  │  Neo4j   │  │ChromaDB  │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │  PROMETHEUS + GRAFANA    │
            │      (Monitoring)        │
            └──────────────────────────┘
```

---

## 📋 Migration Path: Current → Enterprise

### Phase 1: Add Airflow (Week 1)

**Install Airflow**:
```bash
# docker-compose-airflow.yml
version: '3.8'

services:
  airflow-webserver:
    image: apache/airflow:2.8.0
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://airflow:airflow@postgres/airflow
    ports:
      - "8080:8080"
    volumes:
      - ./axiom/pipelines/dags:/opt/airflow/dags
      - ./axiom:/opt/airflow/axiom
    networks:
      - axiom_network

  airflow-scheduler:
    image: apache/airflow:2.8.0
    command: scheduler
    volumes:
      - ./axiom/pipelines/dags:/opt/airflow/dags
      - ./axiom:/opt/airflow/axiom
    networks:
      - axiom_network
```

**Convert pipelines to Airflow DAGs**:
```python
# axiom/pipelines/dags/company_graph_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    'company_graph_builder',
    start_date=datetime(2025, 11, 15),
    schedule_interval='@hourly',
    catchup=False
) as dag:
    
    # Existing LangGraph workflow becomes Airflow tasks
    tasks = create_airflow_tasks_from_langgraph(CompanyGraphBuilderPipeline)
```

---

### Phase 2: Add Kafka (Week 2)

**Deploy Kafka**:
```yaml
# docker-compose-kafka.yml
kafka:
  image: confluentinc/cp-kafka:latest
  ports:
    - "9092:9092"
  environment:
    KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
    KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092

zookeeper:
  image: confluentinc/cp-zookeeper:latest
  ports:
    - "2181:2181"
```

**Refactor to Event-Driven**:
```python
# Data ingestion becomes producer
class KafkaProducerPipeline:
    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers='kafka:9092')
    
    async def ingest_price(self, symbol, price_data):
        # Publish to Kafka instead of direct DB write
        self.producer.send('price-updates', {
            'symbol': symbol,
            'data': price_data,
            'timestamp': datetime.now()
        })

# Multiple consumers process in parallel
class Neo4jKafkaConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer('price-updates')
        self.neo4j = Neo4jGraphClient()
    
    def consume(self):
        for message in self.consumer:
            self.neo4j.update_stock(message.value)

class PostgreSQLKafkaConsumer:
    # Separate consumer for PostgreSQL
    pass

class RedisKafkaConsumer:
    # Separate consumer for Redis
    pass
```

---

### Phase 3: Add Ray for Parallel Processing (Week 3)

**Deploy Ray Cluster**:
```yaml
ray-head:
  image: rayproject/ray:latest
  ports:
    - "8265:8265"  # Ray dashboard
    - "10001:10001"
  command: ray start --head --dashboard-host=0.0.0.0

ray-worker:
  image: rayproject/ray:latest
  command: ray start --address=ray-head:6379
  deploy:
    replicas: 4  # 4 worker nodes
```

**Parallel LangGraph Processing**:
```python
import ray
from ray.util.actor_pool import ActorPool

@ray.remote
class CompanyGraphActor:
    """Each actor processes companies independently."""
    
    def __init__(self):
        self.claude = ChatAnthropic(...)
        self.neo4j = Neo4jGraphClient()
    
    def process_company(self, symbol):
        # Full LangGraph workflow
        return self.run_workflow(symbol)

# Create actor pool
actors = [CompanyGraphActor.remote() for _ in range(10)]
pool = ActorPool(actors)

# Process 1000 companies in parallel
symbols = [...1000 symbols...]
results = list(pool.map(lambda a, s: a.process_company.remote(s), symbols))

# Result: 1000 companies processed in ~1 hour instead of ~8 hours
```

---

### Phase 4: Add FastAPI Control Plane (Week 3)

**Pipeline Control API**:
```python
# axiom/api/pipeline_api.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title="Axiom Pipeline API")

@app.post("/pipeline/company-graph/trigger")
async def trigger_company_graph(
    symbols: List[str],
    background_tasks: BackgroundTasks
):
    """Trigger company graph build for specific symbols."""
    background_tasks.add_task(run_company_graph_pipeline, symbols)
    return {"status": "triggered", "symbols": len(symbols)}

@app.get("/pipeline/company-graph/status")
async def get_company_graph_status():
    """Get current pipeline status."""
    return {
        'running': is_pipeline_running('company-graph'),
        'last_run': get_last_run_time('company-graph'),
        'success_rate': calculate_success_rate('company-graph'),
        'graph_size': get_neo4j_stats()
    }

@app.get("/graph/live-updates")
async def stream_graph_updates():
    """Stream live graph updates via SSE."""
    
    async def event_generator():
        consumer = KafkaConsumer('graph-mutations')
        for message in consumer:
            yield f"data: {json.dumps(message.value)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.get("/metrics/realtime")
async def get_realtime_metrics():
    """Get real-time pipeline metrics."""
    return {
        'records_per_second': get_throughput(),
        'active_pipelines': count_active_pipelines(),
        'graph_density': calculate_graph_density(),
        'claude_api_usage': get_claude_usage()
    }
```

---

### Layer 5: Monitoring - Prometheus + Grafana

**Metrics Exporter**:
```python
# axiom/monitoring/metrics.py
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# Pipeline metrics
pipeline_runs = Counter('pipeline_runs_total', 'Pipeline runs', ['name', 'status'])
pipeline_duration = Histogram('pipeline_duration_seconds', 'Duration', ['name'])
records_processed = Counter('records_processed_total', 'Records', ['pipeline', 'database'])

# Graph metrics
graph_nodes = Gauge('neo4j_nodes_total', 'Node count', ['label'])
graph_relationships = Gauge('neo4j_relationships_total', 'Edge count', ['type'])

# LangGraph metrics
langgraph_agent_calls = Counter('langgraph_agent_calls', 'Agent calls', ['agent', 'pipeline'])
claude_api_calls = Counter('claude_api_calls_total', 'Claude calls', ['model'])
claude_api_cost = Counter('claude_api_cost_dollars', 'Claude cost')

# Start metrics server
start_http_server(9090)  # Metrics at localhost:9090/metrics
```

**Grafana Dashboard**:
```
Panel 1: Pipeline Throughput
- Records/second by pipeline
- Success rate over time

Panel 2: Graph Growth
- Nodes over time (by type)
- Relationships over time (by type)
- Graph density trend

Panel 3: LangGraph Performance
- Agent execution time
- Claude API latency
- Workflow success rate

Panel 4: Cost Monitoring
- Claude API calls/hour
- Estimated daily cost
- Cost per pipeline
```

---

## 🚀 Complete Enterprise Stack

### Docker Compose (All Services):

```yaml
version: '3.8'

services:
  # ============================================
  # ORCHESTRATION
  # ============================================
  airflow-webserver:
    image: apache/airflow:2.8.0
    ports:
      - "8080:8080"
    
  airflow-scheduler:
    image: apache/airflow:2.8.0
    command: scheduler
    
  airflow-worker:
    image: apache/airflow:2.8.0
    command: celery worker
    deploy:
      replicas: 4  # 4 parallel workers
  
  # ============================================
  # STREAMING
  # ============================================
  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
  
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    ports:
      - "2181:2181"
  
  # ============================================
  # DISTRIBUTED COMPUTE
  # ============================================
  ray-head:
    image: rayproject/ray:latest-gpu  # GPU support
    ports:
      - "8265:8265"  # Ray dashboard
    command: ray start --head
  
  ray-worker:
    image: rayproject/ray:latest-gpu
    command: ray start --address=ray-head:6379
    deploy:
      replicas: 4
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  
  # ============================================
  # API LAYER
  # ============================================
  pipeline-api:
    build:
      context: .
      dockerfile: axiom/api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - AIRFLOW_API=http://airflow-webserver:8080
      - KAFKA_BOOTSTRAP=kafka:9092
      - RAY_ADDRESS=ray-head:10001
  
  # ============================================
  # MONITORING
  # ============================================
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/dashboards:/var/lib/grafana/dashboards
  
  # ============================================
  # DATABASES (existing)
  # ============================================
  postgres:
    # ... existing config
  
  redis:
    # ... existing config
  
  neo4j:
    # ... existing config
  
  chromadb:
    # ... existing config
```

---

## 📊 Performance Comparison

### Current (Basic asyncio):
```
Throughput: 5 symbols/minute (sequential)
Scalability: Limited to single container
Fault tolerance: Restart on failure
Monitoring: Basic logs
Cost visibility: None
```

### With Enterprise Stack:
```
Throughput: 1000 symbols/minute (Ray parallel + Kafka)
Scalability: Horizontal (add more Ray workers)
Fault tolerance: Automatic retries, dead letter queues
Monitoring: Real-time Grafana dashboards
Cost visibility: Per-pipeline cost tracking
Observability: Distributed tracing, metrics, logs
```

---

## 🎯 Migration Strategy

### Week 1: Add Airflow
```bash
# 1. Deploy Airflow
docker compose -f docker-compose-airflow.yml up -d

# 2. Convert existing pipelines to DAGs
# Keep LangGraph logic, wrap in Airflow tasks

# 3. Test both systems in parallel
```

### Week 2: Add Kafka
```bash
# 1. Deploy Kafka + Zookeeper
docker compose -f docker-compose-kafka.yml up -d

# 2. Create topics
kafka-topics --create --topic price-updates
kafka-topics --create --topic graph-mutations

# 3. Refactor pipelines to produce/consume from Kafka
```

### Week 3: Add Ray
```bash
# 1. Deploy Ray cluster
docker compose -f docker-compose-ray.yml up -d

# 2. Convert company graph to use Ray actors
# 3. Benchmark: Sequential vs Parallel
```

### Week 4: Add Monitoring
```bash
# 1. Deploy Prometheus + Grafana
# 2. Add metrics to all pipelines
# 3. Create dashboards
# 4. Set up alerts
```

---

## 💡 Immediate Upgrade (This Week)

**Option A: Airflow Only** (Quickest ROI)
```
Effort: 2-3 days
Benefit: 
- Professional orchestration
- Web UI monitoring
- Automatic retries
- Easy scheduling changes
```

**Option B: Kafka + Basic Streaming** (Medium effort)
```
Effort: 3-5 days
Benefit:
- Real-time streaming
- Decoupled architecture
- Multiple consumers
- Replay capability
```

**Option C: Full Enterprise Stack** (Comprehensive)
```
Effort: 2-3 weeks
Benefit:
- Production-grade
- Horizontal scaling
- Full observability
- GPU-accelerated
```

---

## 🎯 Recommended: Staged Rollout

**This Week**: Add Airflow
**Next Week**: Add Kafka
**Month 1**: Add Ray for parallel processing
**Month 2**: Add full monitoring stack

**Rationale**: Incremental upgrades, validate at each stage, minimal disruption

---

## 📚 Learning Resources

**Apache Airflow**:
- Docs: https://airflow.apache.org/docs/
- Course: "Data Engineering with Airflow"

**Kafka**:
- Docs: https://kafka.apache.org/documentation/
- Book: "Kafka: The Definitive Guide"

**Ray**:
- Docs: https://docs.ray.io/
- Focus: Ray Data for data pipelines

**FastAPI**:
- Docs: https://fastapi.tiangolo.com/
- Tutorial: Build REST API in 1 hour

---

## ✅ Next Session: Start Enterprise Upgrade

**Immediate Action Items**:
1. Deploy Apache Airflow
2. Convert one pipeline to Airflow DAG (proof of concept)
3. Validate monitoring via Airflow UI
4. Plan Kafka integration

**You're right - let's build this on enterprise-grade technology!**