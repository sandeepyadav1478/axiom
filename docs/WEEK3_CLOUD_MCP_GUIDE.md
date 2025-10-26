# Week 3 Cloud MCP Integration Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Setup & Installation](#setup--installation)
3. [AWS MCP Server](#aws-mcp-server)
4. [GCP MCP Server](#gcp-mcp-server)
5. [Notification MCP Server](#notification-mcp-server)
6. [Vector DB MCP Server](#vector-db-mcp-server)
7. [Kubernetes MCP Server](#kubernetes-mcp-server)
8. [Integration Examples](#integration-examples)
9. [Production Deployment](#production-deployment)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

Week 3 MCP servers enable production-grade platform operations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Axiom Platform Core                       │
│                   (LangGraph Workflows)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │   MCP Manager Layer   │
         └───────────┬───────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐      ┌───▼────┐      ┌───▼────┐
│ Cloud  │      │ Comms  │      │Storage │
│ (AWS,  │      │(Email, │      │(Vector,│
│  GCP)  │      │ SMS)   │      │ K8s)   │
└────────┘      └────────┘      └────────┘
    │                │                │
┌───▼────┐      ┌───▼────┐      ┌───▼────┐
│   S3   │      │ SMTP   │      │Pinecone│
│  EC2   │      │Twilio  │      │Chroma  │
│Lambda  │      │SendGrid│      │ Qdrant │
└────────┘      └────────┘      └────────┘
```

### Benefits

- **Cloud-Agnostic**: Support for AWS and GCP
- **Unified Notifications**: Single API for email/SMS/Slack
- **AI-Powered Search**: Semantic search for financial data
- **Production Orchestration**: Kubernetes for scale
- **Code Reduction**: ~2,150 lines of maintenance code eliminated

---

## Setup & Installation

### 1. Install Dependencies

```bash
# Install all Week 3 dependencies
pip install -r requirements.txt

# Or install specific components:
pip install boto3 botocore  # AWS
pip install google-cloud-storage google-cloud-compute google-cloud-bigquery  # GCP
pip install sendgrid twilio  # Notifications
pip install chromadb qdrant-client pinecone-client  # Vector DBs
pip install kubernetes  # K8s
```

### 2. Configure Environment Variables

Create or update `.env`:

```bash
# AWS Configuration
AWS_MCP_REGION=us-east-1
AWS_MCP_ACCESS_KEY_ID=your_access_key
AWS_MCP_SECRET_ACCESS_KEY=your_secret_key
# AWS_MCP_PROFILE=default  # Alternative to keys

# GCP Configuration
GCP_MCP_PROJECT_ID=your-project-id
GCP_MCP_CREDENTIALS_PATH=/path/to/credentials.json

# Notification Configuration
NOTIFICATION_MCP_SMTP_SERVER=smtp.gmail.com
NOTIFICATION_MCP_SMTP_PORT=587
NOTIFICATION_MCP_SMTP_USER=your-email@gmail.com
NOTIFICATION_MCP_SMTP_PASSWORD=your-app-password
NOTIFICATION_MCP_SMTP_FROM_ADDRESS=noreply@axiom.com

# SendGrid (optional)
NOTIFICATION_MCP_SENDGRID_API_KEY=SG.xxx

# Twilio (optional)
NOTIFICATION_MCP_TWILIO_ACCOUNT_SID=ACxxx
NOTIFICATION_MCP_TWILIO_AUTH_TOKEN=xxx
NOTIFICATION_MCP_TWILIO_FROM_NUMBER=+1234567890

# Vector DB Configuration
VECTOR_DB_MCP_PROVIDER=chromadb  # or pinecone, qdrant, weaviate
VECTOR_DB_MCP_DIMENSION=1536
VECTOR_DB_MCP_CHROMADB_HOST=localhost
VECTOR_DB_MCP_CHROMADB_PORT=8000

# Kubernetes Configuration
K8S_MCP_NAMESPACE=axiom
K8S_MCP_CONTEXT=production
# K8S_MCP_CONFIG_PATH=/path/to/kubeconfig  # Optional
```

### 3. Start Supporting Services

```bash
# Create Docker network
docker network create axiom-network

# Start Week 3 services
docker-compose -f docker/week3-services.yml up -d

# Verify services are running
docker-compose -f docker/week3-services.yml ps
```

### 4. Enable MCP Servers

Update [`axiom/integrations/mcp_servers/config.py`](../axiom/integrations/mcp_servers/config.py):

```python
@dataclass
class MCPEcosystemConfig:
    # Enable Week 3 servers
    use_aws_mcp: bool = True
    use_gcp_mcp: bool = True
    use_vector_db_mcp: bool = True
    use_kubernetes_mcp: bool = True
    use_email_mcp: bool = True
    use_sms_mcp: bool = True
```

---

## AWS MCP Server

### Use Cases

1. **Portfolio Data Storage**
   ```python
   # Store daily portfolio snapshot to S3
   from axiom.integrations.mcp_servers.manager import mcp_manager
   
   portfolio_json = json.dumps(portfolio_data)
   
   result = await mcp_manager.call_tool(
       server_name="aws",
       tool_name="s3_upload",
       bucket="axiom-portfolios",
       key=f"snapshots/{date}/portfolio.json",
       data=portfolio_json,
       metadata={"date": date, "strategy": "momentum"}
   )
   ```

2. **Backtesting on EC2**
   ```python
   # Create EC2 spot instance for backtesting
   result = await mcp_manager.call_tool(
       server_name="aws",
       tool_name="ec2_create_instance",
       image_id="ami-xxxxx",
       instance_type="c5.4xlarge",
       name="backtest-worker",
       user_data=startup_script
   )
   
   instance_id = result["instances"][0]["instance_id"]
   
   # ... run backtest ...
   
   # Stop instance when done
   await mcp_manager.call_tool(
       server_name="aws",
       tool_name="ec2_stop_instance",
       instance_id=instance_id
   )
   ```

3. **Lambda Alerts**
   ```python
   # Deploy price alert Lambda
   result = await mcp_manager.call_tool(
       server_name="aws",
       tool_name="lambda_deploy",
       function_name="price-alert",
       s3_bucket="axiom-lambdas",
       s3_key="price-alert.zip",
       runtime="python3.11",
       handler="handler.check_price",
       role_arn="arn:aws:iam::xxx:role/lambda-exec",
       environment={"ALERT_THRESHOLD": "5.0"}
   )
   
   # Invoke alert check
   result = await mcp_manager.call_tool(
       server_name="aws",
       tool_name="lambda_invoke",
       function_name="price-alert",
       payload={"symbol": "AAPL"}
   )
   ```

### Performance Monitoring

```python
# Get Lambda execution metrics
result = await mcp_manager.call_tool(
    server_name="aws",
    tool_name="cloudwatch_get_metrics",
    namespace="AWS/Lambda",
    metric_name="Duration",
    dimensions=[{"Name": "FunctionName", "Value": "price-alert"}],
    statistic="Average",
    start_time="2024-01-01T00:00:00Z",
    end_time="2024-01-01T23:59:59Z",
    period=3600  # 1 hour intervals
)

for datapoint in result["datapoints"]:
    print(f"{datapoint['timestamp']}: {datapoint['value']}ms")
```

---

## GCP MCP Server

### Use Cases

1. **Multi-Cloud Data Lake**
   ```python
   # Upload to both AWS S3 and GCP Storage
   data = generate_report()
   
   # AWS
   await mcp_manager.call_tool(
       server_name="aws",
       tool_name="s3_upload",
       bucket="axiom-data-us",
       key="reports/daily.json",
       data=data
   )
   
   # GCP
   await mcp_manager.call_tool(
       server_name="gcp",
       tool_name="storage_upload",
       bucket_name="axiom-data-eu",
       blob_name="reports/daily.json",
       data=data
   )
   ```

2. **BigQuery Analytics**
   ```python
   # Load trade data into BigQuery
   result = await mcp_manager.call_tool(
       server_name="gcp",
       tool_name="bigquery_load",
       dataset_id="trading",
       table_id="executions",
       source_uri="gs://axiom-data/trades.csv",
       source_format="CSV",
       write_disposition="WRITE_APPEND"
   )
   
   # Run analytics query
   result = await mcp_manager.call_tool(
       server_name="gcp",
       tool_name="bigquery_query",
       query="""
           SELECT 
               symbol,
               COUNT(*) as trade_count,
               SUM(pnl) as total_pnl
           FROM `project.trading.executions`
           WHERE date = CURRENT_DATE()
           GROUP BY symbol
           ORDER BY total_pnl DESC
       """
   )
   
   for row in result["rows"]:
       print(f"{row['symbol']}: ${row['total_pnl']}")
   ```

3. **Compute Engine for ML Training**
   ```python
   # Start GPU instance for model training
   result = await mcp_manager.call_tool(
       server_name="gcp",
       tool_name="compute_start",
       zone="us-central1-a",
       instance_name="ml-training-gpu"
   )
   
   # ... training completes ...
   
   # Stop instance
   await mcp_manager.call_tool(
       server_name="gcp",
       tool_name="compute_stop",
       zone="us-central1-a",
       instance_name="ml-training-gpu"
   )
   ```

---

## Notification MCP Server

### Use Cases

1. **Trading Alerts**
   ```python
   # Price movement alert
   await mcp_manager.call_tool(
       server_name="notification",
       tool_name="send_alert",
       recipients={
           "email": "trader@example.com",
           "phone": "+1234567890"
       },
       severity="warning",
       title="Price Alert: AAPL",
       message="AAPL moved 5% in last hour to $150.25",
       metadata={
           "symbol": "AAPL",
           "change_pct": "5.2%",
           "current_price": "$150.25"
       }
   )
   ```

2. **Daily Reports**
   ```python
   # Send daily P&L report
   await mcp_manager.call_tool(
       server_name="notification",
       tool_name="send_daily_report",
       to=["trader@example.com", "risk@example.com"],
       report_title="Daily P&L Summary",
       report_data={
           "total_pnl": "$12,450.75",
           "winning_trades": 45,
           "losing_trades": 12,
           "win_rate": "78.9%",
           "sharpe_ratio": "2.34",
           "max_drawdown": "-$2,100.00"
       }
   )
   ```

3. **Risk Limit Breach**
   ```python
   # Critical risk alert
   await mcp_manager.call_tool(
       server_name="notification",
       tool_name="send_alert",
       recipients={
           "email": ["risk@example.com", "cto@example.com"],
           "phone": "+1234567890"
       },
       severity="critical",
       title="VaR Limit Breach",
       message="Portfolio VaR exceeded 2.0% limit, currently at 3.2%",
       metadata={
           "current_var": "3.2%",
           "limit": "2.0%",
           "portfolio_value": "$10,000,000",
           "action_required": "Reduce exposure"
       }
   )
   ```

4. **2FA Authentication**
   ```python
   # Send 2FA code
   import random
   code = f"{random.randint(100000, 999999)}"
   
   await mcp_manager.call_tool(
       server_name="notification",
       tool_name="send_2fa",
       to="+1234567890",
       code=code
   )
   ```

### Severity-Based Routing

The notification server automatically routes alerts based on severity:

| Severity | Channels |
|----------|----------|
| `info` | Email only |
| `warning` | Email only |
| `error` | Email + SMS |
| `critical` | Email + SMS + Slack (if available) |

---

## Vector DB MCP Server

### Use Cases

1. **Company Similarity Search**
   ```python
   # Generate embedding for query
   from openai import OpenAI
   client = OpenAI()
   
   query = "Technology company with strong AI capabilities"
   response = client.embeddings.create(
       model="text-embedding-ada-002",
       input=query
   )
   query_embedding = response.data[0].embedding
   
   # Search for similar companies
   result = await mcp_manager.call_tool(
       server_name="vector_db",
       tool_name="search_similar",
       collection="companies",
       query_embedding=query_embedding,
       limit=10,
       filter={"market_cap": ">10B"}
   )
   
   for match in result["results"]:
       print(f"{match['metadata']['ticker']}: {match['score']:.3f}")
   # Output: NVDA: 0.923, MSFT: 0.891, GOOGL: 0.867, ...
   ```

2. **SEC Filing Analysis**
   ```python
   # Add SEC filings to vector DB
   filings = fetch_sec_filings("AAPL", "10-K")
   
   for filing in filings:
       # Generate embedding
       embedding = generate_embedding(filing.text)
       
       await mcp_manager.call_tool(
           server_name="vector_db",
           tool_name="add_document",
           collection="sec_filings",
           document_id=filing.accession_number,
           text=filing.text,
           embedding=embedding,
           metadata={
               "ticker": "AAPL",
               "filing_type": "10-K",
               "filing_date": filing.date,
               "fiscal_year": filing.year
           }
       )
   
   # Search filings by topic
   query_embedding = generate_embedding("risk factors and contingencies")
   result = await mcp_manager.call_tool(
       server_name="vector_db",
       tool_name="search_similar",
       collection="sec_filings",
       query_embedding=query_embedding,
       limit=5,
       filter={"ticker": "AAPL"}
   )
   ```

3. **News Sentiment Analysis**
   ```python
   # Store news articles with embeddings
   articles = fetch_financial_news(days=7)
   
   for article in articles:
       embedding = generate_embedding(article.content)
       sentiment = analyze_sentiment(article.content)
       
       await mcp_manager.call_tool(
           server_name="vector_db",
           tool_name="add_document",
           collection="news",
           document_id=article.id,
           text=article.content,
           embedding=embedding,
           metadata={
               "title": article.title,
               "source": article.source,
               "date": article.date,
               "sentiment": sentiment,
               "tickers": article.mentioned_tickers
           }
       )
   
   # Find similar news about a company
   query = "Tesla earnings report and guidance"
   query_embedding = generate_embedding(query)
   
   result = await mcp_manager.call_tool(
       server_name="vector_db",
       tool_name="hybrid_search",
       collection="news",
       query_embedding=query_embedding,
       query_text="Tesla earnings",
       limit=20,
       alpha=0.7  # 70% semantic, 30% keyword
   )
   ```

4. **M&A Target Screening**
   ```python
   # Create M&A targets collection
   await mcp_manager.call_tool(
       server_name="vector_db",
       tool_name="create_collection",
       collection="ma_targets",
       dimension=1536
   )
   
   # Add company profiles
   for company in potential_targets:
       profile_text = f"{company.description} {company.products} {company.markets}"
       embedding = generate_embedding(profile_text)
       
       await mcp_manager.call_tool(
           server_name="vector_db",
           tool_name="add_document",
           collection="ma_targets",
           document_id=company.ticker,
           text=profile_text,
           embedding=embedding,
           metadata={
               "ticker": company.ticker,
               "sector": company.sector,
               "revenue": company.revenue,
               "employees": company.employees,
               "founded": company.founded
           }
       )
   
   # Find acquisition targets similar to successful acquisition
   reference_company = "Successful acquisition that fits our strategy"
   ref_embedding = generate_embedding(reference_company)
   
   result = await mcp_manager.call_tool(
       server_name="vector_db",
       tool_name="filter_search",
       collection="ma_targets",
       query_embedding=ref_embedding,
       filters={
           "sector": "Technology",
           "revenue": "<100M",
           "employees": "<500"
       },
       limit=10
   )
   ```

---

## Kubernetes MCP Server

### Use Cases

1. **Production Deployment**
   ```python
   # Deploy Axiom API to production
   result = await mcp_manager.call_tool(
       server_name="kubernetes",
       tool_name="create_deployment",
       name="axiom-api",
       image="axiom/api:v2.0",
       replicas=3,
       namespace="production",
       port=8000,
       env_vars={
           "ENV": "production",
           "DATABASE_URL": "postgresql://...",
           "REDIS_URL": "redis://..."
       },
       resources={
           "requests": {"cpu": "1000m", "memory": "2Gi"},
           "limits": {"cpu": "2000m", "memory": "4Gi"}
       },
       labels={"app": "axiom-api", "tier": "backend"}
   )
   
   # Expose as LoadBalancer
   await mcp_manager.call_tool(
       server_name="kubernetes",
       tool_name="expose_service",
       name="axiom-api-lb",
       deployment_name="axiom-api",
       port=80,
       service_type="LoadBalancer",
       namespace="production"
   )
   ```

2. **Auto-Scaling**
   ```python
   # Monitor load and scale
   usage = await mcp_manager.call_tool(
       server_name="kubernetes",
       tool_name="get_resource_usage",
       namespace="production"
   )
   
   if usage["cpu_usage_pct"] > 80:
       # Scale up
       await mcp_manager.call_tool(
           server_name="kubernetes",
           tool_name="scale_deployment",
           name="axiom-api",
           replicas=5,
           namespace="production"
       )
   ```

3. **Rolling Update**
   ```python
   # Update to new version
   result = await mcp_manager.call_tool(
       server_name="kubernetes",
       tool_name="update_deployment",
       name="axiom-api",
       image="axiom/api:v2.1",
       namespace="production"
   )
   
   # Monitor rollout
   pods = await mcp_manager.call_tool(
       server_name="kubernetes",
       tool_name="list_pods",
       namespace="production",
       label_selector="app=axiom-api"
   )
   
   # Check if rollout successful
   all_ready = all(
       pod["status"] == "Running" 
       for pod in pods["pods"]
   )
   
   if not all_ready:
       # Rollback
       await mcp_manager.call_tool(
           server_name="kubernetes",
           tool_name="rollback_deployment",
           name="axiom-api",
           namespace="production"
       )
   ```

4. **Debugging**
   ```python
   # Get pod logs
   logs = await mcp_manager.call_tool(
       server_name="kubernetes",
       tool_name="get_pod_logs",
       pod_name="axiom-api-7d9f8b6c5d-xyz",
       namespace="production",
       tail_lines=100
   )
   
   print(logs["logs"])
   
   # Execute debug command
   result = await mcp_manager.call_tool(
       server_name="kubernetes",
       tool_name="exec_pod",
       pod_name="axiom-api-7d9f8b6c5d-xyz",
       command=["python", "manage.py", "check_health"],
       namespace="production"
   )
   ```

---

## Integration Examples

### End-to-End Workflow: Automated Trading Alert System

```python
async def trading_alert_workflow(symbol: str, price_change: float):
    """Complete workflow: Monitor → Store → Alert → Log."""
    
    # 1. Store alert data in S3
    alert_data = {
        "symbol": symbol,
        "price_change": price_change,
        "timestamp": datetime.utcnow().isoformat(),
        "triggered_by": "price_monitor"
    }
    
    await mcp_manager.call_tool(
        server_name="aws",
        tool_name="s3_upload",
        bucket="axiom-alerts",
        key=f"alerts/{symbol}/{datetime.utcnow().date()}.json",
        data=json.dumps(alert_data)
    )
    
    # 2. Generate embedding and store in vector DB for pattern analysis
    alert_text = f"{symbol} price changed by {price_change}%"
    embedding = generate_embedding(alert_text)
    
    await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="add_document",
        collection="price_alerts",
        document_id=f"{symbol}_{int(datetime.utcnow().timestamp())}",
        text=alert_text,
        embedding=embedding,
        metadata={
            "symbol": symbol,
            "change_pct": price_change,
            "date": datetime.utcnow().date().isoformat()
        }
    )
    
    # 3. Send notification based on severity
    severity = "critical" if abs(price_change) > 10 else "warning"
    
    await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_alert",
        recipients={
            "email": "traders@example.com",
            "phone": "+1234567890" if severity == "critical" else None
        },
        severity=severity,
        title=f"Price Alert: {symbol}",
        message=f"{symbol} moved {price_change:+.2f}% to ${get_current_price(symbol)}",
        metadata=alert_data
    )
    
    # 4. Log to BigQuery for analytics
    await mcp_manager.call_tool(
        server_name="gcp",
        tool_name="bigquery_load",
        dataset_id="trading",
        table_id="price_alerts",
        data=[{
            "symbol": symbol,
            "price_change": price_change,
            "timestamp": alert_data["timestamp"],
            "severity": severity
        }]
    )
```

### Portfolio Backup & Recovery

```python
async def backup_portfolio_multi_cloud(portfolio_data: dict):
    """Backup portfolio to multiple cloud providers."""
    
    timestamp = datetime.utcnow().isoformat()
    filename = f"portfolio_{timestamp}.json"
    data = json.dumps(portfolio_data, indent=2)
    
    results = {}
    
    # Backup to AWS S3
    results["aws"] = await mcp_manager.call_tool(
        server_name="aws",
        tool_name="s3_upload",
        bucket="axiom-backups",
        key=f"portfolios/{filename}",
        data=data,
        metadata={"backup_time": timestamp}
    )
    
    # Backup to GCP Storage
    results["gcp"] = await mcp_manager.call_tool(
        server_name="gcp",
        tool_name="storage_upload",
        bucket_name="axiom-backups-gcp",
        blob_name=f"portfolios/{filename}",
        data=data,
        metadata={"backup_time": timestamp}
    )
    
    # Send confirmation
    if all(r["success"] for r in results.values()):
        await mcp_manager.call_tool(
            server_name="notification",
            tool_name="send_email",
            to="admin@example.com",
            subject="Portfolio Backup Complete",
            body=f"Portfolio backed up successfully at {timestamp}"
        )
    
    return results
```

### Kubernetes Auto-Deploy Pipeline

```python
async def deploy_new_version(version: str, image_tag: str):
    """Deploy new version with health checks."""
    
    # 1. Build and push image (external process)
    # docker build -t axiom/api:${version} .
    # docker push axiom/api:${version}
    
    # 2. Create new deployment
    result = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="create_deployment",
        name=f"axiom-api-{version}",
        image=f"axiom/api:{image_tag}",
        replicas=1,  # Start with 1 for testing
        namespace="staging",
        port=8000,
        env_vars={"VERSION": version}
    )
    
    if not result["success"]:
        await mcp_manager.call_tool(
            server_name="notification",
            tool_name="send_alert",
            recipients={"email": "devops@example.com"},
            severity="error",
            title="Deployment Failed",
            message=f"Failed to deploy {version}: {result['error']}"
        )
        return False
    
    # 3. Wait for pods to be ready
    await asyncio.sleep(10)
    
    # 4. Check pod health
    pods = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="list_pods",
        namespace="staging",
        label_selector=f"app=axiom-api-{version}"
    )
    
    healthy = all(pod["status"] == "Running" for pod in pods["pods"])
    
    if healthy:
        # 5. Scale to full capacity
        await mcp_manager.call_tool(
            server_name="kubernetes",
            tool_name="scale_deployment",
            name=f"axiom-api-{version}",
            replicas=3,
            namespace="staging"
        )
        
        # 6. Notify success
        await mcp_manager.call_tool(
            server_name="notification",
            tool_name="send_email",
            to="devops@example.com",
            subject=f"Deployment Success: {version}",
            body=f"Version {version} deployed successfully to staging"
        )
    else:
        # Rollback
        await mcp_manager.call_tool(
            server_name="kubernetes",
            tool_name="delete_deployment",
            name=f"axiom-api-{version}",
            namespace="staging"
        )
        
        await mcp_manager.call_tool(
            server_name="notification",
            tool_name="send_alert",
            recipients={"email": "devops@example.com"},
            severity="error",
            title="Deployment Health Check Failed",
            message=f"Version {version} failed health checks, rolled back"
        )
    
    return healthy
```

---

## Production Deployment

### AWS Production Setup

```bash
# 1. Configure AWS credentials
aws configure

# 2. Create S3 buckets
aws s3 mb s3://axiom-data-prod --region us-east-1
aws s3 mb s3://axiom-backups-prod --region us-east-1

# 3. Create IAM role for Lambda
aws iam create-role \
  --role-name axiom-lambda-exec \
  --assume-role-policy-document file://lambda-trust-policy.json

# 4. Attach permissions
aws iam attach-role-policy \
  --role-name axiom-lambda-exec \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### GCP Production Setup

```bash
# 1. Authenticate
gcloud auth login

# 2. Set project
gcloud config set project axiom-prod

# 3. Create Cloud Storage buckets
gsutil mb -l us-central1 gs://axiom-data-prod
gsutil mb -l europe-west1 gs://axiom-data-eu

# 4. Create BigQuery dataset
bq mk --dataset \
  --location=US \
  axiom-prod:trading

# 5. Create service account
gcloud iam service-accounts create axiom-api \
  --display-name="Axiom API Service Account"

# 6. Grant permissions
gcloud projects add-iam-policy-binding axiom-prod \
  --member="serviceAccount:axiom-api@axiom-prod.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

### Kubernetes Production Cluster

```bash
# 1. Create GKE cluster
gcloud container clusters create axiom-prod \
  --num-nodes=3 \
  --machine-type=n1-standard-4 \
  --region=us-central1 \
  --enable-autoscaling \
  --min-nodes=3 \
  --max-nodes=10

# 2. Get credentials
gcloud container clusters get-credentials axiom-prod --region=us-central1

# 3. Create namespace
kubectl create namespace axiom

# 4. Deploy application
kubectl apply -f k8s/production/ -n axiom

# 5. Check deployment
kubectl get deployments -n axiom
kubectl get pods -n axiom
kubectl get services -n axiom
```

---

## Monitoring & Observability

### CloudWatch Dashboard

```python
async def create_monitoring_dashboard():
    """Create CloudWatch dashboard for monitoring."""
    
    # Get Lambda metrics
    lambda_metrics = await mcp_manager.call_tool(
        server_name="aws",
        tool_name="cloudwatch_get_metrics",
        namespace="AWS/Lambda",
        metric_name="Duration",
        statistic="Average"
    )
    
    # Get EC2 metrics
    ec2_metrics = await mcp_manager.call_tool(
        server_name="aws",
        tool_name="cloudwatch_get_metrics",
        namespace="AWS/EC2",
        metric_name="CPUUtilization",
        dimensions=[{"Name": "InstanceId", "Value": "i-xxx"}],
        statistic="Average"
    )
    
    # Send daily summary
    await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_daily_report",
        to="ops@example.com",
        report_title="Infrastructure Metrics",
        report_data={
            "lambda_avg_duration": f"{lambda_metrics['datapoints'][-1]['value']}ms",
            "ec2_cpu_usage": f"{ec2_metrics['datapoints'][-1]['value']}%",
            "active_pods": "12",
            "cluster_health": "Healthy"
        }
    )
```

### Kubernetes Health Monitoring

```python
async def monitor_cluster_health():
    """Monitor Kubernetes cluster health."""
    
    # Get cluster info
    cluster_info = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="get_cluster_info"
    )
    
    # Get recent events
    events = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="get_events",
        namespace="axiom",
        limit=50
    )
    
    # Check for errors
    error_events = [e for e in events["events"] if e["type"] == "Warning"]
    
    if error_events:
        await mcp_manager.call_tool(
            server_name="notification",
            tool_name="send_alert",
            recipients={"email": "k8s-admin@example.com"},
            severity="warning",
            title=f"Kubernetes Warnings ({len(error_events)})",
            message=f"Found {len(error_events)} warning events in cluster",
            metadata={"events": error_events[:5]}
        )
```

---

## Troubleshooting

### AWS Issues

**Problem**: S3 upload fails with "Access Denied"
```python
# Solution: Check IAM permissions
aws iam get-user
aws s3api get-bucket-policy --bucket axiom-data

# Ensure bucket policy allows uploads
```

**Problem**: EC2 instance won't start
```python
# Check instance status
result = await mcp_manager.call_tool(
    server_name="aws",
    tool_name="ec2_list_instances",
    filters=[{"Name": "instance-id", "Values": ["i-xxx"]}]
)

# Check CloudWatch logs
logs = await mcp_manager.call_tool(
    server_name="aws",
    tool_name="cloudwatch_get_metrics",
    namespace="AWS/EC2",
    metric_name="StatusCheckFailed",
    dimensions=[{"Name": "InstanceId", "Value": "i-xxx"}]
)
```

### Vector DB Issues

**Problem**: ChromaDB connection refused
```bash
# Check if ChromaDB is running
docker ps | grep chroma

# Check logs
docker logs chromadb

# Restart if needed
docker-compose -f docker/week3-services.yml restart chromadb
```

**Problem**: Search returns no results
```python
# Verify collection exists
result = await mcp_manager.call_tool(
    server_name="vector_db",
    tool_name="list_collections"
)

# Check document count
# Add test document
await mcp_manager.call_tool(
    server_name="vector_db",
    tool_name="add_document",
    collection="test",
    document_id="test1",
    text="test",
    embedding=[0.1] * 1536
)
```

### Kubernetes Issues

**Problem**: Pods stuck in Pending state
```python
# Get pod details
pods = await mcp_manager.call_tool(
    server_name="kubernetes",
    tool_name="list_pods",
    namespace="axiom"
)

# Check events
events = await mcp_manager.call_tool(
    server_name="kubernetes",
    tool_name="get_events",
    namespace="axiom"
)

# Common issues:
# - Insufficient resources
# - Image pull errors
# - Config errors
```

**Problem**: Service not accessible
```bash
# Check service
kubectl get svc -n axiom

# Check endpoints
kubectl get endpoints -n axiom

# Test connectivity from pod
kubectl run -it --rm debug --image=busybox --restart=Never -- wget -O- http://service-name:port
```

---

## Best Practices

### 1. Cloud Resource Management

- **Use resource tags** for cost tracking
- **Enable auto-scaling** for variable loads
- **Implement backup strategies** (multi-cloud)
- **Monitor costs** with CloudWatch/GCP monitoring
- **Use spot instances** for non-critical workloads

### 2. Notification Strategy

- **Route by severity** (info→email, critical→SMS)
- **Avoid notification fatigue** (aggregate similar alerts)
- **Include actionable information** in alerts
- **Test notification channels** regularly
- **Implement escalation** for unacknowledged critical alerts

### 3. Vector Database Optimization

- **Choose appropriate dimension** (1536 for OpenAI)
- **Use metadata filters** to narrow search space
- **Batch document additions** for efficiency
- **Monitor index size** and query performance
- **Regular index optimization**

### 4. Kubernetes Operations

- **Use namespaces** for environment isolation
- **Implement health checks** (liveness/readiness probes)
- **Set resource limits** to prevent resource exhaustion
- **Use ConfigMaps/Secrets** for configuration
- **Monitor cluster metrics** continuously

---

## Performance Optimization

### Vector Search Optimization

```python
# Use filters to reduce search space
result = await mcp_manager.call_tool(
    server_name="vector_db",
    tool_name="filter_search",
    collection="large_collection",
    query_embedding=embedding,
    filters={"date": "2024-01-01", "type": "filing"},  # Reduces search space
    limit=10
)

# Adjust alpha for hybrid search based on use case
# alpha=1.0: Pure semantic (slower, more relevant)
# alpha=0.5: Balanced (faster, good relevance)
# alpha=0.0: Pure keyword (fastest, less relevant)
```

### S3 Upload Optimization

```python
# Use multipart upload for large files
# The MCP server handles this automatically

# For many small files, batch them
files_to_upload = [...]
tasks = []
for file_data in files_to_upload:
    task = mcp_manager.call_tool(
        server_name="aws",
        tool_name="s3_upload",
        bucket="axiom-data",
        key=file_data["key"],
        data=file_data["content"]
    )
    tasks.append(task)

results = await asyncio.gather(*tasks)
```

---

## Security Checklist

- [ ] AWS credentials stored securely (not in code)
- [ ] GCP service account with minimal permissions
- [ ] Email credentials using app-specific passwords
- [ ] Twilio credentials secured
- [ ] Vector DB API keys rotated regularly
- [ ] Kubernetes RBAC configured
- [ ] TLS/SSL enabled for all communications
- [ ] Network policies in place
- [ ] Audit logging enabled
- [ ] Secrets management system in use

---

## Summary

Week 3 MCP servers provide:

✅ **59 new tools** across 5 servers  
✅ **Cloud-agnostic** infrastructure  
✅ **Unified** notification system  
✅ **AI-powered** semantic search  
✅ **Production-scale** orchestration  
✅ **~2,150 lines** of code eliminated  
✅ **Enterprise-grade** security and monitoring  

**Total (Weeks 1-3)**: 125 tools, ~5,000 lines eliminated

For server-specific details, see [`README_WEEK3.md`](../axiom/integrations/mcp_servers/README_WEEK3.md).