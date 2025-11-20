# Airflow Deployment Guide - Complete Checklist

## 📋 Pre-Deployment Checklist

### ✅ Prerequisites

- [ ] PostgreSQL running (for Airflow metadata + stock data)
- [ ] Neo4j running (for knowledge graph)
- [ ] Redis running (for caching)
- [ ] Claude API key in `.env` file
- [ ] Python 3.11+ available
- [ ] Docker & Docker Compose installed

### ✅ Verify Existing Systems

```bash
# Check databases are running
docker ps | grep -E "(postgres|neo4j|redis)"

# Should see:
# - axiom-postgres
# - axiom-neo4j  
# - axiom-redis
```

## 🚀 Deployment Steps

### Step 1: Create Airflow Database

```bash
# Connect to PostgreSQL
docker exec -it axiom-postgres psql -U axiom -d postgres

# In PostgreSQL prompt:
CREATE DATABASE airflow;
\q
```

### Step 2: Set Up Directory Structure

```bash
cd /home/sandeep/pertinent/axiom/axiom/pipelines/airflow

# Create directories (if not already created)
mkdir -p logs plugins

# Set permissions for Airflow
chmod 777 logs plugins

# Verify DAGs are in place
ls -la dags/
# Should show:
# - company_graph_dag.py
# - events_tracker_dag.py
# - correlation_analyzer_dag.py
# - data_ingestion_dag.py
```

### Step 3: Set Airflow UID

```bash
# Export Airflow user ID
export AIRFLOW_UID=50000

# Or add to .env file
echo "AIRFLOW_UID=50000" >> /home/sandeep/pertinent/axiom/.env
```

### Step 4: Start Airflow

```bash
# From axiom/pipelines/airflow directory
docker compose -f docker-compose-airflow.yml up -d

# Watch initialization
docker logs -f axiom-airflow-init
```

**Expected Output**:
```
DB: postgresql://airflow:***@localhost:5432/airflow
Initializing database
Database initialized
Admin user created
✅ Airflow initialization complete
```

### Step 5: Verify Services Running

```bash
docker ps | grep airflow

# Should see 3 containers:
# - axiom-airflow-webserver
# - axiom-airflow-scheduler
# - axiom-airflow-init (will exit after completion)
```

### Step 6: Access Airflow UI

1. Open browser: http://localhost:8090
2. Login credentials:
   - **Username**: `admin`
   - **Password**: `admin123`

3. You should see the **Airflow Dashboard**

### Step 7: Enable DAGs

In the Airflow UI:

1. Navigate to **DAGs** page
2. Find these 4 DAGs:
   - `data_ingestion`
   - `company_graph_builder`
   - `events_tracker`
   - `correlation_analyzer`

3. Toggle each DAG to **ON** (blue switch)

4. DAGs will start executing automatically according to their schedules

### Step 8: Monitor First Runs

Click on each DAG and watch the tasks execute:

**data_ingestion** (runs every minute):
- `fetch_data` → GREEN
- `store_postgresql` → GREEN
- `cache_redis` → GREEN
- `update_neo4j` → GREEN

**company_graph_builder** (runs hourly):
- `initialize_pipeline` → GREEN
- `fetch_companies` → GREEN
- `identify_relationships` → GREEN (Claude analyzing)
- `generate_cypher` → GREEN
- `execute_neo4j` → GREEN
- `validate_graph` → GREEN

## 🔍 Verification Steps

### Verify Data Ingestion

```bash
# Check PostgreSQL has new data
docker exec -it axiom-postgres psql -U axiom -d axiom_finance -c \
  "SELECT symbol, close, timestamp FROM stock_prices ORDER BY timestamp DESC LIMIT 10;"

# Should show recent price data
```

### Verify Company Graph

```bash
# Check Neo4j has relationships
docker exec -it axiom-neo4j cypher-shell -u neo4j -p axiom_neo4j \
  "MATCH (c:Company)-[r:COMPETES_WITH]->() RETURN c.symbol, count(r) ORDER BY count(r) DESC LIMIT 10;"

# Should show companies with competitor counts
```

### Verify Airflow Metrics

```bash
# Check task success rates
docker exec axiom-airflow-webserver airflow dags list-runs

# Should show successful runs
```

## 📊 What to Monitor

### In Airflow UI

**DAG View**:
- All tasks GREEN = successful
- Any RED = failed (click for logs)
- Yellow/Orange = running

**Grid View**:
- Rows = DAG runs
- Columns = tasks
- Colors = status

**Graph View**:
- Visual task dependencies
- Execution flow

### Key Metrics

1. **Success Rate**: Should be > 95%
2. **Execution Time**: Watch for slowdowns
3. **Queue Length**: Should be near zero
4. **Failed Tasks**: Investigate immediately

## 🐛 Troubleshooting

### Airflow Won't Start

**Problem**: Containers exit immediately

**Solution**:
```bash
# Check logs
docker logs axiom-airflow-init

# Common issue: PostgreSQL not ready
# Ensure PostgreSQL is running first
docker compose -f ../../database/docker-compose.yml ps
```

### DAGs Not Visible

**Problem**: No DAGs showing in UI

**Solution**:
```bash
# Check if DAGs folder is mounted
docker exec axiom-airflow-webserver ls /opt/airflow/dags

# Check for syntax errors
docker exec axiom-airflow-webserver python /opt/airflow/dags/company_graph_dag.py

# Refresh DAGs
docker exec axiom-airflow-webserver airflow dags list-import-errors
```

### Tasks Failing

**Problem**: Tasks show RED in UI

**Solution**:
1. Click on failed task
2. Click "Log" button
3. Read error message
4. Common issues:
   - Missing ANTHROPIC_API_KEY
   - Database connection failed
   - Neo4j not accessible

### Claude API Errors

**Problem**: "Could not resolve authentication"

**Solution**:
```bash
# Verify API key in container
docker exec axiom-airflow-webserver env | grep ANTHROPIC

# Should show: ANTHROPIC_API_KEY=sk-...

# If not, check .env file is mounted
docker inspect axiom-airflow-webserver | grep -A 10 Mounts
```

## 🎯 Success Indicators

You'll know everything is working when:

1. ✅ Airflow UI loads at http://localhost:8090
2. ✅ All 4 DAGs visible and enabled
3. ✅ `data_ingestion` runs every minute
4. ✅ `company_graph_builder` completed at least once
5. ✅ PostgreSQL contains price data
6. ✅ Neo4j contains company relationships
7. ✅ No RED tasks in Grid view
8. ✅ Task logs show "✅" success messages

## 📈 Performance Tuning

### Adjust DAG Schedules

Edit the DAG files to change frequency:

```python
# Every 30 minutes instead of hourly
schedule_interval='*/30 * * * *'

# Every 2 hours
schedule_interval='0 */2 * * *'

# Daily at midnight
schedule_interval='@daily'
```

### Increase Parallelism

Edit `docker-compose-airflow.yml`:

```yaml
environment:
  - AIRFLOW__CORE__PARALLELISM=32  # Default: 16
  - AIRFLOW__CORE__DAG_CONCURRENCY=16  # Per-DAG limit
```

### Add More Workers

For heavy workloads, switch to Celery Executor:

```yaml
airflow-worker:
  <<: *airflow-common
  command: celery worker
  deploy:
    replicas: 4  # 4 parallel workers
```

## 🔄 Switching Between Modes

### Run Both Systems Simultaneously

**Keep existing LangGraph containers** for now:
```bash
# LangGraph containers still running
docker ps | grep axiom-pipeline

# Plus Airflow
docker ps | grep airflow
```

**Benefits**:
- Zero downtime migration
- Compare performance
- Gradual transition

### Full Switch to Airflow

Once confident, stop LangGraph containers:
```bash
docker compose -f ../docker-compose-langgraph.yml down
```

## 📊 Monitoring Dashboard

### Create Airflow Overview Script

```bash
cat > /home/sandeep/pertinent/axiom/scripts/check_airflow.sh << 'EOF'
#!/bin/bash
echo "=== Airflow Pipeline Status ==="
echo ""

# Check services
docker ps --format "table {{.Names}}\t{{.Status}}" | grep airflow

echo ""
echo "=== DAG Status ==="
docker exec axiom-airflow-webserver airflow dags list | grep -v "example"

echo ""
echo "=== Recent Runs ==="
docker exec axiom-airflow-webserver airflow dags list-runs | head -20
EOF

chmod +x /home/sandeep/pertinent/axiom/scripts/check_airflow.sh
```

## 🎓 Learning Airflow UI

### Main Sections

**DAGs**: List of all workflows
**Grid**: Historical run matrix
**Graph**: Visual task dependencies
**Calendar**: Run schedule visualization
**Task Duration**: Performance analysis
**Code**: View DAG source code

### Key Actions

**Trigger DAG**: Click "Play" button
**View Logs**: Click task → Log button
**View XCom**: Click task → XCom tab
**Mark Success/Failed**: Click task → Mark Success/Failed

## 🔐 Security (Production)

### Change Default Password

```bash
docker exec axiom-airflow-webserver airflow users create \
  --username your_username \
  --firstname Your \
  --lastname Name \
  --role Admin \
  --email you@company.com \
  --password secure_password

# Delete default admin
docker exec axiom-airflow-webserver airflow users delete --username admin
```

### Enable RBAC

In `docker-compose-airflow.yml`:
```yaml
environment:
  - AIRFLOW__WEBSERVER__RBAC=True
  - AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth
```

## 📝 Maintenance

### View Logs

```bash
# Webserver logs
docker logs -f axiom-airflow-webserver

# Scheduler logs
docker logs -f axiom-airflow-scheduler

# Specific DAG logs
docker exec axiom-airflow-webserver \
  airflow tasks test company_graph_builder fetch_companies 2025-11-15
```

### Clean Up Old Logs

```bash
# In Airflow UI: Admin → Configurations
# Set: logging_config_class = airflow.config_templates.airflow_local_settings.DEFAULT_LOGGING_CONFIG

# Or manually
cd /home/sandeep/pertinent/axiom/axiom/pipelines/airflow/logs
find . -name "*.log" -mtime +7 -delete  # Delete logs older than 7 days
```

### Database Maintenance

```bash
# Clean up old DAG runs
docker exec axiom-airflow-webserver \
  airflow db clean --clean-before-timestamp "2025-11-01"
```

## 🚀 Next Phase: Kafka

After Airflow is stable, next upgrade is Kafka streaming.

See [`KAFKA_INTEGRATION_PLAN.md`](./KAFKA_INTEGRATION_PLAN.md) for details.

## 💡 Tips

1. **Start Small**: Enable one DAG at a time
2. **Watch Logs**: Monitor first few runs closely  
3. **Test Failures**: Manually trigger with bad data to test retries
4. **Backup DAGs**: Keep DAG files in version control
5. **Document Changes**: Add comments to DAG code

## 🆘 Getting Help

### Check Airflow Health

```bash
# Scheduler health
docker exec axiom-airflow-scheduler airflow jobs check --job-type SchedulerJob

# Webserver health
curl http://localhost:8090/health
```

### Full Diagnostic

```bash
# Run all checks
./scripts/check_airflow.sh

# Review scheduler logs
docker logs axiom-airflow-scheduler --tail 100

# Check database connection
docker exec axiom-airflow-webserver airflow db check
```

## ✅ Deployment Complete!

Once all steps complete, you have:

- ✅ Professional DAG-based orchestration
- ✅ Visual monitoring in Airflow UI
- ✅ Automatic retries and error handling
- ✅ Task dependency management
- ✅ SLA monitoring
- ✅ Production-ready pipeline infrastructure

**Welcome to enterprise-grade data orchestration!** 🎉