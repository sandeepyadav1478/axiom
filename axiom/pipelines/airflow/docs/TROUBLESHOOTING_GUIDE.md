# Troubleshooting Guide - Apache Airflow

## 🔍 Diagnostic Framework

### Step-by-Step Troubleshooting Process

```
Problem Identified
  │
  ▼
┌──────────────────────────────────┐
│ 1. Gather Information            │
│    - What failed?                │
│    - When did it start?          │
│    - What changed recently?      │
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│ 2. Check Logs                    │
│    - Task logs in UI             │
│    - Container logs              │
│    - Database logs               │
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│ 3. Isolate Component             │
│    - Is it Airflow?              │
│    - Is it database?             │
│    - Is it external API?         │
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│ 4. Apply Fix                     │
│    - Use appropriate runbook     │
│    - Test thoroughly             │
│    - Document resolution         │
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│ 5. Prevent Recurrence            │
│    - Update monitoring           │
│    - Add alerts                  │
│    - Document lesson learned     │
└──────────────────────────────────┘
```

---

## 🚨 Common Error Messages & Solutions

### Error: "DAG not found"

**Full Error**:
```
airflow.exceptions.DagNotFound: Dag id 'company_graph_builder' not found
```

**Cause**: DAG file not loaded or syntax error

**Diagnosis**:
```bash
# 1. Check if DAG file exists
ls -la axiom/pipelines/airflow/dags/company_graph_dag.py

# 2. Check for import errors
docker exec axiom-airflow-webserver airflow dags list-import-errors

# 3. Test DAG syntax
docker exec axiom-airflow-webserver \
  python /opt/airflow/dags/company_graph_dag.py
```

**Solution**:
```bash
# If syntax error, fix the Python code
# If import error, install missing package
# If file not mounted, check docker-compose volumes

# Restart to force reload
docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml restart
```

---

### Error: "Connection timeout" to PostgreSQL

**Full Error**:
```
sqlalchemy.exc.OperationalError: could not connect to server: Connection timed out
```

**Cause**: Network/firewall issue or PostgreSQL not running

**Diagnosis**:
```bash
# 1. Check if PostgreSQL running
docker ps --filter "name=postgres"

# 2. Test connection from Airflow container
docker exec axiom-airflow-webserver \
  psql -h localhost -U axiom -d axiom_finance -c "SELECT 1;"

# 3. Check network mode
docker inspect axiom-airflow-webserver | grep NetworkMode
# Should show: "host"
```

**Solution**:
```bash
# Option 1: Restart PostgreSQL
docker compose -f axiom/database/docker-compose.yml restart postgres

# Option 2: Verify network mode is "host"
# In docker-compose-airflow.yml, ensure:
#   network_mode: "host"

# Option 3: Check firewall
sudo ufw status
# Ensure port 5432 is accessible
```

---

### Error: "ModuleNotFoundError" in DAG

**Full Error**:
```
ModuleNotFoundError: No module named 'pipelines'
```

**Cause**: Python path not configured or package not installed

**Diagnosis**:
```bash
# 1. Check Python path in container
docker exec axiom-airflow-webserver \
  python -c "import sys; print('\n'.join(sys.path))"

# 2. Check if axiom directory mounted
docker exec axiom-airflow-webserver ls /opt/airflow/axiom

# 3. Check if module exists
docker exec axiom-airflow-webserver \
  ls /opt/airflow/axiom/pipelines/
```

**Solution**:
```python
# In DAG file, add proper Python path:
import sys
sys.path.insert(0, '/opt/airflow/axiom')

# Or use fully-qualified imports:
from axiom.pipelines.companies.company_graph_builder import CompanyGraphBuilder
```

---

### Error: Claude API "authentication failed"

**Full Error**:
```
anthropic.AuthenticationError: Could not resolve authentication method
```

**Cause**: ANTHROPIC_API_KEY not set or invalid

**Diagnosis**:
```bash
# 1. Check if env var set in container
docker exec axiom-airflow-webserver env | grep ANTHROPIC

# 2. Check .env file
grep ANTHROPIC_API_KEY .env

# 3. Verify key is valid
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-sonnet-20240229","max_tokens":10,"messages":[{"role":"user","content":"test"}]}'
```

**Solution**:
```bash
# 1. Add key to .env file
echo 'ANTHROPIC_API_KEY=sk-ant-api03-your-key-here' >> .env

# 2. Restart Airflow to pick up new env
docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml restart

# 3. Verify key loaded
docker exec axiom-airflow-webserver env | grep ANTHROPIC
```

---

### Error: "Task stuck in running state"

**Full Error**:
```
Task has been running for 2+ hours with no progress
```

**Cause**: Infinite loop, deadlock, or resource exhaustion

**Diagnosis**:
```bash
# 1. Check task logs
# UI → Click task → View Log

# 2. Check container resources
docker stats axiom-airflow-scheduler --no-stream

# 3. Check for zombie processes
docker exec axiom-airflow-scheduler \
  ps aux | grep -E "airflow|python"

# 4. Check database locks
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "SELECT pid, usename, state, query FROM pg_stat_activity WHERE state = 'active';"
```

**Solution**:
```bash
# 1. Kill stuck task
# UI → Click task → Mark Failed
# This will trigger retry

# 2. If that doesn't work, kill process
docker exec axiom-airflow-scheduler \
  pkill -f "airflow tasks run company_graph_builder"

# 3. Clear and re-run
docker exec axiom-airflow-webserver \
  airflow tasks clear company_graph_builder \
  --task-regex "stuck_task_name" \
  --start-date 2025-11-20
```

---

### Error: Neo4j "ServiceUnavailable"

**Full Error**:
```
neo4j.exceptions.ServiceUnavailable: Failed to establish connection
```

**Cause**: Neo4j not running or authentication failed

**Diagnosis**:
```bash
# 1. Check Neo4j container
docker ps --filter "name=neo4j"

# 2. Test direct connection
docker exec axiom_neo4j cypher-shell -u neo4j -p axiom_neo4j \
  "RETURN 1;"

# 3. Check from Airflow container
docker exec axiom-airflow-webserver python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'axiom_neo4j'))
driver.verify_connectivity()
print('✅ Connected')
"
```

**Solution**:
```bash
# 1. Restart Neo4j
docker compose -f axiom/database/docker-compose.yml restart neo4j

# 2. Wait for Neo4j to be ready
sleep 30

# 3. Verify connection
docker exec axiom_neo4j cypher-shell -u neo4j -p axiom_neo4j \
  "MATCH (n) RETURN count(n) LIMIT 1;"

# 4. Retry failed tasks
# UI → Clear task → Auto-retry
```

---

## 📊 Performance Issues

### Issue: DAGs running slower than expected

**Symptoms**:
- DAG duration increasing over time
- Tasks taking 2-3x normal time
- Queue backing up

**Diagnosis**:
```bash
# 1. Check task durations
# UI → Browse → Task Duration

# 2. Check resource usage
docker stats --no-stream axiom-airflow-scheduler

# 3. Check database performance
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size 
   FROM pg_tables 
   WHERE schemaname = 'public' 
   ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
   LIMIT 10;"

# 4. Check for slow queries
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "SELECT query, mean_exec_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_exec_time DESC 
   LIMIT 10;" || echo "pg_stat_statements not enabled"
```

**Solutions**:
```bash
# Option 1: Clean old data
docker exec axiom-airflow-webserver \
  airflow db clean --clean-before-timestamp "$(date -d '30 days ago' '+%Y-%m-%d')"

# Option 2: Optimize queries
# Add indexes to frequently queried tables

# Option 3: Increase resources
# Edit docker-compose, add:
#   deploy:
#     resources:
#       limits:
#         cpus: '2.0'
#         memory: 4G

# Option 4: Add more workers (CeleryExecutor)
# Switch from LocalExecutor to CeleryExecutor
```

---

### Issue: High memory usage

**Symptoms**:
- Container using >2GB RAM
- OOM (Out of Memory) errors
- Scheduler crashes

**Diagnosis**:
```bash
# 1. Check memory usage
docker stats --no-stream axiom-airflow-scheduler

# 2. Check for memory leaks
docker exec axiom-airflow-scheduler \
  python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# 3. Check large XCom data
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "SELECT dag_id, task_id, pg_size_pretty(length(value::text)::bigint) as size
   FROM xcom 
   ORDER BY length(value::text) DESC 
   LIMIT 10;"
```

**Solutions**:
```bash
# Option 1: Reduce XCom data size
# Instead of storing full datasets in XCom:
# - Store file paths
# - Use external storage (S3, filesystem)
# - Serialize efficiently

# Option 2: Increase memory limit
# In docker-compose-airflow.yml:
deploy:
  resources:
    limits:
      memory: 4G

# Option 3: Clean XCom table
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "DELETE FROM xcom WHERE timestamp < NOW() - INTERVAL '7 days';"
```

---

## 🔄 DAG-Specific Issues

### Data Ingestion DAG Issues

**Problem**: Yahoo Finance API rate limiting

**Error**: `429 Too Many Requests`

**Solution**:
```python
# Add delay between requests
import time

for symbol in SYMBOLS:
    fetch_price(symbol)
    time.sleep(0.5)  # 500ms delay
```

---

**Problem**: PostgreSQL write errors

**Error**: `duplicate key value violates unique constraint`

**Solution**:
```python
# Use UPSERT (ON CONFLICT) instead of INSERT
cur.execute("""
    INSERT INTO stock_prices (symbol, timestamp, close, timeframe)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (symbol, timestamp, timeframe) 
    DO UPDATE SET close = EXCLUDED.close
""", (symbol, timestamp, close, 'MINUTE_1'))
```

---

### Company Graph DAG Issues

**Problem**: Claude timeout

**Error**: `ReadTimeout: The read operation timed out`

**Solution**:
```python
# Increase timeout in DAG
from langchain_anthropic import ChatAnthropic

claude = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    timeout=60,  # Increase from default 30s
    max_retries=3
)
```

---

**Problem**: Too many competitors identified

**Error**: Creates 100+ relationships per company

**Solution**:
```python
# Limit Claude response
prompt = f"""Identify EXACTLY 5 direct competitors for {company}.
Return ONLY ticker symbols, comma-separated.
Example: MSFT,GOOGL,AMZN,META,ORCL"""

response = claude.invoke(prompt)
competitors = response.content.strip().split(',')[:5]  # Limit to 5
```

---

## 🔧 Configuration Issues

### Issue: DAG not appearing in UI

**Checklist**:
```bash
# 1. File in correct location?
ls -la axiom/pipelines/airflow/dags/my_dag.py

# 2. Valid Python syntax?
python axiom/pipelines/airflow/dags/my_dag.py

# 3. Any import errors?
docker exec axiom-airflow-webserver airflow dags list-import-errors

# 4. Scheduler running?
docker ps --filter "name=scheduler"

# 5. Wait 60 seconds for DAG detection
sleep 60
docker exec axiom-airflow-webserver airflow dags list | grep my_dag
```

---

### Issue: Scheduler not picking up new DAG runs

**Symptoms**:
- DAG enabled but not executing
- No runs in Grid view
- Schedule shows but doesn't trigger

**Solution**:
```bash
# 1. Check scheduler is running
curl http://localhost:8080/health | jq '.scheduler'

# 2. Check DAG start_date
# Must be in the past!
start_date=days_ago(1)  # ✅ Good
start_date=datetime(2025, 12, 1)  # ❌ Future date won't trigger

# 3. Check if catchup=False
# If catchup=True and start_date is old, might create many backfill runs

# 4. Manually trigger to test
docker exec axiom-airflow-webserver airflow dags trigger my_dag

# 5. Restart scheduler
docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml restart airflow-scheduler
```

---

## 💾 Database Issues

### Issue: Airflow metadata database full

**Symptoms**:
- Disk space warnings
- Slow UI performance
- Backup failures

**Diagnosis**:
```bash
# Check database size
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "SELECT pg_size_pretty(pg_database_size('airflow'));"

# Check largest tables
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "SELECT tablename, pg_size_pretty(pg_total_relation_size(tablename::regclass))
   FROM pg_tables 
   WHERE schemaname = 'public' 
   ORDER BY pg_total_relation_size(tablename::regclass) DESC 
   LIMIT 10;"
```

**Solution**:
```bash
# Clean old DAG runs (keep last 30 days)
docker exec axiom-airflow-webserver \
  airflow db clean --clean-before-timestamp "$(date -d '30 days ago' '+%Y-%m-%d')"

# Clean XCom data
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "DELETE FROM xcom WHERE timestamp < NOW() - INTERVAL '7 days';"

# Vacuum database
docker exec axiom_postgres psql -U airflow -d airflow -c "VACUUM FULL;"
```

---

## 🌐 Network & Connectivity Issues

### Issue: Can't reach external APIs (Yahoo Finance, Claude)

**Symptoms**:
- fetch_data tasks fail with connection errors
- "Network is unreachable"
- Timeout errors

**Diagnosis**:
```bash
# 1. Test internet from container
docker exec axiom-airflow-webserver curl -I https://www.google.com

# 2. Test Yahoo Finance specifically
docker exec axiom-airflow-webserver \
  python -c "import yfinance as yf; print(yf.Ticker('AAPL').info['longName'])"

# 3. Check network mode
docker inspect axiom-airflow-webserver | grep NetworkMode
```

**Solution**:
```bash
# Ensure host network mode in docker-compose-airflow.yml:
network_mode: "host"

# This allows container to use host's internet connection

# Restart after change
docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml up -d --force-recreate
```

---

## 🎯 Task-Specific Debugging

### Debug any task independently

**Test task in isolation**:
```bash
# Run task without scheduler
docker exec axiom-airflow-webserver \
  airflow tasks test company_graph_builder fetch_companies 2025-11-20

# This runs task immediately without:
# - Dependencies
# - Retry logic
# - State saving
# Good for debugging!
```

**Check task state**:
```bash
# Get task state for specific run
docker exec axiom-airflow-webserver \
  airflow tasks state company_graph_builder fetch_companies 2025-11-20
```

**View task output**:
```bash
# Get return value from XCom
docker exec axiom-airflow-webserver \
  airflow tasks render company_graph_builder fetch_companies 2025-11-20
```

---

## 🛡️ Security Issues

### Issue: Unauthorized access to Airflow UI

**Symptoms**:
- Login page not appearing
- Anyone can access without password

**Solution**:
```yaml
# In docker-compose-airflow.yml, ensure:
environment:
  - AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session
  - AIRFLOW__WEBSERVER__AUTHENTICATE=True
  - AIRFLOW__WEBSERVER__AUTH_BACKEND=airflow.www.auth.backend.password_auth

# Restart
docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml restart
```

---

### Issue: Need to change admin password

**Current**: admin/admin123 (default, insecure)

**Solution**:
```bash
# Create new admin user
docker exec axiom-airflow-webserver \
  airflow users create \
    --username your_username \
    --firstname Your \
    --lastname Name \
    --role Admin \
    --email you@company.com \
    --password SecurePassword123!

# Delete default admin
docker exec axiom-airflow-webserver \
  airflow users delete --username admin

# Verify
docker exec axiom-airflow-webserver airflow users list
```

---

## 📈 Monitoring & Alerting Issues

### Issue: Not receiving failure alerts

**Cause**: Email not configured

**Solution**:
```yaml
# Add to docker-compose-airflow.yml environment:
- AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
- AIRFLOW__SMTP__SMTP_USER=your@gmail.com
- AIRFLOW__SMTP__SMTP_PASSWORD=your_app_password
- AIRFLOW__SMTP__SMTP_PORT=587
- AIRFLOW__SMTP__SMTP_SSL=False
- AIRFLOW__SMTP__SMTP_STARTTLS=True
- AIRFLOW__SMTP__SMTP_MAIL_FROM=airflow@axiom.com

# Test email sending
docker exec axiom-airflow-webserver \
  airflow tasks test company_graph_builder fetch_companies 2025-11-20 \
  --email your@email.com
```

---

## 🐛 Advanced Debugging

### Enable debug logging

```yaml
# In docker-compose-airflow.yml:
environment:
  - AIRFLOW__LOGGING__LOGGING_LEVEL=DEBUG

# Restart
docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml restart

# View debug logs
docker logs -f axiom-airflow-scheduler | grep DEBUG
```

---

### Access Airflow Python console

```bash
# Interactive Python in Airflow container
docker exec -it axiom-airflow-webserver python

# Then test imports
>>> from airflow import DAG
>>> from langchain_anthropic import ChatAnthropic
>>> import neo4j
>>> # Test your code interactively
```

---

### Check Airflow configuration

```bash
# View all configuration
docker exec axiom-airflow-webserver \
  airflow config list

# Check specific setting
docker exec axiom-airflow-webserver \
  airflow config get-value core executor

# Test configuration
docker exec axiom-airflow-webserver \
  airflow config show
```

---

## 📚 Diagnostic Commands Cheat Sheet

```bash
# === Services ===
docker ps --filter "name=airflow"                    # Check running
docker logs axiom-airflow-scheduler --tail 50        # View logs
curl http://localhost:8080/health | jq              # Health check

# === DAGs ===
docker exec axiom-airflow-webserver airflow dags list              # List all
docker exec axiom-airflow-webserver airflow dags list-import-errors # Check errors
docker exec axiom-airflow-webserver airflow dags show my_dag        # DAG structure

# === Tasks ===
docker exec axiom-airflow-webserver \
  airflow tasks test my_dag my_task 2025-11-20      # Test task
docker exec axiom-airflow-webserver \
  airflow tasks state my_dag my_task 2025-11-20     # Check state
docker exec axiom-airflow-webserver \
  airflow tasks clear my_dag --start-date 2025-11-20 # Retry task

# === Database ===
docker exec axiom_postgres psql -U airflow -d airflow # Connect to DB
docker exec axiom-airflow-webserver airflow db check  # Check DB health
docker exec axiom-airflow-webserver airflow db clean  # Clean old data

# === Debugging ===
docker exec -it axiom-airflow-webserver bash         # Container shell
docker exec axiom-airflow-webserver airflow version  # Version info
docker exec axiom-airflow-webserver env              # Environment vars
```

---

## 🎯 Problem Resolution Matrix

| Symptom | Most Likely Cause | Quick Fix | Runbook |
|---------|-------------------|-----------|---------|
| All DAGs RED | Database/Scheduler down | Restart services | RUNBOOK-001 |
| Single DAG failing | Code error in DAG | Check logs, fix code | - |
| Task stuck (yellow) | Infinite loop/deadlock | Kill task, retry | RUNBOOK-002 |
| "Module not found" | Import error | Fix Python path | - |
| "Auth failed" | Missing API key | Add to .env | - |
| Slow performance | Database bloat | Clean old data | - |
| High memory | XCom data large | Clean XCom, optimize | - |
| Can't connect to DB | Network/credentials | Check docker network | RUNBOOK-004 |

---

## 💡 Prevention Best Practices

1. **Monitor proactively**: Run health checks hourly
2. **Clean regularly**: Delete logs >7 days, DAG runs >30 days
3. **Test thoroughly**: Use `airflow tasks test` before deploying
4. **Version control**: All DAG changes in git
5. **Validate early**: Check import errors immediately
6. **Resource limits**: Set memory/CPU limits in docker-compose
7. **Backup often**: Daily database backups
8. **Document issues**: Log all problems and solutions

---

This troubleshooting guide provides systematic approaches to diagnosing and resolving common Airflow issues with professional operational procedures.