# Operational Runbooks - Apache Airflow

## 🚨 Emergency Procedures

### RUNBOOK-001: All DAGs Failing

**Severity**: P0 - Critical
**Impact**: Complete pipeline failure

**Symptoms**:
- All DAGs showing RED in UI
- Multiple email alerts
- No data flowing to databases

**Diagnosis**:
```bash
# 1. Check Airflow services
docker ps --filter "name=airflow"

# 2. Check scheduler health
curl http://localhost:8080/health | jq '.scheduler'

# 3. Check database connectivity
docker exec axiom-airflow-webserver airflow db check

# 4. Check recent errors
docker logs axiom-airflow-scheduler --tail 100 | grep ERROR
```

**Resolution Steps**:
1. **Check database connection**:
   ```bash
   docker exec axiom_postgres psql -U airflow -d airflow -c "SELECT 1;"
   ```
   
2. **Restart Airflow services**:
   ```bash
   docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml restart
   ```
   
3. **Verify services recovered**:
   ```bash
   sleep 30
   curl http://localhost:8080/health
   ```

4. **Re-trigger failed DAG runs**:
   - Open UI: http://localhost:8080
   - Click on failed DAG run
   - Click "Clear" to retry

**Prevention**:
- Set up database connection pooling
- Add health check monitoring
- Configure auto-restart policies

---

### RUNBOOK-002: Single DAG Stuck

**Severity**: P1 - High
**Impact**: One pipeline not processing

**Symptoms**:
- Specific DAG showing yellow (running) for >2 hours
- No progress in task execution
- Other DAGs running normally

**Diagnosis**:
```bash
# 1. Check running tasks
docker exec axiom-airflow-webserver \
  airflow tasks states-for-dag-run company_graph_builder <run_id>

# 2. Check task logs
# In UI: Click task → View Log

# 3. Check for deadlocks
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

**Resolution Steps**:
1. **Kill stuck task**:
   ```bash
   # In UI: Click task → Mark Failed
   # This will trigger retry
   ```

2. **Clear task and retry**:
   ```bash
   docker exec axiom-airflow-webserver \
     airflow tasks clear company_graph_builder \
     --task-regex ".*" \
     --start-date 2025-11-20
   ```

3. **Check for zombie processes**:
   ```bash
   docker exec axiom-airflow-scheduler \
     ps aux | grep airflow
   ```

**Prevention**:
- Set execution_timeout on all tasks
- Monitor task duration metrics
- Add task-level health checks

---

### RUNBOOK-003: Claude API Rate Limiting

**Severity**: P2 - Medium
**Impact**: Slower processing, increased costs

**Symptoms**:
- Tasks failing with "rate_limit_error"
- Logs show 429 status codes
- identify_relationships task retrying frequently

**Diagnosis**:
```bash
# 1. Check recent Claude API errors
docker logs axiom-airflow-scheduler | grep -i "rate_limit\|429"

# 2. Count API calls in last hour
# Check Airflow UI → company_graph_builder → Task Duration
```

**Resolution Steps**:
1. **Reduce DAG frequency temporarily**:
   ```python
   # Edit company_graph_dag.py
   schedule_interval='@daily'  # Instead of '@hourly'
   ```

2. **Add rate limiting to Claude calls**:
   ```python
   import time
   
   # Add delay between Claude calls
   time.sleep(2)  # 2 seconds between requests
   ```

3. **Batch process symbols**:
   ```python
   # Process in smaller batches
   SYMBOLS = SYMBOLS[:10]  # Reduce from 30 to 10
   ```

**Prevention**:
- Implement token bucket rate limiter
- Add exponential backoff
- Monitor Claude API usage
- Set budget alerts

---

### RUNBOOK-004: Neo4j Connection Lost

**Severity**: P1 - High
**Impact**: Graph updates failing

**Symptoms**:
- execute_neo4j task failing
- Logs show "ServiceUnavailable" or connection timeout
- Neo4j browser not accessible

**Diagnosis**:
```bash
# 1. Check Neo4j container
docker ps --filter "name=neo4j"

# 2. Test Neo4j connection
docker exec axiom_neo4j cypher-shell -u neo4j -p axiom_neo4j -d neo4j \
  "RETURN 1;"

# 3. Check Neo4j logs
docker logs axiom_neo4j --tail 50
```

**Resolution Steps**:
1. **Restart Neo4j**:
   ```bash
   docker compose -f axiom/database/docker-compose.yml restart neo4j
   ```

2. **Verify connection from Airflow**:
   ```bash
   docker exec axiom-airflow-webserver python -c "
   from neo4j import GraphDatabase
   driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'axiom_neo4j'))
   driver.verify_connectivity()
   print('✅ Connection successful')
   "
   ```

3. **Retry failed tasks**:
   - UI → Clear failed tasks
   - Or: `airflow tasks clear company_graph_builder --task-regex execute_neo4j`

**Prevention**:
- Add connection pooling
- Implement circuit breaker
- Set up Neo4j monitoring
- Configure auto-restart

---

### RUNBOOK-005: Disk Space Full

**Severity**: P0 - Critical
**Impact**: All services failing

**Symptoms**:
- "No space left on device" errors
- Containers crashing
- Logs not writing

**Diagnosis**:
```bash
# 1. Check disk usage
df -h

# 2. Check Docker disk usage
docker system df

# 3. Find largest files
du -h axiom/pipelines/airflow/logs | sort -rh | head -20
```

**Resolution Steps**:
1. **Clean old Airflow logs**:
   ```bash
   # Keep only last 7 days
   find axiom/pipelines/airflow/logs -name "*.log" -mtime +7 -delete
   ```

2. **Clean Docker system**:
   ```bash
   docker system prune -a --volumes -f
   # WARNING: This removes all unused containers/images/volumes
   ```

3. **Clean Airflow database**:
   ```bash
   docker exec axiom-airflow-webserver \
     airflow db clean --clean-before-timestamp "$(date -d '30 days ago' '+%Y-%m-%d')"
   ```

**Prevention**:
- Set up log rotation (logrotate)
- Configure Airflow log retention (7 days)
- Monitor disk usage with alerts
- Automated cleanup cron job

---

## 📅 Routine Maintenance Procedures

### MAINT-001: Daily Health Check

**Frequency**: Daily at 9 AM
**Duration**: 5 minutes

**Checklist**:
```bash
# 1. Verify all services running
docker ps --filter "name=airflow" --format "{{.Names}}: {{.Status}}"

# 2. Check scheduler health
curl -s http://localhost:8080/health | jq

# 3. Review failed DAG runs (last 24h)
docker exec axiom-airflow-webserver \
  airflow dags list-runs --state failed --no-backfill | head -20

# 4. Check disk space
df -h | grep -E "/$|/var"

# 5. Review error logs
docker logs axiom-airflow-scheduler --since 24h | grep ERROR | tail -20
```

**Action Items**:
- If >3 failures: Investigate root cause
- If disk >80%: Clean logs
- If scheduler unhealthy: Restart services

---

### MAINT-002: Weekly Performance Review

**Frequency**: Weekly on Mondays
**Duration**: 15 minutes

**Metrics to Review**:
```bash
# 1. DAG success rates
docker exec axiom-airflow-webserver airflow dags list

# 2. Average task duration
# Check UI → Browse → Task Duration

# 3. Queue lengths
# Check UI → Browse → Jobs

# 4. Database size
docker exec axiom_postgres psql -U airflow -d airflow -c \
  "SELECT pg_size_pretty(pg_database_size('airflow'));"
```

**Analysis**:
- Success rate should be >95%
- Task duration should be stable
- Queue should be near zero
- Database growth should be linear

**Action Items**:
- If success rate <95%: Review error patterns
- If task duration increasing: Optimize queries
- If queue backing up: Add workers
- If database growing fast: Enable cleanup

---

### MAINT-003: Monthly Security Audit

**Frequency**: First Monday of each month
**Duration**: 30 minutes

**Security Checklist**:
```bash
# 1. Review user accounts
docker exec axiom-airflow-webserver airflow users list

# 2. Check for default passwords
# Verify admin password was changed from admin123

# 3. Review DAG permissions
# Check UI → Security → List Roles

# 4. Audit API access logs
docker logs axiom-airflow-webserver | grep "POST /api" | tail -50

# 5. Check for exposed secrets
grep -r "password\|api_key\|secret" axiom/pipelines/airflow/dags/
```

**Action Items**:
- Rotate passwords quarterly
- Remove unused user accounts
- Review and restrict DAG access
- Update secrets in environment variables

---

## 🔧 Configuration Changes

### CONFIG-001: Change DAG Schedule

**Use Case**: Adjust pipeline frequency

**Procedure**:
1. **Edit DAG file**:
   ```python
   # In axiom/pipelines/airflow/dags/company_graph_dag.py
   schedule_interval='@daily'  # Change from '@hourly'
   ```

2. **Airflow auto-detects change** (30-60 seconds)

3. **Verify new schedule**:
   ```bash
   docker exec axiom-airflow-webserver \
     airflow dags show company_graph_builder
   ```

**No restart needed** - Airflow picks up DAG changes automatically!

---

### CONFIG-002: Add Email Alerts

**Use Case**: Get notified on failures

**Procedure**:
1. **Configure SMTP** in docker-compose-airflow.yml:
   ```yaml
   environment:
     - AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
     - AIRFLOW__SMTP__SMTP_USER=your@email.com
     - AIRFLOW__SMTP__SMTP_PASSWORD=your_app_password
     - AIRFLOW__SMTP__SMTP_PORT=587
     - AIRFLOW__SMTP__SMTP_MAIL_FROM=airflow@axiom.com
   ```

2. **Enable in DAG**:
   ```python
   default_args = {
       'email': ['you@company.com'],
       'email_on_failure': True,
       'email_on_retry': False
   }
   ```

3. **Restart Airflow**:
   ```bash
   docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml restart
   ```

---

### CONFIG-003: Increase Parallelism

**Use Case**: Process more tasks concurrently

**Procedure**:
1. **Edit docker-compose-airflow.yml**:
   ```yaml
   environment:
     - AIRFLOW__CORE__PARALLELISM=32  # Default: 16
     - AIRFLOW__CORE__DAG_CONCURRENCY=16  # Per DAG
     - AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG=16
   ```

2. **Restart services**:
   ```bash
   docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml restart
   ```

3. **Monitor resource usage**:
   ```bash
   docker stats axiom-airflow-scheduler
   ```

---

## 📊 Monitoring Procedures

### MONITOR-001: Check DAG Health

**Run this every hour (automated)**:
```bash
#!/bin/bash
# Save as: scripts/check_airflow_health.sh

echo "=== Airflow Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check services
echo "Services:"
docker ps --filter "name=airflow" --format "{{.Names}}: {{.Status}}"
echo ""

# Check scheduler
echo "Scheduler Status:"
curl -s http://localhost:8080/health | jq '.scheduler'
echo ""

# Check recent failures
echo "Recent Failures:"
docker exec axiom-airflow-webserver \
  airflow dags list-runs --state failed --no-backfill | head -5
echo ""

# Check queue
echo "Task Queue:"
docker exec axiom-airflow-webserver \
  airflow jobs check --job-type SchedulerJob --hostname $(hostname)
echo ""

echo "=== Health Check Complete ==="
```

---

### MONITOR-002: Track Claude API Costs

**Run this daily**:
```bash
#!/bin/bash
# Save as: scripts/track_claude_costs.sh

echo "=== Claude API Cost Tracking ==="
echo "Date: $(date +%Y-%m-%d)"
echo ""

# Count Claude API calls today
CALLS=$(docker logs axiom-airflow-scheduler --since 24h | \
  grep -c "POST https://api.anthropic.com")

# Estimate cost (assume avg 1000 tokens per call)
TOKENS=$((CALLS * 1000))
COST=$(echo "scale=2; $TOKENS * 0.00003" | bc)

echo "Claude API Calls (24h): $CALLS"
echo "Estimated Tokens: $TOKENS"
echo "Estimated Cost: \$$COST"
echo ""

# Alert if over budget
if (( $(echo "$COST > 5.00" | bc -l) )); then
  echo "⚠️ WARNING: Daily cost exceeded $5 budget!"
fi
```

---

### MONITOR-003: Neo4j Graph Growth

**Run this weekly**:
```bash
#!/bin/bash
# Save as: scripts/monitor_graph_growth.sh

echo "=== Neo4j Knowledge Graph Metrics ==="
echo "Date: $(date +%Y-%m-%d)"
echo ""

# Count nodes by type
echo "Node Counts:"
docker exec axiom_neo4j cypher-shell -u neo4j -p axiom_neo4j -d neo4j \
  "MATCH (n) RETURN labels(n)[0] as type, count(*) as count ORDER BY count DESC;"
echo ""

# Count relationships by type  
echo "Relationship Counts:"
docker exec axiom_neo4j cypher-shell -u neo4j -p axiom_neo4j -d neo4j \
  "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC;"
echo ""

# Graph density
echo "Graph Density Metrics:"
docker exec axiom_neo4j cypher-shell -u neo4j -p axiom_neo4j -d neo4j \
  "MATCH (n) WITH count(n) as nodes MATCH ()-[r]->() WITH nodes, count(r) as rels RETURN nodes, rels, (rels*1.0/nodes) as density;"
```

---

## 🎯 Standard Operating Procedures

### SOP-001: Enable New DAG

**When**: Adding a new pipeline to Airflow

**Steps**:
1. **Create DAG file** in `axiom/pipelines/airflow/dags/`:
   ```python
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime
   
   with DAG(
       'new_pipeline',
       start_date=datetime(2025, 11, 20),
       schedule_interval='@daily',
       catchup=False
   ) as dag:
       # Define tasks
       pass
   ```

2. **Wait for auto-detection** (30-60 seconds)

3. **Verify DAG loaded**:
   ```bash
   docker exec axiom-airflow-webserver airflow dags list | grep new_pipeline
   ```

4. **Check for import errors**:
   ```bash
   docker exec axiom-airflow-webserver \
     airflow dags list-import-errors
   ```

5. **Enable in UI**:
   - Navigate to http://localhost:8080
   - Find DAG in list
   - Toggle switch to ON (blue)

6. **Trigger test run**:
   - Click "Play" button
   - Monitor execution
   - Verify success

---

### SOP-002: Disable DAG for Maintenance

**When**: Need to pause a pipeline temporarily

**Steps**:
1. **Method A: Via UI** (Preferred):
   - Open http://localhost:8080
   - Find DAG
   - Toggle switch to OFF (grey)

2. **Method B: Via CLI**:
   ```bash
   docker exec axiom-airflow-webserver \
     airflow dags pause company_graph_builder
   ```

3. **Verify paused**:
   ```bash
   docker exec axiom-airflow-webserver \
     airflow dags list | grep company_graph_builder
   # Should show "True" in paused column
   ```

4. **After maintenance, re-enable**:
   ```bash
   docker exec axiom-airflow-webserver \
     airflow dags unpause company_graph_builder
   ```

---

### SOP-003: Manual DAG Trigger

**When**: Need to run pipeline outside schedule

**Steps**:
1. **Via UI**:
   - Open http://localhost:8080
   - Click on DAG name
   - Click "Play" button (top right)
   - Confirm trigger

2. **Via CLI**:
   ```bash
   docker exec axiom-airflow-webserver \
     airflow dags trigger company_graph_builder \
     --conf '{"symbols": ["AAPL", "MSFT", "GOOGL"]}'
   ```

3. **Monitor execution**:
   - UI → Grid View → See new run appear
   - Watch tasks turn green

4. **View logs**:
   - Click on task box
   - Click "Log" button

---

### SOP-004: View Task Logs

**When**: Debugging task failures or checking output

**Steps**:
1. **Via UI** (Preferred):
   - Open http://localhost:8080
   - Click DAG name
   - Click on run (green/red box in Grid)
   - Click on task box
   - Click "Log" button

2. **Via CLI**:
   ```bash
   docker exec axiom-airflow-webserver \
     airflow tasks logs company_graph_builder fetch_companies 2025-11-20
   ```

3. **Stream live logs**:
   ```bash
   docker logs -f axiom-airflow-scheduler | grep company_graph
   ```

---

## 🔄 Backup & Recovery

### BACKUP-001: Backup Airflow Metadata

**Frequency**: Daily at 2 AM
**Retention**: 30 days

**Procedure**:
```bash
#!/bin/bash
# Save as: scripts/backup_airflow_db.sh

BACKUP_DIR="/home/sandeep/pertinent/axiom/backups/airflow"
BACKUP_FILE="airflow_$(date +%Y%m%d_%H%M%S).sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup Airflow database
docker exec axiom_postgres pg_dump -U airflow airflow > "$BACKUP_DIR/$BACKUP_FILE"

# Compress
gzip "$BACKUP_DIR/$BACKUP_FILE"

# Keep only last 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "✅ Backup complete: $BACKUP_FILE.gz"
```

**Automate with cron**:
```bash
# Add to crontab
0 2 * * * /home/sandeep/pertinent/axiom/scripts/backup_airflow_db.sh
```

---

### RECOVERY-001: Restore from Backup

**When**: Airflow metadata corrupted or lost

**Steps**:
1. **Stop Airflow services**:
   ```bash
   docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml down
   ```

2. **Drop and recreate database**:
   ```bash
   docker exec axiom_postgres psql -U axiom -d postgres -c \
     "DROP DATABASE IF EXISTS airflow;"
   docker exec axiom_postgres psql -U axiom -d postgres -c \
     "CREATE DATABASE airflow OWNER airflow;"
   ```

3. **Restore from backup**:
   ```bash
   gunzip -c backups/airflow/airflow_20251120_020000.sql.gz | \
     docker exec -i axiom_postgres psql -U airflow -d airflow
   ```

4. **Restart Airflow**:
   ```bash
   docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml up -d
   ```

5. **Verify recovery**:
   ```bash
   docker exec axiom-airflow-webserver airflow dags list
   # Should show all DAGs
   ```

---

## 📞 Escalation Procedures

### Severity Levels

**P0 - Critical** (Immediate Response):
- All DAGs failing
- Airflow completely down
- Data loss occurring
- **Response**: < 15 minutes
- **Resolution Target**: < 1 hour

**P1 - High** (Urgent):
- Single critical DAG failing
- Database connection issues
- Significant delays (>2x normal)
- **Response**: < 1 hour
- **Resolution Target**: < 4 hours

**P2 - Medium** (Important):
- Non-critical DAG failures
- Performance degradation
- Rate limiting issues
- **Response**: < 4 hours
- **Resolution Target**: < 24 hours

**P3 - Low** (Can Wait):
- UI issues (non-blocking)
- Cosmetic problems
- Feature requests
- **Response**: < 24 hours
- **Resolution Target**: < 1 week

---

## 📋 Incident Response Template

```markdown
## Incident Report: [INCIDENT-YYYY-MM-DD-###]

**Date/Time**: 2025-11-20 14:30 UTC
**Severity**: P1
**Duration**: 45 minutes
**Impact**: Company graph DAG failed

### Timeline
- 14:30: Alert received (DAG failure)
- 14:32: Investigation started
- 14:35: Root cause identified (Claude API timeout)
- 14:40: Fix applied (increased timeout)
- 14:50: DAG re-run successful
- 15:15: Monitoring confirmed stable

### Root Cause
Claude API calls timing out after 30 seconds due to increased API latency.

### Resolution
Increased task execution_timeout from 30s to 60s.

### Prevention
- Add timeout monitoring
- Set up Claude API latency alerts  
- Implement retry with exponential backoff

### Action Items
- [ ] Update all DAGs with 60s timeout
- [ ] Add API latency monitoring
- [ ] Document in troubleshooting guide
```

---

These runbooks provide complete operational procedures for maintaining enterprise-grade Airflow pipelines with professional incident management and monitoring practices.