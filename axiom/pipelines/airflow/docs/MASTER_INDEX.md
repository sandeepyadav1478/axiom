# Apache Airflow - Complete Documentation Index

## 📚 Documentation Structure

This directory contains **complete enterprise-grade documentation** for the Axiom Airflow implementation. All visual aids, runbooks, and procedures are centralized here for easy access.

---

## 🎯 Quick Navigation

### For New Users
1. **[README](../README.md)** - Start here for overview
2. **[AIRFLOW_SETUP](../AIRFLOW_SETUP.md)** - 5-minute quick start
3. **[ARCHITECTURE_DIAGRAMS](./ARCHITECTURE_DIAGRAMS.md)** - Visual system architecture

### For Operators
4. **[OPERATIONAL_RUNBOOKS](./OPERATIONAL_RUNBOOKS.md)** - Emergency procedures & maintenance
5. **[DEPLOYMENT_GUIDE](../DEPLOYMENT_GUIDE.md)** - Production deployment checklist
6. **[TROUBLESHOOTING_GUIDE](./TROUBLESHOOTING_GUIDE.md)** - Common issues & solutions

### For Developers
7. **[DAG_DEVELOPMENT_GUIDE](./DAG_DEVELOPMENT_GUIDE.md)** - Creating new DAGs
8. **[BEST_PRACTICES](./BEST_PRACTICES.md)** - Enterprise patterns
9. **[API_REFERENCE](./API_REFERENCE.md)** - Complete API docs

---

## 📁 File Organization

```
axiom/pipelines/airflow/
├── README.md                          # Overview & quick start
├── AIRFLOW_SETUP.md                   # Getting started guide
├── DEPLOYMENT_GUIDE.md                # Production deployment
├── docker-compose-airflow.yml         # Infrastructure definition
│
├── dags/                              # 4 Production DAGs
│   ├── data_ingestion_dag.py         # Every minute
│   ├── company_graph_dag.py          # Hourly
│   ├── events_tracker_dag.py         # Every 5 minutes
│   └── correlation_analyzer_dag.py   # Hourly
│
├── docs/                              # Comprehensive documentation
│   ├── MASTER_INDEX.md               # This file
│   ├── ARCHITECTURE_DIAGRAMS.md      # System architecture visuals
│   ├── OPERATIONAL_RUNBOOKS.md       # Emergency procedures
│   ├── TROUBLESHOOTING_GUIDE.md      # Problem resolution
│   ├── DAG_DEVELOPMENT_GUIDE.md      # Creating DAGs
│   ├── BEST_PRACTICES.md             # Enterprise patterns
│   ├── API_REFERENCE.md              # Complete API docs
│   └── MONITORING_GUIDE.md           # Observability setup
│
├── scripts/                           # Automation scripts
│   ├── setup_airflow.sh              # Automated deployment
│   ├── monitor_airflow.sh            # Real-time monitoring
│   ├── backup_airflow.sh             # Database backups
│   └── health_check.sh               # Health verification
│
├── logs/                              # Execution logs (auto-created)
└── plugins/                           # Custom Airflow plugins (if needed)
```

---

## 🎨 Visual Documentation Map

### System Architecture
- [Overall Architecture Diagram](./ARCHITECTURE_DIAGRAMS.md#overall-system-architecture)
- [DAG Execution Flow](./ARCHITECTURE_DIAGRAMS.md#dag-execution-flow---company-graph-builder)
- [Data Flow Diagram](./ARCHITECTURE_DIAGRAMS.md#data-flow-through-system)
- [Network Architecture](./ARCHITECTURE_DIAGRAMS.md#network-architecture)

### Operational Flows
- [Error Handling Flow](./ARCHITECTURE_DIAGRAMS.md#retry--error-handling-flow)
- [Task State Machine](./ARCHITECTURE_DIAGRAMS.md#task-state-machine)
- [Parallel Execution](./ARCHITECTURE_DIAGRAMS.md#dag-execution-flow---data-ingestion-parallel)

### Monitoring Dashboards
- [Performance Metrics](./MONITORING_GUIDE.md#performance-metrics)
- [SLA Tracking](./ARCHITECTURE_DIAGRAMS.md#sla--quality-metrics)
- [Cost Monitoring](./OPERATIONAL_RUNBOOKS.md#monitor-002-track-claude-api-costs)

---

## 🚨 Emergency Quick Reference

### Critical Issues (P0)

| Issue | Runbook | Quick Fix |
|-------|---------|-----------|
| All DAGs failing | [RUNBOOK-001](./OPERATIONAL_RUNBOOKS.md#runbook-001-all-dags-failing) | Restart services |
| Disk space full | [RUNBOOK-005](./OPERATIONAL_RUNBOOKS.md#runbook-005-disk-space-full) | Clean logs |
| Database down | Contact DBA | Check database container |

### Common Issues (P1-P2)

| Issue | Runbook | Quick Fix |
|-------|---------|-----------|
| DAG stuck | [RUNBOOK-002](./OPERATIONAL_RUNBOOKS.md#runbook-002-single-dag-stuck) | Kill & retry task |
| Claude rate limit | [RUNBOOK-003](./OPERATIONAL_RUNBOOKS.md#runbook-003-claude-api-rate-limiting) | Reduce frequency |
| Neo4j connection | [RUNBOOK-004](./OPERATIONAL_RUNBOOKS.md#runbook-004-neo4j-connection-lost) | Restart Neo4j |

---

## 📊 The 4 Production DAGs

### 1. Data Ingestion (`data_ingestion`)
**Schedule**: Every 1 minute
**Purpose**: Fetch real-time stock prices
**Tasks**: fetch_data → [store_postgresql, cache_redis, update_neo4j] (parallel)
**Documentation**: [data_ingestion_dag.py](../dags/data_ingestion_dag.py)
**Monitoring**: Real-time price updates in PostgreSQL

### 2. Company Graph Builder (`company_graph_builder`)
**Schedule**: Hourly
**Purpose**: Build company relationship knowledge graph with Claude AI
**Tasks**: initialize → fetch_companies → identify_relationships → generate_cypher → execute_neo4j → validate_graph
**Documentation**: [company_graph_dag.py](../dags/company_graph_dag.py)
**Monitoring**: Neo4j relationship count growth

### 3. Events Tracker (`events_tracker`)
**Schedule**: Every 5 minutes
**Purpose**: Monitor and classify market events
**Tasks**: fetch_news → classify_events → create_event_nodes
**Documentation**: [events_tracker_dag.py](../dags/events_tracker_dag.py)
**Monitoring**: MarketEvent node count

### 4. Correlation Analyzer (`correlation_analyzer`)
**Schedule**: Hourly
**Purpose**: Analyze stock correlations with Claude explanations
**Tasks**: fetch_prices → calculate_correlations → explain_correlations → create_relationships
**Documentation**: [correlation_analyzer_dag.py](../dags/correlation_analyzer_dag.py)
**Monitoring**: CORRELATED_WITH relationship count

---

## 🛠️ Command Reference

### Daily Operations

```bash
# Check Airflow health
curl http://localhost:8080/health | jq

# List all DAGs
docker exec axiom-airflow-webserver airflow dags list

# View recent runs
docker exec axiom-airflow-webserver airflow dags list-runs | head -20

# Trigger manual run
docker exec axiom-airflow-webserver \
  airflow dags trigger company_graph_builder

# View logs
docker logs -f axiom-airflow-scheduler

# Monitor live (script)
./axiom/pipelines/airflow/scripts/monitor_airflow.sh
```

### Maintenance

```bash
# Clean old logs (>7 days)
find axiom/pipelines/airflow/logs -name "*.log" -mtime +7 -delete

# Backup database
./axiom/pipelines/airflow/scripts/backup_airflow.sh

# Health check
./axiom/pipelines/airflow/scripts/health_check.sh

# Restart services
docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml restart
```

### Troubleshooting

```bash
# Check import errors
docker exec axiom-airflow-webserver airflow dags list-import-errors

# Test DAG syntax
docker exec axiom-airflow-webserver \
  python /opt/airflow/dags/company_graph_dag.py

# Clear failed tasks
docker exec axiom-airflow-webserver \
  airflow tasks clear company_graph_builder --start-date 2025-11-20

# View task logs
docker exec axiom-airflow-webserver \
  airflow tasks logs company_graph_builder fetch_companies 2025-11-20
```

---

## 📈 Monitoring & Metrics

### Key Performance Indicators (KPIs)

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| DAG Success Rate | >95% | <90% |
| Average DAG Duration | <10 min | >20 min |
| Task Retry Rate | <5% | >10% |
| Queue Length | <10 tasks | >50 tasks |
| Disk Usage | <70% | >80% |
| Claude API Cost/Day | <$5 | >$10 |

### Monitoring Tools

- **Airflow UI**: http://localhost:8080 (real-time monitoring)
- **Health Endpoint**: `curl http://localhost:8080/health`
- **Live Monitor**: `./axiom/pipelines/airflow/scripts/monitor_airflow.sh`
- **Cost Tracker**: `./axiom/pipelines/airflow/scripts/track_claude_costs.sh`

---

## 🔐 Security & Access

### Default Credentials
- **UI Access**: http://localhost:8080
- **Username**: `admin`
- **Password**: `admin123` (CHANGE IN PRODUCTION!)

### Database Access
- **Host**: localhost:5432
- **Database**: airflow
- **User**: airflow
- **Password**: airflow_pass (from .env)

### API Access
- **Airflow REST API**: http://localhost:8080/api/v1/
- **Auth**: Basic Auth (username/password)
- **Docs**: http://localhost:8080/api/v1/ui/

---

## 📖 Learning Path

### Day 1: Getting Started
1. Read [README](../README.md)
2. Follow [AIRFLOW_SETUP](../AIRFLOW_SETUP.md)
3. Access UI and explore

### Day 2: Understanding DAGs
4. Review [ARCHITECTURE_DIAGRAMS](./ARCHITECTURE_DIAGRAMS.md)
5. Study the 4 production DAGs
6. Trigger manual runs and watch execution

### Day 3: Operations
7. Learn [OPERATIONAL_RUNBOOKS](./OPERATIONAL_RUNBOOKS.md)
8. Practice using scripts
9. Simulate failure scenarios

### Day 4: Customization
10. Read [DAG_DEVELOPMENT_GUIDE](./DAG_DEVELOPMENT_GUIDE.md)
11. Create a test DAG
12. Learn [BEST_PRACTICES](./BEST_PRACTICES.md)

### Week 2: Advanced Topics
13. Set up monitoring dashboards
14. Configure email alerts
15. Implement custom operators

---

## 🌟 Enterprise Features Implemented

### ✅ Core Capabilities
- [x] Professional DAG-based orchestration
- [x] Visual monitoring UI
- [x] Automatic retry logic (3 attempts)
- [x] Task dependency management
- [x] SLA monitoring & alerts
- [x] Parallel task execution
- [x] Backfill support

### ✅ Operational Excellence
- [x] Health check endpoints
- [x] Comprehensive runbooks
- [x] Automated setup scripts
- [x] Real-time monitoring tools
- [x] Backup & recovery procedures
- [x] Incident response templates

### ✅ Developer Experience
- [x] 4 production-ready DAGs
- [x] Complete documentation
- [x] Visual architecture diagrams
- [x] Code examples
- [x] Best practices guide

### 🔜 Future Enhancements
- [ ] Prometheus metrics integration
- [ ] Grafana dashboards
- [ ] Kafka streaming integration
- [ ] Ray parallel processing
- [ ] Custom monitoring UI

---

## 🆘 Getting Help

### Documentation Issues
- Check [TROUBLESHOOTING_GUIDE](./TROUBLESHOOTING_GUIDE.md)
- Review relevant runbook
- Check Airflow official docs

### Technical Issues
- View task logs in UI
- Check container logs: `docker logs axiom-airflow-scheduler`
- Run health check: `./axiom/pipelines/airflow/scripts/health_check.sh`

### Emergency Contact
- **P0 Issues**: Immediate escalation required
- **P1 Issues**: Contact within 1 hour
- **P2 Issues**: Can wait 4 hours

---

## 📊 Success Metrics

### System Health
- ✅ All services running
- ✅ Scheduler healthy
- ✅ Database accessible
- ✅ All 4 DAGs loaded

### Pipeline Performance
- ✅ Success rate >95%
- ✅ Average duration <10 min
- ✅ No stuck tasks
- ✅ Queue length near zero

### Data Quality
- ✅ PostgreSQL receiving prices
- ✅ Neo4j graph growing
- ✅ Redis cache updated
- ✅ No data gaps

---

## 🔄 Continuous Improvement

### Weekly Reviews
- Analyze DAG performance metrics
- Review error patterns
- Identify optimization opportunities
- Update documentation

### Monthly Audits
- Security review
- Cost analysis
- Capacity planning
- Documentation updates

---

## 📝 Document Versioning

| Document | Version | Last Updated | Author |
|----------|---------|--------------|--------|
| MASTER_INDEX.md | 1.0 | 2025-11-20 | Axiom Team |
| ARCHITECTURE_DIAGRAMS.md | 1.0 | 2025-11-20 | Axiom Team |
| OPERATIONAL_RUNBOOKS.md | 1.0 | 2025-11-20 | Axiom Team |
| README.md | 1.0 | 2025-11-20 | Axiom Team |

---

## 🎓 External Resources

### Apache Airflow
- **Official Docs**: https://airflow.apache.org/docs/
- **Best Practices**: https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html
- **Tutorial**: https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html

### Community Support
- **Slack**: Apache Airflow Slack workspace
- **Stack Overflow**: [apache-airflow] tag
- **GitHub Issues**: https://github.com/apache/airflow/issues

---

This master index provides complete navigation to all Airflow documentation, ensuring easy access to guides, runbooks, and procedures for operators, developers, and administrators.