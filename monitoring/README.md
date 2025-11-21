# Production Monitoring System for Axiom AI/ML Platform

**Complete observability stack for monitoring Claude API costs, Airflow DAG performance, and data quality metrics across 24+ production containers.**

## 🎯 Overview

This monitoring system provides enterprise-grade observability for your AI/ML platform with:

- **Real-time Metrics**: Prometheus collects metrics every 15-30 seconds
- **Visual Dashboards**: Grafana provides 3 specialized dashboards
- **Automated Alerting**: 25+ alert rules for costs, failures, and quality degradation
- **Cost Tracking**: Track Claude API spending per DAG/task with projections
- **Data Quality**: Monitor validation pass rates, freshness, and anomalies
- **DAG Monitoring**: Track success rates, durations, and SLA violations

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Grafana Dashboards                        │
│  (AI Platform Overview | Claude Costs | Data Quality)        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     Prometheus                               │
│         (Metrics Collection & Time Series DB)                │
└─┬──────┬──────┬──────┬──────┬──────┬────────┬──────────────┘
  │      │      │      │      │      │        │
  │      │      │      │      │      │        │
  ▼      ▼      ▼      ▼      ▼      ▼        ▼
┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌─────┐
│AI │  │DAG│  │DQ │  │PG │  │Neo│  │Redis  │cAdvisor
│   │  │   │  │   │  │SQL│  │4j │  │      │
└───┘  └───┘  └───┘  └───┘  └───┘  └───┘  └─────┘
9091   9092   9093   9187   2004   9121    8080

Metrics Exporters:
- AI Metrics: LangGraph/DSPy agent execution
- DAG Metrics: Airflow task/DAG performance
- DQ Metrics: Data quality validation results
- DB Metrics: PostgreSQL, Neo4j, Redis stats
- System: cAdvisor (containers), node-exporter (system)
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Existing Axiom platform containers running
- Networks: `axiom-mcp-network` and `database_axiom_network` exist

### 1. Start Monitoring Stack

```bash
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d
```

### 2. Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3000 | admin / admin123 |
| **Prometheus** | http://localhost:9090 | - |
| **Alertmanager** | http://localhost:9093 | - |

### 3. View Metrics

**Grafana Dashboards:**
- AI/ML Platform Overview - Overall system health and performance
- Claude API Costs - Detailed cost tracking and projections
- Data Quality - Validation metrics and freshness

**Direct Metrics Endpoints:**
- AI/ML: http://localhost:9091/metrics
- Airflow DAGs: http://localhost:9092/metrics
- Data Quality: http://localhost:9093/metrics

## 📈 Key Metrics

### Claude API Cost Tracking

```promql
# Daily cost
claude_api_cost_daily_usd

# Monthly cost
claude_api_cost_monthly_usd

# Cost per DAG
sum(rate(claude_api_cost_per_dag_usd[1h])) by (dag_id)

# Projected monthly cost
(claude_api_cost_daily_usd / day_of_month()) * days_in_month()
```

### DAG Performance

```promql
# DAG success rate
rate(airflow_dag_runs_total{state="success"}[5m]) / 
rate(airflow_dag_runs_total[5m])

# Task failures
rate(airflow_task_runs_total{state="failed"}[5m])

# DAG duration (P95)
histogram_quantile(0.95, rate(airflow_dag_run_duration_seconds_bucket[10m]))
```

### Data Quality

```promql
# Quality score
data_quality_score

# Data freshness (minutes)
data_freshness_minutes

# Validation pass rate
data_validation_pass_rate

# Anomalies detected
rate(data_anomalies_detected_total[10m])
```

## 🚨 Automated Alerts

### Alert Categories

1. **Claude API Costs** (6 alerts)
   - High daily cost (>$100)
   - Critical daily cost (>$500)
   - High monthly cost (>$2000)
   - Rapid cost increase
   - Expensive DAG detection

2. **DAG Failures** (7 alerts)
   - High failure rate
   - Consecutive failures
   - DAG not running
   - Duration anomalies
   - Task failure spikes
   - SLA violations
   - Queue backlog

3. **Data Quality** (8 alerts)
   - Low quality score
   - Critical quality score
   - High validation failure rate
   - Stale data
   - Very stale data
   - Anomaly spikes
   - Record count drops
   - Schema validation failures

4. **System Health** (4 alerts)
   - System health degraded
   - Scheduler down
   - High Prometheus cardinality
   - Monitoring data loss

### Alert Routing

- **Critical alerts** → Email to oncall + team
- **Cost alerts** → Email to finance + team
- **Data quality** → Email to data team
- **Pipeline failures** → Email to devops + team
- **SLA violations** → Email to SRE + management

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the `monitoring` directory:

```bash
# Database connections
POSTGRES_HOST=localhost
POSTGRES_USER=axiom
POSTGRES_PASSWORD=your_password
POSTGRES_DB=axiom_finance

# Redis
REDIS_HOST=localhost
REDIS_PASSWORD=your_redis_password

# Grafana
GRAFANA_PASSWORD=admin123

# Alert emails (edit alertmanager.yml)
```

### Customize Alerting

Edit `monitoring/alertmanager/alertmanager.yml`:

```yaml
receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'your-oncall@company.com'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK'
        channel: '#alerts'
```

### Adjust Alert Thresholds

Edit `monitoring/prometheus/alerts/ai_platform_alerts.yml`:

```yaml
# Example: Change Claude cost threshold
- alert: ClaudeAPIHighDailyCost
  expr: claude_api_cost_daily_usd > 100  # Change this value
  for: 5m
```

## 📦 Integration with Existing Containers

The monitoring stack automatically discovers and monitors:

1. **AI/ML Services** (LangGraph, DSPy)
   - Port: 9091
   - Network: `axiom-mcp-network`

2. **Airflow Services** (Webserver, Scheduler)
   - Metrics exporter: 9092
   - Network: `database_axiom_network`

3. **Databases** (PostgreSQL, Neo4j, Redis)
   - Exporters: 9187, 2004, 9121
   - Network: `database_axiom_network`

4. **All Docker Containers**
   - via cAdvisor on port 8080

### Adding New Services

To monitor additional services, add to `prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'my-service'
    static_configs:
      - targets: ['my-service:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

## 🔧 Operations

### View Logs

```bash
# All services
docker-compose -f docker-compose-monitoring.yml logs -f

# Specific service
docker logs -f axiom-prometheus
docker logs -f axiom-grafana
docker logs -f axiom-airflow-metrics-exporter
```

### Restart Services

```bash
# Restart all
docker-compose -f docker-compose-monitoring.yml restart

# Restart specific service
docker restart axiom-prometheus
```

### Update Configuration

```bash
# Reload Prometheus config (no restart needed)
curl -X POST http://localhost:9090/-/reload

# Restart to apply changes
docker-compose -f docker-compose-monitoring.yml restart prometheus
```

### Backup Metrics Data

```bash
# Backup Prometheus data
docker run --rm -v monitoring_prometheus-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/prometheus-backup-$(date +%Y%m%d).tar.gz /data

# Backup Grafana dashboards
docker run --rm -v monitoring_grafana-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/grafana-backup-$(date +%Y%m%d).tar.gz /data
```

## 🐛 Troubleshooting

### No metrics appearing

1. Check exporters are running:
   ```bash
   curl http://localhost:9091/metrics  # AI metrics
   curl http://localhost:9092/metrics  # Airflow metrics
   curl http://localhost:9093/metrics  # Data quality
   ```

2. Check Prometheus targets:
   - Go to http://localhost:9090/targets
   - All should show "UP" status

3. Check networks:
   ```bash
   docker network inspect axiom-mcp-network
   docker network inspect database_axiom_network
   ```

### Grafana not showing data

1. Check Prometheus datasource:
   - Go to Configuration → Data Sources
   - Test the Prometheus connection

2. Verify metrics exist in Prometheus:
   - Go to http://localhost:9090/graph
   - Try query: `up`

### Alerts not firing

1. Check Alertmanager:
   - Go to http://localhost:9093
   - View configured alerts

2. Check email configuration:
   - Ensure MailHog is running: http://localhost:8025

3. Test alert rules:
   ```bash
   promtool check rules prometheus/alerts/*.yml
   ```

## 📚 Best Practices

1. **Regular Backups**: Backup metrics weekly
2. **Alert Tuning**: Adjust thresholds after 1-2 weeks
3. **Dashboard Customization**: Create team-specific views
4. **Cost Monitoring**: Review Claude costs daily
5. **SLA Tracking**: Define and monitor critical DAG SLAs
6. **Capacity Planning**: Monitor trends for scaling decisions

## 🔗 Related Documentation

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alert Manager Guide](https://prometheus.io/docs/alerting/latest/alertmanager/)

## 📞 Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review Prometheus targets: http://localhost:9090/targets
- Inspect Grafana datasources: http://localhost:3000/datasources

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-21  
**Maintained by**: Axiom Platform Team