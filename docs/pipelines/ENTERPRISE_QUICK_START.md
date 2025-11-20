# Enterprise Pipeline Quick Start

## 🚀 5-Minute Setup

### 1. Install Dependencies

```bash
cd /home/sandeep/pertinent/axiom
pip install -r axiom/pipelines/requirements-enterprise.txt
```

### 2. Verify Current Pipelines

The pipelines are already running! Check their health:

```bash
# Company Graph Builder
curl http://localhost:8080/health | jq

# Events Tracker  
curl http://localhost:8081/health | jq

# Correlation Analyzer
curl http://localhost:8082/health | jq
```

### 3. View Live Metrics

```bash
# Prometheus metrics
curl http://localhost:8080/metrics

# Or watch continuously
watch -n 5 'curl -s http://localhost:8080/health | jq ".metrics.cycles"'
```

## 📊 Key Endpoints

Each pipeline exposes:

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `/health` | Detailed health & metrics | `curl localhost:8080/health` |
| `/metrics` | Prometheus format | `curl localhost:8080/metrics` |
| `/ready` | Readiness probe | `curl localhost:8080/ready` |
| `/live` | Liveness probe | `curl localhost:8080/live` |

## 🎯 What You Get

### ✅ Automatic Features

**No configuration needed!** All pipelines now have:

1. **Metrics Tracking**
   - Cycle counts & success rates
   - Processing times
   - Claude API usage & costs
   - Error tracking

2. **Circuit Breakers**
   - Claude API protection
   - Neo4j connection protection
   - Automatic recovery

3. **Retry Logic**
   - 3 attempts with exponential backoff
   - Jitter to prevent thundering herd
   - Per-operation configuration

4. **Rate Limiting**
   - 10 requests/second to Claude
   - Prevents throttling
   - Cost control

5. **Health Checks**
   - HTTP server on each pipeline
   - Kubernetes-ready probes
   - Detailed diagnostics

6. **Structured Logging**
   - JSON format
   - Easy parsing
   - Full context

## 📈 Monitoring Dashboard

### Quick Health Check

```bash
#!/bin/bash
# save as check_pipelines.sh

echo "=== Pipeline Health Status ==="
echo ""

for port in 8080 8081 8082; do
    pipeline=$(curl -s http://localhost:$port/health | jq -r '.pipeline // "unknown"')
    status=$(curl -s http://localhost:$port/health | jq -r '.status // "unknown"')
    echo "Port $port ($pipeline): $status"
done
```

### Cost Monitoring

```bash
#!/bin/bash
# save as check_costs.sh

echo "=== Claude API Costs ==="
echo ""

total_cost=0
for port in 8080 8081 8082; do
    cost=$(curl -s http://localhost:$port/health | jq -r '.metrics.claude.total_cost // 0')
    pipeline=$(curl -s http://localhost:$port/health | jq -r '.pipeline // "unknown"')
    echo "$pipeline: \$$cost"
    total_cost=$(echo "$total_cost + $cost" | bc)
done

echo ""
echo "Total: \$$total_cost"
```

### Performance Check

```bash
#!/bin/bash
# save as check_performance.sh

echo "=== Pipeline Performance ==="
echo ""

for port in 8080 8081 8082; do
    pipeline=$(curl -s http://localhost:$port/health | jq -r '.pipeline // "unknown"')
    cycles=$(curl -s http://localhost:$port/health | jq -r '.metrics.cycles.total // 0')
    success_rate=$(curl -s http://localhost:$port/health | jq -r '.metrics.cycles.success_rate // 0')
    avg_time=$(curl -s http://localhost:$port/health | jq -r '.metrics.performance.avg_cycle_time // 0')
    
    echo "$pipeline:"
    echo "  Cycles: $cycles"
    echo "  Success Rate: $success_rate%"
    echo "  Avg Time: ${avg_time}s"
    echo ""
done
```

## 🔍 Troubleshooting

### Pipeline Not Responding

```bash
# Check if container is running
docker ps | grep axiom-pipeline

# Check logs
docker logs axiom-pipeline-companies --tail 50

# Restart if needed
docker compose -f axiom/pipelines/docker-compose-langgraph.yml restart company-graph
```

### High Error Rate

```bash
# View recent errors
curl -s http://localhost:8080/health | jq '.errors.recent'

# Check circuit breaker status
curl -s http://localhost:8080/health | jq '.circuit_breakers'
```

### Claude API Issues

```bash
# Check if circuit breaker is open
curl -s http://localhost:8080/health | jq '.circuit_breakers.claude'

# If open, wait 60 seconds for automatic recovery
# Or restart the pipeline to force reset
```

## 📊 Grafana Setup (Optional)

### 1. Start Grafana

```bash
docker run -d \
  -p 3000:3000 \
  --name grafana \
  grafana/grafana
```

### 2. Add Prometheus Data Source

- URL: http://localhost:9090
- Access: Browser

### 3. Import Dashboard

Use these queries:

**Success Rate**:
```
rate(pipeline_cycles_successful[5m]) / rate(pipeline_cycles_total[5m]) * 100
```

**Processing Time**:
```
pipeline_avg_cycle_time_seconds
```

**Claude Costs**:
```
pipeline_claude_cost
```

**Error Rate**:
```
rate(pipeline_items_failed[5m]) / rate(pipeline_items_processed[5m]) * 100
```

## 🎓 Learning Path

### Day 1: Health Checks
- Run health check scripts
- Understand metrics structure
- Watch live updates

### Day 2: Monitoring
- Set up Prometheus scraping
- Create basic Grafana dashboard
- Configure alerts

### Day 3: Optimization
- Analyze performance metrics
- Tune rate limits
- Optimize circuit breaker thresholds

### Day 4: Production
- Add log aggregation
- Configure alerting
- Document runbooks

## 💡 Pro Tips

1. **Monitor Costs Daily**: `watch check_costs.sh`
2. **Check Health Before Deploy**: Always verify `/ready` endpoint
3. **Use Structured Logs**: Pipe through `jq` for filtering
4. **Circuit Breaker Alerts**: Set up notifications when breakers open
5. **Baseline Metrics**: Record normal operation metrics for comparison

## 🆘 Getting Help

### Check Current Status

```bash
curl http://localhost:8080/health | jq '{
  status,
  cycles: .metrics.cycles,
  performance: .metrics.performance,
  errors: .errors.counts
}'
```

### Full Diagnostic

```bash
#!/bin/bash
# save as diagnostic.sh

echo "=== Full Pipeline Diagnostic ==="
date
echo ""

for port in 8080 8081 8082; do
    echo "--- Port $port ---"
    curl -s http://localhost:$port/health | jq '
{
  pipeline,
  status,
  cycles: .metrics.cycles.success_rate,
  claude_errors: .metrics.claude.errors,
  neo4j_errors: .metrics.neo4j.errors,
  recent_errors: .errors.recent | length
}'
    echo ""
done
```

## 📚 Next Steps

1. Review [Enterprise Features Guide](./ENTERPRISE_FEATURES_GUIDE.md)
2. Set up monitoring dashboard
3. Configure alerts
4. Optimize for your workload

## 🎉 Success Indicators

You'll know it's working when:

- ✅ All health endpoints return `{"status": "healthy"}`
- ✅ Success rates > 95%
- ✅ No circuit breakers stuck open
- ✅ Costs within budget
- ✅ Processing times stable

**Current Status**: All pipelines are running and healthy! 🚀