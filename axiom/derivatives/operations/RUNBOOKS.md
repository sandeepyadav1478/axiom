# Derivatives Platform - Production Runbooks

## Operational Procedures for 24/7 Production Support

**Audience:** On-call engineers, SREs, DevOps  
**Purpose:** Handle incidents, maintain SLAs, ensure 99.999% uptime

---

## 🚨 CRITICAL ALERTS - IMMEDIATE RESPONSE (<15 minutes)

### Alert: Greeks Latency > 100 Microseconds

**Impact:** Client SLA breach, potential revenue loss  
**Target Response:** <15 minutes

**Diagnosis:**
```bash
# 1. Check current latency
curl http://derivatives-api/stats/engines | jq '.greeks_engine.average_time_microseconds'

# 2. Check GPU status
kubectl exec -it deployment/derivatives-api -- nvidia-smi

# 3. Check pod resources
kubectl top pod -n derivatives

# 4. Check Prometheus
# Query: derivatives_greeks_latency_p95
```

**Common Causes & Fixes:**

**Cause 1: GPU Not Available**
```bash
# Symptoms: Latency 10x higher than normal
# Fix: Restart pod on GPU node
kubectl delete pod <pod-name> -n derivatives
kubectl get pods -n derivatives -o wide  # Verify on GPU node
```

**Cause 2: High Load**
```bash
# Symptoms: Many requests queued
# Fix: Scale up immediately
kubectl scale deployment/derivatives-api --replicas=10 -n derivatives

# Monitor: Should recover in 1-2 minutes
```

**Cause 3: Model Not Cached**
```bash
# Symptoms: First request slow, then fast
# Fix: Model loading issue, restart pod
kubectl rollout restart deployment/derivatives-api -n derivatives
```

**Cause 4: Memory Leak**
```bash
# Symptoms: Gradual degradation over hours
# Fix: Rolling restart
kubectl rollout restart deployment/derivatives-api -n derivatives
# Investigate: Check code for leaks
```

---

### Alert: API Error Rate > 1%

**Impact:** Client unable to calculate Greeks, potential data loss

**Diagnosis:**
```bash
# Check error logs
kubectl logs deployment/derivatives-api -n derivatives --tail=100 | grep ERROR

# Check error breakdown
curl http://derivatives-api/metrics | grep derivatives_api_errors_total
```

**Common Errors:**

**Error: "CUDA out of memory"**
```bash
# Fix: Reduce batch sizes or add more GPU memory
# Temporary: Restart pods
kubectl delete pod <pod-name>

# Permanent: Increase GPU memory in deployment.yaml
# Or: Implement gradient accumulation
```

**Error: "Database connection failed"**
```bash
# Check PostgreSQL
kubectl get pods -n derivatives | grep postgres
kubectl logs <postgres-pod>

# Check connection pool
# If pool exhausted, increase max_connections in postgres config
```

---

### Alert: PostgreSQL Down

**Impact:** Cannot store trades, critical data loss risk

**Fix:**
```bash
# 1. Check pod status
kubectl get pods -n derivatives | grep postgres

# 2. If pod crashed, check logs
kubectl logs <postgres-pod> --previous

# 3. Check disk space
kubectl exec -it <postgres-pod> -- df -h

# 4. If disk full, emergency cleanup
kubectl exec -it <postgres-pod> -- psql -U axiom_prod -d axiom_derivatives
# Run: DELETE FROM greeks_history WHERE timestamp < NOW() - INTERVAL '30 days';

# 5. Restart if needed
kubectl delete pod <postgres-pod>
```

---

### Alert: Large Loss Detected (P&L < -$100K)

**Impact:** Potential model error or adverse market move

**Procedure:**
```bash
# 1. IMMEDIATELY verify it's real
curl http://derivatives-api/stats/engines

# 2. Check all position Greeks
# Query PostgreSQL for current positions

# 3. If Greeks look wrong:
#    - Stop auto-trading immediately
#    - Manually verify Greeks with Black-Scholes
#    - Contact client

# 4. If market move:
#    - Verify with market data
#    - Check if hedges executed
#    - Review auto-hedger logs

# 5. Document incident
#    - What happened
#    - When detected
#    - Actions taken
#    - Resolution
```

---

## ⚠️ HIGH PRIORITY ALERTS (<1 hour)

### Alert: Latency Degrading

**Gradual increase in latency over time**

**Investigation:**
```bash
# 1. Check memory usage trend
kubectl top pods -n derivatives

# 2. Profile running application
kubectl exec -it <pod> -- py-spy top --pid 1

# 3. Check for memory leak
kubectl exec -it <pod> -- python -m memory_profiler

# 4. Review recent deployments
kubectl rollout history deployment/derivatives-api
```

**Fix:** Usually rolling restart resolves

---

### Alert: Cache Miss Rate High

**Cache hit rate < 50%**

**Investigation:**
```bash
# Check Redis
kubectl exec -it <redis-pod> -- redis-cli INFO stats

# Check memory
kubectl exec -it <redis-pod> -- redis-cli INFO memory

# If memory full, increase maxmemory
```

---

## 📊 ROUTINE MAINTENANCE

### Daily Tasks

**Morning (9 AM):**
```bash
# 1. Check overnight performance
# Grafana: Review overnight dashboards

# 2. Check error logs
kubectl logs deployment/derivatives-api --since=24h | grep ERROR | wc -l

# 3. Verify backups
# Check latest PostgreSQL backup timestamp

# 4. Review client usage
# Query: SELECT COUNT(*) FROM option_trades WHERE timestamp > NOW() - INTERVAL '24 hours';
```

**Evening (6 PM):**
```bash
# 1. Daily P&L reconciliation
# Compare system P&L with client reports

# 2. Review performance metrics
# Check if any degradation during day

# 3. Plan next day capacity
# Based on expected volume
```

---

### Weekly Tasks

**Monday Morning:**
```bash
# 1. Review week's performance
# Generate weekly report from Grafana

# 2. Check model accuracy
# Run accuracy validation tests

# 3. Review alerts
# Any patterns? Recurring issues?

# 4. Capacity planning
# Based on growth trends
```

**Friday Afternoon:**
```bash
# 1. Model retraining (RL/DRL)
# Deploy updated models if performance improved

# 2. Database maintenance
# VACUUM, ANALYZE on large tables

# 3. Security patches
# Update dependencies if needed

# 4. Week-over-week comparison
# Performance, usage, errors
```

---

### Monthly Tasks

**First Monday:**
```bash
# 1. Full system audit
# Review all configurations

# 2. Disaster recovery test
# Actually test backup restore

# 3. Cost optimization review
# Cloud costs vs usage

# 4. Client feedback review
# Any feature requests? Issues?
```

---

## 🔧 COMMON PROCEDURES

### Deploy New Model Version

```bash
# 1. Build new image
docker build -t axiom/derivatives:v1.1.0 .
docker push axiom/derivatives:v1.1.0

# 2. Update deployment
kubectl set image deployment/derivatives-api api=axiom/derivatives:v1.1.0 -n derivatives

# 3. Monitor rollout
kubectl rollout status deployment/derivatives-api -n derivatives

# 4. Validate performance
# Run benchmark, compare to baseline

# 5. If worse, rollback
kubectl rollout undo deployment/derivatives-api -n derivatives
```

---

### Scale for High Volume

```bash
# Gradual scaling (preferred)
kubectl scale deployment/derivatives-api --replicas=10 -n derivatives

# Emergency scaling (immediate)
kubectl scale deployment/derivatives-api --replicas=20 -n derivatives

# Monitor: HPA should maintain after manual scale
kubectl get hpa -n derivatives
```

---

### Database Performance Issues

```bash
# 1. Check slow queries
kubectl exec -it <postgres-pod> -- psql -U axiom_prod -d axiom_derivatives
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

# 2. Check table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# 3. Vacuum if needed
VACUUM ANALYZE option_trades;
VACUUM ANALYZE greeks_history;

# 4. Reindex if necessary
REINDEX TABLE option_trades;
```

---

## 📞 ESCALATION

### Level 1: On-Call Engineer (You)
- Respond: <15 minutes (critical), <1 hour (high)
- Authority: Restart pods, scale, basic fixes
- Escalate if: Can't resolve in 1 hour

### Level 2: Senior Engineer
- Respond: <30 minutes
- Authority: Code changes, configuration updates
- Escalate if: Requires architecture change

### Level 3: Engineering Lead
- Respond: <1 hour
- Authority: All technical decisions
- Escalate if: Business decision needed

### Level 4: CTO
- Respond: <2 hours
- Authority: All decisions
- Client communication for critical issues

---

## 📋 INCIDENT REPORT TEMPLATE

```markdown
# Incident Report

## Summary
- Date: YYYY-MM-DD HH:MM UTC
- Duration: X hours Y minutes
- Severity: Critical/High/Medium
- Impact: X clients affected, $Y revenue at risk

## Timeline
- HH:MM - Alert triggered
- HH:MM - Engineer paged
- HH:MM - Initial diagnosis
- HH:MM - Fix attempted
- HH:MM - Issue resolved
- HH:MM - All services normal

## Root Cause
[Detailed explanation]

## Fix Applied
[What was done]

## Preventive Measures
[How to prevent recurrence]

## Lessons Learned
[What we learned]
```

---

**These runbooks ensure 99.999% uptime and <15 minute response to critical issues.**