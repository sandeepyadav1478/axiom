# Production Deployment Guide - 12-Agent System

## Complete deployment strategy for enterprise production environment

This guide covers deploying all 12 professional agents to production with:
- High availability (99.999% uptime)
- Scalability (handle 1M+ requests/day)
- Security (enterprise-grade)
- Monitoring (complete observability)
- Disaster recovery (automated failover)

---

## 🏗️ Architecture Overview

### Deployment Topology

```
Load Balancer (HAProxy/Nginx)
    ↓
API Gateway (FastAPI)
    ↓
Message Bus (Redis/RabbitMQ)
    ↓
┌─────────────────┬──────────────────┬───────────────────┐
│ Trading Cluster │ Analytics Cluster│ Support Cluster   │
│  (5 agents)     │   (3 agents)     │   (4 agents)      │
├─────────────────┼──────────────────┼───────────────────┤
│ - Pricing       │ - Analytics      │ - Compliance      │
│ - Risk          │ - Market Data    │ - Monitoring      │
│ - Strategy      │ - Volatility     │ - Guardrail       │
│ - Execution     │                  │ - Client Interface│
│ - Hedging       │                  │                   │
└─────────────────┴──────────────────┴───────────────────┘
    ↓                   ↓                    ↓
PostgreSQL          ChromaDB            Redis Cache
Vector Store        Metrics DB          Session Store
```

---

## 🐳 Docker Deployment

### 1. Containerize Each Agent

**Example Dockerfile (Pricing Agent):**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY axiom/ai_layer/agents/professional/pricing_agent_v2.py ./agents/
COPY axiom/ai_layer/domain/ ./domain/
COPY axiom/ai_layer/infrastructure/ ./infrastructure/
COPY axiom/ai_layer/messaging/ ./messaging/

# Environment
ENV PYTHONPATH=/app
ENV AGENT_NAME=pricing
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run agent
CMD ["python", "-m", "agents.pricing_agent_v2"]
```

### 2. Docker Compose (All 12 Agents)

**File:** `docker/docker-compose.agents.yml`

```yaml
version: '3.8'

services:
  # Message Bus
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # PostgreSQL (for state persistence)
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: axiom_agents
      POSTGRES_USER: axiom
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U axiom"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Trading Cluster
  pricing-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.pricing
    environment:
      - AGENT_NAME=pricing
      - USE_GPU=false
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://axiom:${DB_PASSWORD}@postgres/axiom_agents
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 3  # High availability
      resources:
        limits:
          cpus: '2'
          memory: 2G

  risk-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.risk
    environment:
      - AGENT_NAME=risk
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 2

  strategy-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.strategy
    environment:
      - AGENT_NAME=strategy
      - USE_GPU=${USE_GPU:-false}
    deploy:
      replicas: 2

  execution-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.execution
    environment:
      - AGENT_NAME=execution
      - FIX_VENUES=${FIX_VENUES}
    deploy:
      replicas: 3  # High availability for execution

  hedging-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.hedging
    environment:
      - AGENT_NAME=hedging
      - USE_GPU=${USE_GPU:-false}
    deploy:
      replicas: 2

  # Analytics Cluster
  analytics-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.analytics
    deploy:
      replicas: 2

  market-data-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.market_data
    environment:
      - OPRA_API_KEY=${OPRA_API_KEY}
      - POLYGON_API_KEY=${POLYGON_API_KEY}
    deploy:
      replicas: 3  # High availability for market data

  volatility-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.volatility
    environment:
      - USE_GPU=${USE_GPU:-false}
    deploy:
      replicas: 2

  # Support Cluster
  compliance-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.compliance
    deploy:
      replicas: 2  # High availability for compliance

  monitoring-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.monitoring
    environment:
      - PROMETHEUS_URL=${PROMETHEUS_URL}
    deploy:
      replicas: 2

  guardrail-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.guardrail
    deploy:
      replicas: 3  # High availability for safety

  client-interface-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.client_interface
    ports:
      - "8000:8000"
    deploy:
      replicas: 3

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

---

## 🚀 Kubernetes Deployment

### Agent Deployment (Example - Pricing Agent)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pricing-agent
  namespace: axiom-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pricing-agent
  template:
    metadata:
      labels:
        app: pricing-agent
        cluster: trading
    spec:
      containers:
      - name: pricing-agent
        image: axiom/pricing-agent:v2.0.0
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: AGENT_NAME
          value: "pricing"
        - name: USE_GPU
          value: "false"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: pricing-agent-service
spec:
  selector:
    app: pricing-agent
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pricing-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pricing-agent
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 📊 Monitoring & Observability

### 1. Prometheus Metrics

**File:** `monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'pricing-agent'
    static_configs:
      - targets: ['pricing-agent:8000']
    metrics_path: '/metrics'
  
  - job_name: 'risk-agent'
    static_configs:
      - targets: ['risk-agent:8000']
  
  # ... all 12 agents
```

### 2. Grafana Dashboards

**Pre-configured dashboards for:**
- Agent performance (latency, throughput, errors)
- System health (CPU, memory, network)
- Business metrics (P&L, risk, compliance)
- Circuit breaker states
- Queue depths
- Error rates by agent

---

## 🔒 Security Configuration

### 1. Environment Variables

```bash
# API Keys (encrypted)
export OPRA_API_KEY="encrypted_key"
export POLYGON_API_KEY="encrypted_key"

# Database
export DB_PASSWORD="secure_password"

# Redis
export REDIS_PASSWORD="secure_password"

# Monitoring
export GRAFANA_PASSWORD="secure_password"
export PROMETHEUS_PASSWORD="secure_password"
```

### 2. Network Security

- Use VPC/private networks
- TLS/SSL for all communication
- API key rotation (automated)
- Rate limiting per client
- DDoS protection
- WAF (Web Application Firewall)

---

## ⚡ Performance Optimization

### 1. GPU Acceleration

```python
# Enable GPU for compute-intensive agents
config = {
    'pricing': {'use_gpu': True},
    'strategy': {'use_gpu': True},
    'volatility': {'use_gpu': True},
    'hedging': {'use_gpu': True}
}
```

### 2. Caching Strategy

- Market data: 1 second TTL
- Greeks: Computed on demand, cached 100ms
- Strategies: Cached 5 minutes
- Reports: Cached 1 hour

### 3. Database Optimization

- Connection pooling (100 connections)
- Read replicas for analytics
- Write-ahead logging
- Index optimization
- Query caching

---

## 🔄 High Availability

### 1. Redundancy

- Each agent: 2-3 replicas minimum
- Critical agents (Pricing, Execution, Guardrail): 3-5 replicas
- Load balancing: Round-robin with health checks
- Failover: Automatic within 1 second

### 2. Data Persistence

- PostgreSQL: Master-replica setup
- Redis: Cluster mode with persistence
- ChromaDB: Replicated vector store
- Backups: Hourly incremental, daily full

---

## 📈 Scaling Strategy

### Horizontal Scaling

**Auto-scale based on:**
- CPU utilization (>70%)
- Memory usage (>80%)
- Request queue depth (>1000)
- Latency (>2x target)

**Per Agent:**
- Pricing: 3-10 replicas
- Risk: 2-5 replicas
- Strategy: 2-5 replicas
- Execution: 3-8 replicas
- Others: 2-4 replicas

### Vertical Scaling

- Start: 2 CPU, 2GB RAM per agent
- Scale: Up to 8 CPU, 16GB RAM for GPU agents
- GPU agents: V100 or A100 GPUs

---

## 🚨 Disaster Recovery

### Backup Strategy

1. **Database:** Continuous replication + hourly snapshots
2. **Configuration:** Version controlled in Git
3. **Models:** Stored in S3 with versioning
4. **Logs:** Retained for 90 days
5. **Metrics:** Retained for 1 year

### Recovery Procedures

- **RTO:** 5 minutes (Recovery Time Objective)
- **RPO:** 1 minute (Recovery Point Objective)
- **Automated failover:** Geographic redundancy
- **Manual intervention:** <15 minutes for critical issues

---

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] All 12 agents tested individually
- [ ] Integration tests passed
- [ ] Performance benchmarks met
- [ ] Security audit completed
- [ ] Load testing passed (1M requests/day)
- [ ] Disaster recovery tested
- [ ] Monitoring dashboards configured
- [ ] Alerts configured
- [ ] Documentation updated
- [ ] Runbooks created

### Deployment Steps

1. Deploy infrastructure (Redis, PostgreSQL, monitoring)
2. Deploy support cluster (Monitoring, Guardrail first)
3. Deploy analytics cluster (Market Data, Volatility, Analytics)
4. Deploy trading cluster (Pricing, Risk, Strategy, Execution, Hedging)
5. Deploy client interface (last)
6. Verify health checks
7. Run smoke tests
8. Enable traffic gradually (canary deployment)
9. Monitor for 1 hour
10. Full rollout

### Post-Deployment

- [ ] All agents reporting healthy
- [ ] All metrics within targets
- [ ] No critical alerts
- [ ] Load balancing working
- [ ] Auto-scaling working
- [ ] Failover tested
- [ ] Client acceptance testing
- [ ] Performance validation

---

## 🔧 Configuration Management

### Environment-Specific Configs

**Development:**
```python
config = {
    'use_gpu': False,
    'log_level': 'DEBUG',
    'enable_tracing': True,
    'replicas': 1
}
```

**Staging:**
```python
config = {
    'use_gpu': False,
    'log_level': 'INFO',
    'enable_tracing': True,
    'replicas': 2
}
```

**Production:**
```python
config = {
    'use_gpu': True,
    'log_level': 'INFO',
    'enable_tracing': True,
    'replicas': 3,
    'enable_circuit_breaker': True,
    'enable_guardrails': True
}
```

---

## 📊 Monitoring Setup

### Key Metrics to Track

**Per Agent:**
- Requests per second
- Average latency (p50, p95, p99)
- Error rate
- Circuit breaker state
- Queue depth
- Memory usage
- CPU usage

**System-Wide:**
- Total throughput
- End-to-end latency
- System availability
- Active alerts
- Compliance rate

### Alerts Configuration

**Critical (Page On-Call):**
- Agent down (>1 minute)
- Error rate >1%
- Latency >10x target
- Compliance violation
- Guardrail blocking >50% actions

**Warning (Email Team):**
- Error rate >0.1%
- Latency >5x target
- Memory usage >80%
- Queue depth >1000

---

## 🔐 Security Best Practices

1. **Network:** VPC, private subnets, security groups
2. **Authentication:** API keys, JWT tokens
3. **Authorization:** Role-based access control
4. **Encryption:** TLS in transit, AES-256 at rest
5. **Secrets:** AWS Secrets Manager / HashiCorp Vault
6. **Audit:** Complete audit trail for all actions
7. **Compliance:** SOC 2, PCI-DSS ready

---

## 🎯 Performance Tuning

### Agent-Specific Optimizations

**Pricing Agent:**
- Batch processing (1000 Greeks in single call)
- GPU acceleration (10x speedup)
- Model quantization (FP16 for speed)

**Risk Agent:**
- Parallel VaR calculation
- Incremental Greeks updates
- Cache portfolio state

**Execution Agent:**
- Connection pooling to venues
- Async order submission
- Pre-flight validation

---

## 📞 Operations Runbooks

### Common Issues

**Issue:** Agent not responding
**Solution:** Check health endpoint, restart if needed, check logs

**Issue:** High latency
**Solution:** Check CPU/memory, scale horizontally, check downstream services

**Issue:** Circuit breaker open
**Solution:** Check error logs, fix root cause, reset circuit breaker

### Maintenance Windows

- Weekly: Rolling updates (zero downtime)
- Monthly: Database optimization
- Quarterly: Model retraining and deployment

---

## ✅ Success Criteria

### Performance SLAs

- **Pricing:** <1ms for 99% of requests
- **Risk:** <5ms for 95% of requests
- **Strategy:** <100ms for 90% of requests
- **Execution:** <10ms for 95% of orders
- **Overall System:** 99.999% uptime

### Business Metrics

- Trades executed: >10,000/day
- P&L tracked: Real-time
- Compliance: 100%
- Client satisfaction: >95%

---

## 🚀 Deployment Commands

### Deploy to Production

```bash
# Build all agents
docker-compose -f docker/docker-compose.agents.yml build

# Deploy
docker-compose -f docker/docker-compose.agents.yml up -d

# Check health
./scripts/check_all_agents_health.sh

# Monitor
docker-compose -f docker/docker-compose.agents.yml logs -f
```

### Rollback

```bash
# Rollback to previous version
docker-compose -f docker/docker-compose.agents.yml down
docker-compose -f docker/docker-compose.agents.yml up -d --force-recreate
```

---

## 📚 Additional Resources

- **Architecture:** [`ALL_12_AGENTS_COMPLETE.md`](ALL_12_AGENTS_COMPLETE.md:1)
- **Configuration:** [`agent_configuration.py`](config/agent_configuration.py:1)
- **Testing:** [`test_all_agents_integration.py`](tests/test_all_agents_integration.py:1)
- **Demo:** [`demo_complete_12_agent_workflow.py`](demos/demo_complete_12_agent_workflow.py:1)

---

**This deployment guide ensures professional production deployment of all 12 agents with enterprise-grade reliability and performance.**