# Production Deployment Guide

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/your-org/axiom.git
cd axiom
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Deploy with Docker Compose
chmod +x scripts/deploy_production.sh
./scripts/deploy_production.sh

# 4. Access services
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

---

## Architecture

**Frontend Layer:**
- Client dashboards (Plotly)
- Web UI (Streamlit)
- Research reports (HTML/PDF)

**API Layer:**
- FastAPI REST endpoints
- Model serving (60 models)
- Batch inference engine
- WebSocket for real-time

**ML Layer:**
- 60 ML models
- Model caching
- LangGraph orchestration
- DSPy optimization

**Infrastructure Layer:**
- MLflow (tracking + registry)
- Feast (feature store)
- Evidently (drift detection)
- Redis (caching)
- PostgreSQL (metadata)

**Monitoring Layer:**
- Prometheus (metrics)
- Grafana (dashboards)
- Alerts (email/Slack)
- Performance tracking

---

## Scaling

**Horizontal Scaling:**
```bash
# Scale API replicas
kubectl scale deployment axiom-api --replicas=10
```

**Vertical Scaling:**
```yaml
# Increase resources in deployment.yaml
resources:
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

---

## Monitoring

**Key Metrics:**
- Request latency (target: <100ms)
- Error rate (target: <1%)
- Model drift score (alert: >0.3)
- Cache hit rate (target: >80%)
- Memory usage (alert: >90%)

**Dashboards:**
- Grafana: http://localhost:3000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090

---

## Backup & Recovery

**Database Backups:**
```bash
# Backup PostgreSQL (MLflow metadata)
docker exec postgres pg_dump mlflow > backup.sql
```

**Model Backups:**
- Models stored in MLflow registry
- Automatic versioning
- S3/GCS sync available

---

## Security

**API Authentication:**
- API key required for all endpoints
- JWT tokens for web UI
- OAuth2 for enterprise

**Network Security:**
- TLS/SSL for all endpoints
- VPC isolation
- Firewall rules

---

## Performance Tuning

**Model Caching:**
- 25 models cached by default
- LRU eviction
- Configurable via MODEL_CACHE_SIZE

**Batch Processing:**
- Default: 32 requests per batch
- Configurable via BATCH_SIZE
- GPU batching for deep learning models

**Database Connection Pool:**
- Min: 5 connections
- Max: 20 connections
- Idle timeout: 300s

---

## Troubleshooting

**High Memory Usage:**
- Reduce MODEL_CACHE_SIZE
- Enable model eviction
- Use CPU-only mode

**Slow Response:**
- Check model cache hit rate
- Increase batch size
- Add more replicas

**Model Drift:**
- Retrain affected models
- Update with recent data
- Monitor drift dashboard

---

## Production Checklist

- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] API keys set
- [ ] Database initialized
- [ ] Models cached
- [ ] Monitoring enabled
- [ ] Alerts configured
- [ ] Backups scheduled
- [ ] Security hardened
- [ ] Load testing completed

---

**Platform:** 60 ML Models + Complete Infrastructure  
**Status:** Production-ready  
**Support:** See docs/ for additional guides