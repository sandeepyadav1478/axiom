#!/bin/bash
# Complete Deployment Script for Derivatives Platform
# Automates entire deployment process

set -e  # Exit on error

echo "============================================================"
echo "AXIOM DERIVATIVES PLATFORM - AUTOMATED DEPLOYMENT"
echo "============================================================"
echo ""

# Configuration
ENVIRONMENT=${1:-staging}  # staging or production
NAMESPACE="derivatives-${ENVIRONMENT}"
IMAGE_TAG=${2:-latest}

echo "Environment: $ENVIRONMENT"
echo "Namespace: $NAMESPACE"
echo "Image Tag: $IMAGE_TAG"
echo ""

# Step 1: Build Docker Image
echo "→ Step 1: Building Docker Image..."
docker build -t axiom/derivatives:${IMAGE_TAG} -f axiom/derivatives/docker/Dockerfile .

if [ $? -eq 0 ]; then
    echo "   ✓ Docker image built successfully"
else
    echo "   ✗ Docker build failed"
    exit 1
fi

# Step 2: Push to Registry
echo ""
echo "→ Step 2: Pushing to Container Registry..."
docker push axiom/derivatives:${IMAGE_TAG}

if [ $? -eq 0 ]; then
    echo "   ✓ Image pushed to registry"
else
    echo "   ✗ Push failed"
    exit 1
fi

# Step 3: Create Kubernetes Namespace (if not exists)
echo ""
echo "→ Step 3: Setting up Kubernetes namespace..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
echo "   ✓ Namespace ready"

# Step 4: Deploy Database (PostgreSQL)
echo ""
echo "→ Step 4: Deploying PostgreSQL..."
kubectl apply -f axiom/derivatives/kubernetes/deployment.yaml -n ${NAMESPACE}

# Wait for postgres to be ready
echo "   Waiting for PostgreSQL..."
kubectl wait --for=condition=ready pod -l app=derivatives-postgres -n ${NAMESPACE} --timeout=300s

if [ $? -eq 0 ]; then
    echo "   ✓ PostgreSQL ready"
else
    echo "   ✗ PostgreSQL failed to start"
    exit 1
fi

# Step 5: Initialize Database Schema
echo ""
echo "→ Step 5: Initializing database schema..."
kubectl exec -it deployment/derivatives-postgres -n ${NAMESPACE} -- psql -U axiom_prod -d axiom_derivatives -f /schema.sql

# Step 6: Deploy Redis
echo ""
echo "→ Step 6: Deploying Redis..."
kubectl wait --for=condition=ready pod -l app=derivatives-redis -n ${NAMESPACE} --timeout=60s
echo "   ✓ Redis ready"

# Step 7: Deploy Application
echo ""
echo "→ Step 7: Deploying Derivatives API..."
kubectl set image deployment/derivatives-api api=axiom/derivatives:${IMAGE_TAG} -n ${NAMESPACE}

# Wait for rollout
kubectl rollout status deployment/derivatives-api -n ${NAMESPACE} --timeout=600s

if [ $? -eq 0 ]; then
    echo "   ✓ API deployment successful"
else
    echo "   ✗ API deployment failed"
    exit 1
fi

# Step 8: Verify Health
echo ""
echo "→ Step 8: Verifying deployment health..."

# Port forward temporarily for health check
kubectl port-forward svc/derivatives-api-svc 8000:80 -n ${NAMESPACE} &
PF_PID=$!
sleep 5

# Health check
HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')

if [ "$HEALTH" == "healthy" ]; then
    echo "   ✓ Health check passed"
else
    echo "   ✗ Health check failed"
    kill $PF_PID
    exit 1
fi

# Test a Greeks calculation
echo ""
echo "→ Testing Greeks endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:8000/greeks \
    -H "Content-Type: application/json" \
    -d '{"spot": 100, "strike": 100, "time_to_maturity": 1.0, "risk_free_rate": 0.03, "volatility": 0.25}')

DELTA=$(echo $RESPONSE | jq -r '.delta')

if [ ! -z "$DELTA" ]; then
    echo "   ✓ Greeks calculation working (Delta: $DELTA)"
else
    echo "   ✗ Greeks calculation failed"
    kill $PF_PID
    exit 1
fi

# Cleanup port forward
kill $PF_PID

# Step 9: Deploy Monitoring
echo ""
echo "→ Step 9: Deploying Prometheus & Grafana..."
kubectl apply -f axiom/derivatives/kubernetes/deployment.yaml -n ${NAMESPACE}
echo "   ✓ Monitoring deployed"

# Step 10: Configure Autoscaling
echo ""
echo "→ Step 10: Configuring autoscaling..."
kubectl autoscale deployment derivatives-api --min=3 --max=10 --cpu-percent=70 -n ${NAMESPACE}
echo "   ✓ HPA configured"

# Summary
echo ""
echo "============================================================"
echo "DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "Deployment Summary:"
echo "  Environment: $ENVIRONMENT"
echo "  Namespace: $NAMESPACE"
echo "  Image: axiom/derivatives:${IMAGE_TAG}"
echo "  Pods: $(kubectl get pods -n ${NAMESPACE} --no-headers | wc -l)"
echo "  Services: $(kubectl get svc -n ${NAMESPACE} --no-headers | wc -l)"
echo ""
echo "Access:"
echo "  API: kubectl port-forward svc/derivatives-api-svc 8000:80 -n ${NAMESPACE}"
echo "  Grafana: kubectl port-forward svc/grafana 3000:3000 -n ${NAMESPACE}"
echo "  Prometheus: kubectl port-forward svc/prometheus 9090:9090 -n ${NAMESPACE}"
echo ""
echo "Next Steps:"
echo "  1. Review Grafana dashboards"
echo "  2. Run load tests"
echo "  3. Monitor performance"
echo "  4. Gradual rollout to clients"
echo ""
echo "✓ Deployment successful!"

# Usage:
# ./scripts/deploy_derivatives_platform.sh staging
# ./scripts/deploy_derivatives_platform.sh production v1.0.0