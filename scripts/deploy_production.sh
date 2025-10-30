#!/bin/bash
# Production Deployment Script

echo "Deploying Axiom Platform to Production"
echo "========================================"

# Build Docker images
echo "Building Docker images..."
docker build -f docker/Dockerfile.api -t axiom/api:latest .

# Start infrastructure
echo "Starting infrastructure..."
docker-compose -f docker/docker-compose.production.yml up -d

# Wait for services
echo "Waiting for services to be ready..."
sleep 10

# Check health
echo "Checking service health..."
curl http://localhost:8000/health

# Deploy to Kubernetes (if configured)
if command -v kubectl &> /dev/null; then
    echo "Deploying to Kubernetes..."
    kubectl apply -f kubernetes/deployment.yaml
fi

echo "Deployment complete!"
echo "API: http://localhost:8000"
echo "MLflow: http://localhost:5000"
echo "Grafana: http://localhost:3000"