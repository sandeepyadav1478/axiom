#!/bin/bash

# Production Monitoring Stack Deployment Script
# Deploys Prometheus, Grafana, Alertmanager, and metrics exporters

set -e

echo "🚀 Deploying Axiom Production Monitoring Stack"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo ""
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"

# Check if networks exist
echo ""
echo "🌐 Checking Docker networks..."

if ! docker network inspect axiom-mcp-network &> /dev/null; then
    echo -e "${YELLOW}⚠️  Creating axiom-mcp-network...${NC}"
    docker network create axiom-mcp-network
fi

if ! docker network inspect database_axiom_network &> /dev/null; then
    echo -e "${YELLOW}⚠️  Creating database_axiom_network...${NC}"
    docker network create database_axiom_network
fi

echo -e "${GREEN}✅ Networks are ready${NC}"

# Check for .env file
echo ""
echo "🔐 Checking environment configuration..."

if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating from template...${NC}"
    cat > .env << 'EOF'
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_USER=axiom
POSTGRES_PASSWORD=change_me
POSTGRES_DB=axiom_finance

# Redis Configuration
REDIS_HOST=localhost
REDIS_PASSWORD=change_me

# Grafana Configuration
GRAFANA_PASSWORD=admin123

# Metrics Configuration
METRICS_SCRAPE_INTERVAL=15
EOF
    echo -e "${YELLOW}⚠️  Please edit .env file with your actual credentials${NC}"
    read -p "Press Enter to continue after editing .env..."
fi

echo -e "${GREEN}✅ Environment configuration ready${NC}"

# Build custom exporters
echo ""
echo "🔨 Building metrics exporters..."

cd ../axiom/pipelines/airflow

if docker build -f Dockerfile.metrics-exporter -t axiom-airflow-metrics-exporter:latest . ; then
    echo -e "${GREEN}✅ Airflow metrics exporter built${NC}"
else
    echo -e "${RED}❌ Failed to build Airflow metrics exporter${NC}"
    exit 1
fi

if docker build -f Dockerfile.quality-exporter -t axiom-data-quality-exporter:latest . ; then
    echo -e "${GREEN}✅ Data quality exporter built${NC}"
else
    echo -e "${RED}❌ Failed to build data quality exporter${NC}"
    exit 1
fi

cd ../../../monitoring

# Deploy monitoring stack
echo ""
echo "🚢 Deploying monitoring stack..."

# Use docker-compose or docker compose based on what's available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

if $COMPOSE_CMD -f docker-compose-monitoring.yml up -d; then
    echo -e "${GREEN}✅ Monitoring stack deployed successfully${NC}"
else
    echo -e "${RED}❌ Failed to deploy monitoring stack${NC}"
    exit 1
fi

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to become healthy..."
sleep 10

# Check service health
echo ""
echo "🏥 Checking service health..."

services=(
    "axiom-prometheus:9090"
    "axiom-grafana:3000"
    "axiom-alertmanager:9093"
    "axiom-airflow-metrics-exporter:9092"
    "axiom-data-quality-exporter:9093"
)

all_healthy=true

for service in "${services[@]}"; do
    name="${service%:*}"
    port="${service##*:}"
    
    if docker ps | grep -q "$name"; then
        echo -e "${GREEN}✅ $name is running${NC}"
    else
        echo -e "${RED}❌ $name is not running${NC}"
        all_healthy=false
    fi
done

# Display access information
echo ""
echo "================================================"
echo "✨ Deployment Complete!"
echo "================================================"
echo ""
echo "📊 Access Your Dashboards:"
echo ""
echo "  Grafana:       http://localhost:3000"
echo "    Username: admin"
echo "    Password: admin123 (or from .env)"
echo ""
echo "  Prometheus:    http://localhost:9090"
echo "  Alertmanager:  http://localhost:9093"
echo ""
echo "📈 Metrics Endpoints:"
echo ""
echo "  AI/ML Metrics:     http://localhost:9091/metrics"
echo "  Airflow Metrics:   http://localhost:9092/metrics"
echo "  Data Quality:      http://localhost:9093/metrics"
echo "  PostgreSQL:        http://localhost:9187/metrics"
echo "  Redis:             http://localhost:9121/metrics"
echo "  Container Metrics: http://localhost:8080/metrics"
echo ""
echo "📚 Documentation:"
echo ""
echo "  Full Guide: monitoring/README.md"
echo ""
echo "🔧 Useful Commands:"
echo ""
echo "  View logs:    $COMPOSE_CMD -f docker-compose-monitoring.yml logs -f"
echo "  Stop:         $COMPOSE_CMD -f docker-compose-monitoring.yml stop"
echo "  Restart:      $COMPOSE_CMD -f docker-compose-monitoring.yml restart"
echo "  Remove:       $COMPOSE_CMD -f docker-compose-monitoring.yml down"
echo ""

if [ "$all_healthy" = true ]; then
    echo -e "${GREEN}✅ All services are healthy and running!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some services may not be healthy. Check logs:${NC}"
    echo "   $COMPOSE_CMD -f docker-compose-monitoring.yml logs"
    exit 1
fi