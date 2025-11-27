#!/bin/bash

# Axiom Streaming API - Quick Start Script
# Launches production streaming infrastructure

set -e

echo "=================================="
echo "Axiom Real-Time Streaming API"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${BLUE}Starting services...${NC}"
echo ""

# Start services
docker-compose up -d

echo ""
echo -e "${GREEN}✅ Services started successfully!${NC}"
echo ""

# Wait for services to be healthy
echo -e "${BLUE}Waiting for services to be ready...${NC}"
sleep 5

# Check health
echo ""
echo -e "${BLUE}Checking service health...${NC}"
health_response=$(curl -s http://localhost:8001/health || echo "failed")

if [[ $health_response != "failed" ]]; then
    echo -e "${GREEN}✅ Streaming API is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Health check failed, services may still be starting...${NC}"
fi

# Display access information
echo ""
echo "=================================="
echo "Services Ready!"
echo "=================================="
echo ""
echo -e "${GREEN}🌐 Dashboard:${NC}      http://localhost:8001/"
echo -e "${GREEN}📡 WebSocket:${NC}      ws://localhost:8001/ws/{client-id}"
echo -e "${GREEN}📊 SSE:${NC}            http://localhost:8001/sse/{client-id}"
echo -e "${GREEN}🏥 Health Check:${NC}   http://localhost:8001/health"
echo -e "${GREEN}📈 Stats:${NC}          http://localhost:8001/stats"
echo ""
echo -e "${BLUE}📊 Monitoring:${NC}"
echo -e "   Prometheus:    http://localhost:9090"
echo -e "   Grafana:       http://localhost:3001 (admin/admin)"
echo ""
echo -e "${YELLOW}To view logs:${NC}      docker-compose logs -f"
echo -e "${YELLOW}To stop:${NC}          docker-compose down"
echo ""

# Optional: Run demo
read -p "Run demo? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}Running streaming demo...${NC}"
    echo ""
    cd ../..
    python demos/demo_streaming_api.py
fi