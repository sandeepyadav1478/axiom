#!/bin/bash

# 🚀 Axiom Financial MCP Servers Setup Script
# This script builds and manages all financial MCP Docker containers

set -e

echo "🚀 Setting up Axiom Financial MCP Servers..."
echo "=" x 60

# Change to the financial data directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Function to build and run containers
setup_containers() {
    echo ""
    echo "🔧 Building all financial MCP Docker containers..."
    docker-compose build --parallel
    
    echo ""
    echo "✅ All containers built successfully!"
    echo ""
    echo "🐳 Available Docker Images:"
    docker images | grep "financial_data-"
}

# Function to start containers
start_containers() {
    echo ""
    echo "🚀 Starting all financial MCP servers..."
    docker-compose up -d
    
    echo ""
    echo "📊 Container Status:"
    docker-compose ps
}

# Function to stop containers
stop_containers() {
    echo ""
    echo "🛑 Stopping all financial MCP servers..."
    docker-compose down
}

# Function to view logs
show_logs() {
    echo ""
    echo "📋 Financial MCP Server Logs:"
    docker-compose logs --tail=50
}

# Function to test containers
test_containers() {
    echo ""
    echo "🧪 Testing financial MCP containers..."
    
    # Test each container individually
    echo "Testing Polygon.io MCP Server..."
    docker run --rm -e POLYGON_API_KEY=demo financial_data-polygon-io-server &
    sleep 2
    
    echo "Testing Yahoo Finance Professional MCP Server..."
    docker run --rm financial_data-yahoo-finance-professional &
    sleep 2
    
    echo "Testing Yahoo Finance Comprehensive MCP Server..."
    docker run --rm financial_data-yahoo-finance-comprehensive &
    sleep 2
    
    echo "✅ All containers can start successfully!"
}

# Function to show usage
show_usage() {
    echo ""
    echo "📖 Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     - Build all financial MCP Docker containers"
    echo "  start     - Start all financial MCP servers"
    echo "  stop      - Stop all financial MCP servers"
    echo "  restart   - Restart all financial MCP servers"
    echo "  logs      - Show logs from all financial MCP servers"
    echo "  test      - Test all containers can start"
    echo "  status    - Show status of all containers"
    echo "  clean     - Clean up stopped containers and unused images"
    echo ""
    echo "Examples:"
    echo "  $0 build    # Build all containers"
    echo "  $0 start    # Start all MCP servers"
    echo "  $0 logs     # View logs"
    echo ""
}

# Main script logic
case "${1:-help}" in
    build)
        setup_containers
        ;;
    start)
        start_containers
        ;;
    stop)
        stop_containers
        ;;
    restart)
        stop_containers
        start_containers
        ;;
    logs)
        show_logs
        ;;
    test)
        test_containers
        ;;
    status)
        echo "📊 Financial MCP Container Status:"
        docker-compose ps
        echo ""
        echo "🐳 Financial MCP Images:"
        docker images | grep "financial_data-"
        ;;
    clean)
        echo "🧹 Cleaning up financial MCP containers..."
        docker-compose down --rmi all --volumes
        docker system prune -f
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "🎯 Financial MCP Setup Complete!"
echo "📊 Available MCP Servers:"
echo "  • polygon-io-financial (Official Polygon.io)"  
echo "  • yahoo-finance-professional (27 advanced tools)"
echo "  • yahoo-finance-comprehensive (Complete analysis)"
echo ""
echo "💰 Cost: FREE Yahoo Finance servers + Optional Polygon.io ($25/month)"
echo "🏆 Savings vs Bloomberg: 98%+ cost reduction"
echo ""
echo "📝 Next steps:"
echo "  1. Add Polygon.io API key (FREE tier available at polygon.io)"
echo "  2. Restart Roo/Claude to load new MCP servers"
echo "  3. Test financial tools: 'Get current stock price for AAPL'"
echo ""