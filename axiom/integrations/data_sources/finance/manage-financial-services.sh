#!/bin/bash

# Axiom Unified Financial Data Services - Management Script
# Manages both MCP Servers and Provider Containers from single interface
# Location: axiom/integrations/data_sources/finance/manage-financial-services.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Print header
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Axiom Unified Financial Data Services Manager                ║${NC}"
    echo -e "${BLUE}║   MCP Servers + Provider Containers                            ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}ERROR: Docker is not running${NC}"
        echo -e "${YELLOW}Please start Docker Desktop and try again${NC}"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${RED}ERROR: .env file not found at $ENV_FILE${NC}"
        echo -e "${YELLOW}Please create a .env file from .env.example:${NC}"
        echo -e "  cd $PROJECT_ROOT"
        echo -e "  cp .env.example .env"
        echo -e "  # Edit .env and add your API keys"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Docker is running${NC}"
    echo -e "${GREEN}✓ Found .env file at: $ENV_FILE${NC}"
    echo ""
}

# Function to check API keys
check_api_keys() {
    echo -e "${BLUE}${BOLD}API Key Configuration Status:${NC}"
    echo ""
    
    echo -e "${CYAN}MCP Servers:${NC}"
    check_api_key "POLYGON_API_KEY" "  Polygon.io API Key"
    check_api_key "FIRECRAWL_API_KEY" "  Firecrawl API Key"
    echo ""
    
    echo -e "${CYAN}Provider Containers:${NC}"
    check_api_key "TAVILY_API_KEY" "  Tavily API Key"
    check_api_key "FINANCIAL_MODELING_PREP_API_KEY\|FMP_API_KEY" "  FMP API Key"
    check_api_key "FINNHUB_API_KEY" "  Finnhub API Key"
    check_api_key "ALPHA_VANTAGE_API_KEY" "  Alpha Vantage API Key"
    echo ""
}

# Helper function to check individual API key
check_api_key() {
    local key_name=$1
    local key_display=$2
    
    if grep -qE "^${key_name}=.+" "$ENV_FILE" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $key_display"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $key_display ${YELLOW}(not configured)${NC}"
        return 1
    fi
}

# Parse command line arguments
parse_arguments() {
    COMMAND=${1:-"help"}
    shift 2>/dev/null || true
    SERVICES=("$@")
    
    # If no services specified, default to "all"
    if [ ${#SERVICES[@]} -eq 0 ]; then
        SERVICES=("all")
    fi
}

# Function to build service profiles from services list
build_profiles() {
    local services=("$@")
    local profiles=""
    
    for service in "${services[@]}"; do
        case "$service" in
            all)
                profiles="--profile mcp --profile providers"
                break
                ;;
            mcp)
                profiles="$profiles --profile mcp"
                ;;
            providers)
                profiles="$profiles --profile providers"
                ;;
            polygon|polygon-io-server)
                profiles="$profiles --profile polygon"
                ;;
            yahoo-pro|yahoo-finance-professional)
                profiles="$profiles --profile yahoo-pro"
                ;;
            yahoo-comp|yahoo-finance-comprehensive)
                profiles="$profiles --profile yahoo-comp"
                ;;
            firecrawl|firecrawl-server)
                profiles="$profiles --profile firecrawl"
                ;;
            tavily|tavily-provider)
                profiles="$profiles --profile tavily"
                ;;
            fmp|fmp-provider)
                profiles="$profiles --profile fmp"
                ;;
            finnhub|finnhub-provider)
                profiles="$profiles --profile finnhub"
                ;;
            alpha-vantage|alpha-vantage-provider)
                profiles="$profiles --profile alpha-vantage"
                ;;
            *)
                # Try as direct service name
                profiles="$profiles $service"
                ;;
        esac
    done
    
    echo "$profiles"
}

# Function to start services
start_services() {
    echo -e "${BLUE}${BOLD}Starting Financial Data Services...${NC}"
    echo ""
    
    local profiles=$(build_profiles "${SERVICES[@]}")
    
    if [ "${SERVICES[0]}" = "all" ]; then
        echo -e "${CYAN}Starting ALL services (MCP + Providers)...${NC}"
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" $profiles up -d
        echo -e "${GREEN}✓ All services started successfully!${NC}"
    elif [[ "$profiles" == *"--profile"* ]]; then
        echo -e "${CYAN}Starting services with profiles: $profiles${NC}"
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" $profiles up -d
        echo -e "${GREEN}✓ Services started successfully!${NC}"
    else
        echo -e "${CYAN}Starting services: ${SERVICES[*]}${NC}"
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d $profiles
        echo -e "${GREEN}✓ Services started: ${SERVICES[*]}${NC}"
    fi
    
    echo ""
    show_status
}

# Function to stop services
stop_services() {
    echo -e "${BLUE}${BOLD}Stopping Financial Data Services...${NC}"
    echo ""
    
    local profiles=$(build_profiles "${SERVICES[@]}")
    
    if [ "${SERVICES[0]}" = "all" ]; then
        echo -e "${CYAN}Stopping ALL services...${NC}"
        docker-compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}✓ All services stopped${NC}"
    elif [[ "$profiles" == *"--profile"* ]]; then
        echo -e "${CYAN}Stopping services with profiles: $profiles${NC}"
        docker-compose -f "$COMPOSE_FILE" $profiles stop
        echo -e "${GREEN}✓ Services stopped${NC}"
    else
        echo -e "${CYAN}Stopping services: ${SERVICES[*]}${NC}"
        docker-compose -f "$COMPOSE_FILE" stop $profiles
        echo -e "${GREEN}✓ Services stopped: ${SERVICES[*]}${NC}"
    fi
}

# Function to restart services
restart_services() {
    stop_services
    sleep 2
    start_services
}

# Function to show logs
show_logs() {
    local profiles=$(build_profiles "${SERVICES[@]}")
    
    if [ "${SERVICES[0]}" = "all" ]; then
        docker-compose -f "$COMPOSE_FILE" logs -f
    elif [[ "$profiles" == *"--profile"* ]]; then
        docker-compose -f "$COMPOSE_FILE" $profiles logs -f
    else
        docker-compose -f "$COMPOSE_FILE" logs -f $profiles
    fi
}

# Function to show status
show_status() {
    echo -e "${BLUE}${BOLD}Service Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
}

# Function to check health of provider containers
check_health() {
    echo -e "${BLUE}${BOLD}Provider Health Check:${NC}"
    echo ""
    
    check_endpoint() {
        local name=$1
        local url=$2
        
        if curl -sf "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $name: ${GREEN}healthy${NC}"
        else
            echo -e "${RED}✗${NC} $name: ${RED}unavailable${NC}"
        fi
    }
    
    echo -e "${CYAN}REST API Providers:${NC}"
    check_endpoint "  Tavily Provider (8001)" "http://localhost:8001/health"
    check_endpoint "  FMP Provider (8002)" "http://localhost:8002/health"
    check_endpoint "  Finnhub Provider (8003)" "http://localhost:8003/health"
    check_endpoint "  Alpha Vantage Provider (8004)" "http://localhost:8004/health"
    echo ""
    
    echo -e "${CYAN}MCP Servers:${NC}"
    echo -e "${YELLOW}  ℹ MCP servers use stdio protocol - check with 'logs' command${NC}"
    echo ""
}

# Function to rebuild containers
rebuild_services() {
    echo -e "${BLUE}${BOLD}Rebuilding Services...${NC}"
    echo ""
    
    local profiles=$(build_profiles "${SERVICES[@]}")
    
    if [ "${SERVICES[0]}" = "all" ]; then
        echo -e "${CYAN}Rebuilding ALL services...${NC}"
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build --no-cache
        echo -e "${GREEN}✓ All services rebuilt${NC}"
    elif [[ "$profiles" == *"--profile"* ]]; then
        echo -e "${CYAN}Rebuilding services with profiles: $profiles${NC}"
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" $profiles build --no-cache
        echo -e "${GREEN}✓ Services rebuilt${NC}"
    else
        echo -e "${CYAN}Rebuilding services: ${SERVICES[*]}${NC}"
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build --no-cache $profiles
        echo -e "${GREEN}✓ Services rebuilt: ${SERVICES[*]}${NC}"
    fi
}

# Function to clean up
clean_services() {
    echo -e "${YELLOW}${BOLD}⚠ WARNING: This will remove all containers, networks, and volumes${NC}"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        echo -e "${BLUE}Cleaning up...${NC}"
        docker-compose -f "$COMPOSE_FILE" down -v --rmi local
        echo -e "${GREEN}✓ Cleanup complete${NC}"
    else
        echo -e "${YELLOW}Cleanup cancelled${NC}"
    fi
}

# Function to show comprehensive info
show_info() {
    echo -e "${BLUE}${BOLD}Service Information:${NC}"
    echo ""
    
    echo -e "${CYAN}${BOLD}MCP SERVERS${NC} (Model Context Protocol - for AI integration):"
    echo -e "  ${BOLD}polygon-io-server${NC}          - Real-time market data (Profile: polygon)"
    echo -e "    Cost: FREE tier (5 calls/min) | Premium: \$25/month"
    echo -e "  ${BOLD}yahoo-finance-professional${NC} - 27 professional tools (Profile: yahoo-pro)"
    echo -e "    Cost: 100% FREE unlimited"
    echo -e "  ${BOLD}yahoo-finance-comprehensive${NC} - Fundamental analysis (Profile: yahoo-comp)"
    echo -e "    Cost: 100% FREE unlimited"
    echo -e "  ${BOLD}firecrawl-server${NC}           - Web scraping (Profile: firecrawl)"
    echo -e "    Cost: FREE tier available"
    echo ""
    
    echo -e "${CYAN}${BOLD}PROVIDER CONTAINERS${NC} (REST APIs - for HTTP access):"
    echo -e "  ${BOLD}tavily-provider${NC} (8001)     - Search & M&A intelligence (Profile: tavily)"
    echo -e "    Cost: Paid service"
    echo -e "  ${BOLD}fmp-provider${NC} (8002)        - Comprehensive financial data (Profile: fmp)"
    echo -e "    Cost: FREE tier (250 calls/day) | Premium: \$14/month"
    echo -e "  ${BOLD}finnhub-provider${NC} (8003)    - Real-time market data (Profile: finnhub)"
    echo -e "    Cost: FREE tier (60 calls/min) | Premium: \$7.99/month"
    echo -e "  ${BOLD}alpha-vantage-provider${NC} (8004) - Financial data (Profile: alpha-vantage)"
    echo -e "    Cost: FREE tier (500 calls/day) | Premium: \$49/month"
    echo ""
    
    echo -e "${GREEN}${BOLD}💰 Cost Summary:${NC}"
    echo -e "  FREE Tier Total: ~750+ calls/day across all providers"
    echo -e "  Premium Total: ~\$71/month for unlimited professional data"
    echo -e "  vs Bloomberg Terminal: \$2,000+/month ${GREEN}(96%+ cost savings!)${NC}"
    echo ""
}

# Function to show help
show_help() {
    echo -e "${BLUE}${BOLD}USAGE:${NC}"
    echo -e "  ./manage-financial-services.sh <command> [service1] [service2] ..."
    echo ""
    
    echo -e "${BLUE}${BOLD}COMMANDS:${NC}"
    echo -e "  ${BOLD}start${NC} [services...]    - Start services (default: all)"
    echo -e "  ${BOLD}stop${NC} [services...]     - Stop services (default: all)"
    echo -e "  ${BOLD}restart${NC} [services...]  - Restart services (default: all)"
    echo -e "  ${BOLD}logs${NC} [services...]     - Show logs (default: all, -f for follow)"
    echo -e "  ${BOLD}status${NC}                 - Show status of all services"
    echo -e "  ${BOLD}health${NC}                 - Check health of provider containers"
    echo -e "  ${BOLD}rebuild${NC} [services...]  - Rebuild containers (default: all)"
    echo -e "  ${BOLD}clean${NC}                  - Remove all containers, networks, and volumes"
    echo -e "  ${BOLD}info${NC}                   - Show detailed service information"
    echo -e "  ${BOLD}keys${NC}                   - Show API key configuration status"
    echo -e "  ${BOLD}help${NC}                   - Show this help message"
    echo ""
    
    echo -e "${BLUE}${BOLD}SERVICE CATEGORIES:${NC}"
    echo -e "  ${BOLD}all${NC}                    - All services (default)"
    echo -e "  ${BOLD}mcp${NC}                    - All MCP servers"
    echo -e "  ${BOLD}providers${NC}              - All provider containers"
    echo ""
    
    echo -e "${BLUE}${BOLD}INDIVIDUAL SERVICES:${NC}"
    echo -e "  ${CYAN}MCP Servers:${NC}"
    echo -e "    polygon, yahoo-pro, yahoo-comp, firecrawl"
    echo -e "  ${CYAN}Providers:${NC}"
    echo -e "    tavily, fmp, finnhub, alpha-vantage"
    echo ""
    
    echo -e "${BLUE}${BOLD}EXAMPLES:${NC}"
    echo -e "  ${GREEN}# Start everything${NC}"
    echo -e "  ./manage-financial-services.sh start"
    echo ""
    echo -e "  ${GREEN}# Start all MCP servers${NC}"
    echo -e "  ./manage-financial-services.sh start mcp"
    echo ""
    echo -e "  ${GREEN}# Start all provider containers${NC}"
    echo -e "  ./manage-financial-services.sh start providers"
    echo ""
    echo -e "  ${GREEN}# Start specific services${NC}"
    echo -e "  ./manage-financial-services.sh start polygon fmp"
    echo -e "  ./manage-financial-services.sh start yahoo-pro tavily"
    echo ""
    echo -e "  ${GREEN}# View logs${NC}"
    echo -e "  ./manage-financial-services.sh logs fmp"
    echo -e "  ./manage-financial-services.sh logs mcp"
    echo ""
    echo -e "  ${GREEN}# Check status and health${NC}"
    echo -e "  ./manage-financial-services.sh status"
    echo -e "  ./manage-financial-services.sh health"
    echo ""
    echo -e "  ${GREEN}# Rebuild services${NC}"
    echo -e "  ./manage-financial-services.sh rebuild fmp"
    echo -e "  ./manage-financial-services.sh rebuild all"
    echo ""
}

# Main script execution
main() {
    print_header
    check_prerequisites
    parse_arguments "$@"
    
    case $COMMAND in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            check_health
            ;;
        health)
            check_health
            ;;
        rebuild)
            rebuild_services
            ;;
        clean)
            clean_services
            ;;
        info)
            show_info
            ;;
        keys)
            check_api_keys
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}ERROR: Unknown command: $COMMAND${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"