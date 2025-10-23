#!/bin/bash
# Axiom Database Setup Script
# Quick setup for PostgreSQL + Vector DB infrastructure

set -e

echo "=================================================="
echo "Axiom Database Infrastructure Setup"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${BLUE}✓ Docker and Docker Compose detected${NC}"
echo ""

# Prompt for setup type
echo "Select database setup:"
echo "  1) PostgreSQL only (basic)"
echo "  2) PostgreSQL + ChromaDB (local development)"
echo "  3) PostgreSQL + Weaviate (self-hosted production)"
echo "  4) PostgreSQL + PgAdmin (with management UI)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo -e "${BLUE}Setting up PostgreSQL only...${NC}"
        docker-compose up -d postgres
        ;;
    2)
        echo -e "${BLUE}Setting up PostgreSQL + ChromaDB...${NC}"
        docker-compose --profile vector-db-light up -d
        ;;
    3)
        echo -e "${BLUE}Setting up PostgreSQL + Weaviate...${NC}"
        docker-compose --profile vector-db up -d
        ;;
    4)
        echo -e "${BLUE}Setting up PostgreSQL + PgAdmin...${NC}"
        docker-compose --profile admin up -d
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}Waiting for services to be ready...${NC}"
sleep 5

# Check PostgreSQL health
echo -e "${BLUE}Checking PostgreSQL health...${NC}"
for i in {1..30}; do
    if docker exec axiom_postgres pg_isready -U axiom &> /dev/null; then
        echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${YELLOW}⚠ PostgreSQL health check timeout${NC}"
    fi
    sleep 1
done

echo ""
echo -e "${GREEN}=================================================="
echo "✓ Database Infrastructure Setup Complete!"
echo "==================================================${NC}"
echo ""
echo "Connection Details:"
echo "  PostgreSQL:"
echo "    Host: localhost"
echo "    Port: 5432"
echo "    Database: axiom_finance"
echo "    User: axiom"
echo ""

if [ "$choice" -eq 2 ]; then
    echo "  ChromaDB:"
    echo "    URL: http://localhost:8000"
    echo ""
fi

if [ "$choice" -eq 3 ]; then
    echo "  Weaviate:"
    echo "    URL: http://localhost:8080"
    echo ""
fi

if [ "$choice" -eq 4 ]; then
    echo "  PgAdmin:"
    echo "    URL: http://localhost:5050"
    echo "    Email: admin@axiom.com"
    echo ""
fi

echo "Next Steps:"
echo "  1. Update .env file with your configuration"
echo "  2. Install Python dependencies:"
echo "     pip install -r requirements.txt"
echo "  3. Initialize database schema:"
echo "     python -c 'from axiom.database import get_migration_manager; get_migration_manager().init_schema()'"
echo "  4. Run demo:"
echo "     python demos/demo_database_integration.py"
echo ""
echo -e "${GREEN}Happy coding!${NC}"