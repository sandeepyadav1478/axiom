#!/bin/bash
# Airflow Setup Automation Script
# Handles complete Airflow deployment from scratch

set -e  # Exit on error

echo "=========================================="
echo "  Axiom Airflow Setup Automation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0[31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="/home/sandeep/pertinent/axiom"

echo "Step 1: Verify Prerequisites"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker installed${NC}"

# Check Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker Compose installed${NC}"

# Check PostgreSQL running
if ! docker ps | grep -q axiom_postgres; then
    echo -e "${RED}❌ PostgreSQL container not running${NC}"
    echo "Start with: docker compose -f axiom/database/docker-compose.yml up -d"
    exit 1
fi
echo -e "${GREEN}✅ PostgreSQL running${NC}"

echo ""
echo "Step 2: Create Required Directories"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

AIRFLOW_DIR="$PROJECT_ROOT/axiom/pipelines/airflow"

mkdir -p "$AIRFLOW_DIR/logs"
mkdir -p "$AIRFLOW_DIR/plugins"
chmod 777 "$AIRFLOW_DIR/logs"
chmod 777 "$AIRFLOW_DIR/plugins"

echo -e "${GREEN}✅ Directories created${NC}"

echo ""
echo "Step 3: Create Airflow Database"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if airflow database exists
DB_EXISTS=$(docker exec axiom_postgres psql -U axiom -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname='airflow'")

if [ "$DB_EXISTS" != "1" ]; then
    docker exec axiom_postgres psql -U axiom -d postgres -c "CREATE DATABASE airflow;"
    echo -e "${GREEN}✅ Airflow database created${NC}"
else
    echo -e "${YELLOW}⚠️  Airflow database already exists${NC}"
fi

echo ""
echo "Step 4: Create Airflow User"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create airflow user if doesn't exist
USER_EXISTS=$(docker exec axiom_postgres psql -U axiom -d postgres -tAc \
  "SELECT 1 FROM pg_roles WHERE rolname='airflow'")

if [ "$USER_EXISTS" != "1" ]; then
    docker exec axiom_postgres psql -U axiom -d postgres -c \
      "CREATE USER airflow WITH PASSWORD 'airflow_pass';"
    echo -e "${GREEN}✅ Airflow user created${NC}"
else
    echo -e "${YELLOW}⚠️  Airflow user already exists${NC}"
fi

# Grant permissions
docker exec axiom_postgres psql -U axiom -d airflow -c \
  "GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;"
docker exec axiom_postgres psql -U axiom -d airflow -c \
  "ALTER DATABASE airflow OWNER TO airflow;"

echo -e "${GREEN}✅ Permissions granted${NC}"

echo ""
echo "Step 5: Set Environment Variables"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if AIRFLOW_UID is set
if ! grep -q "AIRFLOW_UID" "$PROJECT_ROOT/.env"; then
    echo "AIRFLOW_UID=50000" >> "$PROJECT_ROOT/.env"
    echo -e "${GREEN}✅ AIRFLOW_UID added to .env${NC}"
else
    echo -e "${YELLOW}⚠️  AIRFLOW_UID already in .env${NC}"
fi

export AIRFLOW_UID=50000

echo ""
echo "Step 6: Deploy Airflow Services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT"
docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml up -d

echo -e "${GREEN}✅ Airflow services starting...${NC}"

echo ""
echo "Step 7: Wait for Initialization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Waiting for Airflow to initialize (60 seconds)..."
sleep 60

echo ""
echo "Step 8: Verify Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check containers
WEBSERVER_RUNNING=$(docker ps --filter "name=axiom-airflow-webserver" --format "{{.Status}}" | grep -c "Up" || true)
SCHEDULER_RUNNING=$(docker ps --filter "name=axiom-airflow-scheduler" --format "{{.Status}}" | grep -c "Up" || true)

if [ "$WEBSERVER_RUNNING" = "1" ] && [ "$SCHEDULER_RUNNING" = "1" ]; then
    echo -e "${GREEN}✅ Airflow containers running${NC}"
else
    echo -e "${RED}❌ Airflow containers not healthy${NC}"
    docker ps --filter "name=airflow"
    exit 1
fi

# Check health endpoint
HEALTH_STATUS=$(curl -s http://localhost:8080/health | jq -r '.scheduler.status' || echo "unhealthy")

if [ "$HEALTH_STATUS" = "healthy" ]; then
    echo -e "${GREEN}✅ Airflow scheduler healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Scheduler status: $HEALTH_STATUS${NC}"
fi

# List DAGs
echo ""
echo "Available DAGs:"
docker exec axiom-airflow-webserver airflow dags list 2>/dev/null || echo "DAGs still loading..."

echo ""
echo "=========================================="
echo "  ✅ Airflow Setup Complete!"
echo "=========================================="
echo ""
echo "Access Airflow UI:"
echo "  URL: http://localhost:8080"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "Quick Commands:"
echo "  List DAGs:    docker exec axiom-airflow-webserver airflow dags list"
echo "  View Health:  curl http://localhost:8080/health | jq"
echo "  Stop:         docker compose -f axiom/pipelines/airflow/docker-compose-airflow.yml down"
echo ""
echo "Next Steps:"
echo "  1. Open http://localhost:8080 in browser"
echo "  2. Login with admin/admin123"
echo "  3. Enable DAGs by toggling switches"
echo "  4. Monitor execution in Grid View"
echo ""