#!/bin/bash
# Airflow Monitoring Script
# Real-time health monitoring and metrics collection

PROJECT_ROOT="/home/sandeep/pertinent/axiom"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo "=========================================="
echo "  Axiom Airflow Live Monitor"
echo "  Press Ctrl+C to exit"
echo "=========================================="
echo ""

while true; do
    # Timestamp
    echo -e "${GREEN}Last Update: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
    
    # Service Status
    echo "━━━ Service Status ━━━"
    docker ps --filter "name=airflow" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null
    echo ""
    
    # Scheduler Health
    echo "━━━ Scheduler Health ━━━"
    HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null || echo '{}')
    SCHEDULER_STATUS=$(echo $HEALTH | jq -r '.scheduler.status // "unknown"')
    DB_STATUS=$(echo $HEALTH | jq -r '.metadatabase.status // "unknown"')
    
    if [ "$SCHEDULER_STATUS" = "healthy" ]; then
        echo -e "Scheduler: ${GREEN}✅ Healthy${NC}"
    else
        echo -e "Scheduler: ${RED}❌ $SCHEDULER_STATUS${NC}"
    fi
    
    if [ "$DB_STATUS" = "healthy" ]; then
        echo -e "Database:  ${GREEN}✅ Healthy${NC}"
    else
        echo -e "Database:  ${RED}❌ $DB_STATUS${NC}"
    fi
    echo ""
    
    # DAG Status
    echo "━━━ DAG Status ━━━"
    docker exec axiom-airflow-webserver airflow dags list 2>/dev/null | tail -n +2 || echo "DAGs loading..."
    echo ""
    
    # Recent Runs
    echo "━━━ Recent Runs (Last 5) ━━━"
    docker exec axiom-airflow-webserver \
      airflow dags list-runs --no-backfill 2>/dev/null | head -6 || echo "No runs yet"
    echo ""
    
    # Failed Tasks (if any)
    echo "━━━ Failed Tasks (Last 24h) ━━━"
    FAILED=$(docker exec axiom-airflow-webserver \
      airflow dags list-runs --state failed --no-backfill 2>/dev/null | wc -l)
    
    if [ "$FAILED" -gt 1 ]; then
        echo -e "${RED}⚠️  $((FAILED - 1)) failed runs${NC}"
        docker exec axiom-airflow-webserver \
          airflow dags list-runs --state failed --no-backfill 2>/dev/null | head -6
    else
        echo -e "${GREEN}✅ No failures${NC}"
    fi
    echo ""
    
    # Resource Usage
    echo "━━━ Resource Usage ━━━"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
      axiom-airflow-webserver axiom-airflow-scheduler 2>/dev/null || echo "Stats unavailable"
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Refreshing in 10 seconds..."
    sleep 10
    clear
done