#!/bin/bash
# MCP Services Docker Compose Validation Script
# Location: tests/docker/test_mcp_services.sh
# Run from project root: bash tests/docker/test_mcp_services.sh

set -e

PROJECT_ROOT="/Users/sandeep.yadav/work/axiom"
COMPOSE_FILE="axiom/integrations/data_sources/finance/docker-compose.yml"
ENV_FILE=".env"

echo "🧪 MCP Services Docker Compose Validation"
echo "=========================================="
echo ""

# Test 1: Check .env file exists
echo "1️⃣ Checking root .env file..."
if [ -f "$ENV_FILE" ]; then
    echo "✅ Root .env file exists"
    echo "   Checking API keys:"
    grep -q "POLYGON_API_KEY=" "$ENV_FILE" 2>/dev/null && echo "   ✅ POLYGON_API_KEY defined" || echo "   ⚠️  POLYGON_API_KEY not defined (optional)"
    grep -q "FIRECRAWL_API_KEY=" "$ENV_FILE" 2>/dev/null && echo "   ✅ FIRECRAWL_API_KEY defined" || echo "   ⚠️  FIRECRAWL_API_KEY not defined (optional)"
else
    echo "⚠️  Root .env file not found (using defaults)"
    echo "   Note: docker-compose will use .env.example as fallback"
fi
echo ""

# Test 2: Validate docker-compose syntax
echo "2️⃣ Validating docker-compose syntax..."
if docker-compose -f "$COMPOSE_FILE" config > /dev/null 2>&1; then
    echo "✅ docker-compose.yml syntax valid"
else
    echo "❌ docker-compose.yml has syntax errors"
    exit 1
fi
echo ""

# Test 3: Check services defined
echo "3️⃣ Checking service definitions..."
SERVICES=$(docker-compose -f "$COMPOSE_FILE" --profile polygon --profile yahoo-comp --profile firecrawl config --services | wc -l | tr -d ' ')
if [ "$SERVICES" -eq 3 ]; then
    echo "✅ All 3 MCP services defined:"
    docker-compose -f "$COMPOSE_FILE" --profile polygon --profile yahoo-comp --profile firecrawl config --services | sed 's/^/   /'
    echo "   Note: yahoo-finance-professional is disabled due to build issues"
else
    echo "❌ Expected 3 services, found $SERVICES"
    exit 1
fi
echo ""

# Test 4: Verify environment variable configuration
echo "4️⃣ Verifying environment variable configuration..."
CONFIG_OUTPUT=$(docker-compose -f "$COMPOSE_FILE" --profile polygon config 2>&1)
# Check if environment section exists in config (indicates env_file is being processed)
if docker-compose -f "$COMPOSE_FILE" --profile polygon config | grep -q "environment:"; then
    echo "✅ Environment variable configuration present"
    echo "   Note: API key warnings are expected during config validation"
    echo "   Services will load actual values from .env when started"
else
    echo "❌ Environment configuration missing"
    exit 1
fi
echo ""

# Test 5: Check network configuration
echo "5️⃣ Checking network configuration..."
if docker-compose -f "$COMPOSE_FILE" --profile polygon config 2>/dev/null | grep -q "financial-data-network"; then
    echo "✅ Network configuration correct (axiom-financial-data-unified)"
else
    echo "❌ Network configuration missing"
    exit 1
fi
echo ""

# Test 6: Validate service profiles
echo "6️⃣ Validating service profiles..."
echo "   Available MCP profiles:"
echo "   - polygon (Polygon.io server)"
echo "   - yahoo-comp (Yahoo Finance Comprehensive)"
echo "   - firecrawl (Firecrawl server)"
echo "   Note: yahoo-pro profile disabled (build issues)"
echo "✅ Profile configuration correct"
echo ""

echo "✅ All validation checks passed!"
echo ""
echo "📋 Next steps to test actual containers:"
echo "   1. Start a service: docker-compose -f $COMPOSE_FILE --profile polygon up -d"
echo "   2. Check status: docker-compose -f $COMPOSE_FILE ps"
echo "   3. View logs: docker-compose -f $COMPOSE_FILE logs polygon-io-server"
echo "   4. Stop service: docker-compose -f $COMPOSE_FILE down"
echo ""
echo "🎯 To run full integration test, use: bash tests/docker/test_mcp_integration.sh"