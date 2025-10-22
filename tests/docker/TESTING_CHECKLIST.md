# 🧪 Testing Checklist for Docker Compose & Configuration Changes

## ⚠️ CRITICAL: Never conclude work without validation!

This checklist ensures that configuration changes are properly tested before marking tasks complete.

## 📋 Docker Compose Validation Checklist

### 1. **Syntax Validation**
```bash
cd axiom/integrations/data_sources/finance/mcp_servers
docker-compose config
```
✅ **Expected**: Clean YAML output without errors
❌ **If fails**: Fix YAML syntax errors

### 2. **Environment Variable Loading**
```bash
# Check if .env file is accessible
docker-compose config | grep -A 5 "environment:"
```
✅ **Expected**: Environment variables properly substituted
❌ **If fails**: Check .env file path and permissions

### 3. **API Keys Verification**
```bash
# Verify API keys are loaded
docker-compose config | grep "API_KEY"
```
✅ **Expected**: API keys showing as ${VAR} or actual values
❌ **If fails**: Check .env file has required keys

### 4. **Service Definitions**
```bash
# List all services
docker-compose config --services
```
✅ **Expected**: All 4 services listed (polygon-io-server, yahoo-finance-professional, yahoo-finance-comprehensive, firecrawl-server)
❌ **If fails**: Check service definitions in docker-compose.yml

### 5. **Build Test (Dry Run)**
```bash
# Validate services can be built
docker-compose --profile polygon pull 2>&1 || echo "Build context ready"
```
✅ **Expected**: No critical errors
❌ **If fails**: Check build contexts and Dockerfiles

### 6. **Container Startup Test**
```bash
# Try starting one lightweight service
docker-compose --profile polygon up -d --no-build
sleep 5
docker-compose ps
```
✅ **Expected**: Container status "Up" or "running"
❌ **If fails**: Check logs with `docker-compose logs polygon-io-server`

### 7. **API Key Runtime Verification**
```bash
# Check if container can access environment variables
docker-compose exec polygon-io-server env | grep API_KEY
```
✅ **Expected**: API_KEY variables visible inside container
❌ **If fails**: Check env_file path and .env file location

### 8. **Network Connectivity**
```bash
# Verify network is created
docker network ls | grep axiom-financial-mcp
```
✅ **Expected**: Network exists
❌ **If fails**: Check network configuration

### 9. **Cleanup Test**
```bash
# Ensure containers can be stopped properly
docker-compose down
```
✅ **Expected**: Clean shutdown
❌ **If fails**: Force removal with `docker-compose down -v --remove-orphans`

## 📋 Code/Demo Validation Checklist

### 1. **Import Tests**
```bash
# Test all moved demo files can still import correctly
python -c "import sys; sys.path.insert(0, '.'); from demos import simple_demo"
```
✅ **Expected**: No import errors
❌ **If fails**: Fix import paths

### 2. **Demo Execution**
```bash
# Run simple demos to ensure they work from new location
python demos/simple_demo.py --dry-run
```
✅ **Expected**: Demo runs without errors
❌ **If fails**: Fix file paths or dependencies

### 3. **Documentation Links**
```bash
# Verify all markdown links are valid
find . -name "*.md" -exec grep -H "\[.*\](.*)" {} \; | head -20
```
✅ **Expected**: Links point to correct relocated files
❌ **If fails**: Update broken links

## 🎯 Essential Pre-Completion Checks

Before marking ANY task complete:

- [ ] All syntax validations pass
- [ ] Environment variables load correctly
- [ ] API keys are accessible in containers
- [ ] At least one service starts successfully
- [ ] Cleanup works properly
- [ ] Documentation is updated and links work
- [ ] Code can be imported/executed from new locations

## 🚨 Red Flags - DO NOT COMPLETE if:

- ❌ Docker compose config shows errors
- ❌ API keys not loading into containers
- ❌ Services fail to start
- ❌ Environment variables return empty
- ❌ Demo files throw import errors
- ❌ Documentation links are broken

## 💡 Quick Validation Script

Create this file: `scripts/validate-docker-setup.sh`
```bash
#!/bin/bash
set -e

echo "🧪 Running Docker Compose Validation..."

cd axiom/integrations/data_sources/finance/mcp_servers

# 1. Syntax check
echo "1️⃣ Validating YAML syntax..."
docker-compose config > /dev/null && echo "✅ YAML valid" || exit 1

# 2. Check services
echo "2️⃣ Checking services..."
SERVICES=$(docker-compose config --services | wc -l)
[[ $SERVICES -eq 4 ]] && echo "✅ All 4 services defined" || echo "❌ Expected 4 services, found $SERVICES"

# 3. Check .env file
echo "3️⃣ Checking .env file..."
[[ -f ../../../../.env ]] && echo "✅ .env file exists" || echo "❌ .env file not found"

# 4. Check API keys in config
echo "4️⃣ Checking API keys..."
docker-compose config | grep -q "POLYGON_API_KEY" && echo "✅ API keys configured" || echo "❌ API keys missing"

echo ""
echo "🎉 Basic validation complete!"
echo "Next: Run 'docker-compose --profile polygon up -d' to test actual container startup"
```

## 📝 Testing Log Template

When testing, record results:
```
Date: [DATE]
Task: [TASK DESCRIPTION]

✅ Syntax validation: PASS/FAIL
✅ Env var loading: PASS/FAIL  
✅ API keys verified: PASS/FAIL
✅ Service startup: PASS/FAIL
✅ Container logs: CLEAN/ERRORS
✅ Network created: PASS/FAIL
✅ Cleanup: PASS/FAIL

Notes: [ANY ISSUES OR OBSERVATIONS]
```

Remember: **Working code is better than fast completion!**