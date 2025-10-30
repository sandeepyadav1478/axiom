# MCP Complete Restructure - Implementation Plan

## 🎯 Goal
Create professional `axiom/mcp/` structure with clear clients and servers separation

## 📋 Phase-by-Phase Plan

### PHASE 1: Create New Structure (5 min)
```
axiom/mcp/
├── README.md (master guide)
├── clients/ (all MCP clients)
└── servers/ (all MCP servers)
    ├── public/ (9 servers for Claude)
    ├── internal/ (3 servers for system)
    └── shared/ (infrastructure)
```

### PHASE 2: Move Server Files (15 min)
Move from `axiom/mcp_servers/` to `axiom/mcp/servers/`:
- trading/ → servers/public/trading/
- analytics/ → servers/public/analytics/
- compliance/ → servers/public/compliance/
- internal/ → servers/internal/
- shared/ → servers/shared/

### PHASE 3: Move Client Files (10 min)
Consolidate into `axiom/mcp/clients/`:
- integrations/mcp_servers/ → mcp/clients/external_integrations/
- mcp_clients/ → mcp/clients/data_sources/

### PHASE 4: Update Imports (30 min)
Update all imports from:
```python
from axiom.mcp_servers.shared import ...
from axiom.integrations.mcp_servers import ...
```

To:
```python
from axiom.mcp.servers.shared import ...
from axiom.mcp.clients.external_integrations import ...
```

### PHASE 5: Update Docker Files (10 min)
Update docker-compose.yml paths

### PHASE 6: Test & Verify (15 min)
- Run import tests
- Start containers
- Verify all working

### PHASE 7: Documentation (10 min)
Create master README with clear hierarchy

## ⏱️ Total Estimated Time: ~1.5 hours

## 🔄 Rollback Plan
If anything breaks:
```bash
git reset --hard HEAD~1
git checkout main
```

## ✅ Success Criteria
- [ ] All imports working
- [ ] All 12 containers start
- [ ] All tests pass
- [ ] Clear documentation
- [ ] Professional structure

Let's begin!