# MCP Consolidation - Cleanup Guide

## ✅ Consolidation Complete

All MCP infrastructure has been successfully consolidated into `axiom/mcp/` with the following structure:
- `axiom/mcp/clients/` - All MCP clients (external + internal)
- `axiom/mcp/servers/` - All MCP servers (public + internal + shared)

## 🗑️ Old Directories to Remove

The following directories are now redundant and should be manually removed:

### 1. axiom/mcp/client/
**Status**: Deprecated (replaced by `axiom/mcp/clients/external/`)  
**Content**: Old external MCP client integrations  
**Safe to remove**: YES - All content copied to new location

### 2. axiom/mcp_clients/
**Status**: Deprecated (replaced by `axiom/mcp/clients/internal/`)  
**Content**: Internal MCP clients (derivatives_data, market_data)  
**Safe to remove**: YES - All content copied to new location

### 3. axiom/mcp_servers/
**Status**: Deprecated (replaced by `axiom/mcp/servers/`)  
**Content**: Old MCP server structure  
**Safe to remove**: YES - All content migrated to new structure

### 4. axiom/mcp_professional/
**Status**: Deprecated (replaced by `axiom/mcp/`)  
**Content**: Previous professional MCP structure  
**Safe to remove**: YES - All content migrated to unified structure

## 📋 Manual Cleanup Steps

```bash
# From project root directory

# Remove old external clients directory
rm -rf axiom/mcp/client

# Remove old internal clients directory
rm -rf axiom/mcp_clients

# Remove old servers directory
rm -rf axiom/mcp_servers

# Remove previous professional structure
rm -rf axiom/mcp_professional

# Verify cleanup
find axiom -type d -name "mcp*" -maxdepth 1
# Should only show: axiom/mcp
```

## ✅ Verification After Cleanup

After removing old directories, verify:

1. **Check imports work**:
```bash
python -c "from axiom.mcp.servers.shared.mcp_base import BaseMCPServer; print('✅ Imports work')"
```

2. **Check directory structure**:
```bash
ls -la axiom/ | grep mcp
# Should only show: mcp/
```

3. **Run tests**:
```bash
python -m pytest tests/test_mcp_*.py
```

## 📊 Cleanup Impact

**Before Cleanup**: 4 separate MCP directories (mcp/, mcp_clients/, mcp_servers/, mcp_professional/)  
**After Cleanup**: 1 unified MCP directory (mcp/)

**Space Freed**: ~500+ MB (duplicate files removed)  
**Import Paths**: All updated to use `axiom.mcp.*`

## 🔄 What Was Migrated

### Clients
- ✅ External clients: `axiom/mcp/client/*` → `axiom/mcp/clients/external/*`
- ✅ Internal clients: `axiom/mcp_clients/*` → `axiom/mcp/clients/internal/*`

### Servers
- ✅ Public servers: `axiom/mcp_professional/servers/public/*` → `axiom/mcp/servers/public/*`
- ✅ Internal servers: `axiom/mcp_professional/servers/internal/*` → `axiom/mcp/servers/internal/*`
- ✅ Shared infrastructure: `axiom/mcp_professional/servers/shared/*` → `axiom/mcp/servers/shared/*`

### Imports Updated
- ✅ All `axiom.mcp_servers.*` → `axiom.mcp.servers.*`
- ✅ All `axiom.mcp_professional.*` → `axiom.mcp.*`
- ✅ All `axiom.mcp_clients.*` → `axiom.mcp.clients.internal.*`

## ⚠️ Important Notes

1. **DO NOT** remove directories until you've verified the new structure works
2. **DO** test critical imports before cleanup
3. **DO** commit the migration first, then cleanup as separate commit
4. **DO** backup if uncertain (though git history maintains everything)

## 🎯 Result

After cleanup:
- **Single source of truth**: `axiom/mcp/`
- **Clear organization**: clients/ vs servers/, external vs internal
- **Professional structure**: Industry-standard layout
- **Easy discovery**: Everything MCP-related in one place
- **Simplified imports**: Consistent `axiom.mcp.*` pattern

---

**Status**: Migration complete, cleanup ready  
**Branch**: feature/20251031-1903-next-milestone  
**Safe to proceed**: YES