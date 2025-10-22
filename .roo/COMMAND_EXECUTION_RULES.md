# 🚨 Critical Command Execution Rules

## ⛔ NEVER USE `cd` IN COMMANDS

**RULE**: Always execute commands from the project root directory (`/Users/sandeep.yadav/work/axiom`)

### ❌ WRONG Examples:
```bash
cd some/directory && command
cd path && docker-compose up
```

### ✅ CORRECT Examples:
```bash
# Use relative paths from root
docker-compose -f axiom/integrations/data_sources/finance/mcp_servers/docker-compose.yml config

# Use cwd parameter in execute_command
<execute_command>
<command>docker-compose config</command>
<cwd>axiom/integrations/data_sources/finance/mcp_servers</cwd>
</execute_command>

# For file operations, use full relative paths
cat axiom/integrations/data_sources/finance/mcp_servers/.env
ls -la demos/
python demos/simple_demo.py
```

## 📋 Reasons:
1. **Consistency**: Always know current working directory
2. **Reliability**: No path confusion or lost context
3. **Debugging**: Easier to track command execution
4. **Script Safety**: Commands work regardless of where they're run from

## 🔧 Implementation:
- Use `-f` flag for docker-compose to specify file location
- Use full relative paths from project root for all commands
- Use `cwd` parameter in execute_command when needed
- Never chain commands with `cd &&`

This rule applies to ALL commands, no exceptions!