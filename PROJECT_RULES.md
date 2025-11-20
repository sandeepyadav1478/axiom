# Axiom Project - Strict Working Rules

## 🚨 MANDATORY RULES - NO EXCEPTIONS

### Rule #1: NEVER Change Working Directory

**STRICT REQUIREMENT**: All commands MUST be executed from project root:
```
/home/sandeep/pertinent/axiom
```

**FORBIDDEN**:
- ❌ `cd` commands in any form
- ❌ `cwd` parameter in execute_command
- ❌ Changing to subdirectories like axiom/database, axiom/mcp, etc.

**ALLOWED**:
- ✅ Use full relative paths from root: `axiom/database/docker-compose.yml`
- ✅ Commands with path prefixes: `docker compose -f axiom/database/docker-compose.yml`
- ✅ Multi-command with paths: `ls axiom/mcp/servers && cat axiom/database/models.py`

**Examples**:

**WRONG**:
```bash
cd axiom/database && docker compose up -d  # ❌ Changes directory
```

**CORRECT**:
```bash
docker compose -f axiom/database/docker-compose.yml up -d  # ✅ Stays in root
```

**WRONG**:
```bash
cd axiom/mcp && docker compose build  # ❌ Changes directory
```

**CORRECT**:
```bash
docker compose -f axiom/mcp/docker-compose.yml build  # ✅ Stays in root
```

### Rule #2: .env File MUST Be Created Early

**REQUIREMENT**: `.env` file creation is step #3 in setup:
1. Clone repo
2. Create venv
3. **Run `python setup_environment.py`** (creates .env, validates)
4. Install packages
5. Start databases

**Why**: System has fallback defaults that can mask missing .env file. This is dangerous for production.

### Rule #3: Always Use Full Paths for File Operations

**REQUIREMENT**: When reading/writing files, always use paths relative to `/home/sandeep/pertinent/axiom`

**Examples**:
- ✅ `axiom/database/models.py`
- ✅ `demos/demo_complete_data_infrastructure.py`
- ❌ `../axiom/models.py` (relative navigation)
- ❌ `/tmp/test.py` (absolute paths outside project)

### Rule #4: Verify Terminal Output Quickly

**REQUIREMENT**: When terminal shows `<VSCE exit code is undefined>`, command has completed. Proceed immediately, don't wait.

This is a VSCode terminal communication issue, not a command execution issue.

### Rule #5: NEVER Push Directly to Main Branch

**STRICT REQUIREMENT**: All code changes MUST go through feature branches and pull requests.

**Git Workflow (MANDATORY)**:
```bash
# 1. Create feature branch
git checkout -b feature/descriptive-name

# 2. Make changes and commit
git add .
git commit -m "Clear commit message"

# 3. Push to feature branch
git push origin feature/descriptive-name

# 4. Create PR to merge to main
# Only merge to main after PR review/approval
```

**FORBIDDEN**:
- ❌ `git push origin main` (direct push to main)
- ❌ `git commit` on main branch
- ❌ Working directly on main branch

**ALLOWED**:
- ✅ Create feature branches
- ✅ Push to feature branches
- ✅ Merge to main via approved PRs only

**Why**: Protects main branch integrity, enables code review, maintains project history.

### Rule #6: No Temporary Documentation Files

**FORBIDDEN**: Creating multiple .md files for every small task

**ALLOWED**: 
- Official documentation in `docs/`
- Session handoff documents (1 per session)
- Critical architecture documents

**FORBIDDEN**:
- ❌ `NEW_WORKSTATION_SETUP.md`, `COMPLETE_SYSTEM_STATUS.md`, etc. for setup tasks
- ❌ Multiple summary documents for same topic
- ❌ Temporary test/validation .md files

Use README.md or existing docs instead.

### Rule #7: ALWAYS Verify Virtual Environment is Activated

**STRICT REQUIREMENT**: Before executing ANY Python command, verify venv is activated.

**Check Method**:
```bash
# Verify venv is active
which python  # Should show: /home/sandeep/pertinent/axiom/.venv/bin/python
```

**If NOT activated**:
```bash
source .venv/bin/activate  # Manual activation
# Or rely on autoenv if configured
```

**FORBIDDEN**:
- ❌ Running Python commands without venv active
- ❌ Using system Python instead of project venv
- ❌ Assuming venv is active without checking

**REQUIRED for every terminal session**:
```bash
# At start of session, ALWAYS:
python --version  # Should show: Python 3.13.9
which python      # Should show: .venv/bin/python
```

**Why**:
- Using system Python = wrong packages
- Missing dependencies = runtime errors
- Package version conflicts
- Corrupted environment

**Note**: Autoenv (`.autoenv` file) auto-activates venv on `cd` to project root. If not working, always activate manually first.

### Rule #8: FIX ROOT CAUSES - PREVENT RECURRENCE

**CRITICAL PRINCIPLE**: Every fix must address the ROOT CAUSE, not just symptoms.

**MANDATORY Approach**:
- **Identify**: What's the underlying cause, not just the symptom?
- **Design**: How can we prevent this class of issues permanently?
- **Implement**: Fix it once, correctly, completely
- **Validate**: Ensure it can't happen again

**Examples from This Project**:

**Bad** ❌: Fix one Dockerfile path issue
**Good** ✅: Create script to update all 12 Dockerfiles + document pattern

**Bad** ❌: Add missing import to one file
**Good** ✅: Use sed to fix all 10 files + ensure pattern is clear

**Bad** ❌: Manually fix container healthcheck
**Good** ✅: Update docker-compose.yml template + document why

**FORBIDDEN**:
- ❌ Band-aid fixes that need repeating
- ❌ Fixing symptoms instead of causes
- ❌ "Quick fixes" that break later
- ❌ Solving same problem multiple times

**REQUIRED**:
- ✅ Systematic solutions (scripts, templates, patterns)
- ✅ Documentation of WHY (prevent recurrence)
- ✅ Validation that fix is complete
- ✅ Future-proof design

**When fixing bugs**:
1. Ask: "Why did this happen?"
2. Ask: "How can this entire class of bugs be prevented?"
3. Implement systemic fix
4. Document for future developers
5. Add validation/checks to prevent regression


### Rule #9: NEVER Commit Credentials to Git

**CRITICAL SECURITY RULE**: Credentials, API keys, passwords, and secrets MUST NEVER be committed to git.

**MANDATORY Practice**:
- ✅ Store ALL credentials in `.env` file (already gitignored)
- ✅ Use environment variables: `os.getenv('API_KEY')`
- ✅ Provide `.env.example` template (with placeholder values)
- ✅ Verify `.gitignore` includes all credential files

**FORBIDDEN**:
- ❌ Hardcoding API keys in source code
- ❌ Committing .env file to git
- ❌ Passwords in docker-compose.yml (use ${VARIABLE} references)
- ❌ Database credentials in Python files
- ❌ Secrets in configuration files

**Required Files**:
```
.env                    # Real credentials (gitignored) ✅
.env.example            # Template format (committed) ✅
.gitignore              # Must include .env ✅
```

**Examples**:

**WRONG** ❌:
```python
# In source code - NEVER DO THIS
ANTHROPIC_API_KEY = "sk-ant-api03-xxxxx"  # ❌ Exposed in git!
DB_PASSWORD = "axiom_secret_password"     # ❌ Security breach!
```

**CORRECT** ✅:
```python
# Use environment variables
import os
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')  # ✅ From .env
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')        # ✅ Secure
```

**Validation Before Commit**:
```bash
# ALWAYS check before committing
git diff | grep -i "password\|api_key\|secret\|token"

# If any matches found, DO NOT COMMIT
```

### Rule #10: Credential Files Must Have Example Templates

**REQUIREMENT**: Every file containing credentials MUST have a corresponding `.example` template file.

**Pattern**:
```
.env                  # Real credentials (gitignored)
.env.example          # Template (committed to git)

config.json           # Real config (gitignored if contains secrets)
config.json.example   # Template (committed)
```

**Template File Requirements**:
1. **Same structure** as real file
2. **Placeholder values** (not real credentials)
3. **Clear comments** explaining what each value is for
4. **Example format** showing expected value types

**Example - .env.example**:
```bash
# Database Credentials
POSTGRES_USER=axiom
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=axiom_finance

# API Keys (get from provider websites)
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxx
POLYGON_API_KEY=your_polygon_key_here

# Format: Keep variable names, replace values with descriptive placeholders

### Rule #11: ALWAYS Leverage Existing Open-Source Solutions

**CRITICAL PRINCIPLE**: Never reinvent the wheel. Use battle-tested open-source tools instead of writing custom code.

**MANDATORY Approach**:
1. **Search first**: Is there an open-source solution already?
2. **Evaluate**: Does it meet 80%+ of requirements?
3. **Integrate**: Use existing tools via pip/docker/npm
4. **Customize**: Only add small wrapper/config, not full implementation

**FORBIDDEN**:
- ❌ Writing custom implementations of common functionality
- ❌ Building features that exist in popular libraries
- ❌ Reinventing standard tools (logging, monitoring, viz, etc.)
- ❌ "Not invented here" syndrome

**REQUIRED**:
- ✅ Search GitHub, PyPI, npm before coding
- ✅ Use established libraries (Apache, Linux Foundation projects preferred)
- ✅ Contribute improvements back to open-source if needed
- ✅ Write ONLY unique business logic

**Examples from This Project**:

**BAD** ❌: Wrote custom 3D graph viewer (FastAPI + custom viz)
**GOOD** ✅: Should have used:
- Gephi (desktop app, export from Neo4j)
- Graphistry (Docker image, GPU-accelerated)
- yEd (free desktop tool)
- Or Neo4j Bloom (if acceptable)

**BAD** ❌: Writing custom monitoring system
**GOOD** ✅: Use Prometheus + Grafana (industry standard)

**BAD** ❌: Custom job scheduler
**GOOD** ✅: Use Apache Airflow (battle-tested)

**BAD** ❌: Custom streaming solution
**GOOD** ✅: Use Kafka or Redis Streams

**For Neo4j Visualization Specifically**:

**Open-Source Options We Should Use**:
```bash
# Option 1: Gephi (Desktop)
# - Free, powerful, widely used
# - Export from Neo4j, visualize in Gephi
# - No code needed!

# Option 2: Graphistry (Docker)
docker pull graphistry/graphistry-forge-base
# - GPU-accelerated 3D viz
# - Open-source core
# - Docker container ready

# Option 3: NetworkX + Plotly (Python)
pip install networkx plotly
# - Pure Python
# - Existing ecosystem
# - ~20 lines of code vs 200 we wrote

# Option 4: Neo4j Browser (Built-in)
# - Already running at localhost:7474
# - Free, from Neo4j itself
# - No installation needed
```

**The Rule**:
- Search "neo4j 3d visualization open source" BEFORE writing code
- Find: Gephi, Graphistry, etc.
- Use one of those
- Save 90% development time

**How to Check**:
```bash
# Before writing ANY new feature, search:
# 1. GitHub: "topic:neo4j topic:visualization"
# 2. PyPI: search "neo4j graph visualization"
# 3. Docker Hub: search "neo4j visualization"
# 4. Ask Claude: "What are popular open-source tools for [feature]?"

### Rule #12: ALWAYS Use `uv add` for Dependencies

**CRITICAL REQUIREMENT**: All package installations MUST use `uv add`, not `uv pip install`.

**MANDATORY Practice**:
- ✅ `uv add package-name` (updates pyproject.toml + uv.lock)
- ❌ `uv pip install package-name` (doesn't update lock file)

**Why This Matters**:
- `uv add` updates both pyproject.toml and uv.lock
- Lock file ensures reproducible builds
- Other developers get exact same versions
- Prevents "works on my machine" issues

**Examples**:

**WRONG** ❌:
```bash
uv pip install langgraph  # Not tracked in lock file!
uv pip install pyvis      # Dependencies not locked!
```

**CORRECT** ✅:
```bash
uv add langgraph  # Updates pyproject.toml + uv.lock
uv add pyvis      # Locks all dependencies
```

**Lock File Benefits**:
- Exact version reproducibility
- Dependency conflict detection
- Security vulnerability tracking
- Team synchronization
- CI/CD reliability

**When Installing Multiple Packages**:
```bash
# WRONG
uv pip install langgraph langchain-anthropic pyvis

# CORRECT  
uv add langgraph langchain-anthropic pyvis
```

**To sync from lock file** (other developers):
```bash
uv sync  # Installs exact versions from uv.lock
```

**ALWAYS**: After `uv add`, commit both pyproject.toml AND uv.lock to git.

```

**When Custom Code is OK**:
- ✅ Unique business logic (quant models, trading strategies)
- ✅ Integration glue (connecting two systems)
- ✅ Project-specific workflows
- ✅ When truly no existing solution exists

**When Custom Code is NOT OK**:
- ❌ Standard features (viz, monitoring, scheduling, etc.)
- ❌ Common utilities (logging, config, etc.)
- ❌ Infrastructure (databases, queues, etc.)

**This project must maximize leverage of open-source ecosystem** - write ONLY what's unique to Axiom.

```

**REQUIRED When Adding New Secrets**:
1. Add to `.env` (gitignored)
2. Add placeholder to `.env.example` (committed)
3. Update `.gitignore` if new file type
4. Document in setup guide

**Why This Matters**:
- New developers can set up quickly
- No guessing what credentials are needed
- No accidental credential exposure
- Easy credential rotation
- Industry-standard security practice

**This project must be maintenance-free** - fix it once, fix it right, never revisit.

---

## Enforcement

These rules are STRICT and MANDATORY for all work on this project.

Violations will result in:
- Broken terminal state
- File path confusion
- Documentation bloat
- Harder maintenance

**When in doubt, stay in `/home/sandeep/pertinent/axiom` and use relative paths!**