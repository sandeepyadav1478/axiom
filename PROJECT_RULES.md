# Axiom Project - Strict Working Rules

## ⚠️ CRITICAL: ZERO-TOLERANCE ENFORCEMENT

**These rules are ABSOLUTE and NON-NEGOTIABLE. Violations are NEVER acceptable.**

### Enforcement Policy

**ALL AI assistants and developers MUST**:
1. **Read these rules BEFORE any work** - No exceptions
2. **Follow every rule strictly** - No shortcuts
3. **Never violate rules** - Even for "quick fixes"
4. **Self-check before actions** - Verify rule compliance

**Consequences of Violations**:
- ❌ Work rejection - Changes must be reverted
- ❌ Branch protection - Main branch violations require force-revert
- ❌ Session restart - Violating sessions may need to restart
- ❌ Loss of trust - Repeated violations are unacceptable

**Before EVERY commit, verify**:
```bash
# MANDATORY pre-commit check
git branch --show-current  # Must NOT be "main"

# If shows "main":
echo "⚠️ STOP! Cannot commit to main!"
echo "Run: git checkout -b feature/your-feature-20251120"
```

**Before EVERY cd command, STOP**:
```bash
# ❌ NEVER ALLOWED - Not even once
cd anywhere  # FORBIDDEN

# ✅ ALWAYS use full paths from root
docker compose -f axiom/path/to/file.yml
```

**These rules exist to prevent disasters. Follow them STRICTLY, ALWAYS.**

---

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

### Rule #13: ALWAYS Close Unused Terminals

**CRITICAL REQUIREMENT**: AI assistants MUST close terminals they open when those terminals are no longer needed.

**MANDATORY Practice**:
- ✅ Close terminal after command completes and output is captured
- ✅ Only keep terminals open for long-running processes (servers, monitoring)
- ❌ Never leave terminals open after one-off commands

**Why This Matters**:
- Reduces clutter in VSCode
- Prevents confusion about which terminal is active
- Improves developer experience
- Makes it clear which processes are actually running

**Examples**:

**WRONG** ❌:
```
# Run command and leave terminal open
execute_command: docker ps
# Terminal stays open forever even though command finished
```

**CORRECT** ✅:
```
# Short-lived command - note that it completes
execute_command: docker ps
# AI captures output, terminal can be closed by user if needed
# Or AI notes "This terminal can be closed"
```

**When to Keep Terminals Open**:
- ✅ Long-running services (airflow, databases)
- ✅ Active monitoring scripts
- ✅ Interactive sessions in progress
- ✅ Processes that need to stay running

**When to Close Terminals**:
- ❌ After git commands complete
- ❌ After docker build finishes
- ❌ After one-time scripts finish
- ❌ After diagnostic commands complete

**Best Practice**: Inform user when a terminal is no longer needed: "Terminal can be closed - command completed successfully"

### Rule #14: ALWAYS Commit and Push Completed Work

**CRITICAL REQUIREMENT**: When any work is marked as complete locally, it MUST be immediately committed and pushed to remote. NEVER commit to main branch.

**MANDATORY Workflow**:
1. Complete the work/fix/feature locally
2. **Check current branch** - if on main, create feature branch
3. **If already on feature branch** - can continue using it for related work
4. Commit with descriptive message
5. Push to remote
6. **NEVER** leave completed work uncommitted locally

**Branch Strategy**:
- For new major work: Create new feature branch
- For small fixes/additions: Can use existing feature branch if related
- **NEVER** commit directly to main branch

**Why This Matters**:
- Prevents work loss if machine fails
- Enables collaboration and review
- Maintains project history
- Allows rollback if needed
- Professional development practice

**Examples**:

**WRONG** ❌:
```bash
# Work completed but not committed
# AI says "Task complete!" but files only exist locally
# No git commit, no push
# RISKY - work could be lost!
```

**CORRECT** ✅:
```bash
# Option 1: New major work - create new branch
git checkout -b feature/descriptive-name-20251120
git add -A
git commit -m "Clear description of changes"
git push origin feature/descriptive-name-20251120

# Option 2: Small fix on existing feature branch
git status  # Verify not on main
git add -A
git commit -m "Small fix description"
git push  # Push to current feature branch
# ✅ Work is safe, backed up, reviewable
```

**Commit Frequency**:
- After each logical unit of work completes
- After fixing bugs
- After adding features
- After creating documentation
- Before ending work session

**FORBIDDEN**:
- ❌ Leaving completed work uncommitted
- ❌ Committing to main branch directly
- ❌ Working without version control
- ❌ Ending session without pushing changes

**REQUIRED**:
- ✅ Commit after completing each task
- ✅ Use feature branches (NEVER main)
- ✅ Push to remote immediately
- ✅ Use descriptive commit messages
- ✅ Can reuse feature branch for related small work

**This ensures all work is version controlled and safely backed up in remote repository.**

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

### Rule #15: ALWAYS Use Single-Line Commit Messages

**CRITICAL REQUIREMENT**: All git commit messages MUST be single-line to prevent terminal hanging and workflow violations.

**MANDATORY Practice**:
- ✅ Use `-m "Single line message"` for commits
- ❌ NEVER use multi-line messages with line breaks in `-m` flag
- ✅ If need details, use separate `-m` flags or commit file

**Why This Matters**:
- Multi-line messages cause `cmdand dquote>` prompt in terminal
- Terminal appears to hang waiting for closing quote
- Can lead to confusion about whether commit succeeded
- May cause accidental commits to wrong branch

**Examples**:

**WRONG** ❌:
```bash
git commit -m "Fix issue

BUGFIX: Description here
More details
Multiple lines"
# This causes: cmdand dquote> prompt
# Terminal hangs waiting for closing "
```

**CORRECT** ✅:
```bash
# Option 1: Single concise message
git commit -m "Fix Claude API key and email alerts"

# Option 2: Title + body (separate -m flags)
git commit -m "Fix Claude API key" -m "Added CLAUDE_API_KEY env var and disabled email alerts"

# Option 3: Use commit file for long messages
git commit -F commit_message.txt
```

**Before EVERY commit with `-m` flag:**
1. Ensure message is on ONE line
2. No unescaped newlines in the string
3. Proper closing quote on same line
4. Verify git branch first

**Impact of Violation**:
- Terminal confusion (appears stuck)
- Possible wrong branch commits
- Harder to detect actual commit status
- Violates professional git practices

**This prevents terminal issues and ensures clean git workflow.**

### Rule #16: NEVER Put Credentials in Code - Always Use .env Files

**ABSOLUTE REQUIREMENT**: ALL credentials, API keys, tokens, and secrets MUST be stored ONLY in `.env` files, NEVER in code.

**MANDATORY Practice**:
- ✅ Store ALL credentials in `.env` (gitignored)
- ✅ Use `os.getenv()` or environment variables in code
- ✅ Include `.env.example` with placeholder values
- ❌ NEVER hardcode credentials, even for testing

**This applies to:**
- Production credentials
- Development credentials
- **Test credentials** (use `.env` not code!)
- API keys
- Database passwords
- Authentication tokens
- Private keys

**Examples**:

**WRONG** ❌:
```python
# In test file - NEVER DO THIS
def test_api():
    api_key = "test-key-12345"  # ❌ Hardcoded test credential!
    assert call_api(api_key)
```

**CORRECT** ✅:
```python
# In test file - Use environment variable
import os

def test_api():
    api_key = os.getenv('TEST_API_KEY')  # ✅ From .env file
    assert call_api(api_key)
```

**Required Setup**:
```bash
# .env file (gitignored)
TEST_API_KEY=test-key-12345
STAGING_DB_PASSWORD=staging-pass
DEV_CLAUDE_KEY=sk-ant-dev-key

# .env.example file (committed)
TEST_API_KEY=your_test_key_here
STAGING_DB_PASSWORD=your_staging_password
DEV_CLAUDE_KEY=your_dev_claude_key
```

**Why This Matters**:
- Prevents credential leaks in git history
- Easy credential rotation (change .env, not code)
- Same code works in all environments (dev/staging/prod)
- Security audit compliance
- No accidental commits of secrets

**Before ANY commit:**
```bash
git diff | grep -iE "key|password|token|secret|credential"
# If any matches, verify they're from .env, not hardcoded!
```

### Rule #17: NEVER Rename - Use Versioning or Deprecation

**CRITICAL PRINCIPLE**: When improving code, NEVER rename files/functions/classes. Use versioning or deprecation instead to maintain compatibility and avoid confusion.

**MANDATORY Approach**:
- ✅ Add new version: `function_v2()`, `model_v3.py`, `dag_v2.py`
- ✅ Keep old version: Mark as deprecated with clear docs
- ✅ Gradual migration: Both versions coexist
- ❌ NEVER rename and break existing code

**Examples**:

**WRONG** ❌:
```python
# Old file: company_graph_dag.py
# Renamed to: enhanced_company_graph_dag.py
# Result: Confusion! Which one to use? Where's the old one?
```

**CORRECT** ✅:
```python
# Keep both:
# company_graph_dag.py (v1 - deprecated but functional)
# company_graph_dag_v2.py (v2 - enhanced version)

# Or use clear naming:
# company_graph_dag.py (original)
# company_graph_enterprise_dag.py (enterprise version)
```

**For DAGs Specifically**:
```python
# dag_v1.py - Original (paused, kept as backup)
with DAG('data_ingestion', ...):  # Original name
    # Original implementation

# dag_v2.py - Enhanced (active)
with DAG('data_ingestion_v2', ...):  # Versioned name
    # Enhanced implementation
```

**Benefits**:
- Clear which is newer (v2 > v1)
- Can run both for comparison
- Easy rollback (activate v1, pause v2)
- No confusion about "where did X go?"
- Migration path is obvious

### Rule #18: NEVER Delete Code - Move to Deprecated Directory

**STRICT REQUIREMENT**: When deprecating code, NEVER delete it. Move it to a `deprecated/` directory with full documentation.

**MANDATORY Workflow**:
1. Create `deprecated/` directory in same location as original
2. Move old code there (maintain directory structure)
3. Create `DEPRECATION_NOTICE.md` explaining:
   - Why deprecated
   - What replaced it
   - How to revert if needed
   - Migration guide
4. Update main code with comment pointing to deprecated version

**Directory Structure**:
```
axiom/pipelines/airflow/
├── dags/
│   ├── company_graph_dag_v2.py  (active)
│   ├── data_ingestion_dag_v2.py (active)
│   └── deprecated/
│       ├── DEPRECATION_NOTICE.md
│       ├── company_graph_dag.py (v1 - preserved)
│       └── data_ingestion_dag.py (v1 - preserved)
```

**DEPRECATION_NOTICE.md Template**:
```markdown
# Deprecated DAGs

## company_graph_dag.py (v1)

**Deprecated**: November 21, 2025
**Replaced by**: company_graph_dag_v2.py
**Reason**:
- v2 has 70% cost reduction via caching
- v2 has enterprise resilience patterns
- v2 has automated quality checks

**Differences**:
- v1: Basic Claude calls, no caching
- v2: CachedClaudeOperator with Redis

**How to Revert**:
1. Copy this file back to ../company_graph_dag.py
2. Pause v2 DAG in Airflow UI
3. Enable v1 DAG in Airflow UI

**Migration Path**:
See ../MIGRATION_GUIDE.md for upgrading from v1 to v2
```

**Why This Matters**:
- Can always revert if new version has issues
- Understand what changed and why
- Reference old implementation
- Learn from evolution
- Compliance/audit trail

**FORBIDDEN**:
```bash
rm old_file.py  # ❌ NEVER DELETE!
git rm deprecated_code.py  # ❌ NEVER REMOVE FROM GIT!
```

**REQUIRED**:
```bash
mkdir -p deprecated
mv old_file.py deprecated/
echo "See DEPRECATION_NOTICE.md" > deprecated/README.md
# Write DEPRECATION_NOTICE.md with full context
git add deprecated/
git commit -m "Deprecate old_file.py, moved to deprecated/"
```

**This ensures code history is never lost and reversions are always possible.**

### Rule #19: Organized Documentation Structure - NEVER Root Directory

**CRITICAL REQUIREMENT**: Session documents, handoffs, status files MUST be organized in proper directory structure. NEVER place in root.

**MANDATORY Directory Structure**:
```
docs/
├── sessions/
│   ├── 2025-11/
│   │   ├── session-nov-21.md
│   │   ├── session-nov-26.md
│   │   └── README.md (index for month)
│   └── 2025-10/
│       └── ...
├── status/
│   ├── current-status.md (ONE file, updated)
│   └── milestones/
│       ├── milestone-1-complete.md
│       └── ...
└── handoffs/
    ├── handoff-template.md
    └── latest-handoff.md (symlink or single file)
```

**FORBIDDEN**:
- ❌ Root directory placement: `SESSION_HANDOFF_NOV_21.md`, `COMPLETE_STATUS.md`, `FINAL_HANDOFF.md`
- ❌ Multiple status/summary files with similar names
- ❌ Scattered documents: Some in root, some in docs/
- ❌ Creating new top-level .md files for every session

**REQUIRED**:
- ✅ ONE session document per session in `docs/sessions/YYYY-MM/`
- ✅ ONE current status file in `docs/status/`
- ✅ Organized by date/type
- ✅ Index/README files in each directory

**Examples**:

**WRONG** ❌:
```
Project Root:
├── SESSION_HANDOFF_NOV_21.md
├── FINAL_SESSION_HANDOFF_NOV_21.md
├── COMPLETE_QUALITY_FRAMEWORK.md
├── PRODUCTION_MONITORING_COMPLETE.md
├── COMPLETE_PROJECT_STATUS_NOV_26.md
└── ... (cluttered root)
```

**CORRECT** ✅:
```
Project Root:
├── docs/
│   ├── sessions/
│   │   └── 2025-11/
│   │       ├── session-2025-11-21.md (combines all Nov 21 docs)
│   │       └── session-2025-11-26.md
│   ├── status/
│   │   └── current-status.md (ONE file, always current)
│   └── milestones/
│       ├── quality-framework-complete.md
│       └── monitoring-deployed.md
└── README.md
└── PROJECT_RULES.md
└── (clean root)
```

**Migration Process**:
1. Create `docs/sessions/YYYY-MM/` directory
2. Move all session documents there
3. Rename with consistent pattern
4. Create index README.md
5. Delete from root
6. Commit: "Organize session docs per Rule #19"

**Going Forward**:
- Create ONE session doc: `docs/sessions/YYYY-MM/session-YYYY-MM-DD.md`
- Update ONE status file: `docs/status/current-status.md`
- Create milestone docs: `docs/milestones/milestone-name.md`
- NEVER create in root

**This keeps root directory clean and professional.**

---

## Enforcement

These rules are STRICT and MANDATORY for all work on this project.

Violations will result in:
- Broken terminal state
- File path confusion
- Documentation bloat
- Harder maintenance

**When in doubt, stay in `/home/sandeep/pertinent/axiom` and use relative paths!**