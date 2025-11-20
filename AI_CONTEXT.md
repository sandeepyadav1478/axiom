# Axiom Project - AI Assistant Context

## 🚨 READ THIS FIRST - MANDATORY

**Before working on this project, you MUST read [`PROJECT_RULES.md`](PROJECT_RULES.md)**

This document contains 7 STRICT RULES that are non-negotiable for all work on this codebase.

---

## Quick Project Overview

**What**: Institutional quantitative finance platform with 60 ML models
**Tech Stack**: Python 3.13, PyTorch, 4 databases, 30 MCP servers, DSPy
**Hardware**: RTX 4090 GPU, CUDA 12.8
**Key Config**: `.env` file with 11 API providers

---

## Critical Rules Summary

1. **NEVER change directory** - Always work from `/home/sandeep/pertinent/axiom`
2. **`.env` is mandatory** - Created early via `setup_environment.py`
3. **Use full relative paths** - No `cd` commands
4. **Proceed on terminal undefined** - Command already completed
5. **NEVER push to main** - Feature branches + PRs only
6. **No temp docs** - No .md bloat
7. **Verify venv active** - Check `which python` first

**Full details in [`PROJECT_RULES.md`](PROJECT_RULES.md)**

---

## System Status

**To verify system health, run**:
```bash
python system_check.py
```

Expected: 6/6 components PASS (Databases, MCP Servers, MCP Clients, GPU, API Keys, DSPy)

---

## Key Files

**Configuration**:
- `.env` - API keys and settings (209 lines)
- `.env.example` - Template
- `setup_environment.py` - Validates setup

**Infrastructure**:
- `axiom/database/docker-compose.yml` - 4 databases
- `axiom/mcp/docker-compose.yml` - 12 MCP servers
- `system_check.py` - Health verification

**Documentation**:
- `PROJECT_RULES.md` - **READ THIS FIRST**
- `README.md` - Project overview
- `docs/QUICKSTART.md` - Setup guide
- `SESSION_HANDOFF_NOV_06_2025.md` - Last session

---

## Current State

**Operational**:
- ✅ Python 3.13.9 + 383 packages
- ✅ RTX 4090 GPU operational
- ✅ 4 databases running
- ✅ 12 MCP servers (Python modules)
- ✅ 2 MCP clients working
- ✅ DSPy v3.0.3
- ✅ 7 API keys configured

**Needs Work**:
- MCP Docker containers (Dockerfiles need path updates)
- Only 1/12 Dockerfiles updated (pricing_greeks)

---

## For New Sessions

**Start here**:
1. Read [`PROJECT_RULES.md`](PROJECT_RULES.md)
2. Check current branch: `git branch`
3. Verify venv: `which python`
4. Run health check: `python system_check.py`
5. Review last handoff: `SESSION_HANDOFF_NOV_06_2025.md`

**Before making changes**:
1. Create feature branch: `git checkout -b feature/your-feature`
2. Ensure venv active
3. Stay in project root
4. Test changes before committing

---

**REMEMBER**: This is a production codebase with strict workflow rules. Follow [`PROJECT_RULES.md`](PROJECT_RULES.md) without exception!