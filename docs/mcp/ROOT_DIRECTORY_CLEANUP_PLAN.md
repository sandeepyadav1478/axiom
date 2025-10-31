# Root Directory Cleanup Plan

## 🎯 Goal
Organize 60+ root files into professional directory structure

## 📋 Current Situation
Root directory has ~60 markdown files + test scripts mixed together

## ✅ Proposed Organization

### Files to KEEP in Root (Essential Only)
```
/
├── README.md                    ← Main project README
├── LICENSE                      ← License file
├── pyproject.toml              ← Project configuration
├── requirements.txt            ← Dependencies
├── requirements-derivatives.txt ← Derivatives deps
├── install.sh                  ← Installation script
├── .gitignore                  ← Git configuration
├── .env.example                ← Environment template
├── uv.lock                     ← Lock file
└── .python-version             ← Python version
```

### Files to MOVE

#### 1. Session Handoffs → `docs/sessions/`
- SESSION_COMPLETE_HANDOFF_2025_10_30.md
- SESSION_COMPLETE_HANDOFF.md
- SESSION_HANDOFF_DERIVATIVES.md
- SESSION_HANDOFF_MCP_PROFESSIONAL_STRUCTURE.md
- SESSION_OCT_30_MCP_VALIDATION_COMPLETE.md
- SESSION_SUMMARY_COMPREHENSIVE_AGENT_REBUILD.md
- SESSION_VERIFICATION_AND_FIXES.md
- SESSION_WORK_COMPLETE_OCT_29.md
- SESSION_WORK_LOG.txt
- THREAD_COMPLETE_HANDOFF_OCT_29.md
- THREAD_HANDOFF_COMPLETE_SESSION.md
- THREAD_HANDOFF_FOR_NEXT_SESSION.md

#### 2. Old Status/Summary Files → `docs/archive/`
- 35_MODELS_MILESTONE_ACHIEVED.md
- 60_MODELS_ACHIEVEMENT.md
- BATCH_1_IMPLEMENTATION_COMPLETE.md
- BATCHES_1_AND_2_MILESTONE_COMPLETE.md
- COMPLETE_PROJECT_STATUS.md
- COMPLETE_SESSION_ACHIEVEMENT.md
- COMPLETE_SESSION_FINAL.md
- COMPLETE_SESSION_SUMMARY_2025_10_29.md
- COMPLETE_WORK_SUMMARY_OCT_29.md
- COMPREHENSIVE_FINAL_SUMMARY.md
- DERIVATIVES_COMPLETE_SUMMARY.md
- EXTERNAL_LIBRARIES_INTEGRATION_SUMMARY.md
- FINAL_COMPLETE_WORK_LOG.md
- FINAL_COMPREHENSIVE_WORK_SUMMARY.md
- FINAL_MODEL_COUNT.txt
- FINAL_SESSION_COMPREHENSIVE_SUMMARY.md
- FINAL_SESSION_STATUS.md
- FINAL_WORK_SUMMARY.md
- MARATHON_SESSION_COMPLETE.md
- M&A_QUANTITATIVE_MODELS_SUMMARY.md
- PROJECT_COMPLETION_SUMMARY.md
- PROJECT_STATUS_2025_10_29.md
- README_COMPLETE_WORK.md
- README_SESSION_COMPLETE.md
- TEST_FIXES_SUMMARY.md
- ULTIMATE_ACHIEVEMENT_18_MODELS.md
- VERIFICATION_AND_FIXES_COMPLETE.md
- VERIFICATION_REPORT.md
- WORK_CONTINUES.md
- WORK_SUMMARY.txt

#### 3. Current Status Files → `docs/status/`
- CURRENT_PROJECT_STATUS.md
- CURRENT_STATUS.md
- STATUS.txt
- NEXT_PHASE_PLAN.md
- NEXT_PHASE_ROADMAP.md
- NEXT_SESSION_START_HERE.md

#### 4. MCP Documentation → `docs/mcp/`
- ALL_12_MCP_SERVERS_RUNNING.md
- MCP_RESTRUCTURE_IMPLEMENTATION_PLAN.md
- MCP_SERVERS_EXPLAINED.md
- MCP_STRUCTURE_ANALYSIS_AND_PROPOSAL.md
- MCP_VALIDATION_STATUS.md
- PROFESSIONAL_MCP_STRUCTURE_COMPLETE.md

#### 5. Platform Documentation → `docs/platform/`
- PLATFORM_CAPABILITIES_2025.md
- PLATFORM_READY_FOR_CLIENTS.md
- PRODUCTION_PLATFORM_COMPLETE.md
- PROFESSIONAL_AGENT_SYSTEM_HANDOFF.md
- MASTER_INDEX.md
- ML_MODELS_MA_INTEGRATION_GUIDE.md
- RESEARCH_IMPLEMENTATION_PRIORITIES.md

#### 6. Guides → Keep in `guides/` or move to `docs/guides/`
- FIX_MAIN_BRANCH_v2.md
- GENERATE_CHARTS.md
- URGENT_RECOVERY.md

#### 7. Test Scripts → `scripts/` or `tests/`
- test_all_12_mcp_servers.py
- test_all_servers_verified.sh
- test_fixes_verification.py
- test_ma_models_simple.py
- test_mcp_container.sh
- test_mcp_imports.py
- test_mcp_queries.sh
- test_mcp_simple.py
- validate_ma_models.py
- verify_portfolio_optimization.py
- fix_all_nn_imports.sh

## 📊 Summary

**Files in Root**: 85
**After Cleanup**: ~10 essential files
**Files Organized**: 75+

**Result**: Professional, clean root directory!