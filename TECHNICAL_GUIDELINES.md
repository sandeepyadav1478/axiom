# Axiom Development Technical Guidelines
## Critical Development Practices for Future AI Assistants

### 🚨 **CRITICAL GIT WORKFLOW RULES**

#### **NEVER PUSH TO MAIN BRANCH**
```bash
# ❌ NEVER DO THIS:
git push origin main
git commit -m "changes" && git push origin main

# ✅ ALWAYS DO THIS:
git checkout -b feature/new-enhancement
git commit -m "implement new feature"
git push origin feature/new-enhancement
```

#### **BRANCH MANAGEMENT RULES**
1. **Always create new feature branch** for any changes
2. **Check if existing branch is merged** before reusing
3. **Create new branch if old one merged** to main
4. **Use descriptive branch names** like `feature/ma-enhancement-name`
5. **Never commit directly to main branch**

**Example Workflow:**
```bash
# Check current branch
git branch

# If on main or existing branch merged, create new one
git checkout -b feature/new-ma-enhancement-name

# Make changes, then commit and push to feature branch
git add .
git commit -m "implement enhancement"
git push origin feature/new-ma-enhancement-name
```

### ✅ **COMMIT MESSAGE FORMATTING RULES**

#### **Multi-Line Commit Message Format**
```bash
# ❌ WRONG: Long message on single line
git commit -m "Implement comprehensive M&A enhancements with risk assessment and regulatory compliance and PMI planning and advanced modeling..."

# ✅ CORRECT: Use proper multi-line format
git commit -m "Implement comprehensive M&A enhancements

🚀 NEW FEATURES:
- Risk assessment workflows
- Regulatory compliance automation  
- PMI planning and execution
- Advanced financial modeling

✅ BUSINESS VALUE:
- 90% time savings in risk assessment
- $10-50M annual failed deal prevention
- Professional-grade analysis capabilities

Ready for investment banking operations
"
```

#### **Commit Message Best Practices**
1. **Start with descriptive title** (50 characters or less)
2. **Add blank line** after title
3. **Use bullet points** for detailed descriptions
4. **Include business value** and technical details
5. **End with status** or next steps

### 📁 **DIRECTORY STRUCTURE GUIDELINES**

#### **Proper Organization Patterns**
```
✅ CORRECT STRUCTURE:
axiom/
├── workflows/              # Business workflow modules
├── data_sources/          # Data integration sources
│   ├── finance/           # Financial data providers
│   ├── market/           # Market data sources (future)
│   └── regulatory/       # Regulatory data sources (future)
├── ai_client_integrations/ # AI provider abstractions
└── tools/                 # Tool integrations

❌ AVOID MISLEADING NAMES:
axiom/
├── financial_integrations/  # Misleading - sounds like finance workflows
├── api_integrations/        # Too generic
└── data_providers/          # Unclear scope
```

#### **File Naming Conventions**
- **Base classes**: `base_*_provider.py`
- **Provider implementations**: `provider_name_provider.py`
- **Workflow modules**: `workflow_name.py`
- **Documentation**: `TITLE_IN_CAPS.md`

### 🤖 **AI INTEGRATION PATTERNS**

#### **Provider Pattern (Follow Existing Pattern)**
```python
# Follow the pattern from ai_client_integrations/
# 1. Base abstract class
# 2. Specific provider implementations
# 3. Factory pattern for provider management
# 4. Unified interface for all providers
```

#### **Conservative AI Settings for Investment Banking**
```python
# Always use conservative settings for financial decisions
TEMPERATURE_SETTINGS = {
    "due_diligence": 0.03,     # Ultra-conservative
    "valuation": 0.05,         # Conservative
    "risk_assessment": 0.03,   # Ultra-conservative
    "market_analysis": 0.1     # Moderate
}
```

### 💰 **COST OPTIMIZATION PRIORITIES**

#### **Always Prioritize Cost-Effective Solutions**
```python
# ✅ PREFERRED: Free and affordable data sources
- OpenBB (100% FREE)
- SEC Edgar (100% FREE, government data)
- Alpha Vantage (FREE tier or $49/month)
- Financial Modeling Prep ($15/month)

# ❌ AVOID: Expensive enterprise platforms
- Bloomberg Terminal ($24,000/year)
- FactSet Professional ($15,000/year)
```

#### **Cost-Benefit Analysis Required**
- Always analyze cost vs benefit for new integrations
- Prioritize free and open-source alternatives
- Document cost savings in commit messages

### 🧪 **TESTING AND VALIDATION REQUIREMENTS**

#### **Always Test Before Committing**
```bash
# Required tests before any commit:
source .venv/bin/activate
python tests/validate_system.py     # Must show 7/7 passed
python demo_ma_analysis.py          # Must show 5/5 successful
ruff check .                        # Must show "All checks passed!"
black --check .                     # Must show compliance
```

#### **Import Testing Pattern**
```python
# Always test imports after creating new modules
python -c "
try:
    from axiom.new_module import NewClass
    instance = NewClass()
    print('✅ Import and instantiation successful')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

### 📚 **DOCUMENTATION REQUIREMENTS**

#### **Documentation Organization**
```
docs/
├── ma-workflows/          # M&A workflow guides
├── architecture/          # System architecture docs
└── deployment/           # Deployment guides

# Always update documentation index at docs/README.md
```

#### **Code Documentation Standards**
- **Docstrings**: Required for all workflow modules
- **Type hints**: Use modern Python 3.10+ type annotations
- **Comments**: Explain business logic and investment banking concepts

### ⚠️ **COMMON PITFALLS TO AVOID**

#### **Import Path Issues**
```python
# ❌ Wrong: Old import paths after restructuring
from axiom.financial_integrations import Provider

# ✅ Correct: Updated import paths
from axiom.data_sources.finance import Provider
```

#### **Abstract Method Implementation**
```python
# ❌ Wrong: Incomplete abstract class implementation
class MyProvider(BaseProvider):
    def method1(self):
        pass
    # Missing required abstract methods

# ✅ Correct: Implement ALL abstract methods
class MyProvider(BaseProvider):
    def method1(self):
        pass
    def method2(self):  # All abstract methods implemented
        pass
```

### 🎯 **DEVELOPMENT WORKFLOW CHECKLIST**

**Before Starting Any Enhancement:**
- [ ] Create new feature branch (never work on main)
- [ ] Check if existing branch was merged (create new if merged)
- [ ] Review existing code patterns and follow them
- [ ] Plan cost-effective approach vs expensive alternatives

**During Development:**
- [ ] Follow existing patterns (AI providers, workflow modules, etc.)
- [ ] Use modern Python 3.10+ type annotations
- [ ] Test imports and instantiation frequently
- [ ] Keep cost optimization as priority

**Before Committing:**
- [ ] Test all imports and functionality
- [ ] Run validation suite (7/7 checks)
- [ ] Check code quality (ruff + black)
- [ ] Write comprehensive commit message with proper formatting
- [ ] Commit to feature branch only (never main)

**After Implementation:**
- [ ] Push feature branch to remote
- [ ] Update documentation if needed
- [ ] Test comprehensive system functionality
- [ ] Document business value and cost savings

### 🏆 **SUCCESS METRICS TO MAINTAIN**

**Validation Targets:**
- System validation: 7/7 checks passed ✅
- M&A demos: 5/5 demonstrations successful ✅  
- Code quality: ruff + black compliant ✅
- GitHub Actions: All workflows passing ✅

**Cost Optimization Targets:**
- Financial data costs: <$100/month vs $51K/year traditional platforms
- Implementation efficiency: Leverage free/open-source solutions
- Professional capabilities: Investment banking grade analysis

---

**📋 NOTE FOR FUTURE AI ASSISTANTS:**

### 🚨 **CRITICAL TECHNICAL ISSUES FROM THIS THREAD**

#### **Circular Import Problems**
```python
# ❌ CAUSES RuntimeWarning:
# axiom/__init__.py imports from axiom.main
# Then running: python -m axiom.main
# Creates circular import warning

# ✅ SOLUTION: Remove unnecessary imports from __init__.py
# Only import what's actually needed for external API
```

#### **Abstract Method Implementation Errors**
```python
# ❌ ERROR: Missing abstract methods
class MyProvider(BaseProvider):
    def method1(self):
        pass
    # Missing required abstract methods = instantiation error

# ✅ SOLUTION: Implement ALL abstract methods
class MyProvider(BaseProvider):
    def method1(self):
        pass
    def get_comparable_companies(self):  # All methods required
        pass
    def get_market_data(self):
        pass
```

#### **Import Path Updates After Restructuring**
```python
# ❌ OLD PATH: After moving directories
from axiom.financial_integrations import Provider

# ✅ NEW PATH: Update all import paths
from axiom.data_sources.finance import Provider

# ALWAYS test imports after restructuring!
```

#### **Linting and Code Quality Issues**
```bash
# Common ruff/black issues encountered:
# - Deprecated typing imports (Dict->dict, List->list, Optional->X|None)
# - Unused imports and variables (remove with F401 fixes)
# - Import organization (sort imports properly)
# - Trailing whitespace and formatting

# ALWAYS RUN before committing:
source .venv/bin/activate
ruff check . --fix --unsafe-fixes
black .
```

#### **Virtual Environment Issues**
```bash
# ❌ ERROR: Running without venv activation
python -c "import axiom"  # May use wrong Python version

# ✅ ALWAYS: Activate virtual environment first
source .venv/bin/activate
python -c "import axiom"
```

#### **API Key Management**
```python
# ❌ PLACEHOLDER KEYS cause errors:
CLAUDE_API_KEY=sk-ant-api03-placeholder_key_for_development_testing

# ✅ REAL KEYS for live testing:
CLAUDE_API_KEY=sk-ant-api03-your_actual_key_here

# OR comment out for demo mode:
# CLAUDE_API_KEY=your_key_here
```

#### **Validation Threshold Issues**
```python
# ❌ WRONG: Transaction validation too restrictive
elif value > 1000000:  # $1M threshold (too low for M&A)

# ✅ CORRECT: Realistic M&A transaction thresholds  
elif value > 1000000000000:  # $1T threshold (realistic)
```

### 🔧 **DEPENDENCY MANAGEMENT PATTERNS**

#### **Adding New Dependencies**
```bash
# ALWAYS add to BOTH files:
# 1. pyproject.toml [project.dependencies]
# 2. requirements.txt

# Then test installation:
pip install -e .
```

#### **Required Testing Sequence**
```bash
# MANDATORY before any commit:
source .venv/bin/activate
ruff check . --fix --unsafe-fixes  # Fix linting
black .                           # Apply formatting  
python tests/validate_system.py   # 7/7 checks
python demo_ma_analysis.py        # 5/5 demos
```

### 📦 **PACKAGE STRUCTURE MANAGEMENT**

#### **__init__.py Export Management**
```python
# ALWAYS update __all__ when adding new modules:
__all__ = [
    # Core components
    "existing_exports",
    
    # New additions
    "new_workflow_class",
    "new_function",
]
```

#### **Import Organization Rules**
```python
# FOLLOW THIS ORDER:
# 1. Standard library imports
# 2. Third-party imports  
# 3. Local application imports
# 4. Relative imports (from .)
```

### 🎯 **CRITICAL ERROR PATTERNS ENCOUNTERED**

**Based on Actual Issues from This Thread:**
1. **Circular imports** from unnecessary __init__.py imports
2. **Missing abstract method implementations** in provider classes
3. **Import path errors** after directory restructuring  
4. **383 ruff linting errors** from deprecated typing and unused imports
5. **Validation threshold errors** (too restrictive for M&A use cases)
6. **Virtual environment** not activated causing import issues
7. **API key placeholder** errors when testing live functionality
8. **GitHub Actions failures** from code quality issues

### 💡 **LESSONS LEARNED - TECHNICAL BEST PRACTICES**

#### **Always Follow This Development Sequence:**
1. **Plan enhancement** with cost-effective approach
2. **Create feature branch** (never work on main)
3. **Follow existing patterns** (AI providers, workflow modules)
4. **Test frequently** during development
5. **Fix linting issues** with ruff --fix --unsafe-fixes
6. **Apply formatting** with black
7. **Run validation suite** (7/7 checks required)
8. **Test imports** after any restructuring
9. **Commit with proper message formatting** (quotes on new lines)
10. **Push to feature branch** only (never main)

#### **Repository Health Maintenance**
- Keep main branch stable with all GitHub Actions passing
- Use feature branches for all enhancements
- Maintain 7/7 validation checks and 5/5 demo success
- Keep code quality at 100% ruff + black compliance
- Prioritize cost-effective solutions (99.7% savings achieved)

---

**🚨 CRITICAL REMINDER FOR FUTURE AI ASSISTANTS:**
These technical guidelines are based on ACTUAL issues encountered during comprehensive M&A platform development. Following these practices prevents common pitfalls and ensures professional-grade development standards.
These guidelines ensure consistent, professional development practices while maintaining cost optimization and proper git workflow management for the Axiom M&A Investment Banking Analytics platform.