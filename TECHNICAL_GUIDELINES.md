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
These guidelines ensure consistent, professional development practices while maintaining cost optimization and proper git workflow management for the Axiom M&A Investment Banking Analytics platform.