# Axiom Investment Banking Analytics - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### 1. Installation
```bash
# Clone and install
git clone <repository-url>
cd axiom
pip install -e .

# Verify installation
python simple_demo.py
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys (at minimum):
# TAVILY_API_KEY=your_tavily_key_here
# FIRECRAWL_API_KEY=your_firecrawl_key_here
# OPENAI_API_KEY=sk-your_openai_key_here  # OR
# CLAUDE_API_KEY=sk-ant-your_claude_key_here
```

### 3. Run M&A Analysis
```bash
# Test the system
python -m axiom.main "Microsoft acquisition of OpenAI strategic analysis"
```

---

## 🏦 Investment Banking Features

### M&A Analysis (Phase 1 - Ready)
- **Due Diligence**: Financial health, operational risks, strategic fit
- **Valuation**: DCF, comparable analysis, precedent transactions  
- **Strategic Analysis**: Market position, synergies, competitive advantages
- **Risk Assessment**: Integration complexity, regulatory compliance

### AI Provider Configuration
- **Claude**: Optimized for M&A reasoning and strategic analysis
- **OpenAI**: Structured financial analysis and valuation
- **SGLang**: Local inference for NVIDIA systems
- **Multi-Provider**: Consensus mode for critical decisions

---

## 📊 Usage Examples

### M&A Due Diligence
```bash
python -m axiom.main "Comprehensive M&A due diligence analysis of NVIDIA financial health and strategic value"
```

### M&A Valuation 
```bash
python -m axiom.main "Tesla acquisition valuation analysis using DCF and comparable transactions with synergy assessment"
```

### Strategic M&A Analysis
```bash
python -m axiom.main "Microsoft OpenAI merger strategic fit analysis and integration complexity assessment"
```

### Market Intelligence
```bash
python -m axiom.main "Semiconductor industry M&A consolidation trends and valuation multiples analysis"
```

---

## ⚙️ Configuration Options

### AI Provider Setup
```env
# Option 1: OpenAI Only
OPENAI_API_KEY=sk-your_key_here
OPENAI_MODEL_NAME=gpt-4o-mini

# Option 2: Claude Only  
CLAUDE_API_KEY=sk-ant-your_key_here
CLAUDE_MODEL_NAME=claude-3-sonnet-20240229

# Option 3: Multi-Provider (Recommended)
OPENAI_API_KEY=sk-your_openai_key_here
CLAUDE_API_KEY=sk-ant-your_claude_key_here

# Option 4: Local Inference (NVIDIA)
SGLANG_BASE_URL=http://localhost:30000/v1
```

### Investment Banking Parameters
```env
# Analysis Configuration
DUE_DILIGENCE_CONFIDENCE_THRESHOLD=0.8
VALUATION_MODEL_TYPES=dcf,comparable,precedent
RISK_ANALYSIS_ENABLED=true

# Conservative AI Settings (Pre-configured)
# M&A Due Diligence: Temperature=0.03, Consensus=true
# M&A Valuation: Temperature=0.05, Consensus=true  
# Observer Synthesis: Temperature=0.02
```

---

## 🎯 System Validation

### Check System Health
```bash
python simple_demo.py
```

### Expected Output:
```
🎯 KEY FEATURES IMPLEMENTED:
   • Multi-AI Provider System (OpenAI, Claude, SGLang)
   • Investment Banking Workflow Orchestration
   • M&A-Specific Analysis Planning
   • Financial Data Validation & Compliance
   • DSPy Optimization for Financial Queries
   • Comprehensive Error Handling

Demo Score: 4/4 ✅
```

### Run System Validation
```bash
python tests/validate_system.py
```

---

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**No AI Provider Available**
```bash
# Solution: Configure at least one AI provider in .env
OPENAI_API_KEY=sk-your_key_here
# OR
CLAUDE_API_KEY=sk-ant-your_key_here
```

**Low Confidence Results**
```bash
# Solution: Check data sources and increase evidence threshold
DUE_DILIGENCE_CONFIDENCE_THRESHOLD=0.7  # Lower threshold
```

### Getting Help
- Check [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for detailed setup
- Review [`CONTEXT.md`](CONTEXT.md) for architecture details
- Run [`simple_demo.py`](simple_demo.py) for system validation

---

## 🏃‍♂️ Ready for Production

✅ **Core Features**: Multi-AI provider system, M&A analysis, financial validation  
✅ **Architecture**: LangGraph workflow, DSPy optimization, comprehensive error handling  
✅ **Investment Banking**: M&A due diligence, valuation, strategic analysis  
✅ **Quality**: Conservative AI settings, compliance validation, audit trails

**Next**: Configure your API keys and start analyzing M&A transactions!