# 🎯 Axiom Demo Files

This directory contains demonstration files showing how to use the Axiom investment banking analytics platform.

## 📁 Available Demos

### Core Workflow Demos
- **[`demo_complete_ma_workflow.py`](demo_complete_ma_workflow.py)** - Complete M&A workflow demonstration from target screening to deal execution
- **[`demo_enhanced_ma_workflows.py`](demo_enhanced_ma_workflows.py)** - Enhanced M&A workflows with advanced features
- **[`demo_ma_analysis.py`](demo_ma_analysis.py)** - M&A analysis examples and use cases

### Provider & Integration Demos
- **[`demo_enhanced_financial_providers.py`](demo_enhanced_financial_providers.py)** - Enhanced financial data providers demonstration
- **[`test_enhanced_providers.py`](test_enhanced_providers.py)** - Testing enhanced provider integrations

### Quick Start
- **[`simple_demo.py`](simple_demo.py)** - Simple demonstration to get started quickly

## 🚀 Running Demos

### Prerequisites
```bash
# Make sure you're in the project root
cd /Users/sandeep.yadav/work/axiom

# Activate your environment
source .venv/bin/activate  # or your preferred environment

# Ensure dependencies are installed
pip install -r requirements.txt
```

### Running a Demo
```bash
# Run any demo file
python demos/simple_demo.py
python demos/demo_complete_ma_workflow.py
```

## 📖 Documentation

For more information on using these demos, see:
- **[Setup Guide](../docs/SETUP_GUIDE.md)** - Initial setup instructions
- **[Quick Start](../docs/QUICKSTART.md)** - Quick start guide
- **[M&A Workflows](../docs/ma-workflows/)** - Detailed M&A workflow documentation

## 🔧 Configuration

All demos use environment variables from the [`../.env`](../.env) file. Make sure to:
1. Copy `.env.example` to `.env`
2. Add your API keys (see [QUICKSTART.md](../docs/QUICKSTART.md))
3. Configure your preferred AI providers

## 💡 Tips

- Start with [`simple_demo.py`](simple_demo.py) to understand the basics
- Review [`demo_complete_ma_workflow.py`](demo_complete_ma_workflow.py) for comprehensive examples
- Check the docs folder for detailed guides and architecture documentation