#!/bin/bash

# Axiom Research Agent - Installation Script

echo "🚀 Installing Axiom Research Agent..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env

echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Test with: python -m axiom.main 'What is artificial intelligence?'"
echo "3. Run evaluation: python -m axiom.eval.run_eval"
echo ""
echo "For development:"
echo "• Install dev dependencies: pip install -e '.[dev]'"
echo "• Run tests: pytest"
echo "• Format code: black . && ruff check --fix ."
