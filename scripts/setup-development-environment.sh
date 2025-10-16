#!/bin/bash
# Axiom Investment Banking Analytics - Development Environment Setup
# Automated setup with pyenv, direnv, and UV package manager

set -e

echo "🏦 Setting up Axiom Investment Banking Development Environment"
echo "============================================================="

# Check if pyenv is installed
if command -v pyenv 1>/dev/null 2>&1; then
    echo "✅ pyenv found: $(pyenv --version)"
    
    # Install Python 3.13 if not available
    if ! pyenv versions --bare | grep -q "3.13"; then
        echo "📦 Installing Python 3.13 via pyenv..."
        pyenv install 3.13 --skip-existing
    fi
    
    # Create project virtual environment
    if ! pyenv versions --bare | grep -q "axiom-investment-banking"; then
        echo "🏗️ Creating project virtual environment..."
        pyenv virtualenv 3.13 axiom-investment-banking
    fi
    
    # Set local Python version
    pyenv local 3.13
    echo "✅ Python 3.13 set for project"
else
    echo "⚠️ pyenv not found - install with: brew install pyenv"
    echo "Using system Python instead"
fi

# Check if direnv is installed  
if command -v direnv 1>/dev/null 2>&1; then
    echo "✅ direnv found: $(direnv --version)"
    
    # Allow .envrc for auto-activation
    direnv allow .
    echo "✅ Auto-activation enabled with direnv"
else
    echo "⚠️ direnv not found - install with: brew install direnv"
    echo "Add to shell: echo 'eval \"\$(direnv hook bash)\"' >> ~/.bashrc"
fi

# Install UV package manager
if command -v uv 1>/dev/null 2>&1; then
    echo "✅ UV package manager available"
else
    echo "📦 Installing UV package manager..."
    pip install uv
fi

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "🏗️ Creating .venv virtual environment..."
    python -m venv .venv
fi

# Activate environment and install dependencies
echo "⚡ Installing dependencies with UV..."
source .venv/bin/activate && uv pip install -e .

echo ""
echo "🎉 Development Environment Setup Complete!"
echo ""
echo "📋 What was configured:"
echo "• Python 3.13 via pyenv (if available)"
echo "• Project virtual environment: axiom-investment-banking" 
echo "• Auto-activation via .envrc and direnv"
echo "• UV package manager for fast installations"
echo "• Dependencies installed with UV"
echo ""
echo "🚀 Next time you enter this directory:"
echo "• Python 3.13 will be automatically selected"
echo "• Virtual environment will auto-activate"
echo "• All dependencies will be ready"
echo ""
echo "📝 Manual activation if needed:"
echo "source .venv/bin/activate"
echo "# OR"
echo "pyenv activate axiom-investment-banking"