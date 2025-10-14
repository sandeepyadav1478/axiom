#!/bin/bash

# Axiom Research Agent - Installation Script
# Cross-platform installation with proper error handling and best practices

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Virtual environment name (.venv is modern standard)
VENV_DIR=".venv"

# Helper functions
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Detect Python command
detect_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        error "Python is not installed. Please install Python 3.10+ and try again."
    fi
}

# Check Python version
check_python_version() {
    local python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    local major=$(echo $python_version | cut -d'.' -f1)
    local minor=$(echo $python_version | cut -d'.' -f2)
    
    if [[ $major -lt 3 ]] || [[ $major -eq 3 && $minor -lt 10 ]]; then
        error "Python 3.10+ is required. Found: $python_version"
    fi
    
    success "Found Python $python_version"
}

# Main installation function
main() {
    echo -e "${BLUE}🚀 Axiom Research Agent - Installation${NC}"
    echo "=================================================="
    
    # Detect and validate Python
    info "Checking Python installation..."
    detect_python
    check_python_version
    
    # Check if virtual environment already exists
    if [[ -d "$VENV_DIR" ]]; then
        warning "Virtual environment '$VENV_DIR' already exists."
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            info "Using existing virtual environment..."
        fi
    fi
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$VENV_DIR" ]]; then
        info "Creating virtual environment at $VENV_DIR..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        success "Virtual environment created"
    fi
    
    # Activate virtual environment (cross-platform)
    info "Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source "$VENV_DIR/Scripts/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi
    
    # Upgrade pip and essential tools
    info "Upgrading pip and essential tools..."
    pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    echo
    echo "Installation options:"
    echo "1) Development (recommended - includes DSPy optimization + dev tools)"
    echo "2) With SGLang (for NVIDIA systems only - local inference)"
    echo "3) Basic (minimal dependencies)"
    
    read -p "Choose option (1-3) [1]: " -n 1 -r
    echo
    OPTION=${REPLY:-1}
    
    case $OPTION in
        1)
            info "Installing development dependencies..."
            info "Includes: DSPy optimization + web tools + dev tools + cloud inference"
            pip install -e ".[dev]"
            success "Development dependencies installed"
            ;;
        2)
            info "Installing with SGLang..."
            warning "SGLang requires NVIDIA GPU - not recommended for macOS"
            if [[ "$OSTYPE" == "darwin"* ]]; then
                warning "You're on macOS - SGLang will likely fail"
                read -p "Continue anyway? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    info "Installing development dependencies instead..."
                    pip install -e ".[dev]"
                    success "Development setup completed"
                else
                    pip install -e ".[dev,sglang]" || {
                        warning "SGLang installation failed - falling back to dev only"
                        pip install -e ".[dev]"
                    }
                    success "Installation completed"
                fi
            else
                pip install -e ".[dev,sglang]"
                success "NVIDIA setup with SGLang completed"
            fi
            ;;
        3)
            info "Installing basic dependencies..."
            pip install -e .
            success "Basic installation completed"
            ;;
        *)
            warning "Invalid option. Installing development dependencies..."
            pip install -e ".[dev]"
            ;;
    esac
    
    # Handle .env file
    if [[ -f ".env" ]]; then
        warning ".env file already exists."
        read -p "Do you want to backup and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            mv .env .env.backup
            cp .env.example .env
            info "Backed up existing .env to .env.backup"
        fi
    else
        info "Creating .env from template..."
        cp .env.example .env
    fi
    
    # Verify installation
    info "Verifying installation..."
    
    # Test core package
    if python -c "import axiom; print('✓ Axiom package imported successfully')" 2>/dev/null; then
        success "Core package verified"
    else
        error "Core package import failed"
    fi
    
    # Test DSPy (essential for development)
    if python -c "import dspy; print('✓ DSPy imported successfully')" 2>/dev/null; then
        success "DSPy optimization framework verified"
    else
        warning "DSPy not available - query optimization features will be limited"
    fi
    
    # Test SGLang (optional)
    if python -c "import sglang; print('✓ SGLang imported successfully')" 2>/dev/null; then
        success "SGLang local inference available"
    else
        info "SGLang not available - will use cloud APIs for inference"
    fi
    
    # Final instructions
    echo
    echo "=================================================="
    success "Installation complete!"
    echo
    info "Next steps:"
    echo "1. Edit .env with your API keys:"
    echo "   - TAVILY_API_KEY (get from https://tavily.com)"
    echo "   - FIRECRAWL_API_KEY (get from https://firecrawl.dev)"
    echo "   - OPENAI_API_KEY (for cloud inference)"
    echo
    echo "2. Activate the environment:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "   $VENV_DIR\\Scripts\\activate"
    else
        echo "   source $VENV_DIR/bin/activate"
    fi
    echo
    echo "3. Test installation:"
    echo "   python -c \"from axiom.dspy_modules import hyde; print('DSPy modules ready')\""
    echo
    echo "4. Test with a query (after setting up .env):"
    echo "   python -m axiom.main 'What is artificial intelligence?'"
    echo
    echo "5. Run tests:"
    echo "   pytest"
    echo
    echo "6. Run evaluation:"
    echo "   python -m axiom.eval.run_eval"
    echo
    info "For more information, see README.md"
    
    # Platform-specific notes
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo
        info "macOS Development Notes:"
        echo "• DSPy optimization works fully on macOS"
        echo "• Use cloud APIs (OpenAI, Anthropic) for best inference performance"
        echo "• SGLang local inference may be slower without GPU acceleration"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Axiom Research Agent Installation Script"
            echo
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  -h, --help    Show this help message"
            echo "  --clean       Remove existing virtual environment before installing"
            echo
            exit 0
            ;;
        --clean)
            if [[ -d "$VENV_DIR" ]]; then
                info "Removing existing virtual environment..."
                rm -rf "$VENV_DIR"
            fi
            shift
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Run main installation
main
