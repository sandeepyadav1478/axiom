#!/usr/bin/env python3
"""
Axiom Environment Setup Script
Ensures .env file is configured before proceeding with installation.

This script should be run IMMEDIATELY after creating virtual environment.
"""

import os
import sys
from pathlib import Path
import subprocess

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'=' * 70}{RESET}")
    print(f"{BOLD}{BLUE}{text:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ️  {text}{RESET}")

def check_virtual_env():
    """Check if running in virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print_error("Not running in a virtual environment!")
        print_info("Please create and activate virtual environment first:")
        print("  uv venv")
        print("  source .venv/bin/activate")
        sys.exit(1)
    
    print_success("Virtual environment detected")

def check_env_file():
    """Check if .env file exists and is configured."""
    env_path = Path('.env')
    env_example_path = Path('.env.example')
    
    if not env_path.exists():
        print_error(".env file NOT found!")
        print_warning("This file is REQUIRED for the application to work properly")
        
        if env_example_path.exists():
            print_info("Creating .env from .env.example...")
            
            # Copy .env.example to .env
            with open(env_example_path, 'r') as src:
                content = src.read()
            
            with open(env_path, 'w') as dst:
                dst.write(content)
            
            print_success(".env file created from template")
            print_warning("IMPORTANT: You MUST configure the following in .env:")
            print("  1. OPENAI_API_KEY or ANTHROPIC_API_KEY (REQUIRED)")
            print("  2. Database passwords (if needed)")
            print("  3. API secret keys (for production)")
            print()
            print("Edit .env now? [y/N]: ", end='')
            
            response = input().strip().lower()
            if response == 'y':
                editor = os.environ.get('EDITOR', 'nano')
                subprocess.run([editor, '.env'])
                print_success(".env file edited")
            else:
                print_warning("Remember to edit .env before running the application!")
                print(f"  {BOLD}nano .env{RESET} or {BOLD}vim .env{RESET}")
        else:
            print_error(".env.example not found!")
            sys.exit(1)
    else:
        print_success(".env file exists")
        
        # Check if API keys are configured
        with open(env_path, 'r') as f:
            content = f.read()
        
        has_openai = 'OPENAI_API_KEY=sk-' in content
        has_anthropic = 'ANTHROPIC_API_KEY=sk-ant-' in content
        
        if not (has_openai or has_anthropic):
            print_warning("No API keys found in .env file!")
            print_info("You should configure at least one AI provider:")
            print("  - OPENAI_API_KEY=sk-your-key")
            print("  - ANTHROPIC_API_KEY=sk-ant-your-key")
            print()
            print("These are REQUIRED for LLM-powered features to work")

def check_dependencies():
    """Check if required system dependencies are installed."""
    print_header("CHECKING SYSTEM DEPENDENCIES")
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print_success(f"Docker: {version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Docker not found or not working")
        print_info("Install Docker: https://docs.docker.com/get-docker/")
        return False
    
    # Check Docker Compose
    try:
        result = subprocess.run(['docker', 'compose', 'version'], 
                              capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print_success(f"Docker Compose: {version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Docker Compose not found")
        print_info("Install Docker Compose v2")
        return False
    
    # Check nvidia-smi (optional)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, check=True)
        gpu_name = result.stdout.strip()
        print_success(f"GPU: {gpu_name}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("No NVIDIA GPU detected (optional)")
    
    return True

def main():
    """Main setup validation."""
    print_header("AXIOM ENVIRONMENT SETUP")
    
    print(f"{BOLD}This script validates your environment setup.{RESET}")
    print(f"{BOLD}Run this AFTER creating virtual environment, BEFORE installing packages.{RESET}\n")
    
    # Step 1: Check virtual environment
    print_header("STEP 1: Virtual Environment")
    check_virtual_env()
    
    # Step 2: Check .env file (CRITICAL)
    print_header("STEP 2: Environment Configuration (.env)")
    check_env_file()
    
    # Step 3: Check system dependencies
    deps_ok = check_dependencies()
    
    # Final summary
    print_header("SETUP VALIDATION COMPLETE")
    
    print(f"\n{BOLD}Next Steps:{RESET}")
    print(f"  1. {BOLD}Edit .env{RESET} and add your API keys (REQUIRED)")
    print(f"     nano .env")
    print(f"")
    print(f"  2. {BOLD}Install Python dependencies{RESET}")
    print(f"     uv pip install numpy")
    print(f"     uv pip install --no-build-isolation pmdarima")
    print(f"     uv pip install -r requirements.txt")
    print(f"     uv pip install neo4j")
    print(f"     uv pip install -e .")
    print(f"")
    print(f"  3. {BOLD}Start databases{RESET}")
    print(f"     cd axiom/database")
    print(f"     docker compose up -d postgres")
    print(f"     docker compose --profile cache up -d redis")
    print(f"     docker compose --profile vector-db-light up -d chromadb")
    print(f"     docker compose --profile graph-db up -d neo4j")
    print(f"")
    print(f"  4. {BOLD}Run verification demos{RESET}")
    print(f"     python demos/demo_complete_data_infrastructure.py")
    print(f"     python demos/demo_multi_database_architecture.py")
    
    if not deps_ok:
        print_warning("\nSome system dependencies are missing - install them first!")
        sys.exit(1)
    
    print(f"\n{GREEN}{BOLD}✅ Environment validation passed!{RESET}")
    print(f"{YELLOW}{BOLD}⚠️  Don't forget to configure .env before proceeding!{RESET}\n")

if __name__ == "__main__":
    main()