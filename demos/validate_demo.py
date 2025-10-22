#!/usr/bin/env python3
"""
Validation script for Quantitative Finance Integration Demo
Checks if all dependencies are available and demo structure is correct.
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'yfinance',
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} missing")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All required packages installed!")
        return True

def check_axiom_modules():
    """Check if Axiom modules are accessible."""
    print("\nChecking Axiom modules:")
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    required_modules = [
        ('axiom.models.risk.var_models', 'VaR Models'),
        ('axiom.models.portfolio.optimization', 'Portfolio Optimization'),
        ('axiom.integrations.data_sources.finance.yahoo_finance_provider', 'Yahoo Finance Provider'),
        ('axiom.core.logging.axiom_logger', 'Axiom Logger')
    ]
    
    all_ok = True
    for module_path, name in required_modules:
        try:
            __import__(module_path)
            print(f"✓ {name} accessible")
        except ImportError as e:
            print(f"✗ {name} failed: {e}")
            all_ok = False
    
    if all_ok:
        print("\n✅ All Axiom modules accessible!")
    else:
        print("\n❌ Some Axiom modules not accessible")
    
    return all_ok

def main():
    """Run all validation checks."""
    print("="*60)
    print("Quantitative Finance Demo Validation")
    print("="*60)
    
    print("\nStep 1: Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\nStep 2: Checking Axiom modules...")
    modules_ok = check_axiom_modules()
    
    print("\n" + "="*60)
    if deps_ok and modules_ok:
        print("✅ All validations passed!")
        print("\nYou can now run the demo:")
        print("  python demos/demo_integrated_quant_finance.py")
    else:
        print("❌ Some validations failed")
        print("\nPlease fix the issues above before running the demo")
    print("="*60)

if __name__ == "__main__":
    main()