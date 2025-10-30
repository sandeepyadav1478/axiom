"""
Validation Script for Derivatives Platform Setup

Checks that all components are properly installed and configured:
- Python dependencies
- GPU availability
- Database connections
- Redis cache
- Model files
- Configuration
- Performance baseline

Run before deploying to production to catch issues early.
"""

import sys
import subprocess
from typing import Tuple, List


class SetupValidator:
    """Validates complete derivatives platform setup"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0
    
    def check(self, name: str, condition: bool, error_msg: str = "", warning: bool = False):
        """Check a condition and track result"""
        self.checks_total += 1
        
        if condition:
            print(f"   ✓ {name}")
            self.checks_passed += 1
        else:
            marker = "⚠" if warning else "✗"
            print(f"   {marker} {name}")
            if warning:
                self.warnings.append(error_msg or name)
            else:
                self.errors.append(error_msg or name)
    
    def validate_python_environment(self):
        """Check Python version and core packages"""
        print("\n→ Validating Python Environment:")
        
        # Python version
        py_version = sys.version_info
        self.check(
            "Python 3.10+",
            py_version.major == 3 and py_version.minor >= 10,
            f"Python {py_version.major}.{py_version.minor} found, need 3.10+"
        )
        
        # Core packages
        packages = [
            ('torch', 'PyTorch'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('fastapi', 'FastAPI'),
            ('sqlalchemy', 'SQLAlchemy'),
            ('redis', 'Redis client'),
            ('chromadb', 'ChromaDB'),
            ('langgraph', 'LangGraph'),
            ('prometheus_client', 'Prometheus client')
        ]
        
        for package, name in packages:
            try:
                __import__(package)
                self.check(f"{name} installed", True)
            except ImportError:
                self.check(f"{name} installed", False, f"{name} not found - install with pip")
    
    def validate_gpu_setup(self):
        """Check GPU availability and CUDA"""
        print("\n→ Validating GPU Setup:")
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            self.check(
                "CUDA available",
                cuda_available,
                "GPU not available - will run on CPU (10x slower)",
                warning=True
            )
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.check(
                    f"GPU detected: {gpu_name}",
                    True
                )
                
                # Check CUDA version
                cuda_version = torch.version.cuda
                self.check(
                    f"CUDA {cuda_version}",
                    True
                )
        except Exception as e:
            self.check("GPU setup", False, str(e))
    
    def validate_database_connections(self):
        """Check database connectivity"""
        print("\n→ Validating Database Connections:")
        
        # PostgreSQL
        try:
            import psycopg2
            # Try to connect (would use actual credentials in production)
            # For now, just check package
            self.check("PostgreSQL client installed", True)
        except ImportError:
            self.check("PostgreSQL client installed", False, "Install: pip install psycopg2-binary")
        
        # Redis
        try:
            import redis
            # Try to connect to localhost
            r = redis.Redis(host='localhost', port=6379, socket_timeout=1)
            r.ping()
            self.check("Redis connection", True)
        except Exception as e:
            self.check("Redis connection", False, "Redis not running on localhost:6379", warning=True)
    
    def validate_model_files(self):
        """Check that model weights exist"""
        print("\n→ Validating Model Files:")
        
        import os
        
        model_dir = "models/derivatives/"
        
        self.check(
            "Model directory exists",
            os.path.exists(model_dir),
            f"Create directory: mkdir -p {model_dir}",
            warning=True
        )
        
        # Check for trained weights (optional for initial setup)
        weight_files = [
            'greeks_model.pth',
            'barrier_pinn.pth',
            'asian_vae.pth'
        ]
        
        for weight_file in weight_files:
            path = os.path.join(model_dir, weight_file)
            self.check(
                f"{weight_file}",
                os.path.exists(path),
                f"Train model and save to {path}",
                warning=True
            )
    
    def validate_configuration(self):
        """Check configuration files"""
        print("\n→ Validating Configuration:")
        
        import os
        
        config_files = [
            '.env',
            'axiom/derivatives/schema.sql',
            'requirements-derivatives.txt'
        ]
        
        for config_file in config_files:
            self.check(
                f"{config_file} exists",
                os.path.exists(config_file),
                f"Missing: {config_file}",
                warning=(config_file == '.env')  # .env is optional
            )
    
    def run_smoke_tests(self):
        """Run quick smoke tests"""
        print("\n→ Running Smoke Tests:")
        
        try:
            from axiom.derivatives.ultra_fast_greeks import UltraFastGreeksEngine
            
            engine = UltraFastGreeksEngine(use_gpu=False)  # CPU for validation
            
            # Quick test
            greeks = engine.calculate_greeks(100, 100, 1.0, 0.03, 0.25)
            
            self.check(
                "Greeks calculation working",
                greeks.delta > 0 and greeks.delta < 1.0,
                "Greeks calculation returned invalid values"
            )
            
            self.check(
                "Performance acceptable",
                greeks.calculation_time_us < 10000,  # <10ms on CPU is acceptable
                f"Too slow: {greeks.calculation_time_us:.0f}us"
            )
        except Exception as e:
            self.check("Smoke test", False, str(e))
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\nChecks: {self.checks_passed}/{self.checks_total} passed")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.errors:
            print("\n✅ VALIDATION PASSED")
            print("\nSystem is ready for deployment!")
            print("\nNext steps:")
            print("  1. Run benchmarks: ./scripts/run_derivatives_benchmarks.sh")
            print("  2. Start API: python -m axiom.derivatives.api.endpoints")
            print("  3. Run tests: pytest tests/derivatives/ -v")
            return 0
        else:
            print("\n❌ VALIDATION FAILED")
            print(f"\nFix {len(self.errors)} error(s) before proceeding")
            return 1
    
    def run_all_validations(self) -> int:
        """Run complete validation suite"""
        print("="*60)
        print("DERIVATIVES PLATFORM - SETUP VALIDATION")
        print("="*60)
        
        self.validate_python_environment()
        self.validate_gpu_setup()
        self.validate_database_connections()
        self.validate_model_files()
        self.validate_configuration()
        self.run_smoke_tests()
        
        return self.print_summary()


if __name__ == "__main__":
    validator = SetupValidator()
    exit_code = validator.run_all_validations()
    sys.exit(exit_code)

# Run with: python scripts/validate_derivatives_setup.py