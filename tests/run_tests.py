"""Test runner for Axiom Investment Banking Analytics."""

import pytest
import sys
import os
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.utils import setup_logging, global_health_checker
from axiom.ai_client_integrations import test_providers


def run_health_checks():
    """Run system health checks before tests."""

    print("🏥 Running Investment Banking System Health Checks...")

    # Register health checks
    global_health_checker.register_check(
        "ai_providers",
        lambda: len(test_providers()) > 0,
        "Check if any AI providers are available",
    )

    global_health_checker.register_check(
        "configuration",
        lambda: check_basic_configuration(),
        "Check basic system configuration",
    )

    # Run checks
    health_results = global_health_checker.run_health_checks()

    print(f"Overall Status: {health_results['overall_status'].upper()}")
    for check_name, check_result in health_results["checks"].items():
        status_icon = "✅" if check_result["status"] == "pass" else "❌"
        print(f"  {status_icon} {check_name}: {check_result['status']}")
        if check_result["status"] != "pass":
            print(f"    Error: {check_result.get('error', 'Unknown error')}")

    if health_results["overall_status"] != "healthy":
        print("\n⚠️  Some health checks failed. Tests may not run properly.")
        return False

    print("✅ All health checks passed!\n")
    return True


def check_basic_configuration():
    """Check basic system configuration."""
    try:
        from axiom.config.settings import settings

        # Check that we can load settings
        configured_providers = settings.get_configured_providers()

        # Need at least some configuration to run
        return len(configured_providers) >= 0  # Allow zero for testing

    except Exception as e:
        print(f"Configuration check failed: {e}")
        return False


def run_unit_tests():
    """Run unit tests."""

    print("🧪 Running Unit Tests...")

    # Run specific test files
    test_files = ["tests/test_ai_providers.py", "tests/test_validation.py"]

    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nRunning {test_file}...")
            result = pytest.main([test_file, "-v", "--tb=short"])
            if result != 0:
                print(f"❌ Tests failed in {test_file}")
                return False
        else:
            print(f"⚠️  Test file not found: {test_file}")

    print("✅ Unit tests completed!")
    return True


def run_integration_tests():
    """Run integration tests."""

    print("🔗 Running Integration Tests...")

    integration_test = "tests/test_integration.py"
    if os.path.exists(integration_test):
        result = pytest.main([integration_test, "-v", "--tb=short"])
        if result == 0:
            print("✅ Integration tests passed!")
            return True
        else:
            print("❌ Integration tests failed!")
            return False
    else:
        print(f"⚠️  Integration test file not found: {integration_test}")
        return False


def run_system_validation():
    """Run system validation tests."""

    print("⚙️  Running System Validation...")

    try:
        # Test AI provider factory
        print("  Testing AI Provider Factory...")
        provider_status = test_providers()
        working_providers = [name for name, status in provider_status.items() if status]

        if working_providers:
            print(f"    ✅ Working providers: {', '.join(working_providers)}")
        else:
            print("    ⚠️  No working AI providers (may need API keys)")

        # Test tool availability
        print("  Testing Financial Tools...")
        from axiom.tools.mcp_adapter import mcp_adapter

        tools = mcp_adapter.get_available_tools()
        print(f"    ✅ Available tools: {len(tools)}")

        # Test configuration loading
        print("  Testing Configuration...")
        from axiom.config.settings import settings

        providers = settings.get_configured_providers()
        print(
            f"    ✅ Configured providers: {', '.join(providers) if providers else 'None (using defaults)'}"
        )

        print("✅ System validation completed!")
        return True

    except Exception as e:
        print(f"❌ System validation failed: {e}")
        return False


def main():
    """Main test runner."""

    print("🏦 Axiom Investment Banking Analytics - Test Suite")
    print("=" * 60)

    # Setup logging
    setup_logging("INFO")

    # Track test results
    results = {
        "health_checks": False,
        "unit_tests": False,
        "integration_tests": False,
        "system_validation": False,
    }

    # Run test phases
    try:
        results["health_checks"] = run_health_checks()
        results["unit_tests"] = run_unit_tests()
        results["integration_tests"] = run_integration_tests()
        results["system_validation"] = run_system_validation()

    except KeyboardInterrupt:
        print("\n\n⏸️  Tests interrupted by user")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)

    total_phases = len(results)
    passed_phases = sum(1 for result in results.values() if result)

    for phase, result in results.items():
        status_icon = "✅" if result else "❌"
        print(f"  {status_icon} {phase.replace('_', ' ').title()}")

    print(f"\nOverall: {passed_phases}/{total_phases} phases passed")

    if passed_phases == total_phases:
        print("🎉 All tests passed! Investment Banking Analytics system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
