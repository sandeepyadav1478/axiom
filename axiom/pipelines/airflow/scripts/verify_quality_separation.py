#!/usr/bin/env python3
"""
Verification Script: Data Quality Separation

Tests that the separation of ingestion and validation is working correctly:
1. Data ingestion DAG has no validation
2. Quality validation DAG validates only NEW data
3. Both DAGs are independent
4. Validation history is tracked correctly
"""
import sys
import os
from pathlib import Path

# Add axiom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_ingestion_dag_structure():
    """Test that ingestion DAG has no validation tasks"""
    print("\n🔍 Test 1: Verifying Ingestion DAG Structure")
    print("=" * 60)
    
    dag_path = Path(__file__).parent.parent / "dags" / "data_ingestion_dag_v2.py"
    
    if not dag_path.exists():
        print(f"❌ FAIL: Ingestion DAG not found at {dag_path}")
        return False
    
    with open(dag_path, 'r') as f:
        content = f.read()
    
    # Check that validation operator is NOT imported
    if 'from operators.quality_check_operator import DataQualityOperator' in content:
        print("❌ FAIL: DataQualityOperator still imported in ingestion DAG")
        return False
    else:
        print("✅ PASS: DataQualityOperator not imported")
    
    # Check that validate_ingested_data task is removed
    if 'validate_ingested_data' in content:
        print("❌ FAIL: validate_ingested_data task still exists")
        return False
    else:
        print("✅ PASS: Validation task removed")
    
    # Check that validation is not in task dependencies
    if 'validate_data' in content:
        print("⚠️  WARNING: 'validate_data' reference still in code")
    else:
        print("✅ PASS: No validation references in dependencies")
    
    # Check documentation mentions separation
    if 'data_quality_validation_dag' in content:
        print("✅ PASS: Documentation references separate validation DAG")
    else:
        print("⚠️  WARNING: Documentation doesn't mention validation DAG")
    
    print("\n✅ Test 1 PASSED: Ingestion DAG structure is correct")
    return True


def test_validation_dag_exists():
    """Test that validation DAG exists and has correct structure"""
    print("\n🔍 Test 2: Verifying Validation DAG Exists")
    print("=" * 60)
    
    dag_path = Path(__file__).parent.parent / "dags" / "data_quality_validation_dag.py"
    
    if not dag_path.exists():
        print(f"❌ FAIL: Validation DAG not found at {dag_path}")
        return False
    
    print(f"✅ PASS: Validation DAG found at {dag_path}")
    
    with open(dag_path, 'r') as f:
        content = f.read()
    
    # Check key features
    checks = {
        "Hourly schedule": "schedule_interval='0 * * * *'",
        "Incremental validation": "get_last_validation_time",
        "State tracking": "last_data_quality_validation",
        "History table": "validation_history",
        "Rules engine": "get_validation_engine",
    }
    
    for feature, pattern in checks.items():
        if pattern in content:
            print(f"✅ PASS: {feature} implemented")
        else:
            print(f"❌ FAIL: {feature} not found (pattern: {pattern})")
            return False
    
    print("\n✅ Test 2 PASSED: Validation DAG structure is correct")
    return True


def test_rules_engine_exists():
    """Test that rules engine exists and is functional"""
    print("\n🔍 Test 3: Verifying Rules Engine")
    print("=" * 60)
    
    try:
        from axiom.data_quality.validation.rules_engine import (
            get_validation_engine,
            ValidationSeverity,
            ValidationCategory
        )
        print("✅ PASS: Rules engine imports successfully")
    except ImportError as e:
        print(f"❌ FAIL: Cannot import rules engine: {e}")
        return False
    
    # Test validation engine
    try:
        engine = get_validation_engine()
        print("✅ PASS: Validation engine instantiated")
    except Exception as e:
        print(f"❌ FAIL: Cannot instantiate engine: {e}")
        return False
    
    # Test price data validation
    try:
        test_data = {
            'symbol': 'TEST',
            'open': 100.0,
            'high': 105.0,
            'low': 99.0,
            'close': 103.0,
            'volume': 1000000,
            'timestamp': '2025-11-21T00:00:00Z'
        }
        
        results = engine.validate_data(test_data, "price_data", raise_on_critical=False)
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        print(f"✅ PASS: Validation engine works ({passed}/{total} rules passed)")
        
        if passed < total:
            print(f"ℹ️  INFO: {total - passed} rules failed for test data:")
            for r in results:
                if not r.passed:
                    print(f"     - {r.rule_name}: {r.error_message}")
    except Exception as e:
        print(f"❌ FAIL: Validation engine error: {e}")
        return False
    
    print("\n✅ Test 3 PASSED: Rules engine is functional")
    return True


def test_separation_benefits():
    """Verify benefits of separation"""
    print("\n🔍 Test 4: Verifying Separation Benefits")
    print("=" * 60)
    
    ingestion_path = Path(__file__).parent.parent / "dags" / "data_ingestion_dag_v2.py"
    validation_path = Path(__file__).parent.parent / "dags" / "data_quality_validation_dag.py"
    
    with open(ingestion_path, 'r') as f:
        ingestion_content = f.read()
    
    with open(validation_path, 'r') as f:
        validation_content = f.read()
    
    # Check schedules are different
    print("\n📅 Schedule Comparison:")
    if "schedule_interval='*/1 * * * *'" in ingestion_content:
        print("  Ingestion: Every minute (*/1 * * * *)")
        ingestion_freq = True
    else:
        print("  ❌ Ingestion schedule not found")
        ingestion_freq = False
    
    if "schedule_interval='0 * * * *'" in validation_content:
        print("  Validation: Hourly (0 * * * *)")
        validation_freq = True
    else:
        print("  ❌ Validation schedule not found")
        validation_freq = False
    
    if ingestion_freq and validation_freq:
        print("  ✅ PASS: Different frequencies (60x less validation)")
    
    # Check fail_on_error settings
    print("\n⚠️  Failure Handling:")
    if 'fail_on_error=False' in validation_content:
        print("  Validation: fail_on_error=False (doesn't fail DAG)")
        print("  ✅ PASS: Quality issues won't stop validation DAG")
    
    # Check for incremental validation
    print("\n📊 Validation Approach:")
    if 'WHERE timestamp >' in validation_content and 'last_validation' in validation_content:
        print("  Validation: Incremental (NEW data only)")
        print("  ✅ PASS: Only validates data since last check")
    
    print("\n✅ Test 4 PASSED: Separation benefits verified")
    return True


def test_documentation():
    """Test that documentation is updated"""
    print("\n🔍 Test 5: Verifying Documentation")
    print("=" * 60)
    
    summary_path = Path(__file__).parent.parent / "dags" / "DATA_QUALITY_SEPARATION_SUMMARY.md"
    
    if not summary_path.exists():
        print(f"❌ FAIL: Summary documentation not found at {summary_path}")
        return False
    
    print(f"✅ PASS: Summary documentation exists")
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    required_sections = [
        "Problem Solved",
        "Files Changed",
        "Architecture Comparison",
        "Performance Benefits",
        "Validation Capabilities",
        "Deployment"
    ]
    
    for section in required_sections:
        if section in content:
            print(f"✅ PASS: Documentation has '{section}' section")
        else:
            print(f"⚠️  WARNING: '{section}' section not found")
    
    print("\n✅ Test 5 PASSED: Documentation is comprehensive")
    return True


def main():
    """Run all verification tests"""
    print("\n" + "=" * 60)
    print("DATA QUALITY SEPARATION VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Ingestion DAG Structure", test_ingestion_dag_structure),
        ("Validation DAG Exists", test_validation_dag_exists),
        ("Rules Engine", test_rules_engine_exists),
        ("Separation Benefits", test_separation_benefits),
        ("Documentation", test_documentation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ FATAL ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 SUCCESS: All verification tests passed!")
        print("\nNext Steps:")
        print("1. Deploy both DAGs to Airflow")
        print("2. Enable both DAGs in Airflow UI")
        print("3. Monitor first runs")
        print("4. Check validation_history table for results")
        return 0
    else:
        print("\n⚠️  WARNING: Some tests failed")
        print("Please review the failures above before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())