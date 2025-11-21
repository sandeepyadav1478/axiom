"""
Test script to validate centralized DAG configuration
Run this to ensure dag_config.yaml is properly loaded
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import (
    dag_config,
    get_symbols_for_dag,
    get_data_sources,
    build_postgres_conn_params,
    build_redis_conn_params,
    build_neo4j_conn_params
)


def test_config_loading():
    """Test that configuration loads successfully"""
    print("🔍 Testing Configuration Loading...")
    print("=" * 60)
    
    try:
        config = dag_config.config
        print("✅ Configuration loaded successfully")
        print(f"   Config keys: {list(config.keys())}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    return True


def test_global_settings():
    """Test global settings"""
    print("\n🌍 Testing Global Settings...")
    print("=" * 60)
    
    try:
        owner = dag_config.get_global('owner')
        email = dag_config.get_global('email')
        catchup = dag_config.get_global('catchup')
        
        print(f"✅ Owner: {owner}")
        print(f"✅ Email: {email}")
        print(f"✅ Catchup: {catchup}")
        return True
    except Exception as e:
        print(f"❌ Failed to get global settings: {e}")
        return False


def test_symbol_lists():
    """Test symbol list retrieval"""
    print("\n📊 Testing Symbol Lists...")
    print("=" * 60)
    
    try:
        primary = dag_config.get_symbols('primary')
        extended = dag_config.get_symbols('extended')
        
        print(f"✅ Primary symbols ({len(primary)}): {primary[:5]}...")
        print(f"✅ Extended symbols ({len(extended)}): {extended[:5]}...")
        return True
    except Exception as e:
        print(f"❌ Failed to get symbol lists: {e}")
        return False


def test_dag_configs():
    """Test individual DAG configurations"""
    print("\n🔧 Testing DAG Configurations...")
    print("=" * 60)
    
    dags_to_test = [
        'data_ingestion',
        'data_quality_validation',
        'company_graph_builder',
        'correlation_analyzer',
        'events_tracker'
    ]
    
    for dag_name in dags_to_test:
        try:
            config = dag_config.get_dag_config(dag_name)
            schedule = dag_config.get_schedule(dag_name)
            tags = dag_config.get_tags(dag_name)
            
            print(f"\n✅ {dag_name}:")
            print(f"   Schedule: {schedule}")
            print(f"   Tags: {tags[:3]}..." if len(tags) > 3 else f"   Tags: {tags}")
        except Exception as e:
            print(f"\n❌ {dag_name}: {e}")
            return False
    
    return True


def test_default_args():
    """Test default_args generation"""
    print("\n⚙️  Testing Default Args...")
    print("=" * 60)
    
    try:
        args = dag_config.get_default_args('data_ingestion')
        print(f"✅ Data Ingestion Args:")
        print(f"   Owner: {args.get('owner')}")
        print(f"   Retries: {args.get('retries')}")
        print(f"   Execution timeout: {args.get('execution_timeout')}")
        return True
    except Exception as e:
        print(f"❌ Failed to get default args: {e}")
        return False


def test_batch_config():
    """Test batch validation configuration"""
    print("\n📦 Testing Batch Validation Config...")
    print("=" * 60)
    
    try:
        batch_config = dag_config.get_batch_config()
        thresholds = dag_config.get_validation_thresholds()
        
        print(f"✅ Batch Configuration:")
        print(f"   Enabled: {batch_config.get('enabled')}")
        print(f"   Window minutes: {batch_config.get('window_minutes')}")
        print(f"   Min records: {batch_config.get('min_records_to_validate')}")
        
        print(f"\n✅ Validation Thresholds:")
        print(f"   Data freshness: {thresholds.get('data_freshness_minutes')} minutes")
        print(f"   Min symbols: {thresholds.get('min_symbols_with_recent_data')}")
        print(f"   Price range: ${thresholds.get('price_min')} - ${thresholds.get('price_max')}")
        return True
    except Exception as e:
        print(f"❌ Failed to get batch config: {e}")
        return False


def test_circuit_breaker_configs():
    """Test circuit breaker configurations"""
    print("\n🔌 Testing Circuit Breaker Configs...")
    print("=" * 60)
    
    dags = ['data_ingestion', 'company_graph_builder', 'correlation_analyzer']
    
    for dag_name in dags:
        try:
            cb_config = dag_config.get_circuit_breaker_config(dag_name)
            print(f"\n✅ {dag_name}:")
            print(f"   Failure threshold: {cb_config.get('failure_threshold')}")
            print(f"   Recovery timeout: {cb_config.get('recovery_timeout_seconds')}s")
        except Exception as e:
            print(f"\n❌ {dag_name}: {e}")
            return False
    
    return True


def test_claude_configs():
    """Test Claude API configurations"""
    print("\n🤖 Testing Claude API Configs...")
    print("=" * 60)
    
    dags = ['company_graph_builder', 'correlation_analyzer', 'events_tracker']
    
    for dag_name in dags:
        try:
            claude_config = dag_config.get_claude_config(dag_name)
            print(f"\n✅ {dag_name}:")
            print(f"   Cache TTL: {claude_config.get('cache_ttl_hours')} hours")
            print(f"   Max tokens: {claude_config.get('max_tokens')}")
            print(f"   Track cost: {claude_config.get('track_cost')}")
        except Exception as e:
            print(f"\n❌ {dag_name}: {e}")
            return False
    
    return True


def test_helper_functions():
    """Test helper functions"""
    print("\n🛠️  Testing Helper Functions...")
    print("=" * 60)
    
    try:
        # Test get_symbols_for_dag
        symbols = get_symbols_for_dag('data_ingestion')
        print(f"✅ get_symbols_for_dag('data_ingestion'): {len(symbols)} symbols")
        
        # Test get_data_sources
        sources = get_data_sources()
        print(f"✅ get_data_sources():")
        print(f"   Primary: {sources.get('primary')}")
        print(f"   Fallback: {sources.get('fallback')}")
        
        return True
    except Exception as e:
        print(f"❌ Failed helper functions: {e}")
        return False


def test_connection_builders():
    """Test connection parameter builders"""
    print("\n🔗 Testing Connection Builders...")
    print("=" * 60)
    
    try:
        # These will use environment variables
        pg_params = build_postgres_conn_params()
        redis_params = build_redis_conn_params()
        neo4j_params = build_neo4j_conn_params()
        
        print(f"✅ PostgreSQL params: {list(pg_params.keys())}")
        print(f"✅ Redis params: {list(redis_params.keys())}")
        print(f"✅ Neo4j params: {list(neo4j_params.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Failed connection builders: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CENTRALIZED DAG CONFIGURATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Global Settings", test_global_settings),
        ("Symbol Lists", test_symbol_lists),
        ("DAG Configurations", test_dag_configs),
        ("Default Args", test_default_args),
        ("Batch Config", test_batch_config),
        ("Circuit Breaker Configs", test_circuit_breaker_configs),
        ("Claude API Configs", test_claude_configs),
        ("Helper Functions", test_helper_functions),
        ("Connection Builders", test_connection_builders),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Configuration is valid.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    exit(main())