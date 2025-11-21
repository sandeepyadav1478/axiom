"""
Centralized Configuration Loader for Airflow DAGs
Loads and provides access to dag_config.yaml settings
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import timedelta


class DagConfig:
    """Centralized DAG configuration loader"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to load config once"""
        if cls._instance is None:
            cls._instance = super(DagConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent / "dag_configs" / "dag_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration"""
        return self._config
    
    def get_global(self, key: str, default: Any = None) -> Any:
        """Get global configuration value"""
        return self._config.get('global', {}).get(key, default)
    
    def get_symbols(self, list_name: str = 'primary') -> List[str]:
        """Get symbol list by name (primary or extended)"""
        symbols = self._config.get('symbols', {})
        if list_name == 'primary':
            return symbols.get('primary', [])
        elif list_name == 'extended':
            return symbols.get('primary', []) + symbols.get('extended', [])
        return []
    
    def get_dag_config(self, dag_name: str) -> Dict[str, Any]:
        """Get configuration for specific DAG"""
        return self._config.get(dag_name, {})
    
    def get_default_args(self, dag_name: str = None) -> Dict[str, Any]:
        """Get default_args for a DAG"""
        global_config = self._config.get('global', {})
        dag_config = self._config.get(dag_name, {}) if dag_name else {}
        
        # Build default_args from global and DAG-specific settings
        default_args = {
            'owner': global_config.get('owner', 'axiom'),
            'depends_on_past': False,
            'email': global_config.get('email', ['admin@axiom.com']),
            'email_on_failure': dag_config.get('email_on_failure', global_config.get('email_on_failure', False)),
            'email_on_retry': global_config.get('email_on_retry', False),
            'retries': dag_config.get('retries', global_config.get('retries', 3)),
            'retry_delay': timedelta(minutes=dag_config.get('retry_delay_minutes', global_config.get('retry_delay_minutes', 5))),
        }
        
        # Add execution timeout if specified
        timeout_minutes = dag_config.get('execution_timeout_minutes')
        if timeout_minutes:
            default_args['execution_timeout'] = timedelta(minutes=timeout_minutes)
        
        return default_args
    
    def get_db_config(self, db_type: str) -> Dict[str, str]:
        """Get database configuration (returns env var names)"""
        global_config = self._config.get('global', {})
        return global_config.get(db_type, {})
    
    def get_schedule(self, dag_name: str) -> str:
        """Get schedule interval for DAG"""
        dag_config = self._config.get(dag_name, {})
        return dag_config.get('schedule_interval', '@hourly')
    
    def get_tags(self, dag_name: str) -> List[str]:
        """Get tags for DAG"""
        dag_config = self._config.get(dag_name, {})
        return dag_config.get('tags', [])
    
    def get_circuit_breaker_config(self, dag_name: str) -> Dict[str, int]:
        """Get circuit breaker configuration for DAG"""
        dag_config = self._config.get(dag_name, {})
        return dag_config.get('circuit_breaker', {
            'failure_threshold': 5,
            'recovery_timeout_seconds': 60
        })
    
    def get_claude_config(self, dag_name: str) -> Dict[str, Any]:
        """Get Claude API configuration for DAG"""
        dag_config = self._config.get(dag_name, {})
        return dag_config.get('claude', {
            'cache_ttl_hours': 24,
            'track_cost': True,
            'max_tokens': 4096
        })
    
    def get_neo4j_config(self, dag_name: str) -> Dict[str, Any]:
        """Get Neo4j configuration for DAG"""
        dag_config = self._config.get(dag_name, {})
        return dag_config.get('neo4j', {})
    
    def get_validation_thresholds(self) -> Dict[str, Any]:
        """Get validation thresholds for data quality DAG"""
        validation_config = self._config.get('data_quality_validation', {})
        return validation_config.get('thresholds', {})
    
    def get_batch_config(self) -> Dict[str, Any]:
        """Get batch validation configuration"""
        validation_config = self._config.get('data_quality_validation', {})
        return validation_config.get('batch', {
            'enabled': True,
            'window_minutes': 5,
            'min_records_to_validate': 1
        })
    
    def get_correlation_config(self) -> Dict[str, Any]:
        """Get correlation analysis configuration"""
        corr_config = self._config.get('correlation_analyzer', {})
        return corr_config.get('correlation', {})
    
    def get_news_config(self) -> Dict[str, Any]:
        """Get news fetching configuration"""
        events_config = self._config.get('events_tracker', {})
        return events_config.get('news', {})
    
    def get_cost_tracking_config(self) -> Dict[str, Any]:
        """Get cost tracking configuration"""
        return self._config.get('cost_tracking', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self._config.get('monitoring', {})


# Global instance for easy import
dag_config = DagConfig()


def get_env_value(env_var_name: str, default: str = None) -> str:
    """Helper to get environment variable value"""
    return os.getenv(env_var_name, default)


def build_postgres_conn_params() -> Dict[str, str]:
    """Build PostgreSQL connection parameters from environment"""
    db_config = dag_config.get_db_config('postgres')
    return {
        'host': get_env_value(db_config.get('host_env', 'POSTGRES_HOST')),
        'user': get_env_value(db_config.get('user_env', 'POSTGRES_USER')),
        'password': get_env_value(db_config.get('password_env', 'POSTGRES_PASSWORD')),
        'database': get_env_value(db_config.get('database_env', 'POSTGRES_DB'))
    }


def build_redis_conn_params() -> Dict[str, str]:
    """Build Redis connection parameters from environment"""
    redis_config = dag_config.get_db_config('redis')
    return {
        'host': get_env_value(redis_config.get('host_env', 'REDIS_HOST')),
        'password': get_env_value(redis_config.get('password_env', 'REDIS_PASSWORD'))
    }


def build_neo4j_conn_params() -> Dict[str, str]:
    """Build Neo4j connection parameters from environment"""
    neo4j_config = dag_config.get_db_config('neo4j')
    return {
        'uri': get_env_value(neo4j_config.get('uri_env', 'NEO4J_URI')),
        'user': get_env_value(neo4j_config.get('user_env', 'NEO4J_USER')),
        'password': get_env_value(neo4j_config.get('password_env', 'NEO4J_PASSWORD'))
    }


# Convenience functions for common operations
def get_symbols_for_dag(dag_name: str) -> List[str]:
    """Get the appropriate symbol list for a DAG"""
    dag_conf = dag_config.get_dag_config(dag_name)
    symbols_list = dag_conf.get('symbols_list', 'primary')
    return dag_config.get_symbols(symbols_list)


def get_data_sources() -> Dict[str, Any]:
    """Get data source configuration for ingestion"""
    ingestion_config = dag_config.get_dag_config('data_ingestion')
    return ingestion_config.get('data_sources', {
        'primary': 'yahoo',
        'fallback': ['polygon', 'finnhub']
    })