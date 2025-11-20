"""
Dynamic DAG Factory - Generate Airflow DAGs from YAML Configuration
Enterprise-grade DAG generation system
"""
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import yaml
import os
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


class DAGFactory:
    """
    Factory class to generate Airflow DAGs from YAML configuration.
    
    Features:
    - YAML-based DAG definition
    - Dynamic operator instantiation
    - Template support
    - Validation
    - Hot-reload support
    """
    
    def __init__(self, config_dir: str = './dag_configs'):
        self.config_dir = Path(config_dir)
        self.operator_registry = self._build_operator_registry()
        
    def _build_operator_registry(self) -> Dict[str, Any]:
        """Build registry of available operators"""
        from axiom.pipelines.airflow.operators import (
            ClaudeOperator,
            CachedClaudeOperator,
            Neo4jQueryOperator,
            Neo4jBulkInsertOperator,
            MarketDataFetchOperator,
            DataQualityOperator,
            CircuitBreakerOperator,
            ResilientAPIOperator
        )
        
        return {
            'claude': ClaudeOperator,
            'cached_claude': CachedClaudeOperator,
            'neo4j_query': Neo4jQueryOperator,
            'neo4j_bulk_insert': Neo4jBulkInsertOperator,
            'market_data': MarketDataFetchOperator,
            'data_quality': DataQualityOperator,
            'circuit_breaker': CircuitBreakerOperator,
            'resilient_api': ResilientAPIOperator,
            'python': PythonOperator
        }
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load and validate YAML configuration"""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure"""
        required_fields = ['dag_id', 'schedule_interval', 'tasks']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(config['tasks'], list):
            raise ValueError("'tasks' must be a list")
        
        for task in config['tasks']:
            if 'task_id' not in task or 'operator' not in task:
                raise ValueError("Each task must have 'task_id' and 'operator'")
    
    def generate_dag(self, config_file: str) -> DAG:
        """Generate DAG from configuration"""
        config = self.load_config(config_file)
        
        # Build default_args
        default_args = config.get('default_args', {})
        default_args.setdefault('owner', 'axiom')
        default_args.setdefault('depends_on_past', False)
        default_args.setdefault('retries', 2)
        default_args.setdefault('retry_delay', timedelta(minutes=5))
        
        # Parse retry_delay if string
        if isinstance(default_args.get('retry_delay'), str):
            default_args['retry_delay'] = self._parse_timedelta(
                default_args['retry_delay']
            )
        
        # Create DAG
        dag = DAG(
            dag_id=config['dag_id'],
            default_args=default_args,
            description=config.get('description', ''),
            schedule_interval=config['schedule_interval'],
            start_date=self._parse_date(config.get('start_date', 'days_ago(1)')),
            catchup=config.get('catchup', False),
            tags=config.get('tags', []),
            max_active_runs=config.get('max_active_runs', 1)
        )
        
        # Create tasks
        tasks = {}
        with dag:
            for task_config in config['tasks']:
                task = self._create_task(task_config, dag)
                tasks[task_config['task_id']] = task
        
        # Set dependencies
        if 'dependencies' in config:
            self._set_dependencies(tasks, config['dependencies'])
        
        return dag
    
    def _create_task(self, task_config: Dict[str, Any], dag: DAG) -> Any:
        """Create task from configuration"""
        operator_type = task_config['operator']
        
        if operator_type not in self.operator_registry:
            raise ValueError(f"Unknown operator: {operator_type}")
        
        operator_class = self.operator_registry[operator_type]
        
        # Extract parameters
        params = task_config.get('params', {})
        
        # Handle special parameters
        if 'python_callable' in params and isinstance(params['python_callable'], str):
            params['python_callable'] = self._resolve_callable(
                params['python_callable']
            )
        
        # Create operator
        task = operator_class(
            task_id=task_config['task_id'],
            **params
        )
        
        return task
    
    def _set_dependencies(self, tasks: Dict[str, Any], dependencies: List[Dict]):
        """Set task dependencies"""
        for dep in dependencies:
            if 'upstream' in dep and 'downstream' in dep:
                upstream_ids = dep['upstream'] if isinstance(dep['upstream'], list) else [dep['upstream']]
                downstream_ids = dep['downstream'] if isinstance(dep['downstream'], list) else [dep['downstream']]
                
                upstream_tasks = [tasks[tid] for tid in upstream_ids]
                downstream_tasks = [tasks[tid] for tid in downstream_ids]
                
                for upstream in upstream_tasks:
                    for downstream in downstream_tasks:
                        upstream >> downstream
    
    def _parse_timedelta(self, delta_str: str) -> timedelta:
        """Parse timedelta from string (e.g., '5m', '1h', '30s')"""
        import re
        
        match = re.match(r'(\d+)([smhd])', delta_str)
        if not match:
            raise ValueError(f"Invalid timedelta format: {delta_str}")
        
        value, unit = match.groups()
        value = int(value)
        
        if unit == 's':
            return timedelta(seconds=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date from string"""
        if date_str.startswith('days_ago('):
            days = int(date_str.split('(')[1].split(')')[0])
            return days_ago(days)
        else:
            return datetime.fromisoformat(date_str)
    
    def _resolve_callable(self, callable_str: str) -> callable:
        """Resolve callable from string (e.g., 'module.function')"""
        parts = callable_str.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid callable format: {callable_str}")
        
        module_name, func_name = parts
        
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    
    def generate_all_dags(self) -> Dict[str, DAG]:
        """Generate all DAGs from config directory"""
        dags = {}
        
        if not self.config_dir.exists():
            return dags
        
        for config_file in self.config_dir.glob('*.yaml'):
            try:
                dag = self.generate_dag(config_file.name)
                dags[dag.dag_id] = dag
            except Exception as e:
                print(f"Error generating DAG from {config_file.name}: {e}")
        
        return dags


# Auto-generate DAGs from config directory
def load_dags_from_configs():
    """
    Load all DAGs from YAML configs.
    This function is called automatically by Airflow.
    """
    config_dir = os.path.join(
        os.path.dirname(__file__),
        '../dag_configs'
    )
    
    factory = DAGFactory(config_dir)
    dags = factory.generate_all_dags()
    
    # Export DAGs to globals so Airflow can find them
    for dag_id, dag in dags.items():
        globals()[dag_id] = dag
    
    return dags


# Template helper for creating common DAG patterns
class DAGTemplate:
    """
    Predefined DAG templates for common patterns.
    """
    
    @staticmethod
    def create_etl_pipeline(
        dag_id: str,
        extract_config: Dict,
        transform_config: Dict,
        load_config: Dict,
        schedule: str = '@hourly'
    ) -> Dict[str, Any]:
        """Create ETL pipeline configuration"""
        return {
            'dag_id': dag_id,
            'schedule_interval': schedule,
            'description': f'ETL Pipeline: {dag_id}',
            'tags': ['etl', 'pipeline'],
            'default_args': {
                'owner': 'axiom',
                'retries': 3,
                'retry_delay': '5m'
            },
            'tasks': [
                {
                    'task_id': 'extract',
                    'operator': extract_config['operator'],
                    'params': extract_config['params']
                },
                {
                    'task_id': 'transform',
                    'operator': transform_config['operator'],
                    'params': transform_config['params']
                },
                {
                    'task_id': 'load',
                    'operator': load_config['operator'],
                    'params': load_config['params']
                }
            ],
            'dependencies': [
                {'upstream': 'extract', 'downstream': 'transform'},
                {'upstream': 'transform', 'downstream': 'load'}
            ]
        }
    
    @staticmethod
    def create_ml_pipeline(
        dag_id: str,
        data_prep_tasks: List[Dict],
        training_task: Dict,
        evaluation_task: Dict,
        deployment_task: Dict,
        schedule: str = '@daily'
    ) -> Dict[str, Any]:
        """Create ML training pipeline configuration"""
        tasks = data_prep_tasks + [training_task, evaluation_task, deployment_task]
        
        dependencies = []
        # Data prep tasks run in parallel, then training
        for prep_task in data_prep_tasks:
            dependencies.append({
                'upstream': prep_task['task_id'],
                'downstream': training_task['task_id']
            })
        
        # Training -> Evaluation -> Deployment
        dependencies.extend([
            {'upstream': training_task['task_id'], 'downstream': evaluation_task['task_id']},
            {'upstream': evaluation_task['task_id'], 'downstream': deployment_task['task_id']}
        ])
        
        return {
            'dag_id': dag_id,
            'schedule_interval': schedule,
            'description': f'ML Pipeline: {dag_id}',
            'tags': ['ml', 'training', 'pipeline'],
            'tasks': tasks,
            'dependencies': dependencies
        }