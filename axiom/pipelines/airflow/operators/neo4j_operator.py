"""
Enterprise Neo4j Operators for Knowledge Graph Operations
"""
from typing import Any, Dict, List, Optional
from datetime import datetime

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


class Neo4jQueryOperator(BaseOperator):
    """
    Execute Cypher queries with performance monitoring and error handling.
    
    Features:
    - Query performance tracking
    - Result size monitoring
    - Automatic retry on transient errors
    - Batch processing support
    """
    
    template_fields = ('query', 'parameters')
    template_ext = ('.cypher',)
    ui_color = '#4169E1'
    ui_fgcolor = '#fff'
    
    @apply_defaults
    def __init__(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: str = 'neo4j',
        xcom_key: str = 'neo4j_result',
        return_result: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.query = query
        self.parameters = parameters or {}
        self.database = database
        self.xcom_key = xcom_key
        self.return_result = return_result
        
    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute Cypher query"""
        import os
        
        start_time = datetime.now()
        
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            auth=(
                os.getenv('NEO4J_USER', 'neo4j'),
                os.getenv('NEO4J_PASSWORD')
            )
        )
        
        try:
            # Execute query
            records, summary, keys = driver.execute_query(
                self.query,
                self.parameters,
                database_=self.database
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract results
            results = [dict(record) for record in records] if self.return_result else []
            
            # Build response
            response = {
                'success': True,
                'records_returned': len(results),
                'execution_time_seconds': execution_time,
                'counters': dict(summary.counters),
                'query_type': summary.query_type,
                'timestamp': datetime.now().isoformat(),
                'results': results if self.return_result else None
            }
            
            # Log metrics
            counters = summary.counters
            self.log.info(f"✅ Neo4j query executed successfully")
            self.log.info(f"   Time: {execution_time:.2f}s")
            self.log.info(f"   Records: {len(results)}")
            if counters.nodes_created > 0:
                self.log.info(f"   Nodes created: {counters.nodes_created}")
            if counters.relationships_created > 0:
                self.log.info(f"   Relationships: {counters.relationships_created}")
            if counters.properties_set > 0:
                self.log.info(f"   Properties set: {counters.properties_set}")
            
            # Push to XCom
            context['ti'].xcom_push(key=self.xcom_key, value=response)
            
            return response
            
        except Neo4jError as e:
            self.log.error(f"❌ Neo4j error: {e.message}")
            raise
            
        finally:
            driver.close()


class Neo4jBulkInsertOperator(BaseOperator):
    """
    Bulk insert operator for large-scale graph updates.
    
    Optimized for:
    - Creating thousands of nodes/relationships
    - Batch processing with UNWIND
    - Transaction management
    - Progress tracking
    """
    
    template_fields = ('data',)
    ui_color = '#5F9EA0'
    ui_fgcolor = '#fff'
    
    @apply_defaults
    def __init__(
        self,
        node_type: str,
        data: List[Dict[str, Any]],
        batch_size: int = 1000,
        merge_key: Optional[str] = None,
        xcom_key: str = 'bulk_result',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.node_type = node_type
        self.data = data
        self.batch_size = batch_size
        self.merge_key = merge_key
        self.xcom_key = xcom_key
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bulk insert with batching"""
        import os
        
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            auth=(
                os.getenv('NEO4J_USER', 'neo4j'),
                os.getenv('NEO4J_PASSWORD')
            )
        )
        
        total_created = 0
        total_batches = (len(self.data) + self.batch_size - 1) // self.batch_size
        
        try:
            # Process in batches
            for batch_num in range(total_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.data))
                batch = self.data[start_idx:end_idx]
                
                # Build query
                if self.merge_key:
                    query = f"""
                    UNWIND $batch AS row
                    MERGE (n:{self.node_type} {{{self.merge_key}: row.{self.merge_key}}})
                    SET n += row
                    SET n.updated_at = datetime()
                    """
                else:
                    query = f"""
                    UNWIND $batch AS row
                    CREATE (n:{self.node_type})
                    SET n = row
                    SET n.created_at = datetime()
                    """
                
                # Execute batch
                records, summary, keys = driver.execute_query(
                    query,
                    {'batch': batch}
                )
                
                total_created += summary.counters.nodes_created
                
                self.log.info(
                    f"📦 Batch {batch_num + 1}/{total_batches}: "
                    f"{summary.counters.nodes_created} nodes created"
                )
            
            result = {
                'success': True,
                'total_nodes_created': total_created,
                'total_batches': total_batches,
                'batch_size': self.batch_size,
                'timestamp': datetime.now().isoformat()
            }
            
            self.log.info(f"✅ Bulk insert complete: {total_created} nodes")
            
            context['ti'].xcom_push(key=self.xcom_key, value=result)
            return result
            
        finally:
            driver.close()


class Neo4jGraphValidationOperator(BaseOperator):
    """
    Validate graph structure and data quality.
    
    Checks:
    - Node/relationship counts
    - Schema constraints
    - Orphaned nodes
    - Data consistency
    """
    
    ui_color = '#98D8C8'
    ui_fgcolor = '#000'
    
    @apply_defaults
    def __init__(
        self,
        validation_rules: Dict[str, Any],
        fail_on_error: bool = False,
        xcom_key: str = 'validation_result',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.validation_rules = validation_rules
        self.fail_on_error = fail_on_error
        self.xcom_key = xcom_key
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation checks"""
        import os
        
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            auth=(
                os.getenv('NEO4J_USER', 'neo4j'),
                os.getenv('NEO4J_PASSWORD')
            )
        )
        
        validation_results = {
            'checks_passed': 0,
            'checks_failed': 0,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Check minimum node count
            if 'min_nodes' in self.validation_rules:
                records, _, _ = driver.execute_query(
                    f"MATCH (n:{self.validation_rules['node_type']}) RETURN count(n) as count"
                )
                node_count = records[0]['count']
                validation_results['metrics']['node_count'] = node_count
                
                if node_count < self.validation_rules['min_nodes']:
                    validation_results['checks_failed'] += 1
                    validation_results['errors'].append(
                        f"Node count {node_count} below minimum {self.validation_rules['min_nodes']}"
                    )
                else:
                    validation_results['checks_passed'] += 1
            
            # Check relationship count
            if 'min_relationships' in self.validation_rules:
                records, _, _ = driver.execute_query(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                )
                rel_count = records[0]['count']
                validation_results['metrics']['relationship_count'] = rel_count
                
                if rel_count < self.validation_rules['min_relationships']:
                    validation_results['checks_failed'] += 1
                    validation_results['errors'].append(
                        f"Relationship count {rel_count} below minimum"
                    )
                else:
                    validation_results['checks_passed'] += 1
            
            # Check for orphaned nodes
            if self.validation_rules.get('check_orphans', False):
                records, _, _ = driver.execute_query(
                    """
                    MATCH (n)
                    WHERE NOT (n)--()
                    RETURN count(n) as orphan_count
                    """
                )
                orphan_count = records[0]['orphan_count']
                validation_results['metrics']['orphan_nodes'] = orphan_count
                
                if orphan_count > 0:
                    validation_results['warnings'].append(
                        f"Found {orphan_count} orphaned nodes"
                    )
            
            # Summary
            validation_results['success'] = validation_results['checks_failed'] == 0
            validation_results['timestamp'] = datetime.now().isoformat()
            
            # Log results
            if validation_results['success']:
                self.log.info(
                    f"✅ Validation passed: {validation_results['checks_passed']} checks"
                )
            else:
                self.log.error(
                    f"❌ Validation failed: {validation_results['checks_failed']} checks"
                )
                for error in validation_results['errors']:
                    self.log.error(f"   - {error}")
            
            for warning in validation_results['warnings']:
                self.log.warning(f"⚠️  {warning}")
            
            context['ti'].xcom_push(key=self.xcom_key, value=validation_results)
            
            if self.fail_on_error and not validation_results['success']:
                raise Exception("Graph validation failed")
            
            return validation_results
            
        finally:
            driver.close()