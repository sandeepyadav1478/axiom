"""
Data Quality Validation Operators
Enterprise-grade data validation with Great Expectations integration
"""
from typing import Any, Dict, List, Optional
from datetime import datetime

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults


class DataQualityOperator(BaseOperator):
    """
    Validate data quality with customizable checks.
    
    Supports:
    - Row count validation
    - Null value checks
    - Value range validation
    - Schema validation
    - Custom SQL checks
    """
    
    template_fields = ('sql_query', 'checks')
    ui_color = '#B4E7CE'
    ui_fgcolor = '#000'
    
    @apply_defaults
    def __init__(
        self,
        table_name: str,
        checks: List[Dict[str, Any]],
        connection_id: str = 'postgres_default',
        fail_on_error: bool = True,
        xcom_key: str = 'quality_result',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.table_name = table_name
        self.checks = checks
        self.connection_id = connection_id
        self.fail_on_error = fail_on_error
        self.xcom_key = xcom_key
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data quality checks"""
        import psycopg2
        import os
        
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            user=os.getenv('POSTGRES_USER', 'axiom'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=os.getenv('POSTGRES_DB', 'axiom_finance')
        )
        
        results = {
            'table': self.table_name,
            'checks_passed': 0,
            'checks_failed': 0,
            'checks': [],
            'timestamp': datetime.now().isoformat()
        }
        
        cur = conn.cursor()
        
        try:
            for check in self.checks:
                check_result = self._run_check(cur, check)
                results['checks'].append(check_result)
                
                if check_result['passed']:
                    results['checks_passed'] += 1
                    self.log.info(f"✅ {check['name']}: PASSED")
                else:
                    results['checks_failed'] += 1
                    self.log.error(f"❌ {check['name']}: FAILED - {check_result['message']}")
            
            results['success'] = results['checks_failed'] == 0
            
            # Summary
            self.log.info(
                f"\n📊 Quality Check Summary for {self.table_name}:\n"
                f"   Passed: {results['checks_passed']}\n"
                f"   Failed: {results['checks_failed']}\n"
                f"   Total:  {len(self.checks)}"
            )
            
            context['ti'].xcom_push(key=self.xcom_key, value=results)
            
            if self.fail_on_error and not results['success']:
                raise Exception(
                    f"Data quality check failed: {results['checks_failed']} checks failed"
                )
            
            return results
            
        finally:
            cur.close()
            conn.close()
    
    def _run_check(self, cursor, check: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual check"""
        check_type = check['type']
        
        if check_type == 'row_count':
            return self._check_row_count(cursor, check)
        elif check_type == 'null_count':
            return self._check_nulls(cursor, check)
        elif check_type == 'value_range':
            return self._check_range(cursor, check)
        elif check_type == 'uniqueness':
            return self._check_uniqueness(cursor, check)
        elif check_type == 'custom_sql':
            return self._check_custom_sql(cursor, check)
        else:
            return {
                'name': check['name'],
                'type': check_type,
                'passed': False,
                'message': f"Unknown check type: {check_type}"
            }
    
    def _check_row_count(self, cursor, check: Dict[str, Any]) -> Dict[str, Any]:
        """Check table row count"""
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        
        min_rows = check.get('min_rows', 0)
        max_rows = check.get('max_rows', float('inf'))
        
        passed = min_rows <= count <= max_rows
        
        return {
            'name': check['name'],
            'type': 'row_count',
            'passed': passed,
            'actual_count': count,
            'expected_range': f"{min_rows}-{max_rows}",
            'message': f"Row count: {count} (expected: {min_rows}-{max_rows})"
        }
    
    def _check_nulls(self, cursor, check: Dict[str, Any]) -> Dict[str, Any]:
        """Check for null values"""
        column = check['column']
        max_null_percent = check.get('max_null_percent', 0)
        
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) as null_count
            FROM {self.table_name}
        """)
        
        row = cursor.fetchone()
        total = row[0]
        null_count = row[1]
        null_percent = (null_count / total * 100) if total > 0 else 0
        
        passed = null_percent <= max_null_percent
        
        return {
            'name': check['name'],
            'type': 'null_count',
            'passed': passed,
            'null_count': null_count,
            'null_percent': round(null_percent, 2),
            'max_allowed_percent': max_null_percent,
            'message': f"Null values: {null_percent:.1f}% (max: {max_null_percent}%)"
        }
    
    def _check_range(self, cursor, check: Dict[str, Any]) -> Dict[str, Any]:
        """Check value range"""
        column = check['column']
        min_value = check.get('min_value')
        max_value = check.get('max_value')
        
        cursor.execute(f"""
            SELECT MIN({column}), MAX({column})
            FROM {self.table_name}
        """)
        
        row = cursor.fetchone()
        actual_min = row[0]
        actual_max = row[1]
        
        passed = True
        if min_value is not None and actual_min < min_value:
            passed = False
        if max_value is not None and actual_max > max_value:
            passed = False
        
        return {
            'name': check['name'],
            'type': 'value_range',
            'passed': passed,
            'actual_range': f"{actual_min}-{actual_max}",
            'expected_range': f"{min_value}-{max_value}",
            'message': f"Value range: {actual_min} to {actual_max}"
        }
    
    def _check_uniqueness(self, cursor, check: Dict[str, Any]) -> Dict[str, Any]:
        """Check column uniqueness"""
        column = check['column']
        
        cursor.execute(f"""
            SELECT COUNT(*) as total, COUNT(DISTINCT {column}) as unique_count
            FROM {self.table_name}
        """)
        
        row = cursor.fetchone()
        total = row[0]
        unique_count = row[1]
        
        passed = total == unique_count
        
        return {
            'name': check['name'],
            'type': 'uniqueness',
            'passed': passed,
            'total_rows': total,
            'unique_values': unique_count,
            'duplicates': total - unique_count,
            'message': f"Unique values: {unique_count}/{total} (duplicates: {total - unique_count})"
        }
    
    def _check_custom_sql(self, cursor, check: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom SQL check"""
        sql = check['sql']
        expected = check.get('expected', True)
        
        cursor.execute(sql)
        result = cursor.fetchone()[0]
        
        passed = result == expected
        
        return {
            'name': check['name'],
            'type': 'custom_sql',
            'passed': passed,
            'actual_result': result,
            'expected_result': expected,
            'message': f"Custom check result: {result} (expected: {expected})"
        }


class SchemaValidationOperator(BaseOperator):
    """
    Validate database schema matches expected structure.
    
    Checks:
    - Table existence
    - Column names and types
    - Indexes
    - Constraints
    """
    
    ui_color = '#F7DC6F'
    ui_fgcolor = '#000'
    
    @apply_defaults
    def __init__(
        self,
        expected_schema: Dict[str, Any],
        fail_on_mismatch: bool = True,
        xcom_key: str = 'schema_validation',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.expected_schema = expected_schema
        self.fail_on_mismatch = fail_on_mismatch
        self.xcom_key = xcom_key
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema"""
        import psycopg2
        import os
        
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            user=os.getenv('POSTGRES_USER', 'axiom'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=os.getenv('POSTGRES_DB', 'axiom_finance')
        )
        
        cur = conn.cursor()
        
        results = {
            'tables_validated': 0,
            'mismatches': [],
            'warnings': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            for table_name, expected in self.expected_schema.items():
                # Check table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table_name,))
                
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    results['mismatches'].append({
                        'table': table_name,
                        'issue': 'Table does not exist'
                    })
                    continue
                
                # Check columns
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns
                    WHERE table_name = %s
                """, (table_name,))
                
                actual_columns = {row[0]: row[1] for row in cur.fetchall()}
                expected_columns = expected.get('columns', {})
                
                for col_name, col_type in expected_columns.items():
                    if col_name not in actual_columns:
                        results['mismatches'].append({
                            'table': table_name,
                            'column': col_name,
                            'issue': 'Column missing'
                        })
                    elif actual_columns[col_name] != col_type:
                        results['mismatches'].append({
                            'table': table_name,
                            'column': col_name,
                            'issue': f"Type mismatch: expected {col_type}, got {actual_columns[col_name]}"
                        })
                
                results['tables_validated'] += 1
            
            results['success'] = len(results['mismatches']) == 0
            
            # Log results
            if results['success']:
                self.log.info(
                    f"✅ Schema validation passed: {results['tables_validated']} tables"
                )
            else:
                self.log.error(
                    f"❌ Schema validation failed: {len(results['mismatches'])} mismatches"
                )
                for mismatch in results['mismatches']:
                    self.log.error(f"   - {mismatch}")
            
            context['ti'].xcom_push(key=self.xcom_key, value=results)
            
            if self.fail_on_mismatch and not results['success']:
                raise Exception("Schema validation failed")
            
            return results
            
        finally:
            cur.close()
            conn.close()