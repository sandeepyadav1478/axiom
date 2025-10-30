"""PostgreSQL MCP Server Implementation.

Provides PostgreSQL database operations through MCP protocol:
- Execute queries
- Schema management
- Transaction handling
- Connection pooling
"""

import asyncio
import logging
from typing import Any, Optional

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

logger = logging.getLogger(__name__)


class PostgreSQLMCPServer:
    """PostgreSQL MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg is required for PostgreSQL MCP server. Install with: pip install asyncpg")
        
        self.config = config
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.database = config.get("database")
        self.user = config.get("user")
        self.password = config.get("password")
        self.max_connections = config.get("max_connections", 10)
        self._pool: Optional["asyncpg.Pool"] = None

    async def _ensure_pool(self) -> "asyncpg.Pool":
        """Ensure connection pool is created.

        Returns:
            Connection pool
        """
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=1,
                max_size=self.max_connections,
            )
        return self._pool

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute_query(
        self,
        query: str,
        parameters: Optional[list[Any]] = None,
        fetch_mode: str = "all",
    ) -> dict[str, Any]:
        """Execute SQL query.

        Args:
            query: SQL query string
            parameters: Query parameters
            fetch_mode: Result fetch mode (all, one, none)

        Returns:
            Query results
        """
        try:
            pool = await self._ensure_pool()
            
            async with pool.acquire() as conn:
                if fetch_mode == "all":
                    if parameters:
                        results = await conn.fetch(query, *parameters)
                    else:
                        results = await conn.fetch(query)
                    
                    # Convert to dict list
                    rows = [dict(row) for row in results]
                    
                    return {
                        "success": True,
                        "query": query,
                        "rows": rows,
                        "row_count": len(rows),
                        "fetch_mode": fetch_mode,
                    }
                
                elif fetch_mode == "one":
                    if parameters:
                        result = await conn.fetchrow(query, *parameters)
                    else:
                        result = await conn.fetchrow(query)
                    
                    row = dict(result) if result else None
                    
                    return {
                        "success": True,
                        "query": query,
                        "row": row,
                        "fetch_mode": fetch_mode,
                    }
                
                else:  # none
                    if parameters:
                        status = await conn.execute(query, *parameters)
                    else:
                        status = await conn.execute(query)
                    
                    return {
                        "success": True,
                        "query": query,
                        "status": status,
                        "fetch_mode": fetch_mode,
                    }

        except asyncpg.PostgresError as e:
            logger.error(f"PostgreSQL query failed: {e}")
            return {
                "success": False,
                "error": f"PostgreSQL error: {str(e)}",
                "query": query,
                "error_code": e.sqlstate if hasattr(e, 'sqlstate') else None,
            }
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "query": query,
            }

    async def get_schema(self, table_name: Optional[str] = None) -> dict[str, Any]:
        """Get database schema information.

        Args:
            table_name: Specific table name (None for all tables)

        Returns:
            Schema information
        """
        try:
            pool = await self._ensure_pool()
            
            async with pool.acquire() as conn:
                if table_name:
                    # Get specific table schema
                    query = """
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = $1
                        ORDER BY ordinal_position
                    """
                    columns = await conn.fetch(query, table_name)
                    
                    # Get constraints
                    constraint_query = """
                        SELECT constraint_name, constraint_type
                        FROM information_schema.table_constraints
                        WHERE table_name = $1
                    """
                    constraints = await conn.fetch(constraint_query, table_name)
                    
                    return {
                        "success": True,
                        "table_name": table_name,
                        "columns": [dict(col) for col in columns],
                        "constraints": [dict(const) for const in constraints],
                    }
                else:
                    # Get all tables
                    query = """
                        SELECT table_name, table_type
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                        ORDER BY table_name
                    """
                    tables = await conn.fetch(query)
                    
                    return {
                        "success": True,
                        "database": self.database,
                        "tables": [dict(table) for table in tables],
                        "table_count": len(tables),
                    }

        except asyncpg.PostgresError as e:
            logger.error(f"Schema query failed: {e}")
            return {
                "success": False,
                "error": f"Schema query failed: {str(e)}",
                "table_name": table_name,
            }
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return {
                "success": False,
                "error": f"Failed to get schema: {str(e)}",
                "table_name": table_name,
            }

    async def execute_transaction(
        self, queries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute multiple queries in a transaction.

        Args:
            queries: List of query dictionaries with 'query' and optional 'parameters'

        Returns:
            Transaction result
        """
        try:
            pool = await self._ensure_pool()
            results = []
            
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for query_dict in queries:
                        query = query_dict.get("query")
                        parameters = query_dict.get("parameters", [])
                        
                        if parameters:
                            result = await conn.execute(query, *parameters)
                        else:
                            result = await conn.execute(query)
                        
                        results.append({
                            "query": query,
                            "status": result,
                        })
            
            return {
                "success": True,
                "query_count": len(queries),
                "results": results,
            }

        except asyncpg.PostgresError as e:
            logger.error(f"Transaction failed: {e}")
            return {
                "success": False,
                "error": f"Transaction failed: {str(e)}",
                "query_count": len(queries),
            }
        except Exception as e:
            logger.error(f"Transaction execution failed: {e}")
            return {
                "success": False,
                "error": f"Transaction execution failed: {str(e)}",
                "query_count": len(queries),
            }

    async def get_table_stats(self, table_name: str) -> dict[str, Any]:
        """Get table statistics.

        Args:
            table_name: Table name

        Returns:
            Table statistics
        """
        try:
            pool = await self._ensure_pool()
            
            async with pool.acquire() as conn:
                # Get row count
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = await conn.fetchrow(count_query)
                row_count = count_result['count']
                
                # Get table size
                size_query = """
                    SELECT pg_size_pretty(pg_total_relation_size($1)) as size
                """
                size_result = await conn.fetchrow(size_query, table_name)
                table_size = size_result['size']
                
                return {
                    "success": True,
                    "table_name": table_name,
                    "row_count": row_count,
                    "table_size": table_size,
                }

        except asyncpg.PostgresError as e:
            logger.error(f"Stats query failed: {e}")
            return {
                "success": False,
                "error": f"Stats query failed: {str(e)}",
                "table_name": table_name,
            }
        except Exception as e:
            logger.error(f"Failed to get table stats: {e}")
            return {
                "success": False,
                "error": f"Failed to get table stats: {str(e)}",
                "table_name": table_name,
            }


def get_server_definition() -> dict[str, Any]:
    """Get PostgreSQL MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "postgres",
        "category": "storage",
        "description": "PostgreSQL database operations (queries, schema, transactions)",
        "tools": [
            {
                "name": "execute_query",
                "description": "Execute SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute",
                        },
                        "parameters": {
                            "type": "array",
                            "description": "Query parameters for parameterized queries",
                        },
                        "fetch_mode": {
                            "type": "string",
                            "enum": ["all", "one", "none"],
                            "description": "Result fetch mode",
                            "default": "all",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_schema",
                "description": "Get database schema information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Specific table name (omit for all tables)",
                        }
                    },
                },
            },
            {
                "name": "execute_transaction",
                "description": "Execute multiple queries in a transaction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "parameters": {"type": "array"},
                                },
                                "required": ["query"],
                            },
                            "description": "List of queries to execute",
                        }
                    },
                    "required": ["queries"],
                },
            },
            {
                "name": "get_table_stats",
                "description": "Get table statistics (row count, size)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Table name",
                        }
                    },
                    "required": ["table_name"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "critical",
            "category": "storage",
            "requires": ["asyncpg"],
        },
    }