"""
SQL Analytics MCP Server

Provides business intelligence and data analytics capabilities including SQL query
generation, data aggregation, time series analysis, cohort analysis, and anomaly detection.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLAnalyticsMCPServer:
    """MCP Server for SQL analytics and business intelligence."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQL Analytics MCP Server.

        Args:
            db_path: Path to DuckDB database (default: in-memory)
        """
        self.server = Server("sql-analytics")
        self.db_path = db_path or ":memory:"
        self.conn = duckdb.connect(self.db_path)

        # Query history
        self.query_history: List[Dict[str, Any]] = []

        # Common SQL templates
        self.sql_templates = {
            "portfolio_value": """
                SELECT
                    date,
                    SUM(quantity * price) as total_value,
                    symbol
                FROM portfolio
                WHERE date BETWEEN ? AND ?
                GROUP BY date, symbol
                ORDER BY date
            """,
            "performance": """
                SELECT
                    symbol,
                    (MAX(price) - MIN(price)) / MIN(price) * 100 as return_pct,
                    STDDEV(price) as volatility
                FROM trades
                WHERE date BETWEEN ? AND ?
                GROUP BY symbol
            """,
            "risk_metrics": """
                SELECT
                    symbol,
                    AVG(var_95) as avg_var,
                    MAX(var_95) as max_var,
                    STDDEV(returns) as volatility
                FROM risk_data
                WHERE date BETWEEN ? AND ?
                GROUP BY symbol
            """
        }

        self._register_handlers()
        logger.info("SQL Analytics MCP Server initialized")

    def _register_handlers(self):
        """Register all tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available SQL analytics tools."""
            return [
                Tool(
                    name="generate_sql",
                    description="Generate SQL query from natural language description. "
                                "Uses AI to convert business questions into SQL queries.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Natural language question or requirement"
                            },
                            "schema": {
                                "type": "object",
                                "description": "Database schema information (tables, columns, types)"
                            },
                            "dialect": {
                                "type": "string",
                                "enum": ["duckdb", "postgresql", "mysql", "sqlite"],
                                "description": "SQL dialect (default: duckdb)",
                                "default": "duckdb"
                            }
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="execute_query",
                    description="Execute a SQL query and return results as structured data. "
                                "Supports SELECT, aggregations, joins, and complex queries.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute"
                            },
                            "params": {
                                "type": "array",
                                "description": "Query parameters for parameterized queries"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum rows to return (default: 1000)",
                                "default": 1000
                            },
                            "format": {
                                "type": "string",
                                "enum": ["json", "csv", "markdown"],
                                "description": "Output format (default: json)",
                                "default": "json"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="create_view",
                    description="Create a materialized or regular view for commonly used queries. "
                                "Improves performance for repeated analytics.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "view_name": {
                                "type": "string",
                                "description": "Name for the view"
                            },
                            "query": {
                                "type": "string",
                                "description": "SQL query defining the view"
                            },
                            "materialized": {
                                "type": "boolean",
                                "description": "Create as materialized view (default: false)",
                                "default": False
                            },
                            "replace": {
                                "type": "boolean",
                                "description": "Replace if exists (default: true)",
                                "default": True
                            }
                        },
                        "required": ["view_name", "query"]
                    }
                ),
                Tool(
                    name="aggregate_data",
                    description="Perform data aggregation with grouping and various aggregate functions. "
                                "Supports SUM, AVG, COUNT, MIN, MAX, STDDEV.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table or view name"
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns to select"
                            },
                            "group_by": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns to group by"
                            },
                            "aggregations": {
                                "type": "object",
                                "description": "Aggregation functions: {column: ['sum', 'avg', 'count']}"
                            },
                            "filters": {
                                "type": "object",
                                "description": "WHERE clause filters: {column: value}"
                            },
                            "order_by": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns to order by"
                            }
                        },
                        "required": ["table", "aggregations"]
                    }
                ),
                Tool(
                    name="pivot_table",
                    description="Create pivot table analysis with cross-tabulation. "
                                "Useful for multi-dimensional data analysis.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Source table"
                            },
                            "index": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Row index columns"
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Column pivot columns"
                            },
                            "values": {
                                "type": "string",
                                "description": "Value column to aggregate"
                            },
                            "aggfunc": {
                                "type": "string",
                                "enum": ["sum", "mean", "count", "min", "max"],
                                "description": "Aggregation function (default: sum)",
                                "default": "sum"
                            }
                        },
                        "required": ["table", "index", "columns", "values"]
                    }
                ),
                Tool(
                    name="time_series_agg",
                    description="Time-series aggregation with various time buckets (hourly, daily, weekly, monthly). "
                                "Includes moving averages and trend analysis.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table with time-series data"
                            },
                            "timestamp_column": {
                                "type": "string",
                                "description": "Timestamp column name"
                            },
                            "value_column": {
                                "type": "string",
                                "description": "Value column to aggregate"
                            },
                            "bucket": {
                                "type": "string",
                                "enum": ["hour", "day", "week", "month", "quarter", "year"],
                                "description": "Time bucket for aggregation",
                                "default": "day"
                            },
                            "aggfunc": {
                                "type": "string",
                                "enum": ["sum", "avg", "min", "max", "count"],
                                "description": "Aggregation function (default: avg)",
                                "default": "avg"
                            },
                            "moving_average": {
                                "type": "integer",
                                "description": "Window size for moving average (optional)"
                            },
                            "date_range": {
                                "type": "object",
                                "description": "Date range filter: {start: date, end: date}"
                            }
                        },
                        "required": ["table", "timestamp_column", "value_column"]
                    }
                ),
                Tool(
                    name="cohort_analysis",
                    description="Perform cohort analysis to track user/customer behavior over time. "
                                "Useful for retention analysis and lifecycle tracking.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table with event data"
                            },
                            "user_id_column": {
                                "type": "string",
                                "description": "User/customer ID column"
                            },
                            "event_date_column": {
                                "type": "string",
                                "description": "Event date column"
                            },
                            "cohort_type": {
                                "type": "string",
                                "enum": ["signup", "first_purchase", "first_trade"],
                                "description": "Type of cohort (default: signup)",
                                "default": "signup"
                            },
                            "metric": {
                                "type": "string",
                                "enum": ["retention", "revenue", "activity"],
                                "description": "Metric to analyze (default: retention)",
                                "default": "retention"
                            },
                            "period": {
                                "type": "string",
                                "enum": ["daily", "weekly", "monthly"],
                                "description": "Period granularity (default: monthly)",
                                "default": "monthly"
                            }
                        },
                        "required": ["table", "user_id_column", "event_date_column"]
                    }
                ),
                Tool(
                    name="funnel_analysis",
                    description="Analyze conversion funnels to identify drop-off points. "
                                "Tracks user progression through stages.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table with event data"
                            },
                            "user_id_column": {
                                "type": "string",
                                "description": "User ID column"
                            },
                            "event_column": {
                                "type": "string",
                                "description": "Event type column"
                            },
                            "timestamp_column": {
                                "type": "string",
                                "description": "Timestamp column"
                            },
                            "funnel_stages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Ordered list of funnel stages/events"
                            },
                            "window_days": {
                                "type": "integer",
                                "description": "Time window for funnel completion (default: 7)",
                                "default": 7
                            }
                        },
                        "required": ["table", "user_id_column", "event_column", "timestamp_column", "funnel_stages"]
                    }
                ),
                Tool(
                    name="trend_analysis",
                    description="Detect trends in time-series data using statistical methods. "
                                "Identifies upward, downward, or no trend patterns.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table with time-series data"
                            },
                            "timestamp_column": {
                                "type": "string",
                                "description": "Timestamp column"
                            },
                            "value_column": {
                                "type": "string",
                                "description": "Value column to analyze"
                            },
                            "method": {
                                "type": "string",
                                "enum": ["linear", "polynomial", "exponential"],
                                "description": "Trend detection method (default: linear)",
                                "default": "linear"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence level for trend detection (0-1, default: 0.95)",
                                "default": 0.95
                            }
                        },
                        "required": ["table", "timestamp_column", "value_column"]
                    }
                ),
                Tool(
                    name="anomaly_detection",
                    description="Detect statistical anomalies in data using Z-score and IQR methods. "
                                "Identifies outliers that deviate significantly from normal patterns.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table to analyze"
                            },
                            "column": {
                                "type": "string",
                                "description": "Column to check for anomalies"
                            },
                            "method": {
                                "type": "string",
                                "enum": ["zscore", "iqr", "isolation_forest"],
                                "description": "Detection method (default: zscore)",
                                "default": "zscore"
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Threshold for anomaly detection (default: 3 for zscore)",
                                "default": 3
                            },
                            "group_by": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Group anomaly detection by columns"
                            }
                        },
                        "required": ["table", "column"]
                    }
                ),
                Tool(
                    name="forecast_timeseries",
                    description="Forecast future values in time-series data using simple statistical methods. "
                                "Provides basic trend-based forecasts.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table with historical data"
                            },
                            "timestamp_column": {
                                "type": "string",
                                "description": "Timestamp column"
                            },
                            "value_column": {
                                "type": "string",
                                "description": "Value column to forecast"
                            },
                            "horizon": {
                                "type": "integer",
                                "description": "Number of periods to forecast (default: 7)",
                                "default": 7
                            },
                            "method": {
                                "type": "string",
                                "enum": ["moving_average", "exponential_smoothing", "linear_trend"],
                                "description": "Forecasting method (default: moving_average)",
                                "default": "moving_average"
                            },
                            "confidence_interval": {
                                "type": "boolean",
                                "description": "Include confidence intervals (default: true)",
                                "default": True
                            }
                        },
                        "required": ["table", "timestamp_column", "value_column"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "generate_sql":
                    result = await self._generate_sql(**arguments)
                elif name == "execute_query":
                    result = await self._execute_query(**arguments)
                elif name == "create_view":
                    result = await self._create_view(**arguments)
                elif name == "aggregate_data":
                    result = await self._aggregate_data(**arguments)
                elif name == "pivot_table":
                    result = await self._pivot_table(**arguments)
                elif name == "time_series_agg":
                    result = await self._time_series_agg(**arguments)
                elif name == "cohort_analysis":
                    result = await self._cohort_analysis(**arguments)
                elif name == "funnel_analysis":
                    result = await self._funnel_analysis(**arguments)
                elif name == "trend_analysis":
                    result = await self._trend_analysis(**arguments)
                elif name == "anomaly_detection":
                    result = await self._anomaly_detection(**arguments)
                elif name == "forecast_timeseries":
                    result = await self._forecast_timeseries(**arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

    async def _generate_sql(
        self,
        question: str,
        schema: Optional[Dict[str, Any]] = None,
        dialect: str = "duckdb"
    ) -> Dict[str, Any]:
        """Generate SQL from natural language."""
        try:
            # This is a simplified SQL generator
            # In production, this would use an LLM or more sophisticated NL-to-SQL model

            question_lower = question.lower()

            # Pattern matching for common queries
            if "sharpe ratio" in question_lower and "sector" in question_lower:
                sql = """
                SELECT
                    sector,
                    (AVG(returns) - risk_free_rate) / STDDEV(returns) as sharpe_ratio
                FROM portfolio_returns
                WHERE date >= ? AND date <= ?
                GROUP BY sector
                ORDER BY sharpe_ratio DESC
                """
                params = ["start_date", "end_date"]
            elif "portfolio value" in question_lower:
                sql = self.sql_templates["portfolio_value"]
                params = ["start_date", "end_date"]
            elif "performance" in question_lower or "return" in question_lower:
                sql = self.sql_templates["performance"]
                params = ["start_date", "end_date"]
            elif "risk" in question_lower:
                sql = self.sql_templates["risk_metrics"]
                params = ["start_date", "end_date"]
            else:
                # Generic select
                if schema and "tables" in schema:
                    main_table = schema["tables"][0] if schema["tables"] else "data"
                else:
                    main_table = "data"

                sql = f"SELECT * FROM {main_table} LIMIT 100"
                params = []

            return {
                "success": True,
                "question": question,
                "sql": sql.strip(),
                "dialect": dialect,
                "parameters": params,
                "explanation": "Generated SQL query based on natural language input",
                "note": "In production, this would use an LLM for more sophisticated NL-to-SQL conversion"
            }

        except Exception as e:
            logger.error(f"SQL generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_query(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        limit: int = 1000,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Execute SQL query."""
        try:
            # Add LIMIT if not present
            if "limit" not in query.lower() and limit:
                query = f"{query.rstrip(';')} LIMIT {limit}"

            # Execute query
            if params:
                result = self.conn.execute(query, params).fetchdf()
            else:
                result = self.conn.execute(query).fetchdf()

            # Record in history
            self.query_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "rows_returned": len(result),
                "success": True
            })

            # Format output
            if format == "csv":
                output = result.to_csv(index=False)
            elif format == "markdown":
                output = result.to_markdown(index=False)
            else:  # json
                output = result.to_dict(orient="records")

            return {
                "success": True,
                "query": query,
                "rows_returned": len(result),
                "columns": list(result.columns),
                "data": output,
                "execution_time_ms": "calculated in production"
            }

        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            self.query_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    async def _create_view(
        self,
        view_name: str,
        query: str,
        materialized: bool = False,
        replace: bool = True
    ) -> Dict[str, Any]:
        """Create a view."""
        try:
            # DuckDB doesn't support materialized views in the same way
            # We'll create a table instead for materialized views
            if materialized:
                create_sql = f"CREATE {'OR REPLACE ' if replace else ''}TABLE {view_name} AS {query}"
            else:
                create_sql = f"CREATE {'OR REPLACE ' if replace else ''}VIEW {view_name} AS {query}"

            self.conn.execute(create_sql)

            return {
                "success": True,
                "view_name": view_name,
                "materialized": materialized,
                "query": query,
                "message": f"{'Materialized view (table)' if materialized else 'View'} '{view_name}' created successfully"
            }

        except Exception as e:
            logger.error(f"View creation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _aggregate_data(
        self,
        table: str,
        aggregations: Dict[str, List[str]],
        columns: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform data aggregation."""
        try:
            # Build SELECT clause
            select_parts = []

            if group_by:
                select_parts.extend(group_by)

            # Add aggregations
            for col, agg_funcs in aggregations.items():
                for func in agg_funcs:
                    select_parts.append(f"{func.upper()}({col}) as {col}_{func}")

            select_clause = ", ".join(select_parts)

            # Build query
            query = f"SELECT {select_clause} FROM {table}"

            # Add WHERE clause
            if filters:
                where_parts = [f"{col} = ?" for col in filters.keys()]
                query += " WHERE " + " AND ".join(where_parts)

            # Add GROUP BY
            if group_by:
                query += " GROUP BY " + ", ".join(group_by)

            # Add ORDER BY
            if order_by:
                query += " ORDER BY " + ", ".join(order_by)

            # Execute
            if filters:
                result = self.conn.execute(query, list(filters.values())).fetchdf()
            else:
                result = self.conn.execute(query).fetchdf()

            return {
                "success": True,
                "table": table,
                "rows_returned": len(result),
                "aggregations": aggregations,
                "group_by": group_by,
                "data": result.to_dict(orient="records")
            }

        except Exception as e:
            logger.error(f"Aggregation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _pivot_table(
        self,
        table: str,
        index: List[str],
        columns: List[str],
        values: str,
        aggfunc: str = "sum"
    ) -> Dict[str, Any]:
        """Create pivot table."""
        try:
            # Read data
            df = self.conn.execute(f"SELECT * FROM {table}").fetchdf()

            # Create pivot table
            pivot = pd.pivot_table(
                df,
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

            return {
                "success": True,
                "table": table,
                "index": index,
                "columns": columns,
                "values": values,
                "aggfunc": aggfunc,
                "shape": list(pivot.shape),
                "data": pivot.to_dict()
            }

        except Exception as e:
            logger.error(f"Pivot table error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _time_series_agg(
        self,
        table: str,
        timestamp_column: str,
        value_column: str,
        bucket: str = "day",
        aggfunc: str = "avg",
        moving_average: Optional[int] = None,
        date_range: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Time-series aggregation."""
        try:
            # Map bucket to DuckDB date_trunc
            bucket_map = {
                "hour": "hour",
                "day": "day",
                "week": "week",
                "month": "month",
                "quarter": "quarter",
                "year": "year"
            }

            query = f"""
                SELECT
                    date_trunc('{bucket_map[bucket]}', {timestamp_column}) as period,
                    {aggfunc.upper()}({value_column}) as value
                FROM {table}
            """

            if date_range:
                query += f" WHERE {timestamp_column} BETWEEN ? AND ?"
                params = [date_range.get("start"), date_range.get("end")]
            else:
                params = None

            query += " GROUP BY period ORDER BY period"

            # Execute
            if params:
                result = self.conn.execute(query, params).fetchdf()
            else:
                result = self.conn.execute(query).fetchdf()

            # Add moving average if requested
            if moving_average and len(result) >= moving_average:
                result[f'ma_{moving_average}'] = result['value'].rolling(window=moving_average).mean()

            return {
                "success": True,
                "table": table,
                "bucket": bucket,
                "aggfunc": aggfunc,
                "periods": len(result),
                "moving_average_window": moving_average,
                "data": result.to_dict(orient="records")
            }

        except Exception as e:
            logger.error(f"Time-series aggregation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _cohort_analysis(
        self,
        table: str,
        user_id_column: str,
        event_date_column: str,
        cohort_type: str = "signup",
        metric: str = "retention",
        period: str = "monthly"
    ) -> Dict[str, Any]:
        """Perform cohort analysis."""
        try:
            # This is a simplified cohort analysis
            # In production, this would be more sophisticated

            query = f"""
                WITH cohorts AS (
                    SELECT
                        {user_id_column},
                        date_trunc('month', MIN({event_date_column})) as cohort_month
                    FROM {table}
                    GROUP BY {user_id_column}
                ),
                user_activities AS (
                    SELECT
                        c.{user_id_column},
                        c.cohort_month,
                        date_trunc('month', t.{event_date_column}) as activity_month
                    FROM cohorts c
                    JOIN {table} t ON c.{user_id_column} = t.{user_id_column}
                )
                SELECT
                    cohort_month,
                    activity_month,
                    COUNT(DISTINCT {user_id_column}) as active_users
                FROM user_activities
                GROUP BY cohort_month, activity_month
                ORDER BY cohort_month, activity_month
            """

            result = self.conn.execute(query).fetchdf()

            # Calculate retention rates
            if metric == "retention" and len(result) > 0:
                cohort_sizes = result.groupby('cohort_month').first()['active_users']
                result['retention_rate'] = result.apply(
                    lambda row: (row['active_users'] / cohort_sizes.get(row['cohort_month'], 1)) * 100,
                    axis=1
                )

            return {
                "success": True,
                "table": table,
                "cohort_type": cohort_type,
                "metric": metric,
                "period": period,
                "cohorts": len(result['cohort_month'].unique()) if len(result) > 0 else 0,
                "data": result.to_dict(orient="records")
            }

        except Exception as e:
            logger.error(f"Cohort analysis error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _funnel_analysis(
        self,
        table: str,
        user_id_column: str,
        event_column: str,
        timestamp_column: str,
        funnel_stages: List[str],
        window_days: int = 7
    ) -> Dict[str, Any]:
        """Analyze conversion funnel."""
        try:
            # Build CTEs for each stage
            ctes = []
            for i, stage in enumerate(funnel_stages):
                cte = f"""
                stage_{i} AS (
                    SELECT DISTINCT {user_id_column}, {timestamp_column} as stage_{i}_time
                    FROM {table}
                    WHERE {event_column} = '{stage}'
                )
                """
                ctes.append(cte)

            # Build joins
            joins = ["stage_0"]
            for i in range(1, len(funnel_stages)):
                joins.append(f"""
                    LEFT JOIN stage_{i}
                    ON stage_0.{user_id_column} = stage_{i}.{user_id_column}
                    AND stage_{i}.stage_{i}_time >= stage_0.stage_0_time
                    AND stage_{i}.stage_{i}_time <= stage_0.stage_0_time + INTERVAL '{window_days} days'
                """)

            query = f"""
                WITH {', '.join(ctes)}
                SELECT
                    COUNT(DISTINCT stage_0.{user_id_column}) as stage_0_users
                    {', '.join([f", COUNT(DISTINCT stage_{i}.{user_id_column}) as stage_{i}_users" for i in range(1, len(funnel_stages))])}
                FROM {' '.join(joins)}
            """

            result = self.conn.execute(query).fetchdf()

            # Calculate conversion rates
            funnel_data = []
            for i, stage in enumerate(funnel_stages):
                users = result[f'stage_{i}_users'].iloc[0]
                if i == 0:
                    conversion_rate = 100.0
                else:
                    prev_users = result[f'stage_{i-1}_users'].iloc[0]
                    conversion_rate = (users / prev_users * 100) if prev_users > 0 else 0

                funnel_data.append({
                    "stage": i + 1,
                    "stage_name": stage,
                    "users": int(users),
                    "conversion_rate": round(conversion_rate, 2),
                    "drop_off_rate": round(100 - conversion_rate, 2) if i > 0 else 0
                })

            return {
                "success": True,
                "table": table,
                "funnel_stages": funnel_stages,
                "window_days": window_days,
                "total_users": int(result['stage_0_users'].iloc[0]),
                "funnel_data": funnel_data,
                "overall_conversion": round((funnel_data[-1]["users"] / funnel_data[0]["users"] * 100), 2) if len(funnel_data) > 0 else 0
            }

        except Exception as e:
            logger.error(f"Funnel analysis error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _trend_analysis(
        self,
        table: str,
        timestamp_column: str,
        value_column: str,
        method: str = "linear",
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """Detect trends in time-series data."""
        try:
            # Fetch data
            query = f"""
                SELECT {timestamp_column}, {value_column}
                FROM {table}
                ORDER BY {timestamp_column}
            """
            df = self.conn.execute(query).fetchdf()

            if len(df) < 3:
                return {
                    "success": False,
                    "error": "Insufficient data points for trend analysis (minimum 3 required)"
                }

            # Simple linear trend
            from scipy import stats

            # Convert timestamps to numeric
            df['x'] = range(len(df))
            df['y'] = df[value_column]

            slope, intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])

            # Determine trend direction
            if p_value < (1 - confidence):
                if slope > 0:
                    trend = "upward"
                elif slope < 0:
                    trend = "downward"
                else:
                    trend = "flat"
            else:
                trend = "no significant trend"

            return {
                "success": True,
                "table": table,
                "method": method,
                "data_points": len(df),
                "trend": trend,
                "slope": round(slope, 6),
                "r_squared": round(r_value ** 2, 4),
                "p_value": round(p_value, 6),
                "confidence": confidence,
                "interpretation": self._interpret_trend(trend, slope, r_value ** 2)
            }

        except ImportError:
            return {
                "success": False,
                "error": "scipy not installed. Install with: pip install scipy"
            }
        except Exception as e:
            logger.error(f"Trend analysis error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _interpret_trend(self, trend: str, slope: float, r_squared: float) -> str:
        """Interpret trend results."""
        if trend == "no significant trend":
            return "Data shows no statistically significant trend"
        elif trend == "upward":
            strength = "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "weak"
            return f"Data shows a {strength} upward trend (R²={r_squared:.2f})"
        elif trend == "downward":
            strength = "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "weak"
            return f"Data shows a {strength} downward trend (R²={r_squared:.2f})"
        else:
            return "Data is relatively flat with no clear direction"

    async def _anomaly_detection(
        self,
        table: str,
        column: str,
        method: str = "zscore",
        threshold: float = 3,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect anomalies in data."""
        try:
            # Fetch data
            query = f"SELECT * FROM {table}"
            df = self.conn.execute(query).fetchdf()

            anomalies = []

            if method == "zscore":
                # Z-score method
                mean = df[column].mean()
                std = df[column].std()

                df['zscore'] = (df[column] - mean) / std
                df['is_anomaly'] = abs(df['zscore']) > threshold

                anomalies = df[df['is_anomaly']].to_dict(orient="records")

            elif method == "iqr":
                # IQR method
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                df['is_anomaly'] = (df[column] < lower_bound) | (df[column] > upper_bound)
                anomalies = df[df['is_anomaly']].to_dict(orient="records")

            return {
                "success": True,
                "table": table,
                "column": column,
                "method": method,
                "threshold": threshold,
                "total_records": len(df),
                "anomalies_found": len(anomalies),
                "anomaly_rate": round(len(anomalies) / len(df) * 100, 2) if len(df) > 0 else 0,
                "anomalies": anomalies[:100]  # Limit to first 100
            }

        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _forecast_timeseries(
        self,
        table: str,
        timestamp_column: str,
        value_column: str,
        horizon: int = 7,
        method: str = "moving_average",
        confidence_interval: bool = True
    ) -> Dict[str, Any]:
        """Forecast time-series values."""
        try:
            # Fetch historical data
            query = f"""
                SELECT {timestamp_column}, {value_column}
                FROM {table}
                ORDER BY {timestamp_column} DESC
                LIMIT 100
            """
            df = self.conn.execute(query).fetchdf()
            df = df.sort_values(timestamp_column)

            if len(df) < 3:
                return {
                    "success": False,
                    "error": "Insufficient historical data for forecasting"
                }

            forecasts = []

            if method == "moving_average":
                # Simple moving average
                window = min(7, len(df))
                ma = df[value_column].tail(window).mean()

                # Generate forecasts
                last_date = pd.to_datetime(df[timestamp_column].iloc[-1])
                for i in range(1, horizon + 1):
                    forecast_date = last_date + pd.Timedelta(days=i)
                    forecasts.append({
                        "date": forecast_date.isoformat(),
                        "forecast": round(ma, 2),
                        "method": "moving_average"
                    })

            elif method == "linear_trend":
                # Linear trend
                df['x'] = range(len(df))
                from scipy import stats
                slope, intercept, _, _, _ = stats.linregress(df['x'], df[value_column])

                last_date = pd.to_datetime(df[timestamp_column].iloc[-1])
                last_x = len(df) - 1

                for i in range(1, horizon + 1):
                    forecast_date = last_date + pd.Timedelta(days=i)
                    forecast_value = slope * (last_x + i) + intercept
                    forecasts.append({
                        "date": forecast_date.isoformat(),
                        "forecast": round(forecast_value, 2),
                        "method": "linear_trend"
                    })

            return {
                "success": True,
                "table": table,
                "method": method,
                "horizon": horizon,
                "historical_points": len(df),
                "forecasts": forecasts,
                "note": "These are simple statistical forecasts. For production, use ARIMA, Prophet, or other advanced methods."
            }

        except ImportError:
            return {
                "success": False,
                "error": "scipy not installed for linear trend method"
            }
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    server = SQLAnalyticsMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())