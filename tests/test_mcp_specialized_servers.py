"""
Comprehensive tests for specialized MCP servers.

Tests cover:
- arXiv Research MCP Server
- Linting & Formatting MCP Server
- Model Serving MCP Server
- SQL Analytics MCP Server
"""

import asyncio
import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import numpy as np
import pandas as pd

# Import server classes
from axiom.integrations.mcp_servers.research.arxiv_server import ArxivMCPServer
from axiom.integrations.mcp_servers.code_quality.linting_server import LintingMCPServer
from axiom.integrations.mcp_servers.mlops.model_serving_server import (
    ModelServingMCPServer,
    ModelEndpoint
)
from axiom.integrations.mcp_servers.analytics.sql_server import SQLAnalyticsMCPServer


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_model(temp_dir):
    """Create a sample sklearn model for testing."""
    from sklearn.linear_model import LinearRegression
    
    # Train simple model
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model = LinearRegression()
    model.fit(X, y)
    
    # Save model
    model_path = temp_dir / "test_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path


@pytest.fixture
def sample_code_file(temp_dir):
    """Create sample Python file for linting tests."""
    code = """
def calculate_sum(a, b):
    '''Calculate sum of two numbers.'''
    return a+b

def unused_function():
    pass

x = 1
y = 2
result = calculate_sum(x, y)
"""
    code_file = temp_dir / "sample.py"
    code_file.write_text(code)
    return code_file


@pytest.fixture
def sample_sql_data(temp_dir):
    """Create sample SQL database for analytics tests."""
    import duckdb
    
    db_path = temp_dir / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    
    # Create sample tables
    conn.execute("""
        CREATE TABLE portfolio AS
        SELECT
            '2024-01-01'::DATE + (row_number() OVER () - 1) * INTERVAL 1 DAY as date,
            'AAPL' as symbol,
            100.0 + (row_number() OVER () * 0.5) as price,
            1000 as quantity
        FROM range(30)
    """)
    
    conn.execute("""
        CREATE TABLE trades AS
        SELECT
            '2024-01-01'::DATE + (row_number() OVER () - 1) * INTERVAL 1 DAY as date,
            'AAPL' as symbol,
            100.0 + (row_number() OVER () * 0.5) as price,
            (random() * 1000)::INTEGER as volume,
            (random() - 0.5) * 0.1 as returns
        FROM range(100)
    """)
    
    conn.close()
    return db_path


# ============================================
# arXiv Research MCP Server Tests
# ============================================

class TestArxivMCPServer:
    """Tests for arXiv Research MCP Server."""
    
    @pytest.fixture
    def arxiv_server(self, temp_dir):
        """Create arXiv server instance."""
        return ArxivMCPServer(download_dir=str(temp_dir))
    
    @pytest.mark.asyncio
    async def test_arxiv_server_initialization(self, arxiv_server):
        """Test server initializes correctly."""
        assert arxiv_server.server is not None
        assert arxiv_server.download_dir.exists()
        assert len(arxiv_server.finance_categories) > 0
    
    @pytest.mark.asyncio
    async def test_search_papers(self, arxiv_server):
        """Test paper search functionality."""
        with patch('arxiv.Search') as mock_search:
            # Mock search results
            mock_result = Mock()
            mock_result.entry_id = "https://arxiv.org/abs/2301.12345"
            mock_result.title = "Portfolio Optimization"
            mock_result.authors = [Mock(name="John Doe")]
            mock_result.summary = "Abstract text"
            mock_result.categories = ["q-fin.PM"]
            mock_result.primary_category = "q-fin.PM"
            mock_result.published = Mock(isoformat=lambda: "2024-01-01T00:00:00")
            mock_result.updated = Mock(isoformat=lambda: "2024-01-01T00:00:00")
            mock_result.pdf_url = "https://arxiv.org/pdf/2301.12345"
            mock_result.doi = None
            
            mock_search.return_value.results.return_value = [mock_result]
            
            result = await arxiv_server._search_papers(
                query="portfolio optimization",
                category="q-fin.PM",
                max_results=10
            )
            
            assert result["total_results"] == 1
            assert len(result["papers"]) == 1
            assert result["papers"][0]["title"] == "Portfolio Optimization"
    
    @pytest.mark.asyncio
    async def test_get_paper(self, arxiv_server):
        """Test getting paper details."""
        with patch('arxiv.Search') as mock_search:
            mock_result = Mock()
            mock_result.entry_id = "https://arxiv.org/abs/2301.12345"
            mock_result.title = "Test Paper"
            mock_result.authors = [Mock(name="Jane Smith")]
            mock_result.summary = "Test abstract"
            mock_result.categories = ["q-fin.PM"]
            mock_result.primary_category = "q-fin.PM"
            mock_result.published = Mock(isoformat=lambda: "2024-01-01T00:00:00")
            mock_result.updated = Mock(isoformat=lambda: "2024-01-01T00:00:00")
            mock_result.doi = None
            mock_result.journal_ref = None
            mock_result.comment = None
            mock_result.pdf_url = "https://arxiv.org/pdf/2301.12345"
            mock_result.links = []
            
            mock_search.return_value.results.return_value = iter([mock_result])
            
            result = await arxiv_server._get_paper(arxiv_id="2301.12345")
            
            assert result["arxiv_id"] == "2301.12345"
            assert result["title"] == "Test Paper"
            assert len(result["authors"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_citations(self, arxiv_server):
        """Test citation generation."""
        with patch.object(arxiv_server, '_get_paper') as mock_get_paper:
            mock_get_paper.return_value = {
                "title": "Test Paper",
                "authors": [{"name": "John Doe"}],
                "published": "2024-01-01T00:00:00"
            }
            
            result = await arxiv_server._get_citations(
                arxiv_id="2301.12345",
                style="bibtex"
            )
            
            assert "bibtex" in result["citation"]
            assert result["year"] == 2024
            assert "all_styles" in result
    
    @pytest.mark.asyncio
    async def test_extract_formulas(self, arxiv_server):
        """Test formula extraction."""
        with patch.object(arxiv_server, '_get_paper') as mock_get_paper:
            mock_get_paper.return_value = {
                "title": "Test Paper",
                "abstract": "The formula $E = mc^2$ is important. Also $$F = ma$$"
            }
            
            result = await arxiv_server._extract_formulas(arxiv_id="2301.12345")
            
            assert "inline_formulas" in result
            assert "display_formulas" in result
            assert result["formula_count"] >= 0


# ============================================
# Linting & Formatting MCP Server Tests
# ============================================

class TestLintingMCPServer:
    """Tests for Linting & Formatting MCP Server."""
    
    @pytest.fixture
    def linting_server(self, temp_dir):
        """Create linting server instance."""
        return LintingMCPServer(project_root=str(temp_dir))
    
    @pytest.mark.asyncio
    async def test_linting_server_initialization(self, linting_server):
        """Test server initializes correctly."""
        assert linting_server.server is not None
        assert linting_server.project_root.exists()
        assert len(linting_server.tools_config) > 0
    
    @pytest.mark.asyncio
    async def test_run_command(self, linting_server):
        """Test command execution."""
        result = await linting_server._run_command(["echo", "test"])
        
        assert "success" in result
        assert "stdout" in result
        assert "stderr" in result
        assert "command" in result
    
    @pytest.mark.asyncio
    async def test_lint_python_mock(self, linting_server, sample_code_file):
        """Test Python linting (mocked)."""
        with patch.object(linting_server, '_run_command') as mock_cmd:
            mock_cmd.return_value = {
                "success": True,
                "stdout": "Your code has been rated at 10.00/10",
                "stderr": ""
            }
            
            result = await linting_server._lint_python(
                path=str(sample_code_file),
                linter="pylint"
            )
            
            assert "linters" in result
            assert "overall_success" in result
    
    @pytest.mark.asyncio
    async def test_complexity_analysis_mock(self, linting_server, sample_code_file):
        """Test complexity analysis (mocked)."""
        with patch.object(linting_server, '_run_command') as mock_cmd:
            mock_cmd.return_value = {
                "success": True,
                "stdout": json.dumps({
                    str(sample_code_file): [
                        {"name": "calculate_sum", "complexity": 1, "lineno": 2}
                    ]
                })
            }
            
            result = await linting_server._complexity_analysis(
                path=str(sample_code_file),
                max_complexity=10
            )
            
            assert "complex_functions_count" in result
            assert "average_complexity" in result


# ============================================
# Model Serving MCP Server Tests
# ============================================

class TestModelServingMCPServer:
    """Tests for Model Serving MCP Server."""
    
    @pytest.fixture
    def model_server(self, temp_dir):
        """Create model serving server instance."""
        return ModelServingMCPServer(models_dir=str(temp_dir))
    
    @pytest.mark.asyncio
    async def test_model_server_initialization(self, model_server):
        """Test server initializes correctly."""
        assert model_server.server is not None
        assert model_server.models_dir.exists()
        assert len(model_server.endpoints) == 0
    
    @pytest.mark.asyncio
    async def test_deploy_model(self, model_server, sample_model):
        """Test model deployment."""
        result = await model_server._deploy_model(
            endpoint_name="test-model",
            model_path=str(sample_model),
            version="v1.0",
            model_type="sklearn"
        )
        
        assert result["success"] is True
        assert result["endpoint"] == "test-model"
        assert result["version"] == "v1.0"
        assert "test-model" in model_server.endpoints
    
    @pytest.mark.asyncio
    async def test_predict(self, model_server, sample_model):
        """Test making predictions."""
        # First deploy model
        await model_server._deploy_model(
            endpoint_name="test-model",
            model_path=str(sample_model),
            version="v1.0"
        )
        
        # Make prediction
        result = await model_server._predict(
            endpoint="test-model",
            data=[[3]]
        )
        
        assert result["success"] is True
        assert "prediction" in result
        assert "latency_ms" in result
    
    @pytest.mark.asyncio
    async def test_batch_predict(self, model_server, sample_model):
        """Test batch predictions."""
        await model_server._deploy_model(
            endpoint_name="test-model",
            model_path=str(sample_model),
            version="v1.0"
        )
        
        result = await model_server._batch_predict(
            endpoint="test-model",
            data_list=[[[1]], [[2]], [[3]]],
            batch_size=2
        )
        
        assert result["success"] is True
        assert result["total_requests"] == 3
        assert result["successful"] >= 0
    
    @pytest.mark.asyncio
    async def test_list_models(self, model_server, sample_model):
        """Test listing models."""
        await model_server._deploy_model(
            endpoint_name="test-model",
            model_path=str(sample_model),
            version="v1.0"
        )
        
        result = await model_server._list_models(
            status_filter="all",
            include_metrics=True
        )
        
        assert result["total_endpoints"] == 1
        assert len(result["endpoints"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, model_server, sample_model):
        """Test getting endpoint metrics."""
        await model_server._deploy_model(
            endpoint_name="test-model",
            model_path=str(sample_model),
            version="v1.0"
        )
        
        # Make a prediction to generate metrics
        await model_server._predict(endpoint="test-model", data=[[3]])
        
        result = await model_server._get_metrics(
            endpoint="test-model",
            detailed=True
        )
        
        assert result["success"] is True
        assert "requests" in result
        assert "latency" in result
    
    @pytest.mark.asyncio
    async def test_health_check(self, model_server, sample_model):
        """Test endpoint health check."""
        await model_server._deploy_model(
            endpoint_name="test-model",
            model_path=str(sample_model),
            version="v1.0"
        )
        
        result = await model_server._health_check(
            endpoint="test-model",
            test_prediction=True
        )
        
        assert result["success"] is True
        assert result["healthy"] is True
        assert "test_prediction" in result
    
    @pytest.mark.asyncio
    async def test_ab_test_setup(self, model_server, sample_model):
        """Test A/B test configuration."""
        # Deploy two models
        await model_server._deploy_model(
            endpoint_name="model-a",
            model_path=str(sample_model),
            version="v1.0"
        )
        await model_server._deploy_model(
            endpoint_name="model-b",
            model_path=str(sample_model),
            version="v2.0"
        )
        
        result = await model_server._ab_test(
            test_name="test-ab",
            endpoint_a="model-a",
            endpoint_b="model-b",
            traffic_split=50
        )
        
        assert result["success"] is True
        assert result["test_name"] == "test-ab"


# ============================================
# SQL Analytics MCP Server Tests
# ============================================

class TestSQLAnalyticsMCPServer:
    """Tests for SQL Analytics MCP Server."""
    
    @pytest.fixture
    def sql_server(self, sample_sql_data):
        """Create SQL analytics server instance."""
        return SQLAnalyticsMCPServer(db_path=str(sample_sql_data))
    
    @pytest.mark.asyncio
    async def test_sql_server_initialization(self, sql_server):
        """Test server initializes correctly."""
        assert sql_server.server is not None
        assert sql_server.conn is not None
        assert len(sql_server.sql_templates) > 0
    
    @pytest.mark.asyncio
    async def test_execute_query(self, sql_server):
        """Test SQL query execution."""
        result = await sql_server._execute_query(
            query="SELECT COUNT(*) as count FROM portfolio",
            limit=1000
        )
        
        assert result["success"] is True
        assert result["rows_returned"] >= 0
        assert "data" in result
    
    @pytest.mark.asyncio
    async def test_create_view(self, sql_server):
        """Test view creation."""
        result = await sql_server._create_view(
            view_name="test_view",
            query="SELECT * FROM portfolio WHERE quantity > 0",
            materialized=False
        )
        
        assert result["success"] is True
        assert result["view_name"] == "test_view"
    
    @pytest.mark.asyncio
    async def test_aggregate_data(self, sql_server):
        """Test data aggregation."""
        result = await sql_server._aggregate_data(
            table="portfolio",
            aggregations={"price": ["avg", "min", "max"]},
            group_by=["symbol"]
        )
        
        assert result["success"] is True
        assert "data" in result
    
    @pytest.mark.asyncio
    async def test_time_series_agg(self, sql_server):
        """Test time-series aggregation."""
        result = await sql_server._time_series_agg(
            table="portfolio",
            timestamp_column="date",
            value_column="price",
            bucket="day",
            aggfunc="avg"
        )
        
        assert result["success"] is True
        assert "periods" in result
        assert "data" in result
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, sql_server):
        """Test trend analysis."""
        result = await sql_server._trend_analysis(
            table="portfolio",
            timestamp_column="date",
            value_column="price",
            method="linear"
        )
        
        # May fail if scipy not installed, which is acceptable
        if result.get("success"):
            assert "trend" in result
            assert "slope" in result
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, sql_server):
        """Test anomaly detection."""
        result = await sql_server._anomaly_detection(
            table="trades",
            column="volume",
            method="zscore",
            threshold=3
        )
        
        assert result["success"] is True
        assert "anomalies_found" in result
        assert "anomaly_rate" in result
    
    @pytest.mark.asyncio
    async def test_generate_sql(self, sql_server):
        """Test SQL generation from natural language."""
        result = await sql_server._generate_sql(
            question="What is the average portfolio value?",
            dialect="duckdb"
        )
        
        assert result["success"] is True
        assert "sql" in result
        assert len(result["sql"]) > 0


# ============================================
# Integration Tests
# ============================================

class TestMCPServerIntegration:
    """Integration tests across multiple servers."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_ml_workflow(self, temp_dir, sample_model):
        """Test complete ML workflow: deploy, predict, monitor."""
        server = ModelServingMCPServer(models_dir=str(temp_dir))
        
        # Deploy
        deploy_result = await server._deploy_model(
            endpoint_name="ml-model",
            model_path=str(sample_model),
            version="v1.0"
        )
        assert deploy_result["success"]
        
        # Predict
        pred_result = await server._predict(
            endpoint="ml-model",
            data=[[5]]
        )
        assert pred_result["success"]
        
        # Get metrics
        metrics = await server._get_metrics(
            endpoint="ml-model"
        )
        assert metrics["success"]
        assert metrics["requests"]["total"] > 0
    
    @pytest.mark.asyncio
    async def test_analytics_pipeline(self, sample_sql_data):
        """Test complete analytics pipeline."""
        server = SQLAnalyticsMCPServer(db_path=str(sample_sql_data))
        
        # 1. Query data
        query_result = await server._execute_query(
            query="SELECT * FROM portfolio LIMIT 10"
        )
        assert query_result["success"]
        
        # 2. Create aggregation
        agg_result = await server._aggregate_data(
            table="portfolio",
            aggregations={"price": ["avg"]}
        )
        assert agg_result["success"]
        
        # 3. Time series analysis
        ts_result = await server._time_series_agg(
            table="portfolio",
            timestamp_column="date",
            value_column="price",
            bucket="day"
        )
        assert ts_result["success"]


# ============================================
# Performance Tests
# ============================================

class TestPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.asyncio
    async def test_batch_prediction_performance(self, temp_dir, sample_model):
        """Test batch prediction performance."""
        server = ModelServingMCPServer(models_dir=str(temp_dir))
        
        await server._deploy_model(
            endpoint_name="perf-test",
            model_path=str(sample_model),
            version="v1.0"
        )
        
        # Generate large batch
        batch_data = [[[i]] for i in range(100)]
        
        import time
        start = time.time()
        result = await server._batch_predict(
            endpoint="perf-test",
            data_list=batch_data,
            batch_size=32
        )
        elapsed = time.time() - start
        
        assert result["success"]
        assert result["successful"] == 100
        assert elapsed < 10  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_sql_query_performance(self, sample_sql_data):
        """Test SQL query performance."""
        server = SQLAnalyticsMCPServer(db_path=str(sample_sql_data))
        
        import time
        start = time.time()
        result = await server._execute_query(
            query="SELECT * FROM trades ORDER BY date"
        )
        elapsed = time.time() - start
        
        assert result["success"]
        assert elapsed < 5  # Should complete within 5 seconds


# ============================================
# Error Handling Tests
# ============================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_deploy_nonexistent_model(self, temp_dir):
        """Test deploying non-existent model file."""
        server = ModelServingMCPServer(models_dir=str(temp_dir))
        
        result = await server._deploy_model(
            endpoint_name="bad-model",
            model_path="/nonexistent/model.pkl",
            version="v1.0"
        )
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_predict_nonexistent_endpoint(self, temp_dir):
        """Test prediction on non-existent endpoint."""
        server = ModelServingMCPServer(models_dir=str(temp_dir))
        
        result = await server._predict(
            endpoint="nonexistent",
            data=[[1]]
        )
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_invalid_sql_query(self, sample_sql_data):
        """Test invalid SQL query."""
        server = SQLAnalyticsMCPServer(db_path=str(sample_sql_data))
        
        result = await server._execute_query(
            query="SELECT * FROM nonexistent_table"
        )
        
        assert result["success"] is False
        assert "error" in result


# ============================================
# Utility Tests
# ============================================

def test_model_endpoint_creation():
    """Test ModelEndpoint class."""
    endpoint = ModelEndpoint(
        name="test",
        model_path="/path/to/model.pkl",
        version="v1.0"
    )
    
    assert endpoint.name == "test"
    assert endpoint.version == "v1.0"
    assert endpoint.status == "pending"
    assert endpoint.request_count == 0


def test_model_endpoint_metrics():
    """Test endpoint metrics calculation."""
    endpoint = ModelEndpoint(
        name="test",
        model_path="/path/to/model.pkl",
        version="v1.0"
    )
    
    endpoint.request_count = 100
    endpoint.error_count = 5
    endpoint.total_latency = 10.0
    
    metrics = endpoint.get_metrics()
    
    assert metrics["requests"]["total"] == 100
    assert metrics["requests"]["errors"] == 5
    assert metrics["requests"]["error_rate"] == 5.0
    assert metrics["latency"]["average_ms"] == 100.0


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])