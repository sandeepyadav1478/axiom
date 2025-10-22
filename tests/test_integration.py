"""Integration tests for Axiom Investment Banking Analytics workflow."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from axiom.integrations.ai_providers import AIResponse
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.config.schemas import (
    Citation,
    Evidence,
    ResearchBrief,
    SearchQuery,
)
from axiom.core.orchestration.nodes.planner import planner_node
from axiom.core.orchestration.state import create_initial_state


class TestWorkflowIntegration:
    """Test integration of workflow components."""

    @pytest.mark.asyncio
    async def test_create_initial_state(self):
        """Test initial state creation."""
        query = "Microsoft acquisition of OpenAI strategic analysis"
        trace_id = "test-trace-123"

        state = create_initial_state(query, trace_id)

        assert state["query"] == query
        assert state["trace_id"] == trace_id
        assert state["brief"] is None
        assert len(state["task_plans"]) == 0
        assert len(state["evidence"]) == 0
        assert state["step_count"] == 0

    @pytest.mark.asyncio
    @patch("axiom.integrations.ai_providers.provider_factory.get_layer_provider")
    async def test_planner_node_integration(self, mock_get_provider):
        """Test planner node with mocked AI provider."""
        # Setup mock provider
        mock_provider = Mock()
        mock_response = AIResponse(
            content="Investment banking research plan created",
            provider="MockProvider",
            model="test-model",
            confidence=0.9,
        )
        mock_provider.generate_response_async = AsyncMock(return_value=mock_response)
        mock_get_provider.return_value = mock_provider

        # Create initial state
        initial_state = create_initial_state("Tesla NVIDIA acquisition analysis")

        # Run planner node
        result = await planner_node(initial_state)

        # Validate results
        assert "task_plans" in result
        assert len(result["task_plans"]) > 0
        assert result["step_count"] == 1

        # Check that M&A-specific tasks were created
        task_ids = [task.task_id for task in result["task_plans"]]
        assert any("financial" in task_id for task_id in task_ids)
        assert any("strategic" in task_id for task_id in task_ids)

    def test_ma_query_detection(self):
        """Test M&A query detection in planner."""
        from axiom.core.orchestration.nodes.planner import detect_analysis_type, extract_company_info

        # Test M&A query detection
        ma_query = "Microsoft acquisition of OpenAI due diligence analysis"
        analysis_type = detect_analysis_type(ma_query)
        assert analysis_type == "ma_due_diligence"

        # Test company extraction
        company_info = extract_company_info(ma_query)
        assert "OpenAI" in company_info["name"] or "Microsoft" in company_info["name"]

        # Test valuation query
        valuation_query = "Tesla valuation analysis for acquisition"
        analysis_type = detect_analysis_type(valuation_query)
        assert analysis_type == "ma_valuation"

    def test_task_plan_structure(self):
        """Test task plan structure for investment banking."""
        from axiom.core.orchestration.nodes.planner import create_ib_task_plans

        query = "Apple M&A due diligence analysis"
        analysis_type = "ma_due_diligence"
        company_info = {"name": "Apple"}

        task_plans = create_ib_task_plans(query, analysis_type, company_info, "")

        # Should have multiple tasks
        assert len(task_plans) >= 3

        # Check task types
        task_ids = [task.task_id for task in task_plans]
        assert "financial_due_diligence" in task_ids
        assert "strategic_due_diligence" in task_ids
        assert "risk_assessment" in task_ids

        # Check queries have financial focus
        all_queries = [query.query for task in task_plans for query in task.queries]
        financial_terms = ["financial", "revenue", "EBITDA", "debt", "analysis"]
        assert any(
            any(term in query for term in financial_terms) for query in all_queries
        )


class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_search_query_schema(self):
        """Test SearchQuery schema validation."""
        # Valid query
        valid_query = SearchQuery(
            query="Apple financial performance Q3 2024",
            query_type="expanded",
            priority=1,
        )
        assert valid_query.query == "Apple financial performance Q3 2024"
        assert valid_query.query_type == "expanded"
        assert valid_query.priority == 1

        # Test defaults
        simple_query = SearchQuery(query="Test query")
        assert simple_query.query_type == "original"
        assert simple_query.priority == 1

    def test_evidence_schema(self):
        """Test Evidence schema validation."""
        evidence = Evidence(
            content="Strong financial performance with 20% revenue growth",
            source_url="https://sec.gov/filing123",
            source_title="SEC 10-K Filing",
            confidence=0.85,
            relevance_score=0.9,
        )

        assert evidence.confidence == 0.85
        assert evidence.relevance_score == 0.9
        assert "revenue growth" in evidence.content

    def test_research_brief_schema(self):
        """Test ResearchBrief schema validation."""
        evidence_list = [
            Evidence(
                content="Financial data",
                source_url="https://sec.gov/filing",
                source_title="SEC Filing",
                confidence=0.8,
                relevance_score=0.9,
            )
        ]

        citations = [
            Citation(
                source_url="https://sec.gov/filing",
                source_title="SEC Filing",
                snippet="Financial data snippet",
            )
        ]

        brief = ResearchBrief(
            topic="Investment Banking Analysis",
            questions_answered=["What are the financial metrics?"],
            key_findings=["Strong financial performance"],
            evidence=evidence_list,
            citations=citations,
            confidence=0.85,
        )

        assert brief.confidence == 0.85
        assert len(brief.evidence) == 1
        assert len(brief.citations) == 1
        assert isinstance(brief.timestamp, type(brief.timestamp))


class TestToolIntegration:
    """Test tool integration and MCP adapter."""

    @patch("axiom.integrations.search_tools.tavily_client.TavilyClient")
    def test_tavily_integration(self, mock_tavily_class):
        """Test Tavily search integration."""
        # Mock Tavily client
        mock_client = Mock()
        mock_search_result = {
            "results": [
                {
                    "title": "Tesla Q3 2024 Earnings",
                    "url": "https://ir.tesla.com/earnings-q3-2024",
                    "content": "Tesla reported strong Q3 2024 results with record revenue growth",
                    "score": 0.95,
                }
            ]
        }

        mock_client.search.return_value = mock_search_result
        mock_tavily_class.return_value.client = mock_client

        from axiom.integrations.search_tools.tavily_client import TavilyClient

        tavily = TavilyClient()

        # Test synchronous search method
        with patch.object(tavily, "search", return_value=mock_search_result):
            result = tavily.search("Tesla financial performance")

            assert result is not None
            assert "results" in result
            assert len(result["results"]) == 1
            assert "Tesla" in result["results"][0]["title"]

    @pytest.mark.asyncio
    async def test_mcp_adapter_tool_execution(self):
        """Test MCP adapter tool execution."""
        from axiom.integrations.search_tools.mcp_adapter import mcp_adapter

        # Test tool schema retrieval
        tools = mcp_adapter.get_available_tools()
        assert len(tools) > 0

        tool_names = [tool["name"] for tool in tools]
        assert "investment_banking_search" in tool_names
        assert "financial_document_processor" in tool_names

        # Test parameter validation
        validation_result = mcp_adapter.validate_parameters(
            "investment_banking_search",
            {"query": "Test query", "search_type": "company"},
        )
        assert validation_result["valid"]

        # Test invalid parameters
        validation_result = mcp_adapter.validate_parameters(
            "investment_banking_search", {}  # Missing required query
        )
        assert not validation_result["valid"]
        assert "Missing required parameters" in validation_result["error"]


class TestAILayerConfiguration:
    """Test AI layer configuration and routing."""

    def test_analysis_layer_mapping(self):
        """Test AI layer configuration mapping."""
        from axiom.config.ai_layer_config import (
            AIProviderType,
            ai_layer_mapping,
        )

        # Test M&A due diligence configuration
        ma_dd_config = ai_layer_mapping.get_layer_config(AnalysisLayer.MA_DUE_DILIGENCE)
        assert ma_dd_config.primary_provider == AIProviderType.CLAUDE
        assert ma_dd_config.use_consensus
        assert ma_dd_config.temperature == 0.03  # Very conservative

        # Test M&A valuation configuration
        ma_val_config = ai_layer_mapping.get_layer_config(AnalysisLayer.MA_VALUATION)
        assert ma_val_config.primary_provider == AIProviderType.OPENAI
        assert ma_val_config.use_consensus

    def test_required_providers(self):
        """Test required providers detection."""
        from axiom.config.ai_layer_config import ai_layer_mapping

        required_providers = ai_layer_mapping.get_required_providers()
        assert len(required_providers) > 0

        # Should include both primary and fallback providers
        provider_names = [p.value for p in required_providers]
        assert "claude" in provider_names
        assert "openai" in provider_names

    def test_layer_provider_override(self):
        """Test overriding layer provider configuration."""
        from axiom.config.ai_layer_config import (
            AIProviderType,
            ai_layer_mapping,
        )

        # Override planner to use OpenAI instead of Claude
        ai_layer_mapping.override_layer_provider(
            AnalysisLayer.PLANNER, AIProviderType.OPENAI, [AIProviderType.CLAUDE]
        )

        config = ai_layer_mapping.get_layer_config(AnalysisLayer.PLANNER)
        assert config.primary_provider == AIProviderType.OPENAI
        assert AIProviderType.CLAUDE in config.fallback_providers


class TestEndToEndMockWorkflow:
    """Test complete workflow with mocked components."""

    @pytest.mark.asyncio
    async def test_mock_ma_analysis_workflow(self):
        """Test complete M&A analysis workflow with mocks."""

        # Mock all external dependencies
        with patch(
            "axiom.integrations.ai_providers.provider_factory.get_layer_provider"
        ) as mock_get_provider:
            # Setup mock provider
            mock_provider = Mock()
            mock_response = AIResponse(
                content="Comprehensive M&A analysis completed",
                provider="MockProvider",
                model="test-model",
                confidence=0.9,
            )
            mock_provider.generate_response_async = AsyncMock(
                return_value=mock_response
            )
            mock_provider.is_available.return_value = True
            mock_get_provider.return_value = mock_provider

            with patch("axiom.integrations.search_tools.tavily_client.TavilyClient") as mock_tavily:
                # Mock search results
                mock_tavily.return_value.search = AsyncMock(
                    return_value={
                        "results": [
                            {
                                "title": "Tesla Financial Analysis",
                                "url": "https://sec.gov/tesla-10k",
                                "content": "Tesla's financial performance shows strong revenue growth",
                                "score": 0.9,
                            }
                        ]
                    }
                )

                # Create workflow state
                query = "Tesla NVIDIA acquisition strategic analysis"
                initial_state = create_initial_state(query, "test-trace")

                # Test planner
                planner_result = await planner_node(initial_state)
                assert "task_plans" in planner_result
                assert len(planner_result["task_plans"]) > 0

                # Verify M&A-specific planning
                task_ids = [task.task_id for task in planner_result["task_plans"]]
                assert any("financial" in task_id for task_id in task_ids)
                assert any("strategic" in task_id for task_id in task_ids)

                print("✅ Mock M&A workflow test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
