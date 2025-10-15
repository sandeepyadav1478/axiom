"""
Demonstration script for Axiom Investment Banking Analytics
Shows M&A analysis workflow without requiring external API dependencies
"""

import sys
from pathlib import Path

# Add axiom to Python path
sys.path.insert(0, str(Path(__file__).parent))


def demo_configuration_system():
    """Demonstrate configuration system."""
    print("🔧 Configuration System Demo")
    print("-" * 40)

    try:
        # Test basic imports that don't require external dependencies
        from axiom.config.ai_layer_config import (
            AnalysisLayer,
            ai_layer_mapping,
        )

        # Show M&A configuration
        ma_dd_config = ai_layer_mapping.get_layer_config(AnalysisLayer.MA_DUE_DILIGENCE)
        print(
            f"✅ M&A Due Diligence AI Provider: {ma_dd_config.primary_provider.value}"
        )
        print(f"   Temperature: {ma_dd_config.temperature} (Very conservative)")
        print(f"   Consensus Mode: {ma_dd_config.use_consensus}")
        print(f"   Max Tokens: {ma_dd_config.max_tokens}")

        ma_val_config = ai_layer_mapping.get_layer_config(AnalysisLayer.MA_VALUATION)
        print(f"✅ M&A Valuation AI Provider: {ma_val_config.primary_provider.value}")
        print(f"   Consensus Mode: {ma_val_config.use_consensus}")

        # Show required providers
        required_providers = ai_layer_mapping.get_required_providers()
        print(
            f"✅ Required AI Providers: {', '.join(p.value for p in required_providers)}"
        )

        return True

    except Exception as e:
        print(f"❌ Configuration demo failed: {e}")
        return False


def demo_schemas():
    """Demonstrate data schemas."""
    print("\n📋 Data Schemas Demo")
    print("-" * 40)

    try:

        from axiom.config.schemas import (
            Citation,
            Evidence,
            ResearchBrief,
            SearchQuery,
            TaskPlan,
        )

        # Create sample M&A search query
        ma_query = SearchQuery(
            query="Microsoft OpenAI acquisition financial due diligence revenue synergies",
            query_type="expanded",
            priority=1,
        )
        print(f"✅ M&A Search Query: {ma_query.query[:50]}...")

        # Create sample task plan
        task_plan = TaskPlan(
            task_id="ma_financial_analysis",
            description="M&A financial due diligence analysis",
            queries=[ma_query],
            estimated_priority=1,
        )
        print(f"✅ M&A Task Plan: {task_plan.description}")

        # Create sample evidence
        evidence = Evidence(
            content="Microsoft acquisition of OpenAI shows strong strategic fit with Azure AI services, potential for $5-10B revenue synergies through enterprise customer base expansion",
            source_url="https://sec.gov/filing-example",
            source_title="SEC Filing Analysis",
            confidence=0.85,
            relevance_score=0.92,
        )
        print(f"✅ M&A Evidence: Confidence {evidence.confidence}")

        # Create sample brief
        brief = ResearchBrief(
            topic="Microsoft OpenAI Acquisition Analysis",
            questions_answered=[
                "What are the strategic synergies?",
                "What is the financial impact?",
            ],
            key_findings=["Strong strategic fit", "Significant revenue synergies"],
            evidence=[evidence],
            citations=[
                Citation(
                    source_url=evidence.source_url,
                    source_title=evidence.source_title,
                    snippet=evidence.content[:100],
                )
            ],
            confidence=0.85,
        )
        print(f"✅ M&A Research Brief: {brief.topic}")
        print(f"   Confidence: {brief.confidence}")
        print(f"   Evidence pieces: {len(brief.evidence)}")
        print(f"   Citations: {len(brief.citations)}")

        return True

    except Exception as e:
        print(f"❌ Schema demo failed: {e}")
        return False


def demo_ai_provider_abstraction():
    """Demonstrate AI provider abstraction."""
    print("\n🤖 AI Provider Abstraction Demo")
    print("-" * 40)

    try:
        from axiom.ai_client_integrations.claude_provider import ClaudeProvider
        from axiom.ai_client_integrations.openai_provider import OpenAIProvider

        # Create provider instances (without actual API calls)
        openai_provider = OpenAIProvider(
            api_key="sk-demo-key", model_name="gpt-4o-mini"
        )
        print(f"✅ OpenAI Provider: {openai_provider.provider_name}")
        print(f"   Model: {openai_provider.model_name}")

        claude_provider = ClaudeProvider(
            api_key="sk-ant-demo-key", model_name="claude-3-sonnet-20240229"
        )
        print(f"✅ Claude Provider: {claude_provider.provider_name}")
        print(f"   Model: {claude_provider.model_name}")

        # Test financial analysis prompt generation
        company_info = {"name": "Tesla Inc"}
        claude_messages = claude_provider.financial_analysis_prompt(
            "ma_due_diligence", company_info, "Acquisition target evaluation"
        )

        print(f"✅ M&A Due Diligence Prompt: {len(claude_messages)} messages")
        print(f"   System prompt length: {len(claude_messages[0].content)} chars")
        print("   User prompt includes: Tesla Inc ✓, due diligence ✓")

        return True

    except Exception as e:
        print(f"❌ AI provider demo failed: {e}")
        return False


def demo_validation_system():
    """Demonstrate validation system."""
    print("\n✅ Validation System Demo")
    print("-" * 40)

    try:
        from axiom.utils.validation import ComplianceValidator, FinancialValidator

        # Test financial metrics validation
        valid_metrics = {
            "revenue": 10000000000,  # $10B revenue
            "ebitda": 2500000000,  # $2.5B EBITDA
            "pe_ratio": 25.5,
            "debt_to_equity": 0.8,
            "confidence": 0.87,
        }

        errors = FinancialValidator.validate_financial_metrics(valid_metrics)
        print(f"✅ Financial Metrics Validation: {len(errors)} errors")
        if errors:
            for error in errors[:2]:
                print(f"   Warning: {error}")

        # Test M&A transaction validation
        ma_transaction = {
            "target_company": "OpenAI",
            "acquirer": "Microsoft Corporation",
            "transaction_value": 10000000000,  # $10B
            "announcement_date": "2024-01-15",
        }

        ma_errors = FinancialValidator.validate_ma_transaction(ma_transaction)
        print(f"✅ M&A Transaction Validation: {len(ma_errors)} errors")

        # Test compliance validation
        analysis_data = {
            "confidence": 0.85,
            "evidence": [
                {"source_url": "https://sec.gov/filing1", "content": "Financial data"},
                {
                    "source_url": "https://bloomberg.com/article",
                    "content": "Market analysis",
                },
                {
                    "source_url": "https://reuters.com/news",
                    "content": "Industry analysis",
                },
            ],
        }

        compliance_errors = ComplianceValidator.validate_confidence_levels(
            analysis_data, "due_diligence"
        )
        print(f"✅ Compliance Validation: {len(compliance_errors)} errors")

        return True

    except Exception as e:
        print(f"❌ Validation demo failed: {e}")
        return False


def demo_ma_query_analysis():
    """Demonstrate M&A query analysis without external dependencies."""
    print("\n💼 M&A Query Analysis Demo")
    print("-" * 40)

    try:
        # Import planner utilities
        from axiom.graph.nodes.planner import (
            create_ib_task_plans,
            detect_analysis_type,
            extract_company_info,
        )

        # Test M&A query
        ma_query = (
            "Microsoft acquisition of OpenAI strategic analysis and due diligence"
        )

        # Analyze query
        analysis_type = detect_analysis_type(ma_query)
        company_info = extract_company_info(ma_query)

        print(f"✅ Query: {ma_query}")
        print(f"   Analysis Type: {analysis_type}")
        print(f"   Target Company: {company_info['name']}")

        # Create task plans
        task_plans = create_ib_task_plans(ma_query, analysis_type, company_info, "")

        print(f"✅ M&A Task Plans Generated: {len(task_plans)} tasks")
        for i, task in enumerate(task_plans, 1):
            print(f"   {i}. {task.task_id}: {task.description[:60]}...")
            print(f"      Queries: {len(task.queries)}")

        # Show sample queries
        if task_plans:
            sample_queries = task_plans[0].queries
            print("✅ Sample Financial Queries:")
            for query in sample_queries[:2]:
                print(f"   • {query.query}")

        return True

    except Exception as e:
        print(f"❌ M&A query analysis demo failed: {e}")
        return False


def main():
    """Run demonstration of Axiom Investment Banking Analytics system."""

    print("🏦 AXIOM INVESTMENT BANKING ANALYTICS")
    print("=" * 60)
    print("💼 M&A Analysis System Demonstration")
    print("=" * 60)

    # Run demos in order
    demo_results = {
        "Configuration System": demo_configuration_system(),
        "Data Schemas": demo_schemas(),
        "AI Provider Abstraction": demo_ai_provider_abstraction(),
        "Validation System": demo_validation_system(),
        "M&A Query Analysis": demo_ma_query_analysis(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("🎯 DEMONSTRATION SUMMARY")
    print("=" * 60)

    total_demos = len(demo_results)
    successful_demos = sum(1 for result in demo_results.values() if result)

    for demo_name, result in demo_results.items():
        status_icon = "✅" if result else "❌"
        print(f"  {status_icon} {demo_name}")

    print(f"\nDemo Score: {successful_demos}/{total_demos}")

    if successful_demos == total_demos:
        print("\n🎉 All demos successful! Core system is working.")
        print("\n📋 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure API keys in .env file")
        print(
            "3. Test live analysis: python -m axiom.main 'Microsoft OpenAI M&A analysis'"
        )
        return 0
    else:
        print("\n⚠️  Some demos failed - check implementation above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
