"""Investment Banking DSPy HyDE Module - Hypothetical Financial Document Generation."""

import dspy
from typing import List, Dict, Any
from axiom.config.settings import settings


class InvestmentBankingHyDE(dspy.Signature):
    """Generate hypothetical investment banking documents for enhanced financial search."""

    query = dspy.InputField(desc="The investment banking or M&A research query")
    analysis_type = dspy.InputField(
        desc="Type of analysis: ma_due_diligence, ma_valuation, market_analysis, or financial_analysis"
    )
    hypothetical_document = dspy.OutputField(
        desc="A comprehensive hypothetical investment banking document covering financial metrics, strategic analysis, and market intelligence that would contain the target information"
    )


class FinancialQueryEnrichment(dspy.Signature):
    """Enhance investment banking queries with financial context and industry terminology."""

    original_query = dspy.InputField(desc="The original investment banking query")
    company_context = dspy.InputField(desc="Company or sector context if available")
    enriched_query = dspy.OutputField(
        desc="Enhanced query with financial terminology, relevant metrics, and investment banking context"
    )


class MAAnalysisHyDE(dspy.Signature):
    """Generate hypothetical M&A analysis documents for deal evaluation."""

    query = dspy.InputField(desc="M&A analysis query")
    target_company = dspy.InputField(desc="Target company name or description")
    analysis_focus = dspy.InputField(
        desc="Analysis focus: strategic_fit, synergies, valuation, or due_diligence"
    )
    hypothetical_analysis = dspy.OutputField(
        desc="Comprehensive hypothetical M&A analysis document with financial metrics, strategic rationale, synergy assessment, and risk factors"
    )


class InvestmentBankingHyDEModule(dspy.Module):
    """Enhanced HyDE module for investment banking and M&A analysis."""

    def __init__(self):
        super().__init__()
        self.general_hyde = dspy.ChainOfThought(InvestmentBankingHyDE)
        self.ma_hyde = dspy.ChainOfThought(MAAnalysisHyDE)
        self.enrich_query = dspy.ChainOfThought(FinancialQueryEnrichment)

    def forward(
        self,
        query: str,
        analysis_type: str = "financial_analysis",
        target_company: str = "",
        analysis_focus: str = "overview",
    ) -> str:
        """Generate hypothetical investment banking document for enhanced search."""

        try:
            # Determine if this is M&A specific
            if any(
                term in query.lower()
                for term in ["m&a", "merger", "acquisition", "acquire"]
            ):
                result = self.ma_hyde(
                    query=query,
                    target_company=target_company or "target company",
                    analysis_focus=analysis_focus,
                )

                if hasattr(result, "hypothetical_analysis"):
                    return result.hypothetical_analysis

            # General investment banking analysis
            result = self.general_hyde(query=query, analysis_type=analysis_type)

            if hasattr(result, "hypothetical_document"):
                return result.hypothetical_document

            return query  # Fallback to original query

        except Exception as e:
            print(f"Investment Banking HyDE error: {e}")
            return query

    def enrich_financial_query(self, query: str, company_context: str = "") -> str:
        """Enrich query with investment banking terminology and context."""

        try:
            result = self.enrich_query(
                original_query=query, company_context=company_context
            )

            if hasattr(result, "enriched_query"):
                return result.enriched_query

            return query

        except Exception as e:
            print(f"Financial query enrichment error: {e}")
            return query


class FinancialMetricsHyDE(dspy.Signature):
    """Generate hypothetical financial metrics and analysis for company research."""

    company_query = dspy.InputField(desc="Company financial analysis query")
    metrics_focus = dspy.InputField(
        desc="Focus area: profitability, liquidity, leverage, valuation, or growth"
    )
    hypothetical_metrics = dspy.OutputField(
        desc="Comprehensive hypothetical financial metrics document including ratios, trends, peer comparisons, and industry benchmarks"
    )


class SectorAnalysisHyDE(dspy.Signature):
    """Generate hypothetical sector analysis documents for market intelligence."""

    sector_query = dspy.InputField(desc="Industry sector analysis query")
    analysis_scope = dspy.InputField(
        desc="Analysis scope: market_trends, ma_activity, valuation_multiples, or regulatory_environment"
    )
    hypothetical_sector_analysis = dspy.OutputField(
        desc="Detailed hypothetical sector analysis covering market dynamics, competitive landscape, M&A trends, valuation metrics, and growth outlook"
    )


class ComprehensiveFinancialHyDEModule(dspy.Module):
    """Comprehensive HyDE module for all investment banking use cases."""

    def __init__(self):
        super().__init__()
        self.investment_banking_hyde = InvestmentBankingHyDEModule()
        self.financial_metrics = dspy.ChainOfThought(FinancialMetricsHyDE)
        self.sector_analysis = dspy.ChainOfThought(SectorAnalysisHyDE)

    def forward(self, query: str, document_type: str = "general", **kwargs) -> str:
        """Generate hypothetical documents based on document type and context."""

        try:
            if document_type == "financial_metrics":
                result = self.financial_metrics(
                    company_query=query,
                    metrics_focus=kwargs.get("metrics_focus", "profitability"),
                )

                if hasattr(result, "hypothetical_metrics"):
                    return result.hypothetical_metrics

            elif document_type == "sector_analysis":
                result = self.sector_analysis(
                    sector_query=query,
                    analysis_scope=kwargs.get("analysis_scope", "market_trends"),
                )

                if hasattr(result, "hypothetical_sector_analysis"):
                    return result.hypothetical_sector_analysis

            # Default to investment banking analysis
            return self.investment_banking_hyde.forward(
                query=query,
                analysis_type=kwargs.get("analysis_type", "financial_analysis"),
                target_company=kwargs.get("target_company", ""),
                analysis_focus=kwargs.get("analysis_focus", "overview"),
            )

        except Exception as e:
            print(f"Comprehensive Financial HyDE error: {e}")
            return query


class FinancialQueryEnrichmentModule(dspy.Module):
    """Standalone module for financial query enrichment."""

    def __init__(self):
        super().__init__()
        self.enrich = dspy.ChainOfThought(FinancialQueryEnrichment)

    def forward(self, query: str, company_context: str = "") -> str:
        """Enrich investment banking queries with financial context."""

        try:
            result = self.enrich(original_query=query, company_context=company_context)

            if hasattr(result, "enriched_query"):
                return result.enriched_query

            return query

        except Exception as e:
            print(f"Financial query enrichment error: {e}")
            return query


# Legacy aliases for backward compatibility
HyDEModule = InvestmentBankingHyDEModule
QueryEnrichmentModule = FinancialQueryEnrichmentModule
