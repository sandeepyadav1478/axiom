"""DSPy module for Hypothetical Document Embeddings (HyDE)."""

import dspy
from typing import List
from axiom.config.settings import settings


class HypotheticalAnswer(dspy.Signature):
    """Generate a hypothetical answer that might contain the information needed."""

    query = dspy.InputField(desc="The research query")
    hypothetical_answer = dspy.OutputField(desc="A detailed hypothetical answer that covers what the ideal source document might contain")


class HyDEModule(dspy.Module):
    """DSPy module for generating hypothetical documents to improve search."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(HypotheticalAnswer)

    def forward(self, query: str) -> str:
        """Generate a hypothetical document/answer for better semantic search."""

        result = self.generate_answer(query=query)

        if hasattr(result, 'hypothetical_answer'):
            return result.hypothetical_answer

        return query  # Fallback to original query


class QueryEnrichment(dspy.Signature):
    """Enhance a query with additional context and keywords."""

    original_query = dspy.InputField(desc="The original query")
    enriched_query = dspy.OutputField(desc="Enhanced query with additional context and relevant keywords")


class QueryEnrichmentModule(dspy.Module):
    """DSPy module for enriching queries with additional context."""

    def __init__(self):
        super().__init__()
        self.enrich = dspy.ChainOfThought(QueryEnrichment)

    def forward(self, query: str) -> str:
        """Enrich a query with additional context."""

        result = self.enrich(original_query=query)

        if hasattr(result, 'enriched_query'):
            return result.enriched_query

        return query
