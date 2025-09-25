"""DSPy module for multi-query expansion."""

import dspy
from typing import List
from axiom.config.settings import settings


class MultiQueryExpansion(dspy.Signature):
    """Expand a research query into multiple focused sub-queries."""

    original_query = dspy.InputField(desc="The original research query")
    expanded_queries = dspy.OutputField(desc="3-5 focused sub-queries that explore different aspects")


class MultiQueryModule(dspy.Module):
    """DSPy module for expanding queries into multiple focused searches."""

    def __init__(self):
        super().__init__()
        self.expand = dspy.ChainOfThought(MultiQueryExpansion)

    def forward(self, query: str) -> List[str]:
        """Generate multiple focused queries from a single input query."""

        result = self.expand(original_query=query)

        # Parse the expanded queries (this would be enhanced with better parsing)
        if hasattr(result, 'expanded_queries'):
            # Split by newlines or common delimiters
            queries = []
            raw_queries = result.expanded_queries.split('\n')

            for q in raw_queries:
                # Clean up formatting
                q = q.strip('- •*0123456789. ')
                if len(q) > 10:  # Minimum query length
                    queries.append(q)

            return queries[:5]  # Limit to 5 queries

        return [query]  # Fallback to original query


def setup_dspy_client():
    """Configure DSPy with OpenAI-compatible client."""

    dspy.configure(
        lm=dspy.OpenAI(
            model=settings.openai_model_name,
            api_base=settings.openai_base_url,
            api_key=settings.openai_api_key,
            max_tokens=1000
        )
    )
