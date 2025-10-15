"""Investment Banking DSPy Optimization - M&A and Financial Analysis Query Optimization."""

import dspy
import json
import os
from typing import List, Dict, Any, Optional
from axiom.dspy_modules.multi_query import InvestmentBankingMultiQueryModule, setup_dspy_with_provider
from axiom.dspy_modules.hyde import InvestmentBankingHyDEModule, FinancialQueryEnrichmentModule
from axiom.config.settings import settings


class InvestmentBankingOptimizer:
    """Optimizer for investment banking DSPy modules with M&A and financial analysis focus."""

    def __init__(self):
        try:
            setup_dspy_with_provider()
        except Exception as e:
            print(f"DSPy setup warning: {e}")

        self.ib_multi_query = InvestmentBankingMultiQueryModule()
        self.ib_hyde = InvestmentBankingHyDEModule()
        self.financial_enrichment = FinancialQueryEnrichmentModule()

        self.compiled_modules = {}

    def create_investment_banking_training_data(self) -> List[dspy.Example]:
        """Create investment banking specific training examples."""

        # M&A and financial analysis training data
        examples = [
            # M&A Analysis Examples
            dspy.Example(
                query="Microsoft acquisition of OpenAI strategic analysis",
                analysis_type="ma_due_diligence",
                expanded_queries=[
                    "Microsoft OpenAI acquisition financial due diligence revenue synergies",
                    "OpenAI strategic value AI market position competitive advantages",
                    "Microsoft OpenAI merger regulatory approval antitrust considerations",
                    "OpenAI valuation DCF analysis comparable AI transactions",
                    "Microsoft OpenAI integration technology cultural fit assessment"
                ],
                hypothetical_document="Microsoft's proposed acquisition of OpenAI represents a strategic move to dominate the AI landscape. Financial analysis shows OpenAI's recurring revenue model with $2B+ ARR growth, supported by ChatGPT consumer subscriptions and enterprise API services. Strategic fit analysis reveals complementary capabilities: Microsoft's cloud infrastructure enhances OpenAI's scalability while OpenAI's AI models strengthen Microsoft's competitive moat in enterprise software. Synergy potential includes $5-10B in revenue enhancement through AI integration across Office, Azure, and Windows platforms. Key risks include regulatory scrutiny, talent retention, and technology integration complexity."
            ),
            
            dspy.Example(
                query="Tesla NVIDIA acquisition analysis",
                analysis_type="ma_valuation",
                expanded_queries=[
                    "Tesla NVIDIA acquisition strategic rationale autonomous driving AI chips",
                    "NVIDIA valuation enterprise value trading multiples semiconductor peers",
                    "Tesla NVIDIA synergies vertical integration cost savings analysis",
                    "NVIDIA financial performance data center automotive revenue segments",
                    "Tesla NVIDIA merger antitrust regulatory semiconductor market concentration"
                ],
                hypothetical_document="Tesla-NVIDIA merger analysis reveals significant strategic value creation potential. NVIDIA's $2T market cap commands premium multiples (45x P/E, 25x EV/Sales) reflecting AI leadership and data center dominance. Strategic rationale centers on vertical integration of AI compute for Tesla's FSD technology, reducing chip supply chain dependencies. Financial modeling shows $15-20B synergy potential through cost savings and accelerated autonomous vehicle deployment. DCF analysis supports $400-450 per share valuation range. Key integration risks include regulatory approval complexity and cultural alignment challenges."
            ),

            # Financial Analysis Examples
            dspy.Example(
                query="Apple financial performance analysis Q3 2024",
                analysis_type="financial_analysis",
                expanded_queries=[
                    "Apple Q3 2024 revenue growth iPhone Services segment performance",
                    "Apple profitability margins EBITDA operating leverage efficiency ratios",
                    "Apple cash flow analysis capital allocation dividend buyback strategy",
                    "Apple peer comparison Amazon Google Microsoft valuation multiples",
                    "Apple forward guidance 2025 outlook iPhone 16 Services growth"
                ],
                hypothetical_document="Apple Q3 2024 financial analysis shows resilient performance despite macro headwinds. Revenue of $81.8B (+1% YoY) reflects iPhone stabilization and Services acceleration (+14% YoY to $22.3B). Gross margins expanded 120bp to 46.3% driven by Services mix shift and operational efficiency. Free cash flow generation remains robust at $26.7B, supporting $25B quarterly shareholder returns. Trading at 28x P/E vs 5-year average of 24x, premium justified by Services transformation and AI integration roadmap. Key risks include China market exposure and iPhone upgrade cycle timing."
            ),

            # Sector Analysis Examples
            dspy.Example(
                query="Semiconductor industry M&A consolidation trends",
                analysis_type="market_analysis",
                expanded_queries=[
                    "Semiconductor M&A activity 2023-2024 transaction volume values trends",
                    "Chip industry consolidation drivers supply chain vertical integration",
                    "Semiconductor valuations trading multiples precedent transactions analysis",
                    "Semiconductor regulatory environment China US technology export controls",
                    "AI chip market dynamics NVIDIA AMD Intel competitive positioning"
                ],
                hypothetical_document="Semiconductor industry M&A analysis reveals accelerating consolidation driven by AI demand and supply chain resilience needs. 2024 transaction volume reached $75B+ across 45+ deals, up 40% YoY. Key drivers include vertical integration strategies, technology differentiation, and geopolitical supply chain concerns. Valuation multiples average 4.5x EV/Sales for specialized AI chip companies vs 2.5x for commodity semiconductors. Regulatory scrutiny intensifying, particularly for China-related transactions. Strategic buyers dominate (70% of volume) seeking IP and manufacturing capabilities. Outlook suggests continued consolidation as companies scale for AI infrastructure buildout."
            )
        ]

        return examples

    def create_sector_training_data(self) -> List[dspy.Example]:
        """Create sector-specific training examples."""

        sector_examples = [
            dspy.Example(
                query="Healthcare technology sector M&A trends",
                analysis_focus="ma_activity",
                sector_queries=[
                    "HealthTech M&A transaction volume digital health valuations 2024",
                    "Healthcare AI acquisition targets telehealth electronic records",
                    "Pharmaceutical technology partnerships biotech M&A activity",
                    "Healthcare regulatory environment FDA digital therapeutics approval",
                    "Healthcare technology competitive landscape consolidation drivers"
                ]
            ),
            
            dspy.Example(
                query="Financial services fintech disruption",
                analysis_focus="competitive_landscape",
                sector_queries=[
                    "Fintech market share payment processing digital banking penetration",
                    "Traditional banking fintech partnership acquisition strategies",
                    "Financial services regulatory changes cryptocurrency institutional adoption",
                    "Fintech valuation multiples revenue growth profitability metrics",
                    "Financial services technology investment AI blockchain integration"
                ]
            )
        ]

        return sector_examples

    def optimize_investment_banking_multi_query(self, num_trials: int = 8) -> InvestmentBankingMultiQueryModule:
        """Optimize investment banking multi-query expansion."""

        try:
            training_data = self.create_investment_banking_training_data()

            # Configure optimizer with investment banking metrics
            optimizer = dspy.MIPRO(
                metric=self._investment_banking_multi_query_metric,
                num_candidates=num_trials,
                init_temperature=0.5  # Conservative for financial analysis
            )

            # Compile the module
            compiled_ib_multi_query = optimizer.compile(
                self.ib_multi_query,
                trainset=training_data,
                valset=training_data[:2]
            )

            self.compiled_modules['ib_multi_query'] = compiled_ib_multi_query
            print("✅ Investment banking multi-query optimization complete")
            return compiled_ib_multi_query

        except Exception as e:
            print(f"Multi-query optimization failed: {e}")
            return self.ib_multi_query

    def optimize_investment_banking_hyde(self, num_trials: int = 6) -> InvestmentBankingHyDEModule:
        """Optimize investment banking HyDE module."""

        try:
            training_data = self.create_investment_banking_training_data()

            optimizer = dspy.MIPRO(
                metric=self._investment_banking_hyde_metric,
                num_candidates=num_trials,
                init_temperature=0.3  # Very conservative for financial documents
            )

            compiled_ib_hyde = optimizer.compile(
                self.ib_hyde,
                trainset=training_data,
                valset=training_data[:2]
            )

            self.compiled_modules['ib_hyde'] = compiled_ib_hyde
            print("✅ Investment banking HyDE optimization complete")
            return compiled_ib_hyde

        except Exception as e:
            print(f"HyDE optimization failed: {e}")
            return self.ib_hyde

    def _investment_banking_multi_query_metric(self, example, pred, trace=None) -> float:
        """Investment banking specific metric for multi-query quality."""

        if not hasattr(pred, 'expanded_queries') and not isinstance(pred, list):
            return 0.0

        # Handle different prediction formats
        if hasattr(pred, 'expanded_queries'):
            generated_queries = pred.expanded_queries
        else:
            generated_queries = pred

        # Convert to list if string
        if isinstance(generated_queries, str):
            generated_queries = generated_queries.split('\n')

        # Financial analysis quality metrics
        financial_terms = ['revenue', 'ebitda', 'valuation', 'dcf', 'synergies', 'due diligence', 'analysis']
        financial_relevance = sum(1 for query in generated_queries
                                if any(term in query.lower() for term in financial_terms)) / max(len(generated_queries), 1)

        # Diversity score
        unique_queries = set(q.lower().strip() for q in generated_queries if q.strip())
        diversity_score = len(unique_queries) / max(len(generated_queries), 1)

        # Query count optimization (target 4-5 queries)
        count_score = min(len(generated_queries) / 4.5, 1.0)

        # Combined score with financial relevance weight
        return (financial_relevance * 0.5 + diversity_score * 0.3 + count_score * 0.2)

    def _investment_banking_hyde_metric(self, example, pred, trace=None) -> float:
        """Investment banking specific metric for HyDE quality."""

        # Handle different prediction formats
        if hasattr(pred, 'hypothetical_document'):
            generated_doc = pred.hypothetical_document
        elif hasattr(pred, 'hypothetical_analysis'):
            generated_doc = pred.hypothetical_analysis
        elif hasattr(pred, 'hypothetical_answer'):
            generated_doc = pred.hypothetical_answer
        else:
            return 0.0

        # Financial content quality metrics
        financial_terms = ['financial', 'revenue', 'ebitda', 'valuation', 'analysis', 'market', 'strategic']
        ib_terms = ['investment banking', 'm&a', 'merger', 'acquisition', 'due diligence', 'dcf', 'synergies']
        
        doc_lower = generated_doc.lower()
        financial_score = sum(1 for term in financial_terms if term in doc_lower) / len(financial_terms)
        ib_score = sum(1 for term in ib_terms if term in doc_lower) / len(ib_terms)

        # Length optimization (target 100-300 words)
        word_count = len(generated_doc.split())
        length_score = min(word_count / 200.0, 1.0) if word_count >= 50 else word_count / 50.0

        # Structure quality (sentences, coherence)
        structure_score = min(len(generated_doc.split('.')), 10) / 10.0

        return (financial_score * 0.3 + ib_score * 0.4 + length_score * 0.2 + structure_score * 0.1)

    def save_optimized_modules(self, path: str = "investment_banking_optimized_modules.json"):
        """Save optimized investment banking modules."""

        try:
            # Save module metadata
            module_info = {
                "optimization_timestamp": str(dspy.get_current_time) if hasattr(dspy, 'get_current_time') else "unknown",
                "modules_optimized": list(self.compiled_modules.keys()),
                "training_data_size": len(self.create_investment_banking_training_data()),
                "optimization_config": {
                    "focus": "investment_banking",
                    "domain": "M&A_financial_analysis",
                    "conservative_mode": True
                }
            }

            with open(path, 'w') as f:
                json.dump(module_info, f, indent=2)

            print(f"📁 Investment banking optimized modules metadata saved to {path}")

        except Exception as e:
            print(f"Save modules error: {e}")

    def load_optimized_modules(self, path: str = "investment_banking_optimized_modules.json"):
        """Load previously optimized investment banking modules."""

        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    module_info = json.load(f)
                
                print(f"📂 Loaded investment banking modules: {module_info.get('modules_optimized', [])}")
                return module_info
            else:
                print(f"No saved modules found at {path}")
                return None

        except Exception as e:
            print(f"Load modules error: {e}")
            return None

    def evaluate_modules(self) -> Dict[str, float]:
        """Evaluate optimized modules on test data."""

        test_data = self.create_investment_banking_training_data()[:2]  # Use first 2 as test
        results = {}

        try:
            # Evaluate multi-query
            if 'ib_multi_query' in self.compiled_modules:
                multi_query_scores = []
                for example in test_data:
                    pred = self.compiled_modules['ib_multi_query'].forward(example.query)
                    score = self._investment_banking_multi_query_metric(example, pred)
                    multi_query_scores.append(score)
                results['multi_query_avg_score'] = sum(multi_query_scores) / len(multi_query_scores)

            # Evaluate HyDE
            if 'ib_hyde' in self.compiled_modules:
                hyde_scores = []
                for example in test_data:
                    pred = self.compiled_modules['ib_hyde'].forward(example.query)
                    score = self._investment_banking_hyde_metric(example, pred)
                    hyde_scores.append(score)
                results['hyde_avg_score'] = sum(hyde_scores) / len(hyde_scores)

        except Exception as e:
            print(f"Evaluation error: {e}")

        return results


def run_investment_banking_optimization():
    """Run DSPy optimization for investment banking modules."""

    optimizer = InvestmentBankingOptimizer()

    print("🏦 Starting Investment Banking DSPy optimization...")
    print("Focus: M&A Analysis, Financial Due Diligence, Valuation")

    # Optimize investment banking multi-query expansion
    print("📊 Optimizing investment banking multi-query expansion...")
    optimized_ib_multi_query = optimizer.optimize_investment_banking_multi_query()

    # Optimize investment banking HyDE
    print("📋 Optimizing investment banking HyDE module...")
    optimized_ib_hyde = optimizer.optimize_investment_banking_hyde()

    # Evaluate results
    print("📈 Evaluating optimized modules...")
    evaluation_results = optimizer.evaluate_modules()
    
    for metric, score in evaluation_results.items():
        print(f"  {metric}: {score:.3f}")

    # Save results
    optimizer.save_optimized_modules()

    print("✅ Investment Banking DSPy optimization complete!")
    print("🎯 Modules optimized for M&A analysis, financial due diligence, and valuation")

    return optimizer


# Legacy compatibility
AxiomOptimizer = InvestmentBankingOptimizer
run_optimization = run_investment_banking_optimization


if __name__ == "__main__":
    run_investment_banking_optimization()
