"""Investment Banking Research Planner - M&A and Financial Analysis Task Decomposition."""

import re
from typing import Any

from langchain_core.messages import HumanMessage

from axiom.integrations.ai_providers import AIMessage, get_layer_provider
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.config.schemas import SearchQuery, TaskPlan
from axiom.core.orchestration.state import AxiomState
from axiom.tracing.langsmith_tracer import trace_node


def detect_analysis_type(query: str) -> str:
    """Detect the type of investment banking analysis from the query."""
    query_lower = query.lower()

    if any(
        term in query_lower
        for term in ["m&a", "merger", "acquisition", "acquire", "acquiring"]
    ):
        if "due diligence" in query_lower:
            return "ma_due_diligence"
        elif "valuation" in query_lower:
            return "ma_valuation"
        elif "market" in query_lower or "strategic" in query_lower:
            return "ma_market_analysis"
        else:
            return "ma_due_diligence"  # Default M&A analysis
    elif "due diligence" in query_lower:
        return "due_diligence"
    elif "valuation" in query_lower:
        return "valuation"
    elif "market" in query_lower:
        return "market_intelligence"
    else:
        return "ma_due_diligence"  # Default to M&A focus


def extract_company_info(query: str) -> dict[str, str]:
    """Extract company information from the query."""
    # Simple regex patterns - could be enhanced
    company_patterns = [
        r"(?:of|for)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:acquisition|merger)",
        r"([A-Z][A-Za-z]+)\s+(?:acquiring|merging)",
    ]

    for pattern in company_patterns:
        match = re.search(pattern, query)
        if match:
            return {"name": match.group(1).strip()}

    return {"name": "target company"}


@trace_node("investment_banking_planner")
async def planner_node(state: AxiomState) -> dict[str, Any]:
    """Plan investment banking research by decomposing query into specialized tasks."""

    try:
        # Get optimal AI provider for planning
        provider = get_layer_provider(AnalysisLayer.PLANNER)
        if not provider:
            raise Exception("No available AI provider for planning layer")

        # Detect analysis type and extract company info
        analysis_type = detect_analysis_type(state["query"])
        company_info = extract_company_info(state["query"])

        # Create investment banking planning prompt
        planning_messages = [
            AIMessage(
                role="system",
                content="""You are a senior investment banking analyst specializing in M&A transaction planning.

**Task:** Decompose investment banking research queries into 3-4 parallel execution tasks.

**Task Categories:**
1. **Financial Analysis**: Financial health, performance metrics, debt analysis
2. **Strategic Analysis**: Market position, competitive advantages, strategic fit
3. **Risk Assessment**: Business risks, regulatory compliance, integration challenges
4. **Market Intelligence**: Industry trends, comparable transactions, market conditions

**Output Format:**
For each task, provide:
- Task ID (financial_analysis, strategic_analysis, risk_assessment, market_intelligence)
- Detailed description focusing on investment banking objectives
- 2-3 specific search queries targeting financial data sources
- Priority level (1=critical, 2=important, 3=supplementary)

Be specific about financial metrics, regulatory considerations, and M&A-relevant factors.""",
            ),
            AIMessage(
                role="user",
                content=f"""Investment Banking Research Query: {state['query']}

Analysis Type: {analysis_type}
Target Company: {company_info.get('name', 'Not specified')}

Create a comprehensive research plan with parallel tasks:""",
            ),
        ]

        # Generate research plan using optimal AI provider
        response = await provider.generate_response_async(
            planning_messages,
            max_tokens=2000,
            temperature=0.05,  # Very conservative for M&A planning
        )

        plan_text = response.content

        # Create structured investment banking task plans
        task_plans = create_ib_task_plans(
            state["query"], analysis_type, company_info, plan_text
        )

        return {
            "task_plans": task_plans,
            "step_count": state["step_count"] + 1,
            "messages": state["messages"]
            + [
                HumanMessage(
                    content=f"Investment banking analysis planned: {len(task_plans)} parallel tasks for {analysis_type}"
                )
            ],
        }

    except Exception as e:
        error_msg = f"Investment banking planner error: {str(e)}"
        return {
            "error_messages": state["error_messages"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


def create_ib_task_plans(
    query: str, analysis_type: str, company_info: dict, plan_text: str
) -> list[TaskPlan]:
    """Create structured investment banking task plans."""

    company_name = company_info.get("name", "target company")

    # M&A-focused task templates
    task_templates = {
        "ma_due_diligence": [
            TaskPlan(
                task_id="financial_due_diligence",
                description=f"Financial due diligence analysis of {company_name}: revenue quality, profitability trends, cash flow analysis, debt structure, and working capital assessment",
                queries=[
                    SearchQuery(
                        query=f"{company_name} financial statements revenue EBITDA",
                        query_type="original",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} debt structure credit rating financial health",
                        query_type="expanded",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} cash flow working capital financial performance",
                        query_type="expanded",
                        priority=2,
                    ),
                ],
                estimated_priority=1,
            ),
            TaskPlan(
                task_id="strategic_due_diligence",
                description=f"Strategic due diligence of {company_name}: market position, competitive advantages, strategic rationale, and synergy potential",
                queries=[
                    SearchQuery(
                        query=f"{company_name} market share competitive position industry",
                        query_type="original",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} competitive advantages strategic assets",
                        query_type="expanded",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} merger synergies strategic fit",
                        query_type="expanded",
                        priority=2,
                    ),
                ],
                estimated_priority=1,
            ),
            TaskPlan(
                task_id="risk_assessment",
                description=f"Risk assessment for {company_name}: business risks, regulatory compliance, integration complexity, and mitigation strategies",
                queries=[
                    SearchQuery(
                        query=f"{company_name} business risks regulatory compliance",
                        query_type="original",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} integration challenges operational risks",
                        query_type="expanded",
                        priority=2,
                    ),
                    SearchQuery(
                        query=f"{company_name} litigation regulatory issues ESG risks",
                        query_type="expanded",
                        priority=2,
                    ),
                ],
                estimated_priority=2,
            ),
        ],
        "ma_valuation": [
            TaskPlan(
                task_id="dcf_valuation",
                description=f"DCF valuation analysis of {company_name}: cash flow projections, discount rate calculation, terminal value, and sensitivity analysis",
                queries=[
                    SearchQuery(
                        query=f"{company_name} cash flow projections DCF valuation",
                        query_type="original",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} WACC discount rate cost of capital",
                        query_type="expanded",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} terminal value growth rate assumptions",
                        query_type="expanded",
                        priority=2,
                    ),
                ],
                estimated_priority=1,
            ),
            TaskPlan(
                task_id="comparable_analysis",
                description=f"Comparable company and transaction analysis for {company_name}: trading multiples, precedent transactions, and valuation benchmarks",
                queries=[
                    SearchQuery(
                        query=f"{company_name} comparable companies trading multiples",
                        query_type="original",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} industry precedent transactions M&A multiples",
                        query_type="expanded",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} peer group valuation benchmarks",
                        query_type="expanded",
                        priority=2,
                    ),
                ],
                estimated_priority=1,
            ),
            TaskPlan(
                task_id="synergy_analysis",
                description=f"Synergy and accretion analysis for {company_name}: revenue synergies, cost savings, tax benefits, and EPS impact",
                queries=[
                    SearchQuery(
                        query=f"{company_name} merger synergies revenue cost savings",
                        query_type="original",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} acquisition accretion dilution EPS impact",
                        query_type="expanded",
                        priority=1,
                    ),
                    SearchQuery(
                        query=f"{company_name} tax synergies integration costs",
                        query_type="expanded",
                        priority=2,
                    ),
                ],
                estimated_priority=1,
            ),
        ],
    }

    # Return M&A-specific tasks, or fall back to due diligence
    return task_templates.get(analysis_type, task_templates["ma_due_diligence"])
