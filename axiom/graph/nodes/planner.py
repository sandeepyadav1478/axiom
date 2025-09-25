"""Planner node for decomposing research queries into tasks."""

import asyncio
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from axiom.config.settings import settings
from axiom.config.schemas import TaskPlan, SearchQuery
from axiom.graph.state import AxiomState
from axiom.tracing.langsmith_tracer import trace_node


PLANNER_SYSTEM_PROMPT = """You are a research planning expert. Your job is to decompose a complex research query into 2-4 focused sub-tasks that can be executed in parallel.

For each task, provide:
1. A clear task description
2. 2-3 specific search queries that would gather relevant information
3. A priority score (1=highest priority)

Output your plan as a structured list. Be specific and actionable."""


@trace_node("planner")
async def planner_node(state: AxiomState) -> Dict[str, Any]:
    """Plan the research by decomposing the query into tasks."""

    llm = ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model_name,
        temperature=0.1
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Research query: {state['query']}\n\nCreate a research plan:")
    ]

    try:
        response = await llm.ainvoke(messages)
        plan_text = response.content

        # Parse the plan into structured tasks
        # For now, create a simple default plan - this would be enhanced with structured parsing
        task_plans = [
            TaskPlan(
                task_id="background_research",
                description=f"Background research on: {state['query']}",
                queries=[
                    SearchQuery(query=state['query'], query_type="original", priority=1),
                    SearchQuery(query=f"{state['query']} overview", query_type="expanded", priority=2)
                ],
                estimated_priority=1
            ),
            TaskPlan(
                task_id="detailed_analysis", 
                description=f"Detailed analysis of: {state['query']}",
                queries=[
                    SearchQuery(query=f"{state['query']} analysis", query_type="expanded", priority=1),
                    SearchQuery(query=f"{state['query']} impact", query_type="expanded", priority=2)
                ],
                estimated_priority=2
            )
        ]

        return {
            "task_plans": task_plans,
            "step_count": state["step_count"] + 1,
            "messages": state["messages"] + [HumanMessage(content=f"Planning complete: {len(task_plans)} tasks")]
        }

    except Exception as e:
        error_msg = f"Planner error: {str(e)}"
        return {
            "error_messages": state["error_messages"] + [error_msg],
            "step_count": state["step_count"] + 1
        }
