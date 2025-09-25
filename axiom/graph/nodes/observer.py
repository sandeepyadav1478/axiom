"""Observer node for validation and synthesis."""

import asyncio
from typing import Dict, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from axiom.config.settings import settings
from axiom.config.schemas import ResearchBrief, Citation
from axiom.graph.state import AxiomState
from axiom.tracing.langsmith_tracer import trace_node


SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis expert. Your job is to analyze evidence and create a structured research brief.

Create a comprehensive brief that includes:
1. Topic summary
2. Questions answered by the research
3. Key findings with supporting evidence
4. Confidence assessment
5. Remaining knowledge gaps

Be rigorous about citations and only include findings supported by evidence."""


@trace_node("observer")
async def observer_node(state: AxiomState) -> Dict[str, Any]:
    """Synthesize evidence into a final research brief."""

    llm = ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model_name,
        temperature=0.1
    )

    try:
        # Prepare evidence summary
        evidence_summary = "\n".join([
            f"- {e.content} (Confidence: {e.confidence}, Source: {e.source_title})"
            for e in state["evidence"]
        ])

        synthesis_prompt = f"""Research Query: {state['query']}

Evidence Collected:
{evidence_summary}

Create a structured research brief:"""

        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt)
        ]

        response = await llm.ainvoke(messages)
        brief_text = response.content

        # Create structured brief
        # For now, create a basic structure - this would be enhanced with structured parsing
        citations = []
        for evidence in state["evidence"]:
            citation = Citation(
                source_url=evidence.source_url,
                source_title=evidence.source_title,
                snippet=evidence.content[:200],
                access_date=datetime.now()
            )
            citations.append(citation)

        # Calculate overall confidence
        if state["evidence"]:
            avg_confidence = sum(e.confidence for e in state["evidence"]) / len(state["evidence"])
        else:
            avg_confidence = 0.0

        brief = ResearchBrief(
            topic=state["query"],
            questions_answered=[f"What is {state['query']}?", f"What are the implications of {state['query']}?"],
            key_findings=[brief_text[:500]],  # Simplified - would parse structured findings
            evidence=state["evidence"],
            citations=citations,
            remaining_gaps=["More recent data needed", "Additional perspectives required"],
            confidence=avg_confidence,
            timestamp=datetime.now()
        )

        # Validation checks
        validation_passed = True
        validation_errors = []

        if len(brief.evidence) < 3:
            validation_errors.append("Insufficient evidence (minimum 3 pieces required)")
            validation_passed = False

        if brief.confidence < 0.5:
            validation_errors.append("Low confidence score - requires more evidence")
            validation_passed = False

        if not brief.citations:
            validation_errors.append("No citations found")
            validation_passed = False

        result = {
            "brief": brief if validation_passed else None,
            "step_count": state["step_count"] + 1,
            "messages": state["messages"] + [HumanMessage(content="Research synthesis complete")]
        }

        if validation_errors:
            result["error_messages"] = state["error_messages"] + validation_errors

        return result

    except Exception as e:
        error_msg = f"Observer error: {str(e)}"
        return {
            "error_messages": state["error_messages"] + [error_msg],
            "step_count": state["step_count"] + 1
        }
