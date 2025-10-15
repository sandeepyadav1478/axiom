"""Investment Banking Observer - Synthesis and Validation of Financial Analysis."""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.messages import HumanMessage

from axiom.config.settings import settings
from axiom.config.schemas import ResearchBrief, Citation, Evidence
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.graph.state import AxiomState
from axiom.tracing.langsmith_tracer import trace_node
from axiom.ai_client_integrations import get_layer_provider, AIMessage


@trace_node("investment_banking_observer")
async def observer_node(state: AxiomState) -> Dict[str, Any]:
    """Synthesize investment banking evidence into comprehensive research brief with validation."""

    try:
        # Get optimal AI provider for synthesis (Claude preferred for complex reasoning)
        provider = get_layer_provider(AnalysisLayer.OBSERVER)
        if not provider:
            raise Exception("No available AI provider for synthesis")

        # Investment banking validation checks
        validation_passed, validation_errors = validate_investment_banking_evidence(state)
        
        if not validation_passed and len(state["evidence"]) < 2:
            # Critical failure - insufficient evidence for investment decisions
            return {
                "brief": None,
                "error_messages": state["error_messages"] + validation_errors,
                "step_count": state["step_count"] + 1,
                "messages": state["messages"] + [HumanMessage(content="Investment banking analysis failed validation - insufficient evidence for decision-making")]
            }

        # Create comprehensive investment banking synthesis
        brief = await create_investment_banking_brief(provider, state)
        
        # Additional investment banking specific validation
        final_validation_passed, final_validation_errors = validate_investment_brief(brief)
        
        result = {
            "brief": brief if final_validation_passed else None,
            "step_count": state["step_count"] + 1,
            "messages": state["messages"] + [HumanMessage(
                content=f"Investment banking synthesis complete - Confidence: {brief.confidence:.2f}, Evidence: {len(brief.evidence)} pieces, Citations: {len(brief.citations)}"
            )]
        }

        # Combine all validation errors
        all_errors = validation_errors + final_validation_errors
        if all_errors:
            result["error_messages"] = state["error_messages"] + all_errors

        return result

    except Exception as e:
        error_msg = f"Investment banking observer error: {str(e)}"
        return {
            "error_messages": state["error_messages"] + [error_msg],
            "step_count": state["step_count"] + 1
        }


async def create_investment_banking_brief(provider, state: AxiomState) -> ResearchBrief:
    """Create comprehensive investment banking research brief using AI synthesis."""
    
    # Organize evidence by analysis type
    evidence_by_task = organize_evidence_by_task(state["evidence"])
    
    # Prepare detailed evidence summary
    evidence_summary = create_evidence_summary(evidence_by_task)
    
    # Create investment banking synthesis messages
    synthesis_messages = [
        AIMessage(
            role="system",
            content="""You are a senior investment banking managing director preparing a comprehensive research brief for the investment committee.

**Synthesis Requirements:**
- **Executive Summary**: Clear, actionable summary for senior executives
- **Investment Thesis**: Core rationale for transaction/investment
- **Financial Analysis**: Key metrics, valuation range, returns analysis
- **Strategic Rationale**: Market positioning, synergies, competitive advantages
- **Risk Assessment**: Key risks with probability and mitigation strategies
- **Recommendations**: Clear go/no-go with supporting rationale

**Output Standards:**
- Conservative analysis appropriate for fiduciary responsibility
- Quantitative analysis with ranges and confidence intervals
- Complete citations for all key claims
- Professional tone suitable for board-level presentation
- Risk-adjusted recommendations with sensitivity analysis

**Format Structure:**
```json
{
  "executive_summary": "Brief summary",
  "investment_thesis": "Core investment rationale",
  "key_findings": ["Finding 1", "Finding 2", ...],
  "financial_metrics": {"metric": "value", ...},
  "risk_factors": ["Risk 1", "Risk 2", ...],
  "recommendations": "Clear recommendation with rationale",
  "confidence_assessment": "Detailed confidence analysis"
}
```

Provide investment-grade analysis suitable for institutional decision-making."""
        ),
        AIMessage(
            role="user",
            content=f"""Investment Banking Analysis Query: {state['query']}

**Evidence Collected:**
{evidence_summary}

**Task Analysis Summary:**
{create_task_summary(state['task_plans'])}

**Requirements:**
- Synthesize into investment committee-ready brief
- Conservative confidence assessment for M&A decisions
- Complete citations and source validation
- Risk-adjusted recommendations

Create comprehensive investment banking research brief:"""
        )
    ]

    # Generate synthesis using optimal AI provider
    response = await provider.generate_response_async(
        synthesis_messages,
        max_tokens=4000,
        temperature=0.02  # Extremely conservative for investment decisions
    )
    
    synthesis_content = response.content
    
    # Create structured brief
    citations = create_citations_from_evidence(state["evidence"])
    
    # Extract structured findings (simplified - would use structured parsing)
    key_findings = extract_key_findings_from_synthesis(synthesis_content)
    
    # Calculate conservative confidence score for investment decisions
    investment_confidence = calculate_investment_confidence(state["evidence"], state["task_plans"])
    
    # Identify remaining gaps for due diligence
    remaining_gaps = identify_investment_gaps(state["evidence"], state["task_plans"])
    
    brief = ResearchBrief(
        topic=state["query"],
        questions_answered=extract_questions_answered(state["query"], key_findings),
        key_findings=key_findings,
        evidence=state["evidence"],
        citations=citations,
        remaining_gaps=remaining_gaps,
        confidence=investment_confidence,
        timestamp=datetime.now()
    )
    
    return brief


def validate_investment_banking_evidence(state: AxiomState) -> tuple[bool, List[str]]:
    """Validate evidence meets investment banking standards."""
    validation_errors = []
    
    # Minimum evidence requirements for investment decisions
    if len(state["evidence"]) < settings.due_diligence_confidence_threshold * 10:  # Scale threshold
        validation_errors.append(f"Insufficient evidence for investment decision: {len(state['evidence'])} pieces (minimum {int(settings.due_diligence_confidence_threshold * 10)} required)")
    
    # Evidence quality assessment
    high_confidence_evidence = [e for e in state["evidence"] if e.confidence >= 0.7]
    if len(high_confidence_evidence) < 3:
        validation_errors.append("Insufficient high-confidence evidence for investment banking analysis")
    
    # Source diversity check
    unique_sources = set(e.source_url for e in state["evidence"])
    if len(unique_sources) < 3:
        validation_errors.append("Insufficient source diversity for reliable investment analysis")
    
    # Financial source validation
    financial_sources = [e for e in state["evidence"] if any(domain in e.source_url for domain in ['sec.gov', 'bloomberg', 'reuters', 'wsj', 'ft.com'])]
    if not financial_sources:
        validation_errors.append("No authoritative financial sources found - critical for investment banking decisions")
    
    return len(validation_errors) == 0, validation_errors


def validate_investment_brief(brief: ResearchBrief) -> tuple[bool, List[str]]:
    """Validate final investment banking brief meets standards."""
    validation_errors = []
    
    # Confidence threshold for investment decisions
    if brief.confidence < settings.due_diligence_confidence_threshold:
        validation_errors.append(f"Investment confidence {brief.confidence:.2f} below threshold {settings.due_diligence_confidence_threshold}")
    
    # Minimum content requirements
    if len(brief.key_findings) < 3:
        validation_errors.append("Insufficient key findings for investment committee presentation")
    
    if not brief.citations:
        validation_errors.append("No citations provided - unacceptable for investment banking analysis")
    
    # Evidence quality for investment decisions
    if len(brief.evidence) < 5:
        validation_errors.append("Insufficient evidence depth for investment banking standards")
    
    return len(validation_errors) == 0, validation_errors


def organize_evidence_by_task(evidence: List[Evidence]) -> Dict[str, List[Evidence]]:
    """Organize evidence by task type for structured analysis."""
    task_evidence = {}
    
    for e in evidence:
        # Extract task type from evidence content prefix
        if e.content.startswith('['):
            task_type = e.content.split(']')[0][1:].lower().replace(' ', '_')
        else:
            task_type = 'general'
            
        if task_type not in task_evidence:
            task_evidence[task_type] = []
        task_evidence[task_type].append(e)
    
    return task_evidence


def create_evidence_summary(evidence_by_task: Dict[str, List[Evidence]]) -> str:
    """Create structured evidence summary for AI synthesis."""
    summary_parts = []
    
    for task_type, evidence_list in evidence_by_task.items():
        task_summary = f"\n**{task_type.replace('_', ' ').title()} Evidence:**\n"
        
        for i, evidence in enumerate(evidence_list, 1):
            task_summary += f"{i}. {evidence.content[:300]}...\n"
            task_summary += f"   Source: {evidence.source_title} (Confidence: {evidence.confidence:.2f})\n"
        
        summary_parts.append(task_summary)
    
    return "\n".join(summary_parts)


def create_task_summary(task_plans) -> str:
    """Create summary of planned tasks for context."""
    return "\n".join([
        f"- {plan.task_id}: {plan.description}"
        for plan in task_plans
    ])


def create_citations_from_evidence(evidence: List[Evidence]) -> List[Citation]:
    """Create proper citations from evidence."""
    citations = []
    seen_urls = set()
    
    for e in evidence:
        if e.source_url not in seen_urls:
            citations.append(Citation(
                source_url=e.source_url,
                source_title=e.source_title,
                snippet=e.content[:250],
                access_date=datetime.now()
            ))
            seen_urls.add(e.source_url)
    
    return citations


def extract_key_findings_from_synthesis(synthesis_content: str) -> List[str]:
    """Extract structured key findings from AI synthesis."""
    # Simplified extraction - would use more sophisticated parsing
    findings = []
    
    # Look for structured findings in the synthesis
    lines = synthesis_content.split('\n')
    current_finding = ""
    
    for line in lines:
        if line.strip().startswith('-') or line.strip().startswith('•'):
            if current_finding:
                findings.append(current_finding.strip())
            current_finding = line.strip().lstrip('-•').strip()
        elif current_finding and line.strip():
            current_finding += " " + line.strip()
    
    if current_finding:
        findings.append(current_finding.strip())
    
    # Fallback: create findings from synthesis content
    if not findings:
        sentences = synthesis_content.split('.')[:5]
        findings = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
    
    return findings[:8]  # Limit to top findings


def calculate_investment_confidence(evidence: List[Evidence], task_plans) -> float:
    """Calculate conservative confidence score for investment decisions."""
    if not evidence:
        return 0.0
    
    # Base confidence from evidence quality
    avg_evidence_confidence = sum(e.confidence for e in evidence) / len(evidence)
    
    # Adjust for evidence quantity
    evidence_quantity_factor = min(1.0, len(evidence) / 10.0)
    
    # Adjust for source diversity
    unique_sources = len(set(e.source_url for e in evidence))
    source_diversity_factor = min(1.0, unique_sources / 5.0)
    
    # Conservative adjustment for investment banking
    investment_banking_adjustment = 0.85  # Conservative factor
    
    # Combined confidence with conservative bias
    confidence = (
        avg_evidence_confidence * 0.5 +
        evidence_quantity_factor * 0.3 +
        source_diversity_factor * 0.2
    ) * investment_banking_adjustment
    
    return round(min(0.95, confidence), 3)  # Cap at 95% for risk management


def identify_investment_gaps(evidence: List[Evidence], task_plans) -> List[str]:
    """Identify remaining gaps in investment analysis."""
    gaps = []
    
    # Check task coverage
    completed_tasks = set()
    for e in evidence:
        if e.content.startswith('['):
            task_type = e.content.split(']')[0][1:].lower()
            completed_tasks.add(task_type)
    
    planned_tasks = {plan.task_id.replace('_', ' ') for plan in task_plans}
    missing_tasks = planned_tasks - completed_tasks
    
    for task in missing_tasks:
        gaps.append(f"Incomplete {task} analysis")
    
    # Standard investment banking gaps
    standard_gaps = [
        "Management interviews and due diligence calls required",
        "Legal and regulatory review pending",
        "Detailed financial model validation needed",
        "Integration planning and synergy confirmation required"
    ]
    
    gaps.extend(standard_gaps[:2])  # Add most critical gaps
    
    return gaps[:5]  # Limit gaps list


def extract_questions_answered(query: str, findings: List[str]) -> List[str]:
    """Extract questions answered based on query and findings."""
    questions = [f"What are the investment implications of {query}?"]
    
    if 'M&A' in query or 'acquisition' in query.lower():
        questions.extend([
            "What is the strategic rationale for this transaction?",
            "What are the key financial metrics and valuation ranges?",
            "What are the primary risks and mitigation strategies?"
        ])
    elif 'valuation' in query.lower():
        questions.extend([
            "What is the fair value range using multiple methodologies?",
            "What are the key valuation drivers and sensitivities?"
        ])
    else:
        questions.append("What are the key findings from the analysis?")
    
    return questions[:4]  # Limit questions
