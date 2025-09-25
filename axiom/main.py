"""Main entry point for Axiom research agent."""

import asyncio
import json
import sys
from typing import Optional

from axiom.graph.graph import run_research
from axiom.tracing.langsmith_tracer import create_trace_id, get_trace_url
from axiom.config.settings import settings


async def research_query(query: str, trace_id: Optional[str] = None) -> dict:
    """Run a research query and return the results."""

    if not trace_id:
        trace_id = create_trace_id()

    print(f"🔍 Starting research: {query}")
    print(f"📊 Trace ID: {trace_id}")

    if settings.langchain_api_key:
        trace_url = get_trace_url(trace_id)
        print(f"🔗 Trace URL: {trace_url}")

    try:
        # Run the research workflow
        final_state = await run_research(query, trace_id)

        # Extract results
        if final_state.get("brief"):
            brief = final_state["brief"]
            print("\n✅ Research completed successfully!")
            print(f"📝 Topic: {brief.topic}")
            print(f"🎯 Confidence: {brief.confidence:.2f}")
            print(f"📚 Evidence pieces: {len(brief.evidence)}")
            print(f"🔗 Citations: {len(brief.citations)}")

            return {
                "success": True,
                "brief": brief.dict(),
                "trace_id": trace_id,
                "stats": {
                    "evidence_count": len(brief.evidence),
                    "citation_count": len(brief.citations),
                    "confidence": brief.confidence
                }
            }
        else:
            error_msgs = final_state.get("error_messages", ["Unknown error"])
            print(f"\n❌ Research failed: {'; '.join(error_msgs)}")

            return {
                "success": False,
                "errors": error_msgs,
                "trace_id": trace_id
            }

    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")
        return {
            "success": False,
            "errors": [str(e)],
            "trace_id": trace_id
        }


def main():
    """CLI entry point."""

    if len(sys.argv) < 2:
        print("Usage: python -m axiom.main '<research query>'")
        print("Example: python -m axiom.main 'What are the latest developments in quantum computing?'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    # Run the research
    result = asyncio.run(research_query(query))

    # Output results
    if result["success"]:
        print("\n" + "="*60)
        print("RESEARCH BRIEF")
        print("="*60)

        brief = result["brief"]
        print(f"\nTopic: {brief['topic']}")
        print(f"Confidence: {brief['confidence']:.2f}")

        print("\nKey Findings:")
        for finding in brief["key_findings"]:
            print(f"• {finding}")

        print("\nEvidence:")
        for i, evidence in enumerate(brief["evidence"][:5], 1):
            print(f"{i}. {evidence['content'][:200]}...")
            print(f"   Source: {evidence['source_title']}")
            print(f"   Confidence: {evidence['confidence']:.2f}")

        if brief["remaining_gaps"]:
            print("\nRemaining Questions:")
            for gap in brief["remaining_gaps"]:
                print(f"• {gap}")

    else:
        print("\n❌ Research failed:")
        for error in result["errors"]:
            print(f"• {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
