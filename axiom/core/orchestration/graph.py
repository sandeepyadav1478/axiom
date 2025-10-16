"""Main LangGraph workflow orchestration."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from axiom.core.orchestration.nodes.observer import observer_node
from axiom.core.orchestration.nodes.planner import planner_node
from axiom.core.orchestration.nodes.task_runner import task_runner_node
from axiom.core.orchestration.state import AxiomState, create_initial_state


def should_continue(state: AxiomState) -> str:
    """Determine next step based on current state."""

    # Check for errors
    if state["error_messages"]:
        return "observer"  # Go to observer to synthesize what we have

    # Check if we have evidence to synthesize
    if state["evidence"] and len(state["evidence"]) >= 3:
        return "observer"

    # Check if we have task plans but no evidence yet
    if state["task_plans"] and not state["evidence"]:
        return "task_runner"

    # If no plans yet, go to planner
    if not state["task_plans"]:
        return "planner"

    # Default to observer if we're not sure
    return "observer"


def create_research_graph():
    """Create the main research workflow graph."""

    workflow = StateGraph(AxiomState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("task_runner", task_runner_node)
    workflow.add_node("observer", observer_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Add conditional edges
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {"task_runner": "task_runner", "observer": "observer"},
    )

    workflow.add_conditional_edges(
        "task_runner",
        should_continue,
        {
            "observer": "observer",
            "planner": "planner",  # Rare case where we need more planning
        },
    )

    # Observer always ends
    workflow.add_edge("observer", END)

    # Add memory for state persistence
    memory = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=memory)

    return app


async def run_research(query: str, trace_id: str = None) -> AxiomState:
    """Run a complete research workflow."""

    app = create_research_graph()
    initial_state = create_initial_state(query, trace_id)

    # Execute the workflow
    final_state = await app.ainvoke(
        initial_state, config={"configurable": {"thread_id": trace_id or "default"}}
    )

    return final_state
