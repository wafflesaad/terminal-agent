"""StateGraph construction for termagent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from langgraph.graph import END, START, StateGraph

from termagent.agent.nodes import (
    confirm_node,
    make_agent_node,
    make_execute_node,
    make_gate_router,
    route_after_confirm,
)
from termagent.agent.state import AgentState

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from termagent.config import Settings


def build_graph(
    get_model: Callable[[], "BaseChatModel"],
    settings: "Settings",
    checkpointer: "BaseCheckpointSaver",
):
    """Compile and return the agent StateGraph.

    Invoke with config={"configurable": {"thread_id": session_id}}.
    Resume a human-approval interrupt with Command(resume=True|False).
    """
    builder = StateGraph(AgentState)

    builder.add_node("agent", make_agent_node(get_model))
    builder.add_node("confirm", confirm_node)
    builder.add_node("execute", make_execute_node(settings))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        make_gate_router(settings),
        {"execute": "execute", "confirm": "confirm", END: END},
    )
    builder.add_conditional_edges(
        "confirm",
        route_after_confirm,
        {"execute": "execute", "agent": "agent"},
    )
    builder.add_edge("execute", "agent")

    return builder.compile(checkpointer=checkpointer)
