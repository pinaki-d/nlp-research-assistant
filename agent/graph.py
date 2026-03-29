from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from agent.nodes import classify_and_respond, should_use_tool, tool_node


def build_graph():
    """
    Builds and compiles the LangGraph state graph.

    Graph flow:
        [START]
            |
            v
    classify_and_respond   <----+
            |                   |
            v                   |
    should_use_tool?            |
       /         \              |
    "tools"     "end"           |
      |                         |
      v                         |
    tool_node ------------------+
    (executes tool, loops back to classify_and_respond)

    MemorySaver checkpointer persists conversation state
    across multiple turns using a thread_id.
    """

    # MessagesState is a built-in state schema that manages
    # a list of messages with automatic append behaviour
    graph = StateGraph(MessagesState)

    # Add nodes
    graph.add_node("classify_and_respond", classify_and_respond)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("classify_and_respond")

    # Conditional edge: after LLM responds, check if tool needed
    graph.add_conditional_edges(
        "classify_and_respond",
        should_use_tool,
        {
            "tools": "tools",   # go to tool node
            "end": END          # finish turn
        }
    )

    # After tool executes, go back to LLM to synthesise response
    graph.add_edge("tools", "classify_and_respond")

    # MemorySaver stores conversation history in memory
    # keyed by thread_id — each user session gets its own thread
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app


# Single instance used across the Streamlit app
agent_graph = build_graph()
