import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent.graph import agent_graph
import uuid

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 NLP Research Assistant")
st.caption("Multi-turn conversational agent with memory · Powered by LangGraph + GPT-4o-mini")

# ─────────────────────────────────────────────
# Session state initialisation
# Each browser session gets a unique thread_id
# so memory is isolated per user
# ─────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "tool_calls_log" not in st.session_state:
    st.session_state.tool_calls_log = []

# ─────────────────────────────────────────────
# Sidebar — memory state and tool call log
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 Session Memory")
    st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}...`")
    st.metric("Messages in memory", len(st.session_state.chat_history))

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.tool_calls_log = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    st.header("🔧 Tool Call Trace")
    if st.session_state.tool_calls_log:
        for log in reversed(st.session_state.tool_calls_log[-5:]):
            with st.expander(f"Turn {log['turn']}: {log['tool']}", expanded=False):
                st.markdown(f"**Query:** {log['query']}")
                st.markdown(f"**Result preview:** {log['result'][:200]}...")
    else:
        st.caption("No tools called yet.")

# ─────────────────────────────────────────────
# Chat display
# ─────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if msg.get("tool_used"):
                st.caption(f"🔧 Tool used: `{msg['tool_used']}`")

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
user_input = st.chat_input("Ask me anything about NLP research...")

if user_input:
    # Show user message immediately
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.write(user_input)

    # Run the agent graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            result = agent_graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            # Extract the final AI response and any tool calls
            final_response = ""
            tool_used = None

            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and msg.content:
                    final_response = msg.content
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_used = msg.tool_calls[0]["name"]
                if isinstance(msg, ToolMessage):
                    turn_num = len(st.session_state.chat_history)
                    st.session_state.tool_calls_log.append({
                        "turn": turn_num,
                        "tool": tool_used or "unknown",
                        "query": user_input,
                        "result": msg.content
                    })

            st.write(final_response)
            if tool_used:
                st.caption(f"🔧 Tool used: `{tool_used}`")

    # Save assistant response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": final_response,
        "tool_used": tool_used
    })
