from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from agent.tools import tools
import os
from dotenv import load_dotenv

load_dotenv()

# Initialise LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3
)

# Bind tools to LLM so it can decide when to call them
llm_with_tools = llm.bind_tools(tools)

# System prompt that defines the assistant's behaviour
SYSTEM_PROMPT = """You are a helpful NLP research assistant with access to two tools:
1. search_arxiv: Search academic papers on arXiv
2. search_web: Search the web for general information

Guidelines:
- Use search_arxiv when the user asks about research papers, ML models, or academic topics
- Use search_web for current events, general facts, or non-academic queries
- If you already know the answer confidently, respond directly without using tools
- Always remember the conversation history and refer back to it when relevant
- Be concise but informative in your responses
"""


def classify_and_respond(state: dict) -> dict:
    """
    Main node: takes the conversation state, decides whether to
    call a tool or respond directly, and returns updated messages.
    """
    messages = state["messages"]

    # Prepend system message if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # Call LLM — it will either respond or request a tool call
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}


def should_use_tool(state: dict) -> str:
    """
    Router function: checks if the last message contains a tool call.
    Returns 'tools' if yes, 'end' if no.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ToolNode handles executing whichever tool the LLM requested
tool_node = ToolNode(tools)
