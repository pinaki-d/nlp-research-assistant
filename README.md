# Multi-turn NLP Research Assistant with Memory & Tool Use

A stateful conversational agent built with **LangGraph** that retains memory across conversation turns and dynamically routes queries to the right tool — arXiv paper search or web search — based on intent.

## Architecture

```
User Input
    │
    ▼
classify_and_respond (GPT-4o-mini)
    │
    ├── Tool needed? ──► tool_node ──► back to classify_and_respond
    │
    └── No tool needed? ──► Final Response
```

The agent uses a **3-node StateGraph**:
1. `classify_and_respond` — LLM decides to answer directly or call a tool
2. `tools` — executes whichever tool was requested (arXiv or web search)
3. Routes back to `classify_and_respond` after tool execution to synthesise the final answer

**Memory** is handled by LangGraph's `MemorySaver` checkpointer — each session gets a unique `thread_id` and all messages are persisted in memory across turns.

## Features

- Multi-turn conversation with persistent memory
- Dynamic tool routing — arXiv for academic queries, DuckDuckGo for general web search
- Tool call trace visible in sidebar
- Session isolation — each browser tab gets its own memory thread
- Clean Streamlit UI with conversation history

## Tech Stack

| Component | Library |
|-----------|---------|
| Agent framework | LangGraph |
| LLM | GPT-4o-mini (OpenAI) |
| Tools | arXiv API, DuckDuckGo Search |
| Memory | LangGraph MemorySaver |
| Frontend | Streamlit |
| Orchestration | LangChain |

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/pinaki-d/nlp-research-assistant
cd nlp-research-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your-openai-api-key-here
```

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## Example Conversations

**Multi-turn with memory:**
```
User: What is BERT?
Assistant: BERT (Bidirectional Encoder Representations from Transformers)...

User: Who are its authors?        ← remembers context from previous turn
Assistant: BERT was developed by Jacob Devlin, Ming-Wei Chang...

User: Find me the original paper  ← triggers arXiv tool
Assistant: [searches arXiv and returns paper details]
```

## Project Structure

```
nlp-research-assistant/
├── agent/
│   ├── __init__.py
│   ├── graph.py      # LangGraph StateGraph definition
│   ├── nodes.py      # Node functions + LLM setup
│   └── tools.py      # arXiv and web search tools
├── app.py            # Streamlit frontend
├── requirements.txt
├── .env              # API keys (never commit this)
└── README.md
```
