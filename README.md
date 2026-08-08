# Self-Correcting Research Assistant (Groq Accelerated)

An autonomous, corrective retrieval-augmented generation (CRAG) system designed to deliver zero-hallucination answers from local documents and web sources. Powered by Groq LPU inference, LangGraph state machines, ChromaDB, and Tavily Web Search.

## 🎯 Architecture & Design Philosophy

Standard RAG pipelines follow a rigid, single-pass path: **Retrieve → Prompt → Generate**. If the vector database returns irrelevant context or missing information, standard RAG fails by either giving up or hallucinating incorrect facts.

This project implements a **Self-Correcting State Machine** that reflects on its own retrieval and output quality before presenting a final answer:

```
                  [ User Query ]
                        │
                        ▼
            ┌──────────────────────┐
            │   1. Retrieve Docs   │
            └──────────┬───────────┘
                        │
                        ▼
            ┌──────────────────────┐
            │  2. Grade Documents  │ (Pydantic Binary Evaluation)
            └──────────┬───────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   [ Relevant Context ]       [ Irrelevant / Thin ]
          │                           │
          │                           ▼
          │               ┌──────────────────────┐
          │               │ 3. Query Rewriter /  │
          │               │    Tavily Fallback   │
          │               └──────────┬───────────┘
          │                          │
          └─────────────┬────────────┘
                         │
                         ▼
            ┌──────────────────────┐
            │ 4. Synthesize Answer │ (Inline Citations)
            └──────────┬───────────┘
                        │
                        ▼
            ┌──────────────────────┐
            │  5. Self-Correction  │
            │   (Hallucination &   │
            │  Completeness Check) │
            └──────────┬───────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
      [ Passed ]                 [ Failed ]
          │                           │
          ▼                           └───► [ Feedback Loop to Step 4 ]
  [ Final Response ]
```

## Core Innovations

- **Pydantic Structured Graders** — Uses `llama-3.1-8b-instant` via Groq for fast document relevance scoring and hallucination evaluations.
- **Autonomous Query Transformation** — If initial vector search results yield zero relevant chunks, the agent rewrites the query using conversation history and target search terms.
- **Web Search Fallback** — Automatically queries Tavily Web Search when local document coverage is insufficient.
- **Hallucination Circuit Breaker** — Evaluates whether candidate responses are grounded in context facts. If an ungrounded claim is detected, critique feedback is injected back into the generator for correction.

## 📁 Project Structure

```
agentic-research-assistant/
│
├── .env                         # API secrets (GROQ, TAVILY, LANGFUSE)
├── .streamlit/
│   └── config.toml              # Streamlit watcher configuration
├── README.md                    # Project documentation
├── requirements.txt             # Dependency declarations
├── main.py                      # CLI runner harness
├── app.py                       # Streamlit multi-turn chat interface
│
├── config/
│   └── settings.py              # Central Pydantic project configuration
│
├── data/
│   └── raw_docs/                # Local PDF and TXT documents for ingestion
│
├── evals/
│   └── run_evals.py             # Automated Ragas evaluation suite
│
└── src/
    ├── graph/
    │   ├── state.py             # TypedDict AgentState definition
    │   ├── workflow.py          # StateGraph assembly & node links
    │   └── edges.py             # Conditional router functions
    │
    ├── nodes/
    │   ├── retriever.py         # ChromaDB & Tavily search integration
    │   ├── grader.py            # Fast relevance & hallucination evaluators
    │   ├── rewriter.py          # Context-aware query transformer
    │   └── generator.py         # RAG synthesis with feedback loops
    │
    ├── schemas/
    │   └── graders.py           # Pydantic schemas for structured outputs
    │
    └── services/
        ├── llm_factory.py       # Groq client factory (8B & 70B models)
        ├── vectorstore.py       # ChromaDB persistent vector manager
        ├── web_search.py        # Tavily search integration
        └── tracing.py           # Langfuse observability callback
```

## ⚙️ Tech Stack & Key Technologies

| Category | Technology |
|---|---|
| LPU Inference | Groq API (`llama-3.3-70b-versatile` for synthesis, `llama-3.1-8b-instant` for evaluation) |
| Agentic Orchestration | LangGraph, LangChain |
| Vector Store & Embeddings | ChromaDB, `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| Web Search | Tavily API |
| Observability | Langfuse |
| Evaluation | Ragas, Datasets |
| User Interface | Streamlit (multi-turn chat, drag-and-drop document upload) |

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and set up a Python virtual environment (Python 3.10+ recommended):

```powershell
git clone <repository-url>
cd agentic-research-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```ini
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Ingest Local Documents

Place `.txt` or `.pdf` files inside `data/raw_docs/` and execute the ingestion script:

```powershell
python -m src.services.vectorstore
```

## 🧪 Usage

**Run via Interactive Streamlit Dashboard**

Launch the multi-turn web interface with real-time execution tracing and file upload capabilities:

```powershell
streamlit run app.py
```

**Run via Command-Line Interface (CLI)**

Run single-query tests through the terminal execution harness:

```powershell
python main.py
```

**Run Automated Quality Evaluation (Ragas)**

Execute the regression test suite to measure Faithfulness and Answer Relevance:

```powershell
python -m evals.run_evals
```

## 📊 MLOps & Observability

- **Langfuse Tracing** — Passing `config={"callbacks": [langfuse_handler]}` to `app.invoke()` logs execution latency, token counts, and state transitions to your Langfuse cloud dashboard.
- **Automated Metrics** — `evals/run_evals.py` generates `evals/eval_results.csv` containing scores for generated responses across local and web contexts.

## 🔮 What Next? (Future Enhancements)

If you plan to extend or scale this project further, here is the suggested roadmap:

1. **Hybrid Search (BM25 + Dense Vectors)** — Currently, retrieval relies solely on dense vector embeddings (`all-MiniLM-L6-v2`). Integrating sparse keyword search (BM25) alongside dense search via Reciprocal Rank Fusion (RRF) will significantly boost retrieval accuracy for domain-specific terminology, part numbers, and precise technical codes.
2. **Async Parallel Batch Grading** — Update `src/nodes/grader.py` to evaluate retrieved document chunks in a single structured batch call or via `asyncio.gather`. This will reduce total loop latency when handling larger top-k retrieval counts.
3. **Multimodal Document Parsing** — Expand ingestion from text-only extraction (`pypdf`) to layout-aware tools like Unstructured or LlamaParse to parse tables, charts, and embedded images inside scientific PDFs.
4. **FastAPI Endpoint Deployment** — Package the state graph inside a lightweight FastAPI web server. Exposing `/chat` and `/ingest` endpoints allows this agent engine to be integrated into external web dashboards, desktop applications, or mobile frontends.
