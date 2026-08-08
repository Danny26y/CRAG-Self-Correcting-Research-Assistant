# src/graph/edges.py
from typing import Literal
from src.graph.state import AgentState
from config.settings import settings


def decide_after_grading(state: AgentState) -> Literal["generate", "web_search", "rewrite"]:
    """
    Determines next step after document grading:
    1. If relevant docs exist -> generate answer.
    2. If no docs & loop_count >= 1 -> fallback to Tavily Web Search.
    3. If no docs & loop_count == 0 -> rewrite query for 2nd Chroma search.
    """
    filtered_docs = state.get("filtered_documents", [])
    current_loops = state.get("loop_count", 0)

    if not filtered_docs:
        if current_loops >= 1:
            print("🌐 [ROUTER]: Local documents insufficient after retry. Routing to Tavily Web Search...")
            return "web_search"

        print("⚠️ [ROUTER]: Zero relevant documents found. Routing to Query Rewriter...")
        return "rewrite"

    print(f"✅ [ROUTER]: Found {len(filtered_docs)} relevant document(s). Routing to Generator...")
    return "generate"


# src/graph/edges.py

def decide_after_hallucination_check(state: AgentState) -> Literal["finish", "re_generate"]:
    is_grounded = state.get("is_grounded", True)
    regen_count = state.get("regen_count", 0)

    if is_grounded:
        print("✅ [ROUTER]: Answer passed hallucination check! Returning final output.")
        return "finish"

    if regen_count >= 2:
        print(f"⚠️ [ROUTER]: Exceeded max hallucination retries ({regen_count}). Returning best candidate response.")
        return "finish"

    print(
        f"❌ [ROUTER]: Answer contains ungrounded claims (Attempt {regen_count}). Requesting re-generation with feedback...")
    return "re_generate"