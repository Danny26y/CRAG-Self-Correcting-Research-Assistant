# src/graph/state.py
from typing import List, TypedDict, Optional, Dict

class AgentState(TypedDict):
    query: str
    rewritten_query: Optional[str]
    documents: List[str]
    filtered_documents: List[str]
    generation: str
    citations: List[str]
    loop_count: int
    regen_count: int
    hallucination_feedback: Optional[str]
    needs_rewrite: bool
    is_grounded: bool
    messages: List[Dict[str, str]]