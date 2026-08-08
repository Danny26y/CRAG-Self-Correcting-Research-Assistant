# src/graph/state.py
from typing import List, TypedDict, Optional

class AgentState(TypedDict):
    query: str
    rewritten_query: Optional[str]
    documents: List[str]
    filtered_documents: List[str]
    generation: str
    citations: List[str]
    loop_count: int
    regen_count: int               # New: tracks hallucination retry loop count
    hallucination_feedback: Optional[str]  # New: stores evaluator critique
    needs_rewrite: bool
    is_grounded: bool