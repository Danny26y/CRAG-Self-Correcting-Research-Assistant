# src/graph/workflow.py
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.nodes.retriever import RetrieverNode
from src.nodes.grader import EvaluatorEngine
from src.nodes.rewriter import QueryRewriterNode
from src.nodes.generator import GeneratorNode
from src.graph.edges import decide_after_grading, decide_after_hallucination_check

# Initialize components
retriever_node = RetrieverNode()
evaluator_engine = EvaluatorEngine()
rewriter_node = QueryRewriterNode()
generator_node = GeneratorNode()

# --- NODE WRAPPERS ---

def retrieve_node(state: AgentState) -> AgentState:
    return retriever_node.retrieve(state)

def web_search_node(state: AgentState) -> AgentState:
    return retriever_node.web_search_fallback(state)

def grade_documents_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["query"]
    documents = state.get("documents", [])
    filtered = []

    print(f"\n📋 [GRADER]: Evaluating {len(documents)} document chunk(s)...")
    for idx, doc in enumerate(documents, start=1):
        grade = evaluator_engine.grade_document_relevance(query, doc)
        if grade.binary_score == "yes":
            filtered.append(doc)
            print(f"   Chunk #{idx}: ✅ Relevant | Reason: {grade.explanation}")
        else:
            print(f"   Chunk #{idx}: ❌ Irrelevant | Reason: {grade.explanation}")

    state["filtered_documents"] = filtered
    return state

def rewrite_node(state: AgentState) -> AgentState:
    return rewriter_node.rewrite_query(state)

def generate_node(state: AgentState) -> AgentState:
    return generator_node.generate_answer(state)

def check_hallucination_node(state: AgentState) -> AgentState:
    context = "\n".join(state.get("filtered_documents", []))
    generation = state.get("generation", "")
    current_regens = state.get("regen_count", 0)

    print("\n🔍 [HALLUCINATION CHECK]: Verifying answer grounding against context...")
    result = evaluator_engine.check_hallucination(context, generation)

    if result.binary_score == "yes":
        print(f"   Status: ✅ Grounded | {result.explanation}")
        state["is_grounded"] = True
        state["hallucination_feedback"] = None
    else:
        print(f"   Status: ❌ Hallucinated | {result.explanation}")
        state["is_grounded"] = False
        state["hallucination_feedback"] = result.explanation
        state["regen_count"] = current_regens + 1

    return state
# --- GRAPH ASSEMBLY ---

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("query_rewriter", rewrite_node)
    workflow.add_node("generator", generate_node)
    workflow.add_node("hallucination_checker", check_hallucination_node)

    # 2. Entry Point
    workflow.set_entry_point("retriever")

    # 3. Static & Conditional Edges
    workflow.add_edge("retriever", "grade_documents")

    # Route after document grading: [generate | rewrite | web_search]
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generator",
            "rewrite": "query_rewriter",
            "web_search": "web_search"
        }
    )

    # Route from Rewriter back to Retriever
    workflow.add_edge("query_rewriter", "retriever")

    # Route from Web Search directly to Grade Documents
    workflow.add_edge("web_search", "grade_documents")

    # Route from Generator to Hallucination Checker
    workflow.add_edge("generator", "hallucination_checker")

    # Route after Hallucination Check
    workflow.add_conditional_edges(
        "hallucination_checker",
        decide_after_hallucination_check,
        {
            "finish": END,
            "re_generate": "generator"
        }
    )

    return workflow.compile()