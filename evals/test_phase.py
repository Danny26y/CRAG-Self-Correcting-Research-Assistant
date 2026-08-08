# test_phase3.py
from src.graph.state import AgentState
from src.nodes.retriever import RetrieverNode
from src.nodes.rewriter import QueryRewriterNode
from src.nodes.generator import GeneratorNode

# Initialize state
state: AgentState = {
    "query": "What vector database are we using?",
    "rewritten_query": None,
    "documents": [],
    "filtered_documents": [],
    "generation": "",
    "citations": [],
    "loop_count": 0,
    "needs_rewrite": False
}

# Run retrieval
retriever = RetrieverNode()
state = retriever.retrieve(state)

# Simulate filtering
state["filtered_documents"] = state["documents"][:2]

# Run generation
generator = GeneratorNode()
state = generator.generate_answer(state)

print("\n--- GENERATED OUTPUT ---")
print(state["generation"])