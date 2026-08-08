# main.py
import time
from src.graph.workflow import build_graph
from src.graph.state import AgentState

def run_assistant(query: str):
    print("=" * 70)
    print(f"🚀 SELF-CORRECTING RESEARCH ASSISTANT (Groq Accelerated)")
    print(f"   User Query: '{query}'")
    print("=" * 70)

    app = build_graph()

    initial_state: AgentState = {
        "query": query,
        "rewritten_query": None,
        "documents": [],
        "filtered_documents": [],
        "generation": "",
        "citations": [],
        "loop_count": 0,
        "needs_rewrite": False,
        "is_grounded": True
    }

    t0 = time.time()
    final_state = app.invoke(initial_state)
    t1 = time.time()

    print("\n" + "=" * 70)
    print("💡 FINAL RESPONSE:")
    print("=" * 70)
    print(final_state["generation"])
    print("=" * 70)
    print(f"⏱️ Total Execution Time: {t1 - t0:.2f} seconds | Loops Executed: {final_state.get('loop_count', 0)}")
    print("=" * 70)

if __name__ == "__main__":
    test_query = "What is the total population of tokyo"
    run_assistant(test_query)