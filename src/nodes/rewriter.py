# src/nodes/rewriter.py
from src.services.llm_factory import get_groq_model
from src.graph.state import AgentState
from config.settings import settings

class QueryRewriterNode:
    def __init__(self):
        self.llm = get_groq_model(model_name=settings.MODEL_REASONING, temperature=0.2)

    def rewrite_query(self, state: AgentState) -> AgentState:
        original_query = state["query"]
        history = state.get("messages", [])
        current_loop = state.get("loop_count", 0) + 1

        print(f"\n🔄 [REWRITER]: Rewriting query with chat history context (Attempt {current_loop})...")

        # Format past 3 message turns for context
        history_context = ""
        if history:
            history_context = "Past Conversation History:\n"
            for msg in history[-4:]:
                history_context += f"{msg['role'].capitalize()}: {msg['content']}\n"

        prompt = f"""You are a query rewriter for vector database search.
Analyze the user's latest query along with the conversation history (if any) and rewrite it into a self-contained, highly specific search query for vector retrieval.

{history_context}
Latest User Query: {original_query}

Output ONLY the standalone search query string. Do not include quotes or preamble."""

        response = self.llm.invoke(prompt)
        revised_query = response.content.strip()

        print(f"   Original : '{original_query}'")
        print(f"   Revised  : '{revised_query}'")

        state["rewritten_query"] = revised_query
        state["loop_count"] = current_loop
        return state