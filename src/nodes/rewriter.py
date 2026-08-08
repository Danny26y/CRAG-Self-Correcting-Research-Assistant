from src.services.llm_factory import get_groq_model
from src.graph.state import AgentState
from config.settings import settings

class QueryRewriterNode:
    def __init__(self):
        # Use Llama 70B for higher query re-formulation intelligence
        self.llm = get_groq_model(model_name=settings.MODEL_REASONING, temperature=0.2)

    def rewrite_query(self, state: AgentState) -> AgentState:
        """Transforms user query into an optimized search term when retrieval fails/lacks depth."""
        original_query = state["query"]
        current_loop = state.get("loop_count", 0) + 1

        print(f"\n [REWRITER]: Context was insufficient. Rewriting query (Attempt {current_loop})...")

        prompt = f"""You are a query optimization engine for vector search retrieval.
The initial vector search for the user query returned irrelevant or poor context.
Analyze the original user query and formulate a clearer, more specific search query optimized for vector retrieval.

Original Query: {original_query}

Output ONLY the revised query string. Do not include quotes, preamble, or explanation."""

        response = self.llm.invoke(prompt)
        revised_query = response.content.strip()

        print(f"   Original : '{original_query}'")
        print(f"   Revised  : '{revised_query}'")

        state["rewritten_query"] = revised_query
        state["loop_count"] = current_loop
        return state