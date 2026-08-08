# src/nodes/generator.py
from src.services.llm_factory import get_groq_model
from src.graph.state import AgentState
from config.settings import settings

class GeneratorNode:
    def __init__(self):
        self.llm = get_groq_model(model_name=settings.MODEL_REASONING, temperature=0.0)

    def generate_answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        docs = state.get("filtered_documents", [])
        feedback = state.get("hallucination_feedback")
        history = state.get("messages", [])

        print(f"\n⚙️ [GENERATOR]: Synthesizing answer using {len(docs)} verified context chunk(s)...")

        formatted_context = ""
        for idx, doc in enumerate(docs, start=1):
            formatted_context += f"\n--- [Doc {idx}] ---\n{doc}\n"

        # Include recent prior turns so follow-up questions ("what about
        # the second one?") actually have something to resolve against.
        # Capped to the last few turns to keep the prompt bounded.
        history_prompt = ""
        if history:
            recent = history[-6:]
            formatted_history = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in recent
            )
            history_prompt = f"""
Conversation so far (for resolving references like "it" or "the second one" -
answer the CURRENT query below, using this only for context):
{formatted_history}
"""

        feedback_prompt = ""
        if feedback:
            feedback_prompt = f"""
IMPORTANT WARNING: Your previous draft was rejected for hallucination/unsupported claims:
Critique: "{feedback}"
STRICT REQUIREMENT: Fix this issue. Use ONLY numbers, dates, and facts explicitly present in the provided context blocks. Do NOT rely on prior training knowledge.
"""

        prompt = f"""You are a precise research assistant.
Answer the user query based STRICTLY on the provided document context below.
For every claim you make, cite the corresponding document source using bracket notation like [Doc 1], [Doc 2].
{history_prompt}{feedback_prompt}

User Query:
{query}

Provided Context:
{formatted_context}

Answer with inline citations:"""

        response = self.llm.invoke(prompt)
        state["generation"] = response.content.strip()
        return state