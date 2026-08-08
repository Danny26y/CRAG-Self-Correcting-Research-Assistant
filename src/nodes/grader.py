from src.services.llm_factory import get_groq_model
from src.schemas.graders import GradeAnswering, GradeDocument, GradeHallucination
from config.settings import settings

class EvaluatorEngine:
    def __init__(self):
        self.fast_llm = get_groq_model(model_name=settings.MODEL_FAST, temperature=0.0)

        self.doc_grader = self.fast_llm.with_structured_output(GradeDocument)
        self.hallucination_grader = self.fast_llm.with_structured_output(GradeHallucination)
        self.answer_grader = self.fast_llm.with_structured_output(GradeAnswering)

    def grade_document_relevance(self, query: str, document_chunk: str) -> GradeDocument:
        """Evaluates if a single document chunk is relevant to the user's query."""
        prompt = f"""You are a strict retrieval quality evaluator.
        Evaluate if the following document context is relevant to the user query.

        User Query: {query}

        Document Context:
        {document_chunk}
        """
        return self.doc_grader.invoke(prompt)

    def check_hallucination(self, context: str, generation: str) -> GradeHallucination:
        """Checks if generated answer contains hallucinated facts not supported by context."""
        prompt = f"""You are a strict hallucination checker.
        Evaluate if the candidate answer is strictly supported by the provided context facts.

        Context Facts:
        {context}

        Candidate Answer:
        {generation}
        """
        return self.hallucination_grader.invoke(prompt)

    def check_answer_completeness(self, query: str, generation: str) -> GradeAnswering:
        """Checks if generated answer fully resolves the user query."""
        prompt = f"""You are a task completion evaluator.
        Evaluate if the answer fully and directly addresses the user query.

        User Query: {query}

        Candidate Answer:
        {generation}
        """
        return self.answer_grader.invoke(prompt)

if __name__ == "__main__":
    import time

    print("🧪 Testing Evaluator Engine with Groq Fast Model...")
    evaluator = EvaluatorEngine()

    sample_query = "What database are we using for embeddings?"
    sample_context = "We selected ChromaDB for local vector persistence and SentenceTransformers for embeddings."
    sample_irrelevant = "The weather today is sunny with a mild breeze."

    # 1. Test Relevance (Relevant Case)
    t0 = time.time()
    res1 = evaluator.grade_document_relevance(sample_query, sample_context)
    t1 = time.time()
    print(f"\n1. Relevant Context Check ({int((t1 - t0) * 1000)}ms):")
    print(f"   Score: {res1.binary_score} | Reason: {res1.explanation}")

    # 2. Test Relevance (Irrelevant Case)
    t0 = time.time()
    res2 = evaluator.grade_document_relevance(sample_query, sample_irrelevant)
    t1 = time.time()
    print(f"\n2. Irrelevant Context Check ({int((t1 - t0) * 1000)}ms):")
    print(f"   Score: {res2.binary_score} | Reason: {res2.explanation}")


