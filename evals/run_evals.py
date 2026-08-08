
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.graph.workflow import build_graph
from src.graph.state import AgentState
from config.settings import settings


def run_evaluation_suite():
    print("🧪 Starting Automated Ragas Evaluation Suite on Groq...")

    app = build_graph()

    # 1. Define Test Queries
    test_queries = [
        "What classification algorithm was used in the explosive detection system?",
        "What is the total population of Tokyo?",
        "What sensors were used in the subsurface rover hardware?"
    ]

    results_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": []
    }

    # 2. Execute Queries through our Agent StateGraph
    for query in test_queries:
        print(f"\nEvaluating query: '{query}'")
        initial_state: AgentState = {
            "query": query,
            "rewritten_query": None,
            "messages": [],
            "documents": [],
            "filtered_documents": [],
            "generation": "",
            "citations": [],
            "loop_count": 0,
            "regen_count": 0,
            "hallucination_feedback": None,
            "needs_rewrite": False,
            "is_grounded": True
        }

        final_state = app.invoke(initial_state)

        results_data["user_input"].append(query)
        results_data["response"].append(final_state.get("generation", ""))
        results_data["retrieved_contexts"].append(final_state.get("filtered_documents", []))

    # 3. Convert to Dataset format
    dataset = Dataset.from_dict(results_data)

    # 4. Wrap Groq & HuggingFace for Ragas
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.MODEL_REASONING, temperature=0.0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    answer_relevance_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

    # 5. Compute Ragas Benchmark Scores
    print("\n📊 Computing Ragas Metrics...")
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness_metric, answer_relevance_metric]
    )

    print("\n" + "=" * 50)
    print("📊 RAGAS EVALUATION SCORES:")
    print("=" * 50)
    df = scores.to_pandas()
    print(df[["user_input", "faithfulness", "answer_relevance"]])

    os.makedirs("evals", exist_ok=True)
    df.to_csv("evals/eval_results.csv", index=False)
    print("\n✅ Evaluation complete! Saved to 'evals/eval_results.csv'")


if __name__ == "__main__":
    run_evaluation_suite()