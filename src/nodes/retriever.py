# src/nodes/retriever.py
from src.services.vectorstore import VectorStoreManager
from src.services.web_search import WebSearchManager
from src.graph.state import AgentState

class RetrieverNode:
    def __init__(self, k: int = 4):
        self.vectorstore_manager = VectorStoreManager()
        self.vectorstore = self.vectorstore_manager.get_vectorstore()
        self.web_search = WebSearchManager()
        self.k = k

    def retrieve(self, state: AgentState) -> AgentState:
        search_query = state.get("rewritten_query") or state["query"]
        print(f"\n🔍 [RETRIEVER]: Querying ChromaDB for: '{search_query}'")

        docs = self.vectorstore.similarity_search(search_query, k=self.k)
        doc_contents = [doc.page_content for doc in docs]

        state["documents"] = doc_contents
        return state

    def web_search_fallback(self, state: AgentState) -> AgentState:
        """Fallback node executed when local vector retrieval fails quality checks."""
        search_query = state.get("rewritten_query") or state["query"]
        web_results = self.web_search.search(search_query)

        # Append web search snippets to documents
        state["documents"] = web_results
        return state