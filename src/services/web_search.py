
from tavily import TavilyClient
from config.settings import settings


class WebSearchManager:
    def __init__(self):
        if not settings.TAVILY_API_KEY:
            self.client = None
            print("⚠️ TAVILY_API_KEY not set. Web search fallback is disabled.")
        else:
            self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

    def search(self, query: str, max_results: int = 3) -> list[str]:
        """Performs web search and returns snippets."""
        if not self.client:
            return []

        print(f"🌐 [WEB SEARCH]: Querying Tavily for: '{query}'")
        try:
            response = self.client.search(query=query, max_results=max_results)
            results = [res["content"] for res in response.get("results", [])]
            print(f"   Retrieved {len(results)} web result snippet(s).")
            return results
        except Exception as e:
            print(f"[WEB SEARCH ERROR]: {e}")
            return []