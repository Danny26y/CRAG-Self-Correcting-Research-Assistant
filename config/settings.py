import os

from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Self-Correcting Research Assistant"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    MODEL_REASONING: str = "llama-3.3-70b-versatile"
    MODEL_FAST: str = "llama-3.1-8b-instant"

    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    COLLECTION_NAME: str = "research_docs"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    MAX_RETRY_LOOPS: int = 3


settings = Settings()
