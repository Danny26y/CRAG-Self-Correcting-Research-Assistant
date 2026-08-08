
import os
from langfuse.langchain import CallbackHandler
from config.settings import settings

def get_langfuse_handler():
    """Initializes and returns the Langfuse CallbackHandler for LangGraph tracing."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        print("⚠️ Langfuse keys missing. Running agent without external tracing.")
        return None

    return CallbackHandler()