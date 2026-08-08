from langchain_groq import ChatGroq
from config.settings import settings

def get_groq_model(model_name: str= None, temperature: float = 0.0)-> ChatGroq:


    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    selected_model = model_name or settings.MODEL_REASONING

    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=selected_model,
        temperature=temperature
    )

if __name__ == "__main__":
    llm = get_groq_model(settings.MODEL_FAST)
    response = llm.invoke("Ping! Reply with Groq connected successfully")
    print(f"Groq response: {response.content}")



