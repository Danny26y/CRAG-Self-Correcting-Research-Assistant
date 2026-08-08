# app.py
import os
import time
import streamlit as st
from src.graph.workflow import build_graph
from src.graph.state import AgentState
from src.services.vectorstore import VectorStoreManager
from src.services.tracing import get_langfuse_handler

st.set_page_config(page_title="Self-Correcting Research Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Self-Correcting Research Assistant")
st.caption("Powered by Groq LPU, LangGraph, ChromaDB, and Tavily Web Search")


# Initialize Agent & Vector Store
@st.cache_resource
def load_agent():
    return build_graph(), VectorStoreManager()


graph_app, vector_manager = load_agent()

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # List of {"role": "user"/"assistant", "content": "..."}

if "contexts_history" not in st.session_state:
    st.session_state.contexts_history = {}  # Maps message index to retrieved docs

# --- SIDEBAR: KNOWLEDGE BASE INGESTION ---
with st.sidebar:
    st.header("📄 Knowledge Base Ingestion")
    st.write("Upload `.pdf` or `.txt` research documents to index into ChromaDB.")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Process & Index Documents"):
        if uploaded_files:
            docs_dir = "./data/raw_docs"
            os.makedirs(docs_dir, exist_ok=True)

            for file in uploaded_files:
                file_path = os.path.join(docs_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            with st.spinner("Chunking & Embedding documents..."):
                vector_manager.ingest_documents(docs_dir)
            st.success("✅ Knowledge base updated successfully!")
        else:
            st.warning("Please attach at least one PDF or TXT file.")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.contexts_history = {}
        st.rerun()

# --- RENDER PAST CHAT HISTORY ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display sources if assistant message has associated contexts
        if msg["role"] == "assistant" and idx in st.session_state.contexts_history:
            docs = st.session_state.contexts_history[idx]
            if docs:
                with st.expander("📚 View Verified Document Contexts"):
                    for d_idx, doc in enumerate(docs, start=1):
                        st.info(f"**[Doc {d_idx}]**\n{doc}")

# --- USER INPUT & MULTI-TURN EXECUTION ---
if user_query := st.chat_input("Ask a question or follow-up..."):
    # 1. Render User Message
    st.chat_message("user").markdown(user_query)

    # 2. Append User Query to Session State
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 3. Execute LangGraph Agent with Conversation History
    with st.chat_message("assistant"):
        with st.status("🚀 Agent Processing Pipeline...", expanded=True) as status:

            initial_state: AgentState = {
                "query": user_query,
                "rewritten_query": None,
                "messages": st.session_state.messages[:-1],  # Pass prior history
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

            langfuse_handler = get_langfuse_handler()
            config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}

            t0 = time.time()
            final_state = graph_app.invoke(initial_state, config=config)
            t1 = time.time()

            status.update(label=f"✅ Completed in {t1 - t0:.2f} seconds!", state="complete", expanded=False)

        # 4. Display Assistant Output
        response_text = final_state.get("generation", "No response generated.")
        st.markdown(response_text)

        # 5. Display Source Contexts
        filtered_docs = final_state.get("filtered_documents", [])
        if filtered_docs:
            with st.expander("📚 View Verified Document Contexts"):
                for d_idx, doc in enumerate(filtered_docs, start=1):
                    st.info(f"**[Doc {d_idx}]**\n{doc}")

        # 6. Save Assistant Response & Sources to Session State
        assistant_idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.contexts_history[assistant_idx] = filtered_docs