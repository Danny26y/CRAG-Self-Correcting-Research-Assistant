import os
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.settings import settings

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = settings.CHROMA_PERSIST_DIR
        self.collection_name = settings.COLLECTION_NAME

    def get_vectorstore(self)-> Chroma:
        return Chroma(
            collection_name= self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def ingest_documents(self, docs_dir: str = "./data/raw_docs"):
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)
            print(f"⚠️ Created empty folder at '{docs_dir}'. Please place text/pdf files there.")
            return None

        documents = []

        # 1. Load TXT Files
        txt_loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())

        # 2. Load PDF Files
        try:
            pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"⚠️ Warning loading PDFs (make sure 'pypdf' is installed): {e}")

        if not documents:
            print(f"⚠️ No .txt or .pdf documents found in '{docs_dir}' to ingest.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)

        # Fix: Assign return object to `vectorstore`
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        print(f"✅ Ingestion complete! Added {len(chunks)} chunk(s) from {len(documents)} file(s).")
        return vectorstore


if __name__ == "__main__":
    manager = VectorStoreManager()
    manager.ingest_documents()
