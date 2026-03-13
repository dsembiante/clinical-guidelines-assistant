# ingest.py
# Run this script once to load PDFs, chunk them, embed them, and store in Chroma.
# Re-run any time new documents are added to the docs/ directory.
import os

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
)


def ingest_documents():
    """Load PDFs from docs/, chunk them, embed them, and store in Chroma."""
    print("Starting document ingestion...")

    # Configure the global embedding model used by LlamaIndex
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # Validate that the docs directory exists and contains files before proceeding
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        raise FileNotFoundError(
            f"No documents found in {DOCS_DIR}/. "
            "Please add PDF files before running ingestion."
        )

    # Load all PDFs from the docs directory (non-recursive to keep scope explicit)
    print(f"Loading documents from {DOCS_DIR}/")
    documents = SimpleDirectoryReader(
        DOCS_DIR,
        required_exts=[".pdf"],
        recursive=False,
    ).load_data()
    print(f"Loaded {len(documents)} document pages")

    # Split documents into overlapping chunks to preserve context across boundaries
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Initialize the persistent Chroma client so embeddings survive across sessions
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop the existing collection on re-ingestion to avoid stale or duplicate vectors
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Cleared existing vector store")
    except Exception:
        # Collection doesn't exist yet on first run — safe to continue
        pass

    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Generate embeddings for each chunk and persist them to the vector store
    print("Generating embeddings and building index...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    print(f"Ingestion complete. Vector store saved to {CHROMA_DIR}/")
    return index


if __name__ == "__main__":
    ingest_documents()
