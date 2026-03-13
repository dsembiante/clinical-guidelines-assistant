# rag_engine.py
# Core RAG engine that wires together LlamaIndex, Chroma, Ollama, conversation
# memory, and citation extraction into a single interface for the Streamlit UI.
import chromadb
from dataclasses import dataclass
from typing import Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    HIGH_CONFIDENCE_THRESHOLD,
    LLM_MODEL,
    MEMORY_TOKEN_LIMIT,
    OLLAMA_BASE_URL,
    TOP_K_RESULTS,
)


@dataclass
class Citation:
    """Represents a single source citation from a retrieved document chunk."""
    filename: str
    page_number: Optional[int]
    relevance_score: float
    text_snippet: str
    confidence_label: str  # "High", "Medium", or "Low" based on relevance score


@dataclass
class RAGResponse:
    """Complete response object returned to the Streamlit UI."""
    answer: str
    citations: list[Citation]
    query_type: str           # "document_search" or "general_knowledge"
    used_document_search: bool


class RAGEngine:
    """
    Core RAG engine with conversation memory and source citations.
    Maintains chat history across queries within a session.
    """

    def __init__(self):
        # temperature=0.1 keeps responses factual while allowing slight natural variation
        # request_timeout increased to 300s — local LLMs can be slow on first inference
        Settings.llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            request_timeout=300.0,
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # Connect to the persistent Chroma store created by ingest.py
        # Uses get_collection (not get_or_create) to fail fast if ingestion hasn't been run
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

        # Token-limited memory prevents the context window from growing unbounded
        # across a long conversation session
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=MEMORY_TOKEN_LIMIT,
        )

        # CondensePlusContextChatEngine condenses prior chat history into the retrieval
        # query so follow-up questions remain contextually accurate
        retriever = self.index.as_retriever(similarity_top_k=TOP_K_RESULTS)
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            memory=self.memory,
            system_prompt=(
                "You are a helpful healthcare information assistant. "
                "You answer questions based on official CDC, NIH, and CMS guidelines. "
                "Always be accurate and cite the guidelines when possible. "
                "If the provided context does not contain enough information "
                "to answer the question, say so clearly rather than speculating. "
                "Never provide personal medical advice."
            ),
            verbose=False,
        )

        # Store retriever separately so _extract_citations can call it independently
        self.retriever = retriever

    def _extract_citations(self, query: str) -> list[Citation]:
        """Retrieve source nodes and extract citation metadata."""
        nodes = self.retriever.retrieve(query)
        citations = []

        for node in nodes:
            metadata = node.metadata or {}
            score = node.score or 0.0

            filename = metadata.get("file_name", "Unknown document")
            page = metadata.get("page_label", None)

            # Truncate long chunks to a readable snippet for display in the UI
            snippet = node.text[:300] + "..." if len(node.text) > 300 else node.text

            # Label confidence based on thresholds defined in config.py
            confidence_label = (
                "High" if score >= HIGH_CONFIDENCE_THRESHOLD
                else "Medium" if score >= 0.5
                else "Low"
            )

            citations.append(Citation(
                filename=filename,
                page_number=int(page) if page else None,
                relevance_score=round(score, 3),
                text_snippet=snippet,
                confidence_label=confidence_label,
            ))

        # Return citations sorted by relevance so the most pertinent sources appear first
        return sorted(citations, key=lambda c: c.relevance_score, reverse=True)

    def query_documents(self, question: str) -> RAGResponse:
        """Query the document store with conversation memory."""
        response = self.chat_engine.chat(question)
        citations = self._extract_citations(question)

        return RAGResponse(
            answer=str(response),
            citations=citations,
            query_type="document_search",
            used_document_search=True,
        )

    def query_general(self, question: str) -> RAGResponse:
        """Answer from general knowledge using conversation memory.

        Still routes through the chat engine to preserve memory continuity,
        but explicitly instructs the LLM not to rely on document retrieval.
        """
        response = self.chat_engine.chat(
            f"Please answer this question from your general medical knowledge "
            f"(no need to search specific documents): {question}"
        )

        return RAGResponse(
            answer=str(response),
            citations=[],  # No citations for general knowledge responses
            query_type="general_knowledge",
            used_document_search=False,
        )

    def reset_memory(self):
        """Clear conversation history. Called when user starts a new chat session."""
        self.memory.reset()
