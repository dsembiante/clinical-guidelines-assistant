# rag_engine.py
# Core RAG engine that wires together LlamaIndex, Chroma, Groq, conversation
# memory, and citation extraction into a single interface for the Streamlit UI.
import chromadb
from dataclasses import dataclass
from typing import Optional

import re

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    GROQ_API_KEY,
    HIGH_CONFIDENCE_THRESHOLD,
    LLM_MODEL,
    MEMORY_TOKEN_LIMIT,
    TOP_K_RESULTS,
)

# Prompt template for document-grounded answers.
# Context is injected manually so we control exactly what the LLM sees —
# no metadata, no partial sentences, just clean chunk text.
DOCUMENT_QA_PROMPT = """You are a helpful healthcare information assistant.
Using ONLY the document excerpts provided below, answer the question clearly and concisely.
Write in flowing prose. Do NOT include excerpt labels, numbers, or references in your answer.
Do NOT repeat or quote raw data verbatim — synthesize it into readable sentences.
Do NOT add follow-up questions at the end of your answer.
If the excerpts do not contain enough information to answer the question, say so clearly.
Never provide personal medical advice.

Document excerpts:
{context}

Question: {question}

Answer (in plain prose, no labels or follow-up questions):"""


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
        # Groq provides fast cloud inference for Llama 3 — no local GPU required
        # temperature=0.1 keeps responses factual while allowing slight natural variation
        self.llm = Groq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,
        )
        Settings.llm = self.llm

        # HuggingFaceEmbedding runs locally — no API key required
        # Must match the embedding model used during ingestion
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

        # Connect to the persistent Chroma store created by ingest.py
        # Uses get_collection (not get_or_create) to fail fast if ingestion hasn't been run
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

        # Retriever used for both document search and citation extraction
        self.retriever = index.as_retriever(similarity_top_k=TOP_K_RESULTS)

        # Token-limited memory prevents the context window from growing unbounded
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=MEMORY_TOKEN_LIMIT,
        )

        # SimpleChatEngine handles general knowledge queries with conversation memory
        # No retrieval involved — answers come directly from the LLM
        self.chat_engine = SimpleChatEngine.from_defaults(
            memory=self.memory,
            system_prompt=(
                "You are a helpful healthcare information assistant. "
                "You answer questions based on official CDC, NIH, and CMS guidelines. "
                "Always be accurate. If you are unsure, say so clearly. "
                "Never provide personal medical advice."
            ),
        )

    def _build_context(self, nodes) -> str:
        """Extract plain text from retrieved nodes with no metadata.

        MetadataMode.NONE ensures file paths, page labels, and other metadata
        are completely excluded from the text passed to the LLM.
        """
        excerpts = []
        seen: set[str] = set()
        for node in nodes:
            text = node.node.get_content(metadata_mode=MetadataMode.NONE).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            # If the chunk starts mid-sentence (lowercase first char), strip the
            # incomplete leading fragment up to the first complete sentence boundary
            if text and text[0].islower():
                match = re.search(r'[.!?]\s+[A-Z]', text)
                if match:
                    text = text[match.start() + 2:]
                else:
                    continue  # Skip chunks that are entirely a sentence fragment
            excerpts.append(text)
        return "\n\n---\n\n".join(excerpts)

    def _extract_citations(self, nodes) -> list[Citation]:
        """Extract citation metadata from already-retrieved nodes."""
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

            try:
                page_num = int(page) if page else None
            except (ValueError, TypeError):
                page_num = None

            citations.append(Citation(
                filename=filename,
                page_number=page_num,
                relevance_score=round(score, 3),
                text_snippet=snippet,
                confidence_label=confidence_label,
            ))

        # Return citations sorted by relevance so the most pertinent sources appear first
        return sorted(citations, key=lambda c: c.relevance_score, reverse=True)

    def query_documents(self, question: str) -> RAGResponse:
        """Retrieve relevant chunks, build a clean prompt, and call the LLM directly.

        Bypasses LlamaIndex's response synthesizer entirely to guarantee no raw
        chunk text or metadata leaks into the answer.
        """
        nodes = self.retriever.retrieve(question)
        context = self._build_context(nodes)
        citations = self._extract_citations(nodes)

        # Use chat() with explicit system/user roles so the LLM outputs only
        # its answer — complete() can blend prompt text into the response
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are a helpful healthcare information assistant. "
                    "Answer questions using only the provided document excerpts. "
                    "Write in 2-4 clear sentences. Do not repeat yourself. "
                    "Do not quote raw data verbatim. Do not add follow-up questions. "
                    "If the excerpts lack sufficient information, say so briefly."
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Document excerpts:\n{context}\n\nQuestion: {question}",
            ),
        ]
        response = self.llm.chat(messages, max_tokens=1024)

        answer = response.message.content.strip()
        # Remove any leading sentence fragment the LLM may echo from the context
        if answer and answer[0].islower():
            boundary = re.search(r'[.!?]\s+[A-Z]', answer)
            if boundary:
                answer = answer[boundary.start() + 2:]

        return RAGResponse(
            answer=answer,
            citations=citations,
            query_type="document_search",
            used_document_search=True,
        )

    def query_general(self, question: str) -> RAGResponse:
        """Answer from general knowledge using conversation memory.

        Routes through SimpleChatEngine which maintains memory but does
        not perform document retrieval.
        """
        response = self.chat_engine.chat(question)

        return RAGResponse(
            answer=response.response,
            citations=[],  # No citations for general knowledge responses
            query_type="general_knowledge",
            used_document_search=False,
        )

    def reset_memory(self):
        """Clear conversation history. Called when user starts a new chat session."""
        self.memory.reset()
