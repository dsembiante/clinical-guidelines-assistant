# router.py
# Classifies each incoming query before it reaches the RAG engine.
# Determines whether to search the vector store or answer from general LLM knowledge,
# preventing unnecessary document lookups and improving response quality.
from enum import Enum

from llama_index.llms.groq import Groq

from config import GROQ_API_KEY, LLM_MODEL


class QueryType(str, Enum):
    DOCUMENT_SEARCH = "document_search"
    GENERAL_KNOWLEDGE = "general_knowledge"


class QueryRouter:
    """
    Routes incoming queries to either:
    - document_search: search the healthcare PDF vector store
    - general_knowledge: answer directly from LLM without retrieval
    """

    # Prompt instructs the LLM to act as a strict classifier.
    # Temperature is set to 0 to ensure deterministic, consistent routing decisions.
    ROUTER_PROMPT = """You are a query classification system.
    Classify the following question into exactly one of two categories:

    DOCUMENT_SEARCH: The question asks about specific guidelines, policies,
    recommendations, statistics, or content that would be found in official
    healthcare documents (CDC guidelines, NIH recommendations, CMS policies,
    clinical protocols, screening intervals, dosage recommendations, etc.)

    GENERAL_KNOWLEDGE: The question asks for a general definition, explanation
    of a medical concept, or information that a doctor would know from training
    rather than needing to look up in a specific guideline document.

    Question: {question}

    Respond with ONLY one word: either DOCUMENT_SEARCH or GENERAL_KNOWLEDGE."""

    def __init__(self):
        # temperature=0 makes routing deterministic — same query always gets same classification
        self.llm = Groq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
        )

    def route(self, query: str) -> QueryType:
        """Classify a query and return the appropriate QueryType."""
        prompt = self.ROUTER_PROMPT.format(question=query)
        response = self.llm.complete(prompt)
        result = response.text.strip().upper()

        # Use substring check rather than exact match to handle minor LLM formatting variations
        if "DOCUMENT" in result:
            return QueryType.DOCUMENT_SEARCH
        return QueryType.GENERAL_KNOWLEDGE

    def route_with_explanation(self, query: str) -> dict:
        """Route a query and return both the type and a brief explanation.
        Used by the Streamlit UI to show routing decisions."""
        query_type = self.route(query)

        # Human-readable explanation surfaces the routing decision in the UI
        explanation = (
            "Searching healthcare documents for specific guidelines or recommendations"
            if query_type == QueryType.DOCUMENT_SEARCH
            else "Answering from general medical knowledge"
        )
        return {
            "type": query_type,
            "explanation": explanation,
        }
