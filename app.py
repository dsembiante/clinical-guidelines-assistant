# app.py
# Streamlit UI entry point. Run with: streamlit run app.py
# Displays the chat interface, citation cards with confidence scores,
# query routing decisions, and session management controls.
import streamlit as st

from rag_engine import RAGEngine
from router import QueryRouter, QueryType

# --- Auto-Ingest on First Deploy ---
# chroma_db/ is excluded from GitHub, so the vector store won't exist on a fresh deploy.
# This block automatically runs ingestion the first time the app launches in that environment.
import os
from config import CHROMA_DIR

if not os.path.exists(CHROMA_DIR):
    st.info("Building document index for first time. This may take a few minutes...")
    from ingest import ingest_documents
    with st.spinner("Ingesting healthcare documents..."):
        ingest_documents()
    st.success("Document index built successfully!")
    st.rerun()

# --- Page Configuration ---
# Must be the first Streamlit call in the script
st.set_page_config(
    page_title="Healthcare Document Assistant",
    page_icon="🏥",
    layout="wide",
)

# --- Custom CSS ---
# Citation cards use a colored left border to visually communicate confidence:
# green = high, yellow = medium, red = low
st.markdown("""<style>
.citation-card {
    background-color: #f8f9fa;
    border-left: 4px solid #2E5FA3;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 4px;
    font-size: 0.85em;
}
.high-confidence { border-left-color: #28a745; }
.medium-confidence { border-left-color: #ffc107; }
.low-confidence { border-left-color: #dc3545; }
</style>""", unsafe_allow_html=True)


# --- Session State Initialization ---
# Streamlit reruns the entire script on each interaction, so RAGEngine and
# QueryRouter are stored in session_state to avoid re-initializing on every rerun.
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if "router" not in st.session_state:
    st.session_state.router = QueryRouter()
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Header ---
st.title("🏥 Healthcare Document Assistant")
st.markdown(
    "Ask questions about CDC, NIH, and CMS healthcare guidelines. "
    "All answers include source citations with confidence scores."
)


# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown(
        "This assistant answers questions using official public healthcare "
        "documents from the CDC, NIH, and CMS. It cites sources for every "
        "answer so you can verify the information directly."
    )
    st.divider()

    st.header("Settings")
    # show_routing surfaces whether the query hit the vector store or general knowledge
    show_routing = st.toggle("Show query routing decisions", value=True)
    # show_snippets is off by default to keep the UI clean; power users can enable it
    show_snippets = st.toggle("Show source text snippets", value=False)
    st.divider()

    # Clears both the UI message history and the LLM's conversation memory buffer
    if st.button("Start New Conversation", type="primary"):
        st.session_state.messages = []
        st.session_state.rag_engine.reset_memory()
        st.rerun()


# --- Citation Renderer ---
def _render_citations(citations, show_snippets):
    """Render citation cards beneath an assistant message.

    Defined before the chat history loop so it can be called both when
    replaying history and when rendering a fresh response.
    """
    if not citations:
        return

    with st.expander(f"Sources ({len(citations)} documents referenced)"):
        for i, cit in enumerate(citations, 1):
            confidence_class = cit.confidence_label.lower() + "-confidence"
            page_str = f"Page {cit.page_number}" if cit.page_number else "Page unknown"

            st.markdown(
                f'<div class="citation-card {confidence_class}">'
                f"<strong>{i}. {cit.filename}</strong> — {page_str}<br/>"
                f"Relevance: <strong>{cit.confidence_label}</strong> ({cit.relevance_score})"
                f"</div>",
                unsafe_allow_html=True,
            )

            if show_snippets:
                st.caption(f'"{cit.text_snippet}"')


# --- Chat History Replay ---
# Re-render all prior messages so the conversation is visible on each rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-attach citation cards to assistant messages that had them
        if message["role"] == "assistant" and message.get("citations"):
            _render_citations(message["citations"], show_snippets)


# --- Chat Input ---
if prompt := st.chat_input("Ask a question about healthcare guidelines..."):
    # Display and store the user's message immediately for responsiveness
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        # Route the query to decide whether to search documents or use general knowledge
        routing = st.session_state.router.route_with_explanation(prompt)

        if show_routing:
            route_icon = "📚" if routing["type"] == QueryType.DOCUMENT_SEARCH else "🧠"
            st.caption(f"{route_icon} {routing['explanation']}")

        # Dispatch to the appropriate RAG engine method based on routing decision
        if routing["type"] == QueryType.DOCUMENT_SEARCH:
            response = st.session_state.rag_engine.query_documents(prompt)
        else:
            response = st.session_state.rag_engine.query_general(prompt)

    # Render the assistant's answer and any citations
    with st.chat_message("assistant"):
        st.markdown(response.answer)
        if response.citations:
            _render_citations(response.citations, show_snippets)

    # Persist the response to session history so it survives the next rerun
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "citations": response.citations,
    })
