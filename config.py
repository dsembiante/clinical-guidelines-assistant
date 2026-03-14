# config.py
import os

from dotenv import load_dotenv

# Load environment variables from .env for local development.
# On Streamlit Cloud, these are set as secrets instead.
load_dotenv()

# Model settings
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq-hosted Llama 3.3 70B — better synthesis, still free
EMBED_MODEL = "BAAI/bge-small-en-v1.5" # HuggingFace embedding model (free, no API key)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Document settings
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "healthcare_docs"

# RAG settings
CHUNK_SIZE = 256   # Kept small to stay within embedding model context limits
CHUNK_OVERLAP = 25  # Scaled proportionally with chunk size
TOP_K_RESULTS = 3  # Balanced — enough context for thorough answers without repetition loops
MEMORY_TOKEN_LIMIT = 3000

# Confidence threshold
# Scores above this are shown as high confidence
HIGH_CONFIDENCE_THRESHOLD = 0.75
