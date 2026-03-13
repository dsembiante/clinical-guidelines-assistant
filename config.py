# config.py
import os

# Model settings
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Document settings
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "healthcare_docs"

# RAG settings
CHUNK_SIZE = 256   # Reduced from 512 to stay within nomic-embed-text's context window
CHUNK_OVERLAP = 25  # Scaled down proportionally with chunk size
TOP_K_RESULTS = 4
MEMORY_TOKEN_LIMIT = 3000

# Confidence threshold
# Scores above this are shown as high confidence
HIGH_CONFIDENCE_THRESHOLD = 0.75
