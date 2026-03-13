# Healthcare Clinical Guidelines Assistant

A RAG-powered chat application that answers questions about CDC, NIH, and CMS healthcare guidelines — with source citations, confidence scores, and conversation memory.

## Live Demo

**[Launch App](https://YOUR-APP-NAME.streamlit.app)** ← *(update this link after deploying to Streamlit Community Cloud)*

---

## Screenshot

*(Take a screenshot of the app after asking "What are the CDC recommendations for diabetes screening intervals?" with the Sources expander open showing at least two citations, then add the image here)*

---

## Background and Motivation

During my time as an automation engineer at Mercy Hospital, I worked closely with clinical workflows and saw firsthand how much time staff spent manually searching through CDC, NIH, and CMS guideline documents to answer protocol questions. The information exists — it's just buried across dozens of PDFs. I built this tool to make those guidelines instantly queryable, with every answer traceable back to its source document so clinicians and staff can verify what they're reading.

---

## Key Features

### Conversation Memory
The assistant remembers what was said earlier in the session. If you ask "What are the CDC screening intervals for diabetes?" and then follow up with "What about for adults over 65?", it understands the context without you having to repeat yourself. This is implemented using LlamaIndex's `ChatMemoryBuffer` with a token limit to prevent the context window from growing unbounded.

### Source Citations with Confidence Scores
Every answer backed by a document search includes the source filename, page number, and a relevance score labeled High, Medium, or Low. This makes the tool auditable — users can open the original PDF and verify the guideline directly rather than trusting the model's output blindly.

### Intelligent Query Routing
Not every question requires a document search. A dedicated router classifies each query before it reaches the RAG engine, directing factual guideline questions to the vector store and general medical knowledge questions straight to the LLM. This prevents unnecessary retrieval calls and improves response quality for both query types.

---

## Sample Interaction

**User:** What are the CDC recommendations for adult diabetes screening?

**Routing decision:** 📚 Searching healthcare documents for specific guidelines or recommendations

**Assistant:** According to the CDC guidelines, adults aged 35–70 who are overweight or obese should be screened for prediabetes and type 2 diabetes. Screening should be repeated every 3 years if results are normal...

**Sources (3 documents referenced)**
- `cdc_148231_DS1.pdf` — Page 4 | Relevance: **High** (0.87)
- `2023-adult.pdf` — Page 12 | Relevance: **High** (0.81)
- `p776.pdf` — Page 2 | Relevance: **Medium** (0.63)

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM | Llama 3 via Ollama | Generates answers locally — no API key or data sent to the cloud |
| Embeddings | nomic-embed-text via Ollama | Converts document chunks into searchable vectors |
| Vector Store | ChromaDB | Persists and retrieves document embeddings |
| RAG Framework | LlamaIndex | Orchestrates retrieval, memory, and response synthesis |
| UI | Streamlit | Chat interface with citation cards and session controls |
| Query Router | Custom LlamaIndex + Ollama | Classifies queries before retrieval to improve response quality |

---

## Local Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### 1. Pull the required models
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Clone the repo and install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/clinical-guidelines-assistant.git
cd clinical-guidelines-assistant
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 3. Add your PDF documents
Place PDF files in the `docs/` folder.

### 4. Run ingestion
```bash
python ingest.py
```

### 5. Launch the app
```bash
streamlit run app.py
```

---

## What I Would Add Next

- **Authentication** — restrict access to verified clinical staff
- **Multi-collection support** — separate indexes for CDC, NIH, and CMS so users can filter by source agency
- **Feedback mechanism** — thumbs up/down on answers to track accuracy over time
- **Async ingestion** — background re-ingestion when new documents are added without restarting the app
- **Hybrid search** — combine vector similarity with keyword (BM25) search for better recall on acronym-heavy medical queries
