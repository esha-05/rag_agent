# 📄 Challenge B — AI PDF Summarization & QA Agent (RAG System)

An AI-powered Retrieval-Augmented Generation (RAG) system that reads a PDF, indexes it with vector embeddings, and answers user questions using only the document's content. Powered by **Groq (Llama 3.3 70B)**, **FAISS**, and **Sentence Transformers**.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    INDEXING PHASE (once per PDF)             │
│                                                              │
│  PDF Upload                                                  │
│      │                                                       │
│      ▼                                                       │
│  PyMuPDF (fitz)          ← Extract raw text from all pages  │
│      │                                                       │
│      ▼                                                       │
│  Text Chunker             ← Split into 500-word overlapping  │
│  (chunk_size=500,            chunks (50-word overlap)        │
│   overlap=50)                                                │
│      │                                                       │
│      ▼                                                       │
│  SentenceTransformer      ← Convert each chunk to a         │
│  (all-MiniLM-L6-v2)          384-dimensional vector         │
│      │                                                       │
│      ▼                                                       │
│  FAISS Index              ← Store all vectors locally        │
│  (IndexFlatL2)               for fast similarity search      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    QUERY PHASE (per question)                │
│                                                              │
│  User Question                                               │
│      │                                                       │
│      ▼                                                       │
│  SentenceTransformer      ← Embed the question              │
│      │                                                       │
│      ▼                                                       │
│  FAISS Search             ← Find top 4 most similar chunks  │
│      │                                                       │
│      ▼                                                       │
│  Groq LLM                 ← Answer ONLY using retrieved      │
│  (Llama 3.3 70B)             chunks as context              │
│      │                                                       │
│      ▼                                                       │
│  { answer, chunks_used, filename }                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
ai-pdf-rag-agent/
├── .env                  ← API keys (never commit this)
├── .gitignore
├── requirements.txt      ← Python dependencies
├── rag_agent.py          ← Core RAG pipeline
│                            (PDF parse → chunk → embed → FAISS → Groq)
├── main_b.py             ← FastAPI server
└── index.html            ← Sidebar-style chat UI
```

---

## ⚙️ Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | latest | Web server framework |
| `uvicorn` | latest | ASGI server |
| `groq` | latest | Groq LLM client (Llama 3.3 70B) |
| `pymupdf` | latest | PDF text extraction (imported as `fitz`) |
| `sentence-transformers` | latest | Local embedding model (all-MiniLM-L6-v2) |
| `faiss-cpu` | latest | Facebook's vector similarity search library |
| `numpy` | latest | Array operations for FAISS |
| `python-multipart` | latest | Handles file uploads in FastAPI |
| `python-dotenv` | latest | Load `.env` API keys |

---

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A free [Groq API key](https://console.groq.com) — create account → API Keys
- No other paid services required — embeddings and vector search run **locally**

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-pdf-rag-agent.git
cd ai-pdf-rag-agent
```

### 3. Create a virtual environment

```bash
python -m venv venv

# Windows (Git Bash)
source venv/Scripts/activate

# Mac / Linux
source venv/bin/activate
```

### 4. Install dependencies

> ⚠️ First run downloads the `all-MiniLM-L6-v2` embedding model (~80MB). This is cached after that.

```bash
python -m pip install -r requirements.txt
```

### 5. Configure API keys

Create a `.env` file in the root folder:

```
GROQ_API_KEY=gsk_your_groq_key_here
```

### 6. Run the server

```bash
python -m uvicorn main_b:app --reload --port 8001
```

### 7. Open the UI

Visit **http://127.0.0.1:8001** in your browser.

---

## 🖥️ How to Run

```bash
# Start server
python -m uvicorn main_b:app --reload --port 8001

# Upload a PDF
curl -X POST http://localhost:8001/upload \
  -F "file=@your_document.pdf"

# Ask a question
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology was used in the study?"}'

# Get a summary
curl -X POST http://localhost:8001/summarize
```

---

## 📥 Example Input / Output

**Step 1 — Upload PDF:**
```
POST /upload  →  { "message": "PDF processed", "chunks": 42, "filename": "report.pdf" }
```

**Step 2 — Ask a question:**

Input:
```
What methodology was used in the study?
```

Output:
```json
{
  "answer": "The study used a combination of case studies and experimental evaluations across three enterprise environments...",
  "chunks_used": 4,
  "filename": "report.pdf"
}
```

**Step 3 — Summarize:**
```json
{
  "answer": "The document discusses AI-driven automation in enterprise workflows and evaluates productivity improvements...",
  "chunks_used": 6,
  "filename": "report.pdf"
}
```

---

## 🧠 Design Decisions & Trade-offs

| Decision | Reason | Trade-off |
|---|---|---|
| **FAISS over Pinecone/Chroma** | Runs fully locally, zero cost, no cloud setup | Resets when server restarts (no persistence) |
| **all-MiniLM-L6-v2 embeddings** | Local, free, fast, ~80MB, trained for semantic similarity | Slightly less accurate than OpenAI embeddings |
| **Chunk size 500 words / 50 overlap** | Fits within Groq context window; overlap prevents edge-sentence loss | Larger chunks = more context but more tokens consumed |
| **top_k=4 chunks** | ~2000 words of relevant context — enough for accuracy without overloading LLM | May miss info if relevant content spans many sections |
| **Groq over Anthropic** | Free tier with fast inference | 100K token/day rate limit; need to switch to `llama-3.1-8b-instant` if limit hit |
| **Summarize via evenly-spaced chunks** | Covers beginning, middle, and end of document | Not as deep as full-document summarization |
| **In-memory FAISS index** | Simple, zero infrastructure | One PDF at a time; restarts lose index |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `POST` | `/upload` | Upload & index a PDF file |
| `POST` | `/ask` | Ask a question about the PDF |
| `POST` | `/summarize` | Get a document summary |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

## 💡 RAG Explained Simply

> Instead of feeding the entire PDF to the LLM (impossible for long docs), we convert every chunk of text into a mathematical vector (embedding). When you ask a question, we convert it to the same kind of vector, then find the 4 chunks whose vectors are closest in meaning. Only those 4 chunks go to the LLM — so it answers precisely from the right section of the document.

---

## ⚠️ Known Limitations

- Only one PDF is held in memory at a time — uploading a new PDF replaces the previous index
- FAISS index is not persisted to disk; server restart requires re-uploading the PDF
- Groq free tier has a 100K token/day limit — switch model to `llama-3.1-8b-instant` if rate-limited

---

## 🔒 .gitignore

```
.env
venv/
__pycache__/
*.pyc
*.pdf
```
