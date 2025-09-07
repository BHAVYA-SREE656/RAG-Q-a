# 🧠 RAG Q&A: Talk to Your Documents

An end-to-end **Retrieval-Augmented Generation (RAG)** app built with **Streamlit + FAISS + SentenceTransformers**.
Upload PDFs, Notion exports (Markdown/HTML), or paste a wiki URL — then ask questions and get **accurate, contextual answers with citations**.

## ✨ Features
- Upload **PDF / Markdown / HTML** files
- Paste **URL** (uses readability to extract clean text)
- **Chunking** (configurable), **embeddings** (SentenceTransformers), **FAISS** vector store
- **RAG**: retrieve top-k chunks, generate final answer with **OpenAI** (optional)
- **Citations**: shows which chunks were used, with scores
- **Multi-document** support within a single session

## 🛠️ Tech
- **Parsing**: pypdf, BeautifulSoup, html2text, readability-lxml
- **Embeddings**: sentence-transformers (default: `all-MiniLM-L6-v2`)
- **Vector DB**: FAISS (in-memory)
- **LLM**: OpenAI (if `OPENAI_API_KEY` is set), else a simple extractive fallback

## 🚀 Quickstart

1) Clone and install:
```bash
pip install -r requirements.txt
```

2) (Optional) Add your OpenAI key
- Copy `.env.example` → `.env` and set `OPENAI_API_KEY`

3) Run:
```bash
streamlit run app.py
```

## 📁 Supported Inputs
- **PDFs**
- **Markdown** (`.md`)
- **HTML** (`.html`)
- **URLs** (public pages; no auth)

## 🧩 How it works
1. **Ingest**: Parse inputs → clean text
2. **Chunk**: Split into ~400-token chunks with overlap
3. **Embed**: SentenceTransformers → FAISS
4. **Retrieve**: cosine similarity top-k
5. **Generate**: Build prompt with retrieved context → LLM (OpenAI) → answer + citations

## 🧪 Notes
- Everything runs locally except the optional call to OpenAI.
- For very large PDFs, first try a subset due to memory.
- This is a teaching/reference implementation; for production consider persistence (e.g., saving FAISS + metadata), auth, and error handling.

## 📜 License
MIT
