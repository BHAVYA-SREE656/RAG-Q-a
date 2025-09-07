import os
import io
import requests
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple
from pypdf import PdfReader
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument
import html2text as html2text_lib
import markdown as md_lib
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken
from openai import OpenAI

load_dotenv()

# ---------------- CONFIG ----------------
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

st.set_page_config(page_title="RAG Q&A", page_icon="🧠", layout="wide")

# ---------------- HELPERS ----------------
def read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def read_markdown(file: io.BytesIO) -> str:
    content = file.read().decode("utf-8", errors="ignore")
    html = md_lib.markdown(content)
    text = html2text_lib.HTML2Text().handle(html)
    return text

def read_html(file: io.BytesIO) -> str:
    content = file.read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

def fetch_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        doc = ReadabilityDocument(resp.text)
        readable_html = doc.summary()
        soup = BeautifulSoup(readable_html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return text
    except Exception as e:
        return f"[ERROR fetching URL]: {e}"

def tokenizer_len(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def chunk_text(text: str, chunk_size_tokens: int = 400, chunk_overlap: int = 60) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if tokenizer_len(current + "\n" + p) <= chunk_size_tokens:
            current = (current + "\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedder(name: str = DEFAULT_EMBED_MODEL):
    return SentenceTransformer(name)

def build_faiss_index(chunks: List[str], embedder) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    if not chunks:
        raise ValueError("No chunks to index.")
    vectors = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index, vectors

# ---------------- STREAMLIT UI ----------------
st.title("🧠 RAG Q&A — Talk to Your Documents")
st.caption("Upload PDFs / Markdown / HTML or paste a URL. Ask questions and get answers with citations.")

uploaded_files = st.file_uploader(
    "Upload documents (PDF, MD, HTML). You can upload multiple.", 
    type=["pdf", "md", "markdown", "html", "htm"], 
    accept_multiple_files=True
)

url_input = st.text_input("Or paste a URL to ingest")
ingest_btn = st.button("➕ Ingest to Knowledge Base")

# Session state
if "docs" not in st.session_state:
    st.session_state.docs = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "embedder" not in st.session_state:
    st.session_state.embedder = None

# ---------------- INGESTION ----------------
if ingest_btn:
    texts = []

    # Files
    for f in uploaded_files or []:
        file_name = f.name
        suffix = os.path.splitext(file_name)[1].lower()  # ✅ safe
        if suffix == ".pdf":
            text = read_pdf(f)
        elif suffix in [".md", ".markdown"]:
            text = read_markdown(f)
        elif suffix in [".html", ".htm"]:
            text = read_html(f)
        else:
            text = ""
        if text.strip():
            st.session_state.docs.append({"source": file_name, "text": text})
            texts.append((file_name, text))

    # URL
    if url_input.strip():
        t = fetch_url(url_input.strip())
        if not t.startswith("[ERROR"):
            st.session_state.docs.append({"source": url_input.strip(), "text": t})
            texts.append((url_input.strip(), t))
        else:
            st.error(t)

    if texts:
        new_chunks = []
        for source, text in texts:
            chunks = chunk_text(text)
            for c in chunks:
                new_chunks.append({"source": source, "text": c})
        st.session_state.chunks.extend(new_chunks)

        st.session_state.embedder = load_embedder()
        embedder = st.session_state.embedder
        chunks_text = [c["text"] for c in st.session_state.chunks]
        st.session_state.index, _ = build_faiss_index(chunks_text, embedder)
        st.success(f"Ingested {len(new_chunks)} new chunks. Total chunks: {len(st.session_state.chunks)}")
    else:
        st.info("No valid content ingested yet.")

# ---------------- QUESTION INPUT ----------------
st.markdown("---")
st.subheader("Ask a question")

question = st.text_input("Your question here")
ask_btn = st.button("🔎 Retrieve & Answer")

if ask_btn and question.strip():
    if not st.session_state.index or not st.session_state.chunks:
        st.warning("Please ingest some content first.")
    else:
        embedder = st.session_state.embedder
        qvec = embedder.encode([question], normalize_embeddings=True)
        D, I = st.session_state.index.search(qvec, 5)  # top 5
        retrieved = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = st.session_state.chunks[idx]
            retrieved.append((idx, meta["text"], float(score)))

        # Show retrieved chunks
        with st.expander("Retrieved chunks & scores"):
            for idx, txt, score in retrieved:
                st.markdown(f"**Chunk #{idx}** — Score: `{score:.3f}` — Source: `{st.session_state.chunks[idx]['source']}`")
                st.write(txt[:500] + ("..." if len(txt) > 500 else ""))

        # Simple extractive answer
        answer = retrieved[0][1][:1000] if retrieved else "No relevant content found."
        st.markdown("### Answer")
        st.write(answer)

