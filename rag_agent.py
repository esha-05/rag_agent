
import os
import fitz                          
import numpy as np
import faiss                        
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
print("⏳ Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq()

# In-memory storage
faiss_index = None
chunks = []
pdf_filename = ""



# 1: INDEXING


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    page_count = len(doc)          
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    doc.close()
    print(f"📄 Extracted {len(full_text)} characters from {page_count} pages")  
    return full_text


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 30) -> list:
    """
    Split text into overlapping word chunks.
    chunk_size=500 words ≈ 375 tokens (safe for Groq context)
    overlap=50 words — prevents losing sentences at chunk boundaries
    """
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        result.append(chunk)
        i += chunk_size - overlap
    print(f"✂️  Created {len(result)} chunks")
    return result


def build_faiss_index(text_chunks: list):
    """
    Convert chunks → 384-dim vectors → store in FAISS.
    FAISS IndexFlatL2 = exact nearest neighbour search by Euclidean distance.
    """
    global faiss_index, chunks
    chunks = text_chunks

    print("🔢 Generating embeddings...")
    embeddings = embedder.encode(text_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]              # 384
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    print(f"FAISS index built — {faiss_index.ntotal} vectors stored")


def process_pdf(pdf_path: str, filename: str) -> int:
    global pdf_filename
    pdf_filename = filename
    text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text(text)
    build_faiss_index(text_chunks)
    return len(text_chunks)



# 2: QUERY

def retrieve_chunks(question: str, top_k: int = 2) -> list:
    """
    Embed the question → search FAISS → return top_k matching chunks.
    This is the Retrieval part of RAG.
    """
    if faiss_index is None:
        raise ValueError("No PDF indexed yet. Please upload a PDF first.")

    q_vec = embedder.encode([question])
    q_vec = np.array(q_vec, dtype="float32")

    D, I = faiss_index.search(q_vec, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]


def answer_question(question: str) -> dict:
    """
    RAG answer: retrieve relevant chunks → build prompt → Groq generates answer.
    Groq ONLY sees the retrieved chunks, not the entire PDF.
    This prevents hallucination and keeps context focused.
    """
    relevant = retrieve_chunks(question)
    context = "\n\n---\n\n".join([
        f"Excerpt {i+1}:\n{chunk}"
        for i, chunk in enumerate(relevant)
    ])

    system_prompt = """You are an expert document analyst.
Answer questions using ONLY the provided document excerpts.
If the answer is not in the excerpts, say "This information is not available in the document."
Be precise, cite specific details, and keep answers concise."""

    user_msg = f"""Question: {question}

Relevant excerpts from the document:

{context}

Answer based on the excerpts above:"""

    print(f"🤖 Sending to Groq: '{question}'")
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg}
        ]
    )

    answer = response.choices[0].message.content
    return {
        "answer": answer,
        "chunks_used": len(relevant),
        "filename": pdf_filename
    }


def summarize_pdf() -> dict:
    """
    Summarize by sampling chunks spread across the whole document,
    so we cover beginning, middle, and end.
    """
    if not chunks:
        raise ValueError("No PDF indexed yet.")

    total = len(chunks)
    indices = [int(i * total / 3) for i in range(3)]
    sample = [chunks[i] for i in indices if i < total]
    context = "\n\n---\n\n".join([f"Excerpt {i+1}:\n{c}" for i, c in enumerate(sample)])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=800,
        messages=[
            {"role": "system", "content": "You are a document summarizer. Summarize the key points concisely."},
            {"role": "user",   "content": f"Summarize this document based on these excerpts:\n\n{context}"}
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "chunks_used": len(sample),
        "filename": pdf_filename
    }