# main_b.py
# FastAPI server for PDF RAG agent
# Run: uvicorn main_b:app --reload --port 8001

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_agent import process_pdf, answer_question, summarize_pdf

app = FastAPI(title="AI PDF RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes 

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")


    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        num_chunks = process_pdf(tmp_path, file.filename)
    finally:
        os.unlink(tmp_path)   # delete temp file

    return {
        "message": "PDF processed successfully",
        "filename": file.filename,
        "chunks": num_chunks
    }


class QuestionRequest(BaseModel):
    question: str

@app.post("/summarize")
async def summarize():
    try:
        return summarize_pdf()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Groq rate limit reached. Wait a few minutes and try again.")

@app.post("/ask")
async def ask(req: QuestionRequest):
    try:
        return answer_question(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Groq rate limit reached. Wait a few minutes and try again.")


@app.get("/health")
async def health():
    return {"status": "running", "port": 8001}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h2>index.html not found at: " + html_path + "</h2>")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())