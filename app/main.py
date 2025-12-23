from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.embedder import TextEmbedder
from app.matcher import SemanticMatcher
from app.parser import chunk_text
import uvicorn
import os

app = FastAPI(title="JD-Resume Alignment Analyzer")

# Global instances
embedder = None
matcher = None

@app.on_event("startup")
def startup_event():
    global embedder, matcher
    embedder = TextEmbedder()
    matcher = SemanticMatcher()

class AnalyzeRequest(BaseModel):
    jd_text: str
    resume_text: str

class AnalyzeResponse(BaseModel):
    overall_score: float
    top_matches: list
    weak_areas: list

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    if not request.jd_text.strip() or not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Both JD and Resume text must be provided.")

    # 1. Chunking
    jd_chunks = chunk_text(request.jd_text)
    resume_chunks = chunk_text(request.resume_text)

    if not jd_chunks:
        raise HTTPException(status_code=400, detail="Job Description contains no valid text chunks.")
    if not resume_chunks:
        raise HTTPException(status_code=400, detail="Resume contains no valid text chunks.")

    # 2. Embedding
    jd_embeddings = embedder.embed(jd_chunks)
    resume_embeddings = embedder.embed(resume_chunks)

    # 3. Matching
    result = matcher.align(jd_chunks, resume_chunks, jd_embeddings, resume_embeddings)

    return AnalyzeResponse(
        overall_score=result["overall_score"],
        top_matches=result["top_matches"],
        weak_areas=result["weak_areas"]
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
