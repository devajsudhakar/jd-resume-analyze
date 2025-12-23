# JD-Resume Alignment Analyzer

A simple, honest, and interview-defensible tool to analyze the alignment between a Job Description (JD) and a Resume using semantic search.

## Features

- **Semantic Analysis**: Uses `sentence-transformers` (all-MiniLM-L6-v2) to encode text into vector embeddings.
- **Similarity Search**: Uses `FAISS` to compute cosine similarity between JD and Resume chunks.
- **Scoring**: Calculates an alignment score (0-100) based on how well the Resume covers JD requirements.
- **Insights**: Returns specific matches and identifies weak areas where the Resume lacks coverage.
- **API-First**: Provides a clean FastAPI interface for integration.

## How the Alignment Score Works

The alignment score is not a simple keyword match. It works as follows:
1. **Chunking**: Both the Job Description (JD) and Resume are split into logical paragraphs or chunks.
2. **Embedding**: Each chunk is converted into a dense vector representation using a pre-trained language model.
3. **Matching**: For every chunk in the JD, we find the most semantically similar chunk in the Resume.
4. **Scoring**: The overall score is the average of the similarity scores of these best matches. This means if a JD has 10 requirements and the Resume matches 5 perfectly but misses 5, the score will reflect that partial coverage.


## Project Structure

```
jd_resume_analyzer/
│
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── embedder.py      # Wrapper for Sentence Transformers
│   ├── matcher.py       # Logic for FAISS indexing and scoring
│   ├── parser.py        # Text cleaning and chunking
│
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── README.md            # This file
```

## How to Run Locally

### Prerequisites
- Python 3.10+
- pip

### Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   uvicorn app.main:app --reload
   ```
   Or simply:
   ```bash
   python -m app.main
   ```

3. **Access the API**
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Endpoint: `POST /analyze`

### Example Request

```json
POST /analyze
{
  "jd_text": "We are looking for a Python developer with experience in FastAPI and Machine Learning.",
  "resume_text": "Experienced Software Engineer with strong skills in Python, FastAPI, and building ML models using PyTorch."
}
```

## How to Run with Docker

1. **Build the Image**
   ```bash
   docker build -t jd-resume-analyzer .
   ```

2. **Run the Container**
   ```bash
   docker run -p 8000:8000 jd-resume-analyzer
   ```

3. **Access the API**
   The API will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Tech Stack

- **Framework**: FastAPI
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Search**: faiss-cpu
- **Server**: uvicorn
