import re
from typing import List

def clean_chunk(text: str) -> str:
    """Cleans whitespace within a chunk."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str) -> List[str]:
    """
    Splits text into paragraphs or chunks.
    """
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split by newlines
    raw_chunks = [p for p in text.split('\n')]
    
    # Clean and filter
    cleaned_chunks = [clean_chunk(p) for p in raw_chunks]
    # Filter out extremely short chunks (e.g. < 5 chars) to improve embedding quality
    return [p for p in cleaned_chunks if len(p) >= 5]
