from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # This will download the model if not present
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        # Convert to numpy array directly
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)
