import faiss
import numpy as np
from typing import List, Dict, Any

class SemanticMatcher:
    def __init__(self):
        self.index = None
        self.dimension = 0

    def build_index(self, embeddings: np.ndarray):
        """Builds a FAISS index from embeddings."""
        if embeddings.size == 0:
            return
        
        self.dimension = embeddings.shape[1]
        # IndexFlatIP uses Inner Product. 
        # If vectors are normalized, this is equivalent to Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query_embeddings: np.ndarray, top_k: int = 1):
        """
        Search the index for query embeddings.
        Returns distances (similarity scores) and indices.
        """
        if self.index is None or query_embeddings.size == 0:
            return None, None
            
        # Normalize query vectors
        faiss.normalize_L2(query_embeddings)
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices

    def align(self, jd_chunks: List[str], resume_chunks: List[str], jd_embeddings: np.ndarray, resume_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Computes alignment between JD and Resume.
        """
        if not jd_chunks or not resume_chunks:
            return {
                "overall_score": 0.0,
                "top_matches": [],
                "weak_areas": jd_chunks
            }

        # Build index on Resume chunks (we want to find if Resume covers JD requirements)
        self.build_index(resume_embeddings)
        
        # Search for each JD chunk in the Resume index
        # We want to know: "For this JD requirement, what is the best thing in the resume?"
        distances, indices = self.search(jd_embeddings, top_k=1)
        
        matches = []
        missing = []
        total_score = 0.0
        
        # Threshold for "missing" / "weak"
        # 0.4 is a reasonable threshold for MiniLM cosine similarity to say "somewhat related"
        THRESHOLD = 0.4 
        
        for i, (dist_arr, idx_arr) in enumerate(zip(distances, indices)):
            score = float(dist_arr[0])
            jd_text = jd_chunks[i]
            match_idx = idx_arr[0]
            
            if match_idx < 0 or match_idx >= len(resume_chunks):
                continue
                
            resume_text = resume_chunks[match_idx]
            
            total_score += score
            
            match_data = {
                "jd_chunk": jd_text,
                "resume_match": resume_text,
                "similarity": round(score, 3)
            }
            
            if score < THRESHOLD:
                missing.append(match_data)
            else:
                matches.append(match_data)
                
        # Calculate overall score
        # We average the best-match scores for all JD chunks.
        # If JD has 10 chunks, and resume matches 5 perfectly (1.0) and 5 not at all (0.0), score is 50.
        avg_similarity = total_score / len(jd_chunks) if jd_chunks else 0
        
        # Scale to 0-100
        final_score = max(0, min(100, avg_similarity * 100))
        
        # Sort matches by similarity desc
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        # Sort missing by similarity asc (worst matches first)
        missing.sort(key=lambda x: x['similarity'])
        
        return {
            "overall_score": round(final_score, 2),
            "top_matches": matches[:5],
            "weak_areas": missing
        }
