import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, dimension: int = 1536, index_file: str = "faiss_index.bin", metadata_file: str = "metadata.pkl"):
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.metadata = []
        
        # Initialize or load existing index
        self.load_or_create_index()
    
    def load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.load_index()
        else:
            self.create_new_index()
    
    def create_new_index(self):
        """Create a new FAISS index"""
        # Using IndexFlatIP for cosine similarity (inner product)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("Created new FAISS index")
    
    def add_vectors(self, embeddings: List[List[float]], metadata: List[Dict]):
        """Add vectors and metadata to the index"""
        if not embeddings or not metadata:
            logger.warning("No embeddings or metadata provided")
            return
        
        # Convert to numpy array and normalize for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Normalize query vector
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                results.append((self.metadata[idx], float(score)))
        
        return results
    
    def save_index(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved index with {self.index.ntotal} vectors")
    
    def load_index(self):
        """Load index and metadata from disk"""
        self.index = faiss.read_index(self.index_file)
        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata)
        }
