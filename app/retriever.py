import faiss
import pickle
import numpy as np


class Retriever:

    def __init__(self, index_path: str, doc_path: str, top_k: int) -> None:
        self.index = faiss.read_index(index_path)
        with open(doc_path, "rb") as f:
            self.doc_chunks = pickle.load(f)
        self.top_k = top_k
    
    def retrieve(self, query_vec: np.ndarray):
        distances, indices = self.index.search(query_vec, self.top_k)
        return [self.doc_chunks[i] for i in indices[0]]
    
        
