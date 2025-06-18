"""
Vector store implementation using FAISS.
"""

import os
from typing import List, Dict
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_pipeline.config import FAISS_INDEX_PATH

class VectorStore:
    """FAISS vector store for document embeddings and retrieval."""
    
    def __init__(self):
        """Initialize the vector store with a sentence transformer embedding model."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.index = None
    
    def create_index(self, documents: List[Document]) -> None:
        """Create a FAISS index from the provided documents."""
        self.index = FAISS.from_documents(documents, self.embeddings)
        print(f"Created FAISS index with {len(documents)} documents")
    
    def save_index(self, path: str = FAISS_INDEX_PATH) -> None:
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("No index to save. Create an index first.")
        
        self.index.save_local(path)
        print(f"Saved FAISS index to {path}")
    
    def load_index(self, path: str = FAISS_INDEX_PATH) -> bool:
        """Load the FAISS index from disk."""
        if not os.path.exists(path):
            print(f"No index found at {path}")
            return False
        
        try:
            self.index = FAISS.load_local(path, self.embeddings)
            print(f"Loaded FAISS index from {path}")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search(self, query: str, k: int = 10) -> List[Document]:
        """Search the index for documents similar to the query."""
        if self.index is None:
            raise ValueError("No index to search. Create or load an index first.")
        
        documents = self.index.similarity_search(query, k=k)
        return documents
    
    def search_for_multiple_queries(self, queries: List[str], k: int = 10) -> Dict[str, List[Document]]:
        """Search the index for multiple queries and return results for each."""
        results = {}
        for query in queries:
            results[query] = self.search(query, k=k)
        return results
