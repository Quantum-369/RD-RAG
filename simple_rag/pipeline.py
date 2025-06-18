"""
Main pipeline class that orchestrates the Simple RAG flow.
"""

import os
import shutil
from typing import Optional

from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_utils import LLMUtils
from .config import (
    DOCUMENTS_DIR, 
    FAISS_INDEX_PATH, 
    TOP_K
)

class SimpleRAGPipeline:
    """
    A simple RAG pipeline that integrates document processing,
    vector search, and response generation.
    """
    
    def __init__(self):
        """Initialize the Simple RAG pipeline components."""
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_utils = LLMUtils()
    
    def initialize(self, documents_dir: Optional[str] = None, reinitialize: bool = False) -> None:
        """
        Initialize the pipeline by processing documents and creating the vector index.
        
        Args:
            documents_dir: Directory containing documents to process
            reinitialize: Force reprocessing of documents
        """
        documents_dir = documents_dir or DOCUMENTS_DIR
        
        if reinitialize and os.path.exists(FAISS_INDEX_PATH):
            print(f"Reinitializing: removing existing index at {FAISS_INDEX_PATH}")
            shutil.rmtree(FAISS_INDEX_PATH)

        if not self.vector_store.load_index(FAISS_INDEX_PATH):
            print(f"Creating new index from all documents in {documents_dir}")
            documents = self.doc_processor.process_directory(documents_dir)
            
            if documents:
                self.vector_store.create_index(documents)
                self.vector_store.save_index(FAISS_INDEX_PATH)
            else:
                print("No documents processed, index not created.")
    
    def process_query(self, query: str) -> str:
        """
        Process a user query through the entire Simple RAG pipeline.
        
        Args:
            query: The user's query
            
        Returns:
            Generated response
        """
        # Step 1: Initial retrieval
        print("\n=== Initial Retrieval Results ===")
        retrieved_docs = self.vector_store.search(query, k=TOP_K)
        for i, doc in enumerate(retrieved_docs):
            print(f"  Result {i+1}:")
            print(f"    Source: {doc.metadata.get('source', 'N/A')}")
            print(f"    Content: {doc.page_content[:200]}...")
        
        context_docs = [{"document": doc.page_content, "source": doc.metadata.get("source")} for doc in retrieved_docs]

        # Step 2: Generate Response
        print("\n=== Generating Response ===")
        response = self.llm_utils.generate_response(query, context_docs)
        
        print("\n=== Final Answer ===")
        print(response)
        
        return response
