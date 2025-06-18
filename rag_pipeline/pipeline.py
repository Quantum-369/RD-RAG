"""
Main pipeline class that orchestrates the RD-RAG flow.
"""

import os
from typing import List, Dict, Any, Optional
import shutil

from rag_pipeline.document_processor import DocumentProcessor
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.llm_utils import LLMUtils
from rag_pipeline.reranker import VoyageReranker
from rag_pipeline.config import (
    DOCUMENTS_DIR, 
    FAISS_INDEX_PATH, 
    TOP_N_INITIAL, 
    TOP_K_RERANKED
)

class RDRagPipeline:
    """
    Rationale-Driven RAG Pipeline that integrates document processing,
    vector search, rationale extraction, subquery generation, reranking,
    and response generation.
    """
    
    def __init__(self):
        """Initialize the RD-RAG pipeline components."""
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_utils = LLMUtils()
        self.reranker = VoyageReranker()
    
    def initialize(self, documents_dir: Optional[str] = None, reinitialize: bool = False, file_name: Optional[str] = None) -> None:
        """
        Initialize the pipeline by processing documents and creating the vector index.
        
        Args:
            documents_dir: Directory containing documents to process
            reinitialize: Force reprocessing of documents
            file_name: Name of a single file to process
        """
        documents_dir = documents_dir or DOCUMENTS_DIR
        
        if reinitialize and os.path.exists(FAISS_INDEX_PATH):
            print(f"Reinitializing: removing existing index at {FAISS_INDEX_PATH}")
            shutil.rmtree(FAISS_INDEX_PATH)

        # Try to load existing index first
        if not self.vector_store.load_index(FAISS_INDEX_PATH):
            # If loading fails, process documents and create new index
            documents = []
            if file_name:
                print(f"Creating new index from file: {file_name} in {documents_dir}")
                documents = self.doc_processor.process_file(documents_dir, file_name)
            else:
                print(f"Creating new index from all documents in {documents_dir}")
                documents = self.doc_processor.process_directory(documents_dir)
            
            if documents:
                self.vector_store.create_index(documents)
                self.vector_store.save_index(FAISS_INDEX_PATH)
            else:
                print("No documents processed, index not created.")
    
    def format_results_for_reranker(self, results: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Convert vector search results to format needed by reranker.
        
        Args:
            results: Dictionary mapping queries to document objects
            
        Returns:
            Dictionary mapping queries to document texts
        """
        formatted_results = {}
        for query, docs in results.items():
            formatted_results[query] = [doc.page_content for doc in docs]
        return formatted_results
    
    def process_query(self, query: str) -> str:
        """
        Process a user query through the entire RD-RAG pipeline.
        
        Args:
            query: The user's query
            
        Returns:
            Generated response
        """
        # Step 1: Extract rationales
        rationales = self.llm_utils.extract_rationales(query)
        
        # Step 2: Generate subqueries
        subqueries = self.llm_utils.generate_subqueries(rationales)
        
        # Step 3: Initial retrieval for each subquery
        print("\n=== Initial Retrieval Results ===")
        initial_results = self.vector_store.search_for_multiple_queries(subqueries, k=TOP_N_INITIAL)
        for subquery, docs in initial_results.items():
            print(f"\n--- Subquery: {subquery} ---")
            for i, doc in enumerate(docs):
                print(f"  Result {i+1}: {doc.page_content[:150]}...")
        
        # Step 4: Format results for reranker
        formatted_results = self.format_results_for_reranker(initial_results)
        
        # Step 5: Rerank documents using cross-encoder
        print("\n=== Reranked Results ===")
        reranked_results = self.reranker.rerank_for_multiple_queries(formatted_results)
        for subquery, docs in reranked_results.items():
            print(f"\n--- Subquery: {subquery} ---")
            for i, doc_info in enumerate(docs):
                print(f"  Reranked Result {i+1} (Score: {doc_info['relevance_score']:.4f}): {doc_info['document'][:150]}...")

        # Step 6: Prepare context for LLM
        print("\n=== Context for LLM ===")
        context = self.llm_utils.prepare_context(reranked_results, top_k=TOP_K_RERANKED)
        print(context)
        
        # Step 7: Generate response
        response = self.llm_utils.generate_response(query, context)
        
        return response
