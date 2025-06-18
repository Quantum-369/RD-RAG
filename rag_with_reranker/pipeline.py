"""
Main pipeline class that orchestrates the RAG with reranker flow.
"""

import os
from typing import Optional, List, Dict, Any

from .document_processor import DocumentProcessor
from .llm_utils import LLMUtils
from .reranker import VoyageReranker
from .config import (
    DOCUMENTS_DIR,
    TOP_K_RERANKED
)

class RAGWithRerankerPipeline:
    """
    A RAG pipeline with a reranker that integrates document processing,
    reranking, and response generation without a vector store.
    """

    def __init__(self):
        """Initialize the RAG with reranker pipeline components."""
        self.doc_processor = DocumentProcessor()
        self.llm_utils = LLMUtils()
        self.reranker = VoyageReranker()
        self.all_chunks: List[Dict[str, Any]] = []

    def initialize(self, documents_dir: Optional[str] = None, reinitialize: bool = False) -> None:
        """
        Initialize the pipeline by processing documents and loading chunks into memory.
        
        Args:
            documents_dir: Directory containing documents to process.
            reinitialize: This parameter is kept for compatibility but doesn't do anything here.
        """
        documents_dir = documents_dir or DOCUMENTS_DIR
        print(f"Loading and processing documents from {documents_dir}...")
        
        # Process documents and store chunks in memory
        documents = self.doc_processor.process_directory(documents_dir)
        self.all_chunks = documents
        
        if self.all_chunks:
            print(f"Processed {len(self.all_chunks)} chunks from documents.")
        else:
            print("No documents were processed.")

    def process_query(self, query: str) -> str:
        """
        Process a user query through the entire RAG with reranker pipeline.
        
        Args:
            query: The user's query
            
        Returns:
            Generated response
        """
        if not self.all_chunks:
            return "Error: No documents have been processed. Please initialize the pipeline."

        print(f"Reranking {len(self.all_chunks)} chunks directly...")

        doc_texts = [doc.page_content for doc in self.all_chunks]
        original_docs = self.all_chunks

        # Step 1: Rerank
        print("\n=== Reranking Results ===")
        reranked_results = self.reranker.rerank(query, doc_texts, top_k=TOP_K_RERANKED)
        for i, result in enumerate(reranked_results):
            print(f"  Reranked Result {i+1}:")
            print(f"    Score: {result['relevance_score']:.4f}")
            print(f"    Content: {result['document'][:200]}...")

        context_docs = []
        for result in reranked_results:
            original_doc = original_docs[result["index"]]
            context_docs.append({"document": result["document"], "source": original_doc.metadata.get("source")})

        # Step 2: Generate Response
        print("\n=== Generating Response ===")
        response = self.llm_utils.generate_response(query, context_docs)
        
        print("\n=== Final Answer ===")
        print(response)
        
        return response
