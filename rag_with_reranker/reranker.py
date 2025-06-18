"""
VoyageAI reranker implementation for cross-encoder re-ranking.
"""

import os
from typing import List, Dict, Any
import voyageai

class VoyageReranker:
    """Interface to VoyageAI's rerank-2 cross-encoder API."""
    
    def __init__(self):
        """Initialize the VoyageAI reranker."""
        self.api_key = os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            print("Warning: VOYAGE_API_KEY environment variable not set")
        
        self.client = voyageai.Client(api_key=self.api_key) if self.api_key else None
        self.model = "rerank-2"

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using VoyageAI's rerank API.
        
        Args:
            query: The query string
            documents: List of document text strings to be ranked
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with 'index', 'document', and 'relevance_score' keys
        """
        if not self.client:
            print("No VOYAGE_API_KEY set, returning documents in original order")
            return [{"index": i, "document": doc, "relevance_score": 1.0 - (i * 0.1)} 
                    for i, doc in enumerate(documents)]
        
        try:
            reranking = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k,
                truncation=True
            )
            
            reranked_results = []
            for result in reranking.results:
                reranked_results.append({
                    "index": result.index,                    "document": documents[result.index],
                    "relevance_score": result.relevance_score
                })
            
            return reranked_results
                
        except Exception as e:
            print(f"VoyageAI API error: {e}")
            return [{"index": i, "document": doc, "relevance_score": 1.0 - (i * 0.1)} 
                    for i, doc in enumerate(documents)]
