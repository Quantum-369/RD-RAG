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
        
        # Initialize Voyage client if API key is available
        self.client = voyageai.Client(api_key=self.api_key) if self.api_key else None
        self.model = "rerank-2"  # Using the recommended model from docs

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
            # Fallback to original order if no API key or client
            print("No VOYAGE_API_KEY set, returning documents in original order")
            return [{"index": i, "document": doc, "relevance_score": 1.0 - (i * 0.1)} 
                    for i, doc in enumerate(documents)]
        
        try:
            # Use the official voyageai client to rerank documents
            reranking = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k,
                truncation=True
            )
            
            # Convert to our standard format
            reranked_results = []
            for result in reranking.results:
                reranked_results.append({
                    "index": result.index,                    "document": documents[result.index],
                    "relevance_score": result.relevance_score
                })
            
            return reranked_results
                
        except Exception as e:
            print(f"VoyageAI API error: {e}")
            # Fallback to original order
            return [{"index": i, "document": doc, "relevance_score": 1.0 - (i * 0.1)} 
                    for i, doc in enumerate(documents)]
        except Exception as e:
            print(f"Exception during reranking: {e}")
            # Fallback to original order
            return [{"index": i, "document": doc, "relevance_score": 1.0 - (i * 0.1)} 
                    for i, doc in enumerate(documents)]
    
    def rerank_for_multiple_queries(self, queries_docs: Dict[str, List[str]], 
                                    top_k: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Rerank documents for multiple queries.
        
        Args:
            queries_docs: Dictionary mapping queries to lists of documents
            top_k: Number of top results to return per query
            
        Returns:
            Dictionary mapping queries to their reranked results
        """
        results = {}
        for query, documents in queries_docs.items():
            results[query] = self.rerank(query, documents, top_k=top_k)
        return results
