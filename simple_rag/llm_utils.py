"""
LLM utilities for response generation.
"""

import os
from typing import List, Dict, Any
from openai import OpenAI

from .config import (
    OPENAI_MODEL, 
    MAX_CONTEXT_TOKENS
)

class LLMUtils:
    """Utilities for LLM operations like response generation."""
    
    def __init__(self):
        """Initialize the LLM utilities."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY environment variable not set")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.model = OPENAI_MODEL
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM with the given query and context.
        
        Args:
            query: The user's query
            context_docs: List of context documents
            
        Returns:
            The generated response
        """
        if not self.api_key or not self.client:
            return "No OPENAI_API_KEY set. Cannot generate response."

        context = ""
        for doc in context_docs:
            context += f"Source: {doc.get('source', 'N/A')}\nContent: {doc['document']}\n\n"
        
        # Truncate context if it exceeds the token limit
        # This is a simple implementation; more sophisticated truncation might be needed
        if len(context.split()) > MAX_CONTEXT_TOKENS:
            print("Warning: Context exceeds token limit, truncating...")
            context = " ".join(context.split()[:MAX_CONTEXT_TOKENS])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the query."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
            ]
        )
        return response.choices[0].message.content.strip()
