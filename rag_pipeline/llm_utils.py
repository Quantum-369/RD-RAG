"""
LLM utilities for rationale extraction, subquery generation, and response generation.
"""

import os
from typing import List, Dict, Any
from openai import OpenAI

from rag_pipeline.config import (
    OPENAI_MODEL, 
    NUM_SUBQUERIES,
    MAX_CONTEXT_TOKENS
)

class LLMUtils:
    """Utilities for LLM operations like rationale extraction and response generation."""
    
    def __init__(self):
        """Initialize the LLM utilities."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY environment variable not set")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.model = OPENAI_MODEL
    
    def extract_rationales(self, query: str) -> str:
        """
        Extract rationales from user query using LLM.
        
        Args:
            query: The user's query
            
        Returns:
            Extracted rationales as a string
        """
        if not self.api_key or not self.client:
            print("No OPENAI_API_KEY set, returning original query as rationale")
            return query
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Extract the key rationales or search intents from the given query. "
                                                  "Identify the core information needs."},
                {"role": "user", "content": f"Query: {query}"}
            ]
        )
        rationales = response.choices[0].message.content.strip()
        print(f"Extracted rationales: {rationales}")
        return rationales
    
    def generate_subqueries(self, rationales: str, num_queries: int = NUM_SUBQUERIES) -> List[str]:
        """
        Generate subqueries from rationales using LLM.
        
        Args:
            rationales: The extracted rationales
            num_queries: The number of subqueries to generate
            
        Returns:
            List of generated subqueries
        """
        if not self.api_key or not self.client:
            print("No OPENAI_API_KEY set, returning rationales as single subquery")
            return [rationales]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"Based on these rationales, generate {num_queries} different search "
                                                  f"queries that would help retrieve relevant information."},
                {"role": "user", "content": f"Rationales: {rationales}"}
            ]
        )
        subqueries_text = response.choices[0].message.content.strip()
        
        # Parse subqueries from the response
        subqueries = []
        for line in subqueries_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or ':' in line or 
                         any(str(i) in line[:2] for i in range(1, 10))):
                # Extract the actual query, removing numbers, dashes, etc.
                parts = line.split(':', 1) if ':' in line else [line]
                query = parts[-1].strip()
                if query and query not in subqueries:
                    subqueries.append(query)
        
        # Fallback in case parsing fails
        if not subqueries or len(subqueries) < num_queries:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"Generate {num_queries} search queries based on these rationales. "
                                                  f"Format each query on a new line starting with 'Query: '"},
                    {"role": "user", "content": f"Rationales: {rationales}"}
                ]
            )
            subqueries_text = response.choices[0].message.content.strip()
            subqueries = [line.replace('Query:', '').strip() for line in subqueries_text.split('\n') 
                          if 'Query:' in line]
        
        print(f"Generated subqueries: {subqueries}")
        return subqueries[:num_queries]
    
    def prepare_context(self, reranked_results: Dict[str, List[Dict[str, Any]]], 
                        top_k: int) -> str:
        """
        Prepare context for LLM from reranked results.
        
        Args:
            reranked_results: Dictionary mapping queries to their reranked results
            top_k: Number of top documents to include per query
            
        Returns:
            Combined context as a string
        """
        all_docs = []
        seen_docs = set()
        chunk_counter = 1
        
        for query, results in reranked_results.items():
            # Get top K documents for this query
            for i, doc_info in enumerate(results):
                if i >= top_k:
                    break
                
                doc_content = doc_info["document"]
                
                # Add document if not already included
                if doc_content not in seen_docs:
                    # Add chunk identifier for source tracking
                    labeled_doc = f"[Chunk {chunk_counter}]:\n{doc_content}"
                    all_docs.append(labeled_doc)
                    seen_docs.add(doc_content)
                    chunk_counter += 1
        
        # Join all documents into a single context
        context = "\n\n---\n\n".join(all_docs)
        
        return context
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using LLM with provided context.
        
        Args:
            query: The original user query
            context: The prepared context
            
        Returns:
            Generated response as a string
        """
        if not self.api_key or not self.client:
            return "Error: OPENAI_API_KEY not set. Please set the environment variable to generate responses."
        
        system_prompt = """You are a highly precise assistant that only answers based on the provided context.

STRICT RULES:
1. ONLY use information directly stated in the context. Do not introduce external information even if you believe it's factual.
2. If the context doesn't contain enough information to fully answer the question, state clearly: "The provided documents do not contain sufficient information to answer this question."
3. NEVER invent details, tools, methods, names, dates, or statistics that aren't explicitly mentioned in the context.
4. For each piece of information in your answer, identify in your thinking which part of the context it came from.
5. Cite information using [Chunk X] notation where X corresponds to the chunk where information appears.
6. If information appears in multiple chunks, include all relevant chunk citations.

Format your response as follows:
[Internal thinking: Analyze what information is available in the context and which parts directly address the question. Include chunk references for each fact.]

[Final answer: Write your final answer using ONLY information from the context, with appropriate citations to chunks.]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context (divided into chunks):\n{context}\n\nQuestion: {query}\n\nRemember, only use information explicitly stated in the context. If the answer isn't in the context, say so clearly."}
            ]
        )
        answer = response.choices[0].message.content.strip()
        return answer
