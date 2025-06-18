"""
Configuration parameters for the RD-RAG pipeline.
"""

import os

# Paths
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_chunks")
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index")

# Document processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Retrieval parameters
TOP_N_INITIAL = 10  # Number of documents to retrieve in initial retrieval
TOP_K_RERANKED = 5  # Number of documents to send to LLM after reranking
NUM_SUBQUERIES = 3  # Default number of subqueries to generate

# LLM parameters
OPENAI_MODEL = "o3-2025-04-16"  # Replace with o3 model when available

# Maximum context tokens for LLM (reserve space for query and response)
MAX_CONTEXT_TOKENS = 14000  

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
