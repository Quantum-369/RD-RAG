"""
Configuration parameters for the Simple RAG pipeline.
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
TOP_K = 5  # Number of documents to retrieve and send to LLM

# LLM parameters
OPENAI_MODEL = "o3-2025-04-16"  # Replace with o3 model when available

# Maximum context tokens for LLM (reserve space for query and response)
MAX_CONTEXT_TOKENS = 14000  

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
