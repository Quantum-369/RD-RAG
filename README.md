# RD-RAG: Rationale-Driven Retrieval-Augmented Generation

A robust implementation of a Retrieval-Augmented Generation (RAG) pipeline with a focus on rationale extraction and grounded knowledge.

## Overview

This RAG framework processes documents through a multi-stage pipeline:
1. Document processing and chunking
2. Vector embedding and indexing
3. Rationale extraction from user queries
4. Subquery generation
5. Initial retrieval (vector similarity search)
6. Reranking with Voyage AI
7. Response generation with strict grounding

## Key Features

- PDF to Markdown conversion for robust document processing
- Rationale-driven query understanding
- Multi-query retrieval for comprehensive context gathering
- Cross-encoder reranking with Voyage AI
- Source-grounded response generation with citation
- Configurable chunking and retrieval parameters

## Project Structure

- `main.py`: Command-line interface for the pipeline
- `rag_pipeline/`: Core pipeline components
  - `document_processor.py`: Handles document loading, conversion, and chunking
  - `vector_store.py`: FAISS-based vector storage and retrieval
  - `reranker.py`: Implements the Voyage AI reranker
  - `llm_utils.py`: LLM-based components for query understanding and response generation
  - `pipeline.py`: Main pipeline orchestration
  - `config.py`: Configuration parameters

## Setup and Usage

### Installation

```bash
git clone https://github.com/Quantum-369/RD-RAG.git
cd RD-RAG
pip install -r requirements.txt
```

### Environment Variables

Set your OpenAI and Voyage AI API keys:

```bash
# For Windows PowerShell
$env:OPENAI_API_KEY='your-openai-api-key'
$env:VOYAGE_API_KEY='your-voyage-api-key'
```

### Running the Pipeline

Process all documents in a directory:
```bash
python main.py
```

Process a single document:
```bash
python main.py --file_name "HISTORY-ENGLISH.pdf"
```

Reinitialize the vector index:
```bash
python main.py --reinitialize
```

## Requirements

See `requirements.txt` for the full list of dependencies.
