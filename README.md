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

- `main.py`: The command-line interface for running the pipeline. You can use this to process documents, run queries, and enter interactive mode.

### Pipeline Stages

The RD-RAG pipeline consists of the following stages:

1.  **Document Processing**: Documents in the `documents` directory are loaded, converted to Markdown (if they are PDFs), and split into manageable chunks.
2.  **Rationale Extraction**: When a query is received, the pipeline first uses an LLM to extract the key rationales or intents behind the query.
3.  **Subquery Generation**: Based on the extracted rationales, multiple subqueries are generated to cover different aspects of the information need.
4.  **Initial Retrieval**: For each subquery, a set of relevant documents is retrieved from a FAISS vector store.
5.  **Reranking**: The retrieved documents are reranked using VoyageAI's cross-encoder to improve relevance.
6.  **Response Generation**: The final set of reranked documents is passed to an LLM along with the original query to generate a comprehensive answer.

### Other RAG Implementations

This project also includes two other RAG implementations for comparison:

*   **Simple RAG**: A basic RAG pipeline that retrieves documents and sends them directly to the LLM.
*   **RAG with Reranker**: A pipeline that includes a reranking step but does not use rationale extraction or subquery generation.

## Getting Started

### Prerequisites

- Python 3.8+
- An environment with the required packages (see `requirements.txt`)
- API keys for OpenAI and VoyageAI, set as environment variables:
  - `OPENAI_API_KEY`
  - `VOYAGE_API_KEY`

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd RD-RAG
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The `main.py` script provides a command-line interface for interacting with the RAG pipelines.

#### Processing Documents

To process the documents in the `documents` directory and create the FAISS index, run:

```bash
python main.py
```

You can force the reprocessing of documents by using the `--reinitialize` flag:

```bash
python main.py --reinitialize
```

#### Running a Query

You can run a single query from the command line:

```bash
python main.py --query "Your query here"
```

By default, this will use the RD-RAG pipeline. You can select a different pipeline using the `--pipeline` argument:

```bash
python main.py --pipeline simple --query "Your query here"
python main.py --pipeline reranker --query "Your query here"
```

#### Interactive Mode

To enter interactive mode, run `main.py` without a query:

```bash
python main.py
```

You can then enter queries at the prompt. Type `exit` to quit.

## Configuration

The configuration for each pipeline is located in its respective directory:

- `rag_pipeline/config.py`
- `simple_rag/config.py`
- `rag_with_reranker/config.py`

These files contain parameters for document processing, retrieval, and LLM settings.
