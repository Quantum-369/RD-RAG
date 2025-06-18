"""
Command-line interface for the RD-RAG pipeline.
"""

import os
import argparse
from rag_pipeline.pipeline import RDRagPipeline
from simple_rag.pipeline import SimpleRAGPipeline
from rag_with_reranker.pipeline import RAGWithRerankerPipeline
from rag_pipeline.config import DOCUMENTS_DIR

def main():
    """Main entry point for the RD-RAG pipeline CLI."""
    parser = argparse.ArgumentParser(description="Run a RAG pipeline")
    parser.add_argument("--pipeline", type=str, default="rd_rag", 
                        choices=["simple", "reranker", "rd_rag"],
                        help="The pipeline to run.")
    parser.add_argument("--docs_dir", type=str, default=DOCUMENTS_DIR, 
                        help="Directory containing documents to process")
    parser.add_argument("--file_name", type=str, default=None,
                        help="Name of a single file to process from the documents directory")
    parser.add_argument("--query", type=str, 
                        help="Query to process (optional)")
    parser.add_argument("--reinitialize", action="store_true", 
                        help="Force reprocessing of documents even if index exists")
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    if not os.environ.get("VOYAGE_API_KEY") and args.pipeline in ["reranker", "rd_rag"]:
        print("Warning: VOYAGE_API_KEY environment variable not set")
    
    # Initialize pipeline
    if args.pipeline == "simple":
        pipeline = SimpleRAGPipeline()
    elif args.pipeline == "reranker":
        pipeline = RAGWithRerankerPipeline()
    else:
        pipeline = RDRagPipeline()
    
    # Process documents and initialize
    print(f"Initializing {args.pipeline} pipeline with documents from {args.docs_dir}...")
    if hasattr(pipeline, 'initialize'):
        if args.pipeline in ["simple", "reranker"]:
             pipeline.initialize(args.docs_dir, reinitialize=args.reinitialize)
        else:
             pipeline.initialize(args.docs_dir, reinitialize=args.reinitialize, file_name=args.file_name)
    
    # Interactive mode if no query provided
    if args.query:
        response = pipeline.process_query(args.query)
        print("\n=== Response ===")
        print(response)
    else:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                break
            
            response = pipeline.process_query(query)
            print("\n=== Response ===")
            print(response)

if __name__ == "__main__":
    main()
