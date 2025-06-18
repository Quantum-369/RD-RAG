"""
Document processor component for chunking and managing documents.
"""

import os
from typing import List
import tiktoken
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredFileLoader

from rag_pipeline.config import CHUNK_SIZE, CHUNK_OVERLAP, TEMP_DIR

class DocumentProcessor:
    """Handles document loading, chunking, and saving."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def _convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Converts a PDF file to a Markdown string."""
        markdown_content = ""
        try:
            with open(pdf_path, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    markdown_content += f"## Page {page_num + 1}\n\n"
                    page_text = page.extract_text()
                    if page_text:
                        markdown_content += page_text
                    markdown_content += "\n\n"
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
        return markdown_content

    def load_documents_from_directory(self, documents_dir: str) -> List[Document]:
        """Load documents from the specified directory."""
        processed_docs = []
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
            
        for filename in os.listdir(documents_dir):
            file_path = os.path.join(documents_dir, filename)
            if filename.lower().endswith(".pdf"):
                print(f"Converting {filename} to Markdown...")
                markdown_content = self._convert_pdf_to_markdown(file_path)
                
                if not markdown_content:
                    print(f"Skipping {filename} due to conversion error or empty content.")
                    continue

                temp_md_path = os.path.join(TEMP_DIR, f"{os.path.splitext(filename)[0]}.md")
                with open(temp_md_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                
                # Create Document object directly from markdown content
                processed_docs.append(Document(page_content=markdown_content, metadata={"source": file_path}))
                print(f"Loaded {filename} as markdown.")
            elif not (filename.startswith('.') or filename.endswith((".pyc",))):
                try:
                    loader = UnstructuredFileLoader(file_path)
                    processed_docs.extend(loader.load())
                    print(f"Loaded {filename} directly.")
                except ImportError as e:
                    # Fallback for text-based files
                    if file_path.lower().endswith(('.txt', '.md')):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        processed_docs.append(Document(page_content=content, metadata={"source": file_path}))
                        print(f"Loaded {filename} as raw text.")
                    else:
                        print(f"Skipping non-PDF file {filename} due to error: {e}")
                except Exception as e:
                    print(f"Skipping non-PDF file {filename} due to error: {e}")

        print(f"Loaded {len(processed_docs)} documents from {documents_dir}")
        return processed_docs
    
    def load_document_from_file(self, file_path: str) -> List[Document]:
        """Load a single document from the specified file path."""
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

        if file_path.lower().endswith(".pdf"):
            print(f"Converting {os.path.basename(file_path)} to Markdown...")
            markdown_content = self._convert_pdf_to_markdown(file_path)

            if not markdown_content:
                print(f"Skipping {os.path.basename(file_path)} due to conversion error or empty content.")
                return []

            filename = os.path.basename(file_path)
            temp_md_path = os.path.join(TEMP_DIR, f"{os.path.splitext(filename)[0]}.md")
            with open(temp_md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            # Create Document object directly from markdown content
            documents = [Document(page_content=markdown_content, metadata={"source": file_path})]
            print(f"Loaded {filename} as markdown.")
        else:
            try:
                loader = UnstructuredFileLoader(file_path)
                documents = loader.load()
                print(f"Loaded {os.path.basename(file_path)} directly.")
            except ImportError as e:
                # Fallback for text-based files
                if file_path.lower().endswith(('.txt', '.md')):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents = [Document(page_content=content, metadata={"source": file_path})]
                    print(f"Loaded {os.path.basename(file_path)} as raw text.")
                else:
                    print(f"Error loading {file_path}: {e}")
                    documents = []
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                documents = []
                
        print(f"Loaded {len(documents)} document(s) from {file_path}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def save_chunks(self, chunks: List[Document]) -> None:
        """Save chunks to the temporary directory for inspection."""
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        for i, chunk in enumerate(chunks):
            with open(os.path.join(TEMP_DIR, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(chunk.page_content)
        print(f"Saved {len(chunks)} chunks to {TEMP_DIR}")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return len(self.tokenizer.encode(text))
    
    def truncate_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)
    
    def process_directory(self, documents_dir: str) -> List[Document]:
        """Process documents from a directory: load, split into chunks, and save."""
        documents = self.load_documents_from_directory(documents_dir)
        chunks = self.split_documents(documents)
        self.save_chunks(chunks)
        return chunks

    def process_file(self, documents_dir: str, file_name: str) -> List[Document]:
        """Process a single file: load, split into chunks, and save."""
        file_path = os.path.join(documents_dir, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
        documents = self.load_document_from_file(file_path)
        chunks = self.split_documents(documents)
        self.save_chunks(chunks)
        return chunks
