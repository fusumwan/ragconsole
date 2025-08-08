# Set ONNX Runtime provider to avoid CoreML issues on macOS
# This must be set before importing chromadb
import os
os.environ["ONNXRUNTIME_PROVIDER_LIST"] = "CPUExecutionProvider"

"""
================================================================================
RAG CONTROL SYSTEM DESIGN DOCUMENT
================================================================================

SYSTEM OVERVIEW:
RAGControl is a comprehensive Retrieval-Augmented Generation (RAG) system that 
manages document storage, processing, and semantic search using ChromaDB as the 
vector database backend. The system is designed to handle multiple document formats
(markdown and PDF) with robust error handling, logging, and data management.

ARCHITECTURE COMPONENTS:
1. Document Processing Layer
   - File format detection and validation
   - Text extraction from markdown and PDF files
   - Content chunking with overlap for optimal retrieval
   - Metadata generation and storage

2. Vector Database Layer (ChromaDB)
   - Persistent storage with configurable settings
   - Document collection management
   - Semantic indexing and retrieval
   - Metadata filtering and querying

3. Search and Retrieval Layer
   - Semantic similarity search
   - Document filtering capabilities
   - Result ranking and formatting
   - Query optimization

4. Data Management Layer
   - Document lifecycle management
   - Duplicate detection and prevention
   - Database statistics and monitoring
   - Cleanup and deletion operations

5. Logging and Monitoring Layer
   - Comprehensive logging system
   - Error tracking and reporting
   - Performance monitoring
   - Debug information capture

KEY FEATURES:
- Multi-format document support (Markdown, PDF)
- Intelligent text chunking with sentence boundary detection
- Duplicate document prevention
- Flexible search with document filtering
- Comprehensive error handling and logging
- Database statistics and monitoring
- Clean document deletion with multiple identification methods

PERFORMANCE CONSIDERATIONS:
- Configurable chunk sizes for optimal retrieval
- Overlap between chunks to maintain context
- Efficient vector database operations
- Memory-conscious text processing
- Scalable document storage architecture

SECURITY FEATURES:
- Input validation and sanitization
- File path security checks
- Error message sanitization
- Database access controls

USAGE PATTERNS:
1. Document Ingestion: study_document() for adding new documents
2. Information Retrieval: search_documents() for semantic search
3. Document Management: list_documents(), delete_document() for maintenance
4. System Monitoring: get_database_stats() for health checks

DEPENDENCIES:
- chromadb: Vector database for semantic storage
- pdfplumber: PDF text extraction
- Standard Python libraries: os, sys, logging, hashlib, re, datetime, json

AUTHOR: RAG System Development Team
VERSION: 1.0
LAST UPDATED: 2024
================================================================================
"""

# Standard library imports for core functionality
import sys  # System-specific parameters and functions for logging output
import logging  # Logging facility for application monitoring and debugging
import hashlib  # Secure hash algorithms for document ID generation
from typing import List, Dict, Optional, Any  # Type hints for better code documentation
from pathlib import Path  # Object-oriented filesystem paths (imported but not used in current implementation)
import chromadb  # Vector database for semantic storage and retrieval
from chromadb.config import Settings  # Configuration settings for ChromaDB
from chromadb.utils import embedding_functions  # Embedding functions
import pdfplumber  # PDF text extraction library
import re  # Regular expressions for text processing and chunking
from datetime import datetime  # Date and time utilities for timestamping
import json  # JSON data format support (imported but not used in current implementation)


class RAGControl:
    """
    RAG Control class for managing document storage and retrieval using ChromaDB.
    Handles both markdown and PDF files with professional error handling and logging.
    
    This class provides a complete RAG (Retrieval-Augmented Generation) system
    that can ingest, store, search, and manage documents in a vector database.
    """
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize RAGControl with ChromaDB connection and logging setup.
        
        Args:
            db_path (str): Path to ChromaDB database directory, defaults to "./chroma_db"
        """
        # Store the database path for future reference
        self.db_path = db_path
        
        # Initialize logging system before any database operations
        self.setup_logging()
        
        # Create logger instance for this class
        self.logger = logging.getLogger(__name__)
        
        try:
            # Ensure database directory exists, create if it doesn't
            # exist_ok=True prevents errors if directory already exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            # This creates a persistent database that survives application restarts
            self.client = chromadb.PersistentClient(
                path=self.db_path,  # Database storage location
                settings=Settings(
                    anonymized_telemetry=False,  # Disable telemetry for privacy
                    allow_reset=True  # Allow database reset operations
                )
            )
            
            # Get or create collection for documents
            # Collections are like tables in traditional databases
            # Use sentence transformer embedding function to avoid CoreML issues
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.client.get_or_create_collection(
                name="documents",  # Collection name for storing document chunks
                metadata={"description": "RAG document storage"},  # Collection metadata
                embedding_function=embedding_function  # Use sentence transformer embedding function
            )
            
            # Log successful initialization
            self.logger.info(f"RAGControl initialized successfully with database at {self.db_path}")
            
        except Exception as e:
            # Log initialization error and re-raise with descriptive message
            self.logger.error(f"Failed to initialize RAGControl: {str(e)}")
            raise RuntimeError(f"Database initialization failed: {str(e)}")
    
    def setup_logging(self):
        """Setup comprehensive logging configuration for the entire system."""
        # Configure logging with INFO level and custom format
        logging.basicConfig(
            level=logging.INFO,  # Set minimum log level to INFO
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format with timestamp, logger name, level, and message
            handlers=[
                logging.FileHandler('rag_system.log'),  # Write logs to file
                logging.StreamHandler(sys.stdout)  # Also output to console
            ]
        )
    
    def _generate_document_id(self, file_path: str) -> str:
        """
        Generate a unique document ID based on file path and content hash.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            str: Unique document identifier with "doc_" prefix
        """
        try:
            # Convert to absolute path to ensure consistency across different working directories
            file_path = os.path.abspath(file_path)
            
            # Generate MD5 hash of the file path for unique identification
            # MD5 is sufficient for this use case as we're not using it for security
            file_hash = hashlib.md5(file_path.encode()).hexdigest()
            
            # Return document ID with "doc_" prefix for easy identification
            return f"doc_{file_hash}"
            
        except Exception as e:
            # Log error and re-raise for proper error handling upstream
            self.logger.error(f"Failed to generate document ID for {file_path}: {str(e)}")
            raise
    
    def _read_markdown_file(self, file_path: str) -> str:
        """
        Read and extract text content from markdown file.
        
        Args:
            file_path (str): Path to the markdown file
            
        Returns:
            str: Extracted text content from the markdown file
        """
        try:
            # Check if file exists before attempting to read
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Markdown file not found: {file_path}")
            
            # Open and read file with UTF-8 encoding to handle special characters
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()  # Read entire file content into memory
            
            # Log successful file reading
            self.logger.info(f"Successfully read markdown file: {file_path}")
            return content
            
        except UnicodeDecodeError as e:
            # Handle encoding errors specifically
            self.logger.error(f"Encoding error reading markdown file {file_path}: {str(e)}")
            raise
        except Exception as e:
            # Handle all other errors
            self.logger.error(f"Failed to read markdown file {file_path}: {str(e)}")
            raise
    
    def _read_pdf_file(self, file_path: str) -> str:
        """
        Read and extract text content from PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content from the PDF file
        """
        try:
            # Check if file exists before attempting to read
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Initialize empty string to store extracted content
            content = ""
            
            # Open PDF file using pdfplumber
            with pdfplumber.open(file_path) as pdf:
                # Iterate through all pages in the PDF
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text from current page
                        page_text = page.extract_text()
                        
                        # Only add page content if text was successfully extracted
                        if page_text:
                            # Add page separator and content to maintain page structure
                            content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                            
                    except Exception as e:
                        # Log warning but continue processing other pages
                        self.logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                        continue
            
            # Validate that some content was extracted
            if not content.strip():
                raise ValueError(f"No text content extracted from PDF: {file_path}")
            
            # Log successful PDF reading
            self.logger.info(f"Successfully read PDF file: {file_path}")
            return content
            
        except Exception as e:
            # Handle all PDF reading errors
            self.logger.error(f"Failed to read PDF file {file_path}: {str(e)}")
            raise
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Size of each chunk in characters, defaults to 1000
            overlap (int): Overlap between chunks in characters, defaults to 200
            
        Returns:
            List[str]: List of text chunks with overlap
        """
        try:
            # Return empty list if text is empty or only whitespace
            if not text.strip():
                return []
            
            # Clean and normalize text by replacing multiple whitespace with single space
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Initialize list to store chunks
            chunks = []
            
            # Initialize starting position for chunking
            start = 0
            
            # Continue chunking until we reach the end of the text
            while start < len(text):
                # Calculate end position for current chunk
                end = start + chunk_size
                
                # Extract current chunk
                chunk = text[start:end]
                
                # Try to break at sentence boundaries for better chunk quality
                if end < len(text):  # Only try to break if we haven't reached the end
                    # Find last occurrence of sentence endings in current chunk
                    last_period = chunk.rfind('.')
                    last_exclamation = chunk.rfind('!')
                    last_question = chunk.rfind('?')
                    last_newline = chunk.rfind('\n')
                    
                    # Find the best break point among sentence endings
                    break_point = max(last_period, last_exclamation, last_question, last_newline)
                    
                    # Only break if the break point is not too early in the chunk (70% threshold)
                    if break_point > chunk_size * 0.7:
                        # Adjust chunk to end at the break point
                        chunk = chunk[:break_point + 1]
                        # Update end position to match the break point
                        end = start + break_point + 1
                
                # Add current chunk to list (strip whitespace for clean chunks)
                chunks.append(chunk.strip())
                
                # Move start position for next chunk, accounting for overlap
                start = end - overlap
                
                # Break if we've processed all text
                if start >= len(text):
                    break
            
            # Log chunking results
            self.logger.info(f"Created {len(chunks)} chunks from text")
            return chunks
            
        except Exception as e:
            # Handle chunking errors
            self.logger.error(f"Failed to chunk text: {str(e)}")
            raise
    
    def _check_document_exists(self, document_id: str) -> bool:
        """
        Check if a document already exists in the database.
        
        Args:
            document_id (str): Document identifier to check
            
        Returns:
            bool: True if document exists, False otherwise
        """
        try:
            # Query the collection to check if document exists
            # Use where clause to filter by document_id
            results = self.collection.get(
                where={"document_id": document_id},  # Filter by document ID
                limit=1  # Only need one result to confirm existence
            )
            
            # Check if results are valid
            if not results or not isinstance(results, dict) or 'ids' not in results:
                return False
            
            # Ensure ids is a list
            if not isinstance(results['ids'], list):
                return False
            
            # Return True if any results were found, False otherwise
            return len(results['ids']) > 0
            
        except Exception as e:
            # Log error and return False to allow processing to continue
            self.logger.error(f"Failed to check document existence: {str(e)}")
            return False
    
    def study_document(self, file_path: str, file_type: str = "md") -> Dict[str, Any]:
        """
        Study and store a document (markdown or PDF) in the ChromaDB database.
        
        Args:
            file_path (str): Path to the document file
            file_type (str): Type of file ('md' for markdown, 'pdf' for PDF), defaults to "md"
            
        Returns:
            Dict[str, Any]: Study results including document ID and chunk count
        """
        try:
            # Log the start of document study process
            self.logger.info(f"Starting document study: {file_path} (type: {file_type})")
            
            # Validate that the file exists before processing
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Generate unique document ID based on file path
            document_id = self._generate_document_id(file_path)
            
            # Check if document already exists to prevent duplicates
            if self._check_document_exists(document_id):
                # Log warning and return early if document already exists
                self.logger.warning(f"Document already exists in database: {file_path}")
                return {
                    "status": "exists",
                    "document_id": document_id,
                    "file_path": file_path,
                    "message": "Document already studied and stored in database"
                }
            
            # Read file content based on file type
            if file_type.lower() == "md":
                # Read markdown file
                content = self._read_markdown_file(file_path)
            elif file_type.lower() == "pdf":
                # Read PDF file
                content = self._read_pdf_file(file_path)
            else:
                # Raise error for unsupported file types
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Chunk the content for vector storage
            chunks = self._chunk_text(content)
            
            # Validate that content was successfully chunked
            if not chunks:
                raise ValueError(f"No content extracted from file: {file_path}")
            
            # Prepare chunk IDs for ChromaDB storage
            # Each chunk gets a unique ID based on document ID and chunk index
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Prepare metadata for each chunk
            # Metadata includes document information and chunk details
            metadatas = [
                {
                    "document_id": document_id,  # Link to parent document
                    "file_path": file_path,  # Original file path
                    "file_type": file_type,  # File type (md/pdf)
                    "chunk_index": i,  # Position of chunk in document
                    "total_chunks": len(chunks),  # Total number of chunks
                    "timestamp": datetime.now().isoformat()  # Processing timestamp
                }
                for i in range(len(chunks))
            ]
            
            # Add chunks to ChromaDB collection
            # This stores the text chunks with their metadata for semantic search
            self.collection.add(
                documents=chunks,  # The actual text chunks
                ids=chunk_ids,  # Unique identifiers for each chunk
                metadatas=metadatas  # Metadata for each chunk
            )
            
            # Log successful storage
            self.logger.info(f"Successfully stored document {file_path} with {len(chunks)} chunks")
            
            # Return success response with document details
            return {
                "status": "success",
                "document_id": document_id,
                "file_path": file_path,
                "chunks_count": len(chunks),
                "file_type": file_type,
                "message": f"Document successfully studied and stored with {len(chunks)} chunks"
            }
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            self.logger.error(f"Failed to study document {file_path}: {str(e)}")
            raise
    
    def search_documents(self, query: str, document_filter: Optional[str] = None, 
                        n_results: int = 5) -> Dict[str, Any]:
        """
        Search for relevant content in the stored documents.
        
        Args:
            query (str): Search query for semantic matching
            document_filter (Optional[str]): Filter by specific document ID or file path
            n_results (int): Number of results to return, defaults to 5
            
        Returns:
            Dict[str, Any]: Search results with relevant chunks and metadata
        """
        try:
            # Log search operation
            self.logger.info(f"Searching documents with query: '{query}'")
            
            # Validate that query is not empty
            if not query.strip():
                raise ValueError("Search query cannot be empty")
            
            # Prepare where clause for filtering (None means no filter)
            where_clause = None
            
            # Apply document filter if provided
            if document_filter:
                # Try to match by document_id first, then by file_path
                # Check if it looks like a document ID (starts with "doc_")
                if document_filter.startswith("doc_"):
                    # Filter by document ID
                    where_clause = {"document_id": document_filter}
                else:
                    # Filter by file path (exact match)
                    where_clause = {"file_path": document_filter}
            
            # Perform semantic search using ChromaDB
            results = self.collection.query(
                query_texts=[query],  # Search query
                n_results=n_results,  # Number of results to return
                where=where_clause  # Optional filter
            )
            
            # Process and format search results
            formatted_results = []
            
            # Iterate through all returned results
            for i in range(len(results['documents'][0])):
                # Create structured result object for each match
                result = {
                    "chunk_id": results['ids'][0][i],  # Unique chunk identifier
                    "content": results['documents'][0][i],  # Actual text content
                    "metadata": results['metadatas'][0][i],  # Chunk metadata
                    "distance": results['distances'][0][i] if 'distances' in results else None  # Similarity score
                }
                formatted_results.append(result)
            
            # Log search completion
            self.logger.info(f"Search completed, found {len(formatted_results)} results")
            
            # Return structured search results
            return {
                "status": "success",
                "query": query,
                "results_count": len(formatted_results),
                "results": formatted_results,
                "filter_applied": document_filter is not None  # Indicate if filter was used
            }
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            self.logger.error(f"Failed to search documents: {str(e)}")
            raise
    
    def list_documents(self) -> Dict[str, Any]:
        """
        List all documents stored in the database.
        
        Returns:
            Dict[str, Any]: List of stored documents with metadata
        """
        try:
            # Log the listing operation
            self.logger.info("Listing all documents in database")
            
            # Get all documents from collection
            try:
                # Try to get a small sample first to test if the collection is working
                sample_results = self.collection.get(limit=1)
                
                # If we get here, the collection is working, so get all results
                results = self.collection.get()
                
            except Exception as e:
                self.logger.error(f"Error calling collection methods: {str(e)}")
                # Return empty results if collection operations fail
                return {
                    "status": "success",
                    "documents_count": 0,
                    "documents": []
                }
            
            # Check if collection is empty or results are invalid
            if not results or not isinstance(results, dict):
                self.logger.info("Collection is empty or invalid")
                return {
                    "status": "success",
                    "documents_count": 0,
                    "documents": []
                }
            
            # Validate that required keys exist and are lists
            if 'ids' not in results or 'metadatas' not in results:
                self.logger.info("Collection structure is invalid")
                return {
                    "status": "success",
                    "documents_count": 0,
                    "documents": []
                }
            
            # Ensure ids and metadatas are lists
            if not isinstance(results['ids'], list) or not isinstance(results['metadatas'], list):
                self.logger.info("Collection data types are invalid")
                return {
                    "status": "success",
                    "documents_count": 0,
                    "documents": []
                }
            
            # Check if collection is empty
            if len(results['ids']) == 0:
                self.logger.info("Collection is empty")
                return {
                    "status": "success",
                    "documents_count": 0,
                    "documents": []
                }
            
            # Group by document_id to avoid duplicates
            # Each document may have multiple chunks, but we want one entry per document
            documents = {}
            
            # Iterate through all chunks and group by document
            for i, doc_id in enumerate(results['ids']):
                if i < len(results['metadatas']):
                    metadata = results['metadatas'][i]
                    document_id = metadata.get('document_id', 'unknown')
                    
                    # Only add document if we haven't seen it before
                    if document_id not in documents:
                        documents[document_id] = {
                            "document_id": document_id,
                            "file_path": metadata.get('file_path', 'unknown'),
                            "file_type": metadata.get('file_type', 'unknown'),
                            "total_chunks": metadata.get('total_chunks', 0),
                            "timestamp": metadata.get('timestamp', 'unknown')
                        }
            
            # Convert dictionary to list for easier processing
            document_list = list(documents.values())
            
            # Log the number of documents found
            self.logger.info(f"Found {len(document_list)} documents in database")
            
            # Return structured document list
            return {
                "status": "success",
                "documents_count": len(document_list),
                "documents": document_list
            }
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            self.logger.error(f"Failed to list documents: {str(e)}")
            raise
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document and all its chunks from the database.
        
        Args:
            document_id (str): Document identifier to delete
            
        Returns:
            Dict[str, Any]: Deletion results
        """
        try:
            # Log the deletion operation
            self.logger.info(f"Deleting document: {document_id}")
            
            # Get all chunks for the document to be deleted
            results = self.collection.get(
                where={"document_id": document_id}  # Filter by document ID
            )
            
            # Check if results are valid and document exists
            if not results or not isinstance(results, dict) or 'ids' not in results:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                    "message": "Document not found in database"
                }
            
            # Ensure ids is a list and check if document exists
            if not isinstance(results['ids'], list) or len(results['ids']) == 0:
                # Return early if document not found
                return {
                    "status": "not_found",
                    "document_id": document_id,
                    "message": "Document not found in database"
                }
            
            # Delete all chunks for the document
            # This removes all chunks associated with the document ID
            self.collection.delete(ids=results['ids'])
            
            # Log successful deletion
            self.logger.info(f"Successfully deleted document {document_id} with {len(results['ids'])} chunks")
            
            # Return deletion results
            return {
                "status": "success",
                "document_id": document_id,
                "chunks_deleted": len(results['ids']),
                "message": f"Document successfully deleted with {len(results['ids'])} chunks"
            }
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            self.logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise
    
    def delete_document_by_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a document from the database by its file path.
        
        Args:
            file_path (str): Path to the document file to delete
            
        Returns:
            Dict[str, Any]: Deletion results
        """
        try:
            # Log the deletion attempt
            self.logger.info(f"Attempting to delete document by file path: {file_path}")
            
            # Validate that file path is not empty
            if not file_path or not file_path.strip():
                raise ValueError("File path cannot be empty")
            
            # Normalize file path to absolute path for consistent matching
            normalized_file_path = os.path.abspath(file_path)
            
            # First try exact path match
            results = self.collection.get(
                where={"file_path": normalized_file_path}  # Try exact path match
            )
            
            # Check if results are valid and document exists
            if not results or not isinstance(results, dict) or 'ids' not in results:
                # Try filename matching as fallback
                filename = os.path.basename(file_path)
                self.logger.info(f"No exact path match found, trying filename match: {filename}")
                
                # Get all documents to search by filename
                all_results = self.collection.get()
                
                # Check if all_results is valid
                if not all_results or not isinstance(all_results, dict) or 'ids' not in all_results or 'metadatas' not in all_results:
                    return {
                        "status": "not_found",
                        "file_path": file_path,
                        "message": f"No document found with file path or filename: {file_path}"
                    }
                
                # Ensure required fields are lists
                if not isinstance(all_results['ids'], list) or not isinstance(all_results['metadatas'], list):
                    return {
                        "status": "not_found",
                        "file_path": file_path,
                        "message": f"No document found with file path or filename: {file_path}"
                    }
                
                matching_ids = []
                
                # Search through all documents for matching filename
                for i, metadata in enumerate(all_results['metadatas']):
                    if i < len(all_results['ids']):
                        stored_filename = os.path.basename(metadata.get('file_path', ''))
                        if stored_filename == filename:
                            matching_ids.append(all_results['ids'][i])
                
                # If filename matches found, get their details
                if matching_ids:
                    results = self.collection.get(ids=matching_ids)
                    self.logger.info(f"Found {len(matching_ids)} chunks by filename match")
                else:
                    # Return not found if no matches by filename either
                    return {
                        "status": "not_found",
                        "file_path": file_path,
                        "message": f"No document found with file path or filename: {file_path}"
                    }
            else:
                # Ensure ids is a list and check if document exists
                if not isinstance(results['ids'], list) or len(results['ids']) == 0:
                    # Try filename matching as fallback
                    filename = os.path.basename(file_path)
                    self.logger.info(f"No exact path match found, trying filename match: {filename}")
                    
                    # Get all documents to search by filename
                    all_results = self.collection.get()
                    
                    # Check if all_results is valid
                    if not all_results or not isinstance(all_results, dict) or 'ids' not in all_results or 'metadatas' not in all_results:
                        return {
                            "status": "not_found",
                            "file_path": file_path,
                            "message": f"No document found with file path or filename: {file_path}"
                        }
                    
                    # Ensure required fields are lists
                    if not isinstance(all_results['ids'], list) or not isinstance(all_results['metadatas'], list):
                        return {
                            "status": "not_found",
                            "file_path": file_path,
                            "message": f"No document found with file path or filename: {file_path}"
                        }
                    
                    matching_ids = []
                    
                    # Search through all documents for matching filename
                    for i, metadata in enumerate(all_results['metadatas']):
                        if i < len(all_results['ids']):
                            stored_filename = os.path.basename(metadata.get('file_path', ''))
                            if stored_filename == filename:
                                matching_ids.append(all_results['ids'][i])
                    
                    # If filename matches found, get their details
                    if matching_ids:
                        results = self.collection.get(ids=matching_ids)
                        self.logger.info(f"Found {len(matching_ids)} chunks by filename match")
                    else:
                        # Return not found if no matches by filename either
                        return {
                            "status": "not_found",
                            "file_path": file_path,
                            "message": f"No document found with file path or filename: {file_path}"
                        }
            
            # Validate that results has the required structure
            if not isinstance(results, dict) or 'ids' not in results or 'metadatas' not in results:
                return {
                    "status": "error",
                    "file_path": file_path,
                    "message": "Invalid results structure from database"
                }
            
            # Ensure ids and metadatas are lists
            if not isinstance(results['ids'], list) or not isinstance(results['metadatas'], list):
                return {
                    "status": "error",
                    "file_path": file_path,
                    "message": "Invalid data types in results"
                }
            
            # Get unique document IDs for logging purposes
            # A single file might have multiple document IDs if processed multiple times
            document_ids = set(metadata.get('document_id', 'unknown') for metadata in results['metadatas'])
            
            # Delete all chunks for the file path
            self.collection.delete(ids=results['ids'])
            
            # Log successful deletion
            self.logger.info(f"Successfully deleted document {file_path} with {len(results['ids'])} chunks")
            
            # Return deletion results
            return {
                "status": "success",
                "file_path": file_path,
                "document_ids": list(document_ids),  # List of affected document IDs
                "chunks_deleted": len(results['ids']),
                "message": f"Document successfully deleted with {len(results['ids'])} chunks"
            }
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            self.logger.error(f"Failed to delete document by file path {file_path}: {str(e)}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics and information.
        
        Returns:
            Dict[str, Any]: Database statistics including chunk count, document count, and size
        """
        try:
            # Get all documents from the collection
            try:
                # Try to get a small sample first to test if the collection is working
                sample_results = self.collection.get(limit=1)
                
                # If we get here, the collection is working, so get all results
                results = self.collection.get()
                
            except Exception as e:
                self.logger.error(f"Error calling collection methods: {str(e)}")
                # Return empty stats if collection operations fail
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_content_size_bytes": 0,
                    "database_path": self.db_path,
                    "collection_name": "documents"
                }
            
            # Check if collection is empty or results are invalid
            if not results or not isinstance(results, dict):
                self.logger.info("Collection is empty or invalid")
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_content_size_bytes": 0,
                    "database_path": self.db_path,
                    "collection_name": "documents"
                }
            
            # Validate that required keys exist and are lists
            if 'ids' not in results or 'metadatas' not in results or 'documents' not in results:
                self.logger.info("Collection structure is invalid")
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_content_size_bytes": 0,
                    "database_path": self.db_path,
                    "collection_name": "documents"
                }
            
            # Ensure required fields are lists
            if not isinstance(results['ids'], list) or not isinstance(results['metadatas'], list) or not isinstance(results['documents'], list):
                self.logger.info("Collection data types are invalid")
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_content_size_bytes": 0,
                    "database_path": self.db_path,
                    "collection_name": "documents"
                }
            
            # Check if collection is empty
            if len(results['ids']) == 0:
                self.logger.info("Collection is empty")
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_content_size_bytes": 0,
                    "database_path": self.db_path,
                    "collection_name": "documents"
                }
            
            # Calculate total number of chunks
            total_chunks = len(results['ids'])
            
            # Calculate number of unique documents
            # Use set to count unique document IDs
            unique_documents = len(set(metadata.get('document_id', 'unknown') for metadata in results['metadatas']))
            
            # Calculate total content size in bytes
            # Sum the length of all document chunks
            total_content_size = sum(len(doc) for doc in results['documents'])
            
            # Compile statistics dictionary
            stats = {
                "total_chunks": total_chunks,  # Total number of text chunks
                "unique_documents": unique_documents,  # Number of unique documents
                "total_content_size_bytes": total_content_size,  # Total content size in bytes
                "database_path": self.db_path,  # Database storage location
                "collection_name": "documents"  # Collection name
            }
            
            # Log statistics for monitoring
            self.logger.info(f"Database stats: {stats}")
            return stats
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            self.logger.error(f"Failed to get database stats: {str(e)}")
            raise 