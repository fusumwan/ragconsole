#!/usr/bin/env python3
"""
RAG Delete Module

This module provides a professional interface for deleting documents from the RAG system.
It supports deletion of both markdown and PDF files from the ChromaDB database.

Author: RAG AI System
Version: 1.0.0
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import traceback

# Import the RAGService class
from ragcontrolservice import RAGService


class RAGDelete:
    """
    Professional RAG Delete class for removing documents from the ChromaDB database.
    
    This class provides comprehensive error handling, logging, and validation
    for deleting markdown and PDF files from the RAG system.
    """
    
    def __init__(self, db_path: str = "./chroma_db", log_level: str = "INFO"):
        """
        Initialize RAGDelete with database connection and logging setup.
        
        Args:
            db_path (str): Path to ChromaDB database directory
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.db_path = db_path
        self.log_level = log_level
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        try:
            # Initialize RAGService instance
            self.rag_control = RAGService(db_path=db_path)
            self.logger.info(f"RAGDelete initialized successfully with database at {db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAGDelete: {str(e)}")
            raise RuntimeError(f"RAGDelete initialization failed: {str(e)}")
    
    def setup_logging(self):
        """Setup comprehensive logging configuration for RAGDelete."""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file handler
        log_file = f"logs/rag_delete_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Setup root logger
        logging.basicConfig(
            level=log_level,
            handlers=[file_handler, console_handler],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def validate_file_path(self, file_path: str) -> str:
        """
        Validate and normalize file path.
        
        Args:
            file_path (str): File path to validate
            
        Returns:
            str: Normalized absolute file path
            
        Raises:
            ValueError: If file path is invalid
            FileNotFoundError: If file doesn't exist
        """
        try:
            if not file_path or not file_path.strip():
                raise ValueError("File path cannot be empty")
            
            # Normalize and get absolute path
            normalized_path = os.path.abspath(file_path.strip())
            
            # Check if file exists
            if not os.path.exists(normalized_path):
                raise FileNotFoundError(f"File not found: {normalized_path}")
            
            # Check if it's a file (not directory)
            if not os.path.isfile(normalized_path):
                raise ValueError(f"Path is not a file: {normalized_path}")
            
            # Validate file extension
            file_ext = os.path.splitext(normalized_path)[1].lower()
            if file_ext not in ['.md', '.pdf']:
                raise ValueError(f"Unsupported file type: {file_ext}. Only .md and .pdf files are supported.")
            
            self.logger.info(f"File path validated successfully: {normalized_path}")
            return normalized_path
            
        except Exception as e:
            self.logger.error(f"File path validation failed for {file_path}: {str(e)}")
            raise
    
    def check_document_exists(self, file_path: str) -> Dict[str, Any]:
        """
        Check if a document exists in the database.
        
        Args:
            file_path (str): File path to check
            
        Returns:
            Dict[str, Any]: Document existence check results
        """
        try:
            self.logger.info(f"Checking if document exists in database: {file_path}")
            
            # Get all documents from database
            list_result = self.rag_control.list_documents()
            
            if list_result["status"] != "success":
                raise RuntimeError("Failed to list documents from database")
            
            # Normalize the input file path for comparison
            normalized_input_path = os.path.abspath(file_path)
            
            # Check if file path exists in any document
            for doc in list_result["documents"]:
                # Normalize the stored file path for comparison
                stored_path = os.path.abspath(doc["file_path"])
                
                if stored_path == normalized_input_path:
                    self.logger.info(f"Document found in database: {file_path}")
                    return {
                        "exists": True,
                        "document_id": doc["document_id"],
                        "file_type": doc["file_type"],
                        "total_chunks": doc["total_chunks"],
                        "timestamp": doc["timestamp"]
                    }
            
            # If exact match not found, try matching by filename
            input_filename = os.path.basename(file_path)
            for doc in list_result["documents"]:
                stored_filename = os.path.basename(doc["file_path"])
                if stored_filename == input_filename:
                    self.logger.info(f"Document found in database by filename: {input_filename}")
                    return {
                        "exists": True,
                        "document_id": doc["document_id"],
                        "file_type": doc["file_type"],
                        "total_chunks": doc["total_chunks"],
                        "timestamp": doc["timestamp"]
                    }
            
            self.logger.info(f"Document not found in database: {file_path}")
            return {
                "exists": False,
                "file_path": file_path
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check document existence for {file_path}: {str(e)}")
            raise
    
    def delete_document(self, file_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Delete a document from the RAG database.
        
        Args:
            file_path (str): Path to the document file to delete
            force (bool): Force deletion without confirmation
            
        Returns:
            Dict[str, Any]: Deletion results
        """
        try:
            self.logger.info(f"Starting document deletion process: {file_path}")
            
            # Validate file path
            validated_path = self.validate_file_path(file_path)
            
            # Check if document exists in database
            existence_check = self.check_document_exists(validated_path)
            
            if not existence_check["exists"]:
                return {
                    "status": "not_found",
                    "file_path": validated_path,
                    "message": f"Document not found in database: {validated_path}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Log document details
            self.logger.info(f"Document found - ID: {existence_check['document_id']}, "
                           f"Type: {existence_check['file_type']}, "
                           f"Chunks: {existence_check['total_chunks']}")
            
            # Confirm deletion unless forced
            if not force:
                print(f"\nDocument to be deleted:")
                print(f"  File: {validated_path}")
                print(f"  Type: {existence_check['file_type']}")
                print(f"  Chunks: {existence_check['total_chunks']}")
                print(f"  Added: {existence_check['timestamp']}")
                
                confirm = input("\nAre you sure you want to delete this document? (y/N): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    self.logger.info(f"Document deletion cancelled by user: {validated_path}")
                    return {
                        "status": "cancelled",
                        "file_path": validated_path,
                        "message": "Deletion cancelled by user",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Perform deletion using RAGService
            deletion_result = self.rag_control.delete_document_by_file_path(validated_path)
            
            if deletion_result["status"] == "success":
                self.logger.info(f"Document successfully deleted: {validated_path}")
                return {
                    "status": "success",
                    "file_path": validated_path,
                    "chunks_deleted": deletion_result["chunks_deleted"],
                    "document_ids": deletion_result["document_ids"],
                    "message": f"Document successfully deleted with {deletion_result['chunks_deleted']} chunks",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                self.logger.warning(f"Document deletion returned unexpected status: {deletion_result}")
                return deletion_result
            
        except Exception as e:
            error_msg = f"Failed to delete document {file_path}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            return {
                "status": "error",
                "file_path": file_path,
                "error": str(e),
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def delete_multiple_documents(self, file_paths: List[str], force: bool = False) -> Dict[str, Any]:
        """
        Delete multiple documents from the RAG database.
        
        Args:
            file_paths (List[str]): List of file paths to delete
            force (bool): Force deletion without confirmation
            
        Returns:
            Dict[str, Any]: Batch deletion results
        """
        try:
            self.logger.info(f"Starting batch deletion of {len(file_paths)} documents")
            
            results = {
                "status": "batch_completed",
                "total_files": len(file_paths),
                "successful_deletions": 0,
                "failed_deletions": 0,
                "not_found": 0,
                "cancelled": 0,
                "results": [],
                "timestamp": datetime.now().isoformat()
            }
            
            for file_path in file_paths:
                try:
                    result = self.delete_document(file_path, force=force)
                    results["results"].append(result)
                    
                    if result["status"] == "success":
                        results["successful_deletions"] += 1
                    elif result["status"] == "not_found":
                        results["not_found"] += 1
                    elif result["status"] == "cancelled":
                        results["cancelled"] += 1
                    else:
                        results["failed_deletions"] += 1
                        
                except Exception as e:
                    error_result = {
                        "status": "error",
                        "file_path": file_path,
                        "error": str(e),
                        "message": f"Unexpected error during deletion: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    results["results"].append(error_result)
                    results["failed_deletions"] += 1
                    self.logger.error(f"Unexpected error deleting {file_path}: {str(e)}")
            
            self.logger.info(f"Batch deletion completed - "
                           f"Success: {results['successful_deletions']}, "
                           f"Failed: {results['failed_deletions']}, "
                           f"Not Found: {results['not_found']}, "
                           f"Cancelled: {results['cancelled']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to perform batch deletion: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            return {
                "status": "error",
                "error": str(e),
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        try:
            self.logger.info("Retrieving database statistics")
            stats = self.rag_control.get_database_stats()
            self.logger.info(f"Database stats retrieved successfully")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            raise
    
    def list_documents(self) -> Dict[str, Any]:
        """
        List all documents in the database.
        
        Returns:
            Dict[str, Any]: List of all documents
        """
        try:
            self.logger.info("Listing all documents in database")
            documents = self.rag_control.list_documents()
            self.logger.info(f"Successfully listed {documents['documents_count']} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to list documents: {str(e)}")
            raise


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="RAG Delete - Professional document deletion tool for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ragdelete.py document.md
  python ragdelete.py document.pdf --force
  python ragdelete.py --list
  python ragdelete.py --stats
        """
    )
    
    parser.add_argument(
        "file_paths",
        nargs="*",
        help="File paths to delete from the RAG database"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force deletion without confirmation"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all documents in the database"
    )
    
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show database statistics"
    )
    
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path to ChromaDB database (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAGDelete
        rag_delete = RAGDelete(db_path=args.db_path, log_level=args.log_level)
        
        # Handle different operations
        if args.list:
            print("\n=== Documents in Database ===")
            documents = rag_delete.list_documents()
            if documents["status"] == "success":
                for doc in documents["documents"]:
                    print(f"  {doc['file_path']} ({doc['file_type']}) - {doc['total_chunks']} chunks")
            else:
                print(f"Error listing documents: {documents.get('message', 'Unknown error')}")
        
        elif args.stats:
            print("\n=== Database Statistics ===")
            stats = rag_delete.get_database_stats()
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Unique documents: {stats['unique_documents']}")
            print(f"  Total content size: {stats['total_content_size_bytes']:,} bytes")
            print(f"  Database path: {stats['database_path']}")
        
        elif args.file_paths:
            if len(args.file_paths) == 1:
                # Single file deletion
                result = rag_delete.delete_document(args.file_paths[0], force=args.force)
                print(f"\n=== Deletion Result ===")
                print(f"Status: {result['status']}")
                print(f"File: {result['file_path']}")
                print(f"Message: {result['message']}")
                
                if result['status'] == 'success':
                    print(f"Chunks deleted: {result['chunks_deleted']}")
            else:
                # Multiple file deletion
                result = rag_delete.delete_multiple_documents(args.file_paths, force=args.force)
                print(f"\n=== Batch Deletion Result ===")
                print(f"Total files: {result['total_files']}")
                print(f"Successful: {result['successful_deletions']}")
                print(f"Failed: {result['failed_deletions']}")
                print(f"Not found: {result['not_found']}")
                print(f"Cancelled: {result['cancelled']}")
                
                if args.verbose:
                    print("\nDetailed results:")
                    for res in result['results']:
                        print(f"  {res['file_path']}: {res['status']} - {res['message']}")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    