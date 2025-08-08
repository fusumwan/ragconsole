#!/usr/bin/env python3
"""
RAG Study Module
Provides functionality to study and store documents (markdown/PDF) in ChromaDB database.
"""

import sys
import os
import argparse
import logging
from typing import Optional
from pathlib import Path

# Import the RAGControl class
from ragcontrol import RAGControl


class RAGStudy:
    """
    RAG Study class for processing and storing documents in the RAG system.
    Provides command-line interface and comprehensive error handling.
    """
    
    def __init__(self):
        """Initialize RAGStudy with logging setup."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.rag_control = None
    
    def setup_logging(self):
        """Setup logging configuration for RAGStudy."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_study.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def initialize_rag_control(self, db_path: str = "./chroma_db") -> bool:
        """
        Initialize the RAGControl instance.
        
        Args:
            db_path (str): Path to ChromaDB database directory
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing RAGControl with database path: {db_path}")
            self.rag_control = RAGControl(db_path=db_path)
            self.logger.info("RAGControl initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize RAGControl: {str(e)}")
            return False
    
    def validate_file(self, file_path: str, file_type: str) -> bool:
        """
        Validate that the file exists and is of the correct type.
        
        Args:
            file_path (str): Path to the file
            file_type (str): Expected file type ('md' or 'pdf')
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False
            
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_type.lower() == "md" and file_ext not in ['.md', '.markdown']:
                self.logger.error(f"File {file_path} does not have markdown extension")
                return False
            elif file_type.lower() == "pdf" and file_ext != '.pdf':
                self.logger.error(f"File {file_path} does not have PDF extension")
                return False
            
            # Check if file is readable
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_type.lower() == "md":
                        f.read(1024)  # Read a small sample
            except UnicodeDecodeError:
                # For PDF files, we don't need to check encoding
                if file_type.lower() == "md":
                    self.logger.error(f"File {file_path} is not readable as text")
                    return False
            
            self.logger.info(f"File validation successful: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"File validation failed for {file_path}: {str(e)}")
            return False
    
    def study_document(self, file_path: str, file_type: str) -> bool:
        """
        Study and store a document using RAGControl.
        
        Args:
            file_path (str): Path to the document file
            file_type (str): Type of file ('md' or 'pdf')
            
        Returns:
            bool: True if study successful, False otherwise
        """
        try:
            if not self.rag_control:
                self.logger.error("RAGControl not initialized")
                return False
            
            # Validate file
            if not self.validate_file(file_path, file_type):
                return False
            
            # Study the document
            result = self.rag_control.study_document(file_path, file_type)
            
            # Print results
            if result['status'] == 'success':
                print(f"✅ Successfully studied document: {file_path}")
                print(f"   📄 Document ID: {result['document_id']}")
                print(f"   📊 Chunks created: {result['chunks_count']}")
                print(f"   📁 File type: {result['file_type']}")
                print(f"   💾 Message: {result['message']}")
                return True
            elif result['status'] == 'exists':
                print(f"ℹ️  Document already exists: {file_path}")
                print(f"   📄 Document ID: {result['document_id']}")
                print(f"   💾 Message: {result['message']}")
                return True
            else:
                self.logger.error(f"Study failed with status: {result.get('status', 'unknown')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to study document {file_path}: {str(e)}")
            print(f"❌ Error studying document: {str(e)}")
            return False
    
    def print_database_stats(self):
        """Print current database statistics."""
        try:
            if not self.rag_control:
                print("❌ RAGControl not initialized")
                return
            
            stats = self.rag_control.get_database_stats()
            print("\n📊 Database Statistics:")
            print(f"   📄 Total documents: {stats['unique_documents']}")
            print(f"   📝 Total chunks: {stats['total_chunks']}")
            print(f"   💾 Total content size: {stats['total_content_size_bytes']:,} bytes")
            print(f"   🗂️  Database path: {stats['database_path']}")
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            print(f"❌ Error getting database stats: {str(e)}")
    
    def list_stored_documents(self):
        """List all documents currently stored in the database."""
        try:
            if not self.rag_control:
                print("❌ RAGControl not initialized")
                return
            
            result = self.rag_control.list_documents()
            
            if result['status'] == 'success':
                print(f"\n📚 Stored Documents ({result['documents_count']}):")
                for i, doc in enumerate(result['documents'], 1):
                    print(f"   {i}. {doc['file_path']}")
                    print(f"      📄 ID: {doc['document_id']}")
                    print(f"      📊 Chunks: {doc['total_chunks']}")
                    print(f"      📁 Type: {doc['file_type']}")
                    print(f"      ⏰ Added: {doc['timestamp']}")
                    print()
            else:
                print("❌ Failed to list documents")
                
        except Exception as e:
            self.logger.error(f"Failed to list documents: {str(e)}")
            print(f"❌ Error listing documents: {str(e)}")


def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(
        description="RAG Study - Study and store documents in ChromaDB database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ragstudy.py md alice_in_wonderland.md
  python ragstudy.py pdf document.pdf
  python ragstudy.py --list
  python ragstudy.py --stats
        """
    )
    
    parser.add_argument(
        'file_type',
        nargs='?',
        choices=['md', 'pdf'],
        help='Type of file to study (md or pdf)'
    )
    
    parser.add_argument(
        'file_path',
        nargs='?',
        help='Path to the file to study'
    )
    
    parser.add_argument(
        '--db-path',
        default='./chroma_db',
        help='Path to ChromaDB database directory (default: ./chroma_db)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all stored documents'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize RAGStudy
    rag_study = RAGStudy()
    
    # Initialize RAGControl
    if not rag_study.initialize_rag_control(args.db_path):
        print("❌ Failed to initialize RAG system")
        sys.exit(1)
    
    try:
        # Handle different command modes
        if args.list:
            rag_study.list_stored_documents()
        elif args.stats:
            rag_study.print_database_stats()
        elif args.file_type and args.file_path:
            # Study a document
            success = rag_study.study_document(args.file_path, args.file_type)
            if success:
                print("\n📊 Current database statistics:")
                rag_study.print_database_stats()
            else:
                sys.exit(1)
        else:
            print("❌ Please provide file type and file path, or use --list or --stats")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()