#!/usr/bin/env python3
"""
RAG Query Module
Provides functionality to query documents stored in ChromaDB database.
"""

import sys
import os
import argparse
import logging
from typing import Optional, List
from pathlib import Path

# Import the RAGControl class
from ragcontrol import RAGControl


class RAGQuery:
    """
    RAG Query class for searching and retrieving information from stored documents.
    Provides command-line interface and comprehensive error handling.
    """
    
    def __init__(self):
        """Initialize RAGQuery with logging setup."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.rag_control = None
    
    def setup_logging(self):
        """Setup logging configuration for RAGQuery."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_query.log'),
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
    
    def validate_query(self, query: str) -> bool:
        """
        Validate the search query.
        
        Args:
            query (str): Search query to validate
            
        Returns:
            bool: True if query is valid, False otherwise
        """
        try:
            if not query or not query.strip():
                self.logger.error("Query cannot be empty")
                return False
            
            if len(query.strip()) < 3:
                self.logger.warning("Query is very short, may not return relevant results")
            
            self.logger.info(f"Query validation successful: '{query}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Query validation failed: {str(e)}")
            return False
    
    def search_documents(self, query: str, document_filter: Optional[str] = None, 
                        n_results: int = 5) -> bool:
        """
        Search for relevant content in stored documents.
        
        Args:
            query (str): Search query
            document_filter (Optional[str]): Filter by specific document
            n_results (int): Number of results to return
            
        Returns:
            bool: True if search successful, False otherwise
        """
        try:
            if not self.rag_control:
                self.logger.error("RAGControl not initialized")
                return False
            
            # Validate query
            if not self.validate_query(query):
                return False
            
            # Perform search
            result = self.rag_control.search_documents(query, document_filter, n_results)
            
            # Print results
            if result['status'] == 'success':
                print(f"\n🔍 Search Results for: '{query}'")
                if document_filter:
                    print(f"📄 Filtered by: {document_filter}")
                print(f"📊 Found {result['results_count']} relevant chunks\n")
                
                if result['results_count'] == 0:
                    print("❌ No relevant results found.")
                    print("💡 Try:")
                    print("   - Using different keywords")
                    print("   - Checking if documents are stored in the database")
                    print("   - Using --list to see available documents")
                    return True
                
                # Display results
                for i, search_result in enumerate(result['results'], 1):
                    print(f"📝 Result {i}:")
                    print(f"   📄 Source: {search_result['metadata']['file_path']}")
                    print(f"   📊 Chunk: {search_result['metadata']['chunk_index'] + 1}/{search_result['metadata']['total_chunks']}")
                    if search_result['distance'] is not None:
                        print(f"   🎯 Relevance: {1 - search_result['distance']:.3f}")
                    print(f"   📅 Added: {search_result['metadata']['timestamp']}")
                    print(f"   📖 Content:")
                    
                    # Format and display content
                    content = search_result['content']
                    # Truncate if too long
                    if len(content) > 500:
                        content = content[:500] + "..."
                    
                    # Split into lines and indent
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip():
                            print(f"      {line}")
                    print()
                
                return True
            else:
                self.logger.error(f"Search failed with status: {result.get('status', 'unknown')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to search documents: {str(e)}")
            print(f"❌ Error searching documents: {str(e)}")
            return False
    
    def list_available_documents(self) -> bool:
        """
        List all documents available for querying.
        
        Returns:
            bool: True if listing successful, False otherwise
        """
        try:
            if not self.rag_control:
                self.logger.error("RAGControl not initialized")
                return False
            
            result = self.rag_control.list_documents()
            
            if result['status'] == 'success':
                if result['documents_count'] == 0:
                    print("📚 No documents found in database.")
                    print("💡 Use 'python ragstudy.py md <filename>' to add documents first.")
                    return True
                
                print(f"\n📚 Available Documents ({result['documents_count']}):")
                print("=" * 60)
                
                for i, doc in enumerate(result['documents'], 1):
                    print(f"{i:2d}. 📄 {doc['file_path']}")
                    print(f"     📊 ID: {doc['document_id']}")
                    print(f"     📝 Chunks: {doc['total_chunks']}")
                    print(f"     📁 Type: {doc['file_type']}")
                    print(f"     ⏰ Added: {doc['timestamp']}")
                    print()
                
                print("💡 Usage examples:")
                print("   python ragquery.py 'your search query'")
                print("   python ragquery.py 'your search query' --filter alice_in_wonderland")
                print("   python ragquery.py 'your search query' --filter doc_abc123")
                
                return True
            else:
                self.logger.error("Failed to list documents")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to list documents: {str(e)}")
            print(f"❌ Error listing documents: {str(e)}")
            return False
    
    def get_database_stats(self) -> bool:
        """
        Get and display database statistics.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.rag_control:
                self.logger.error("RAGControl not initialized")
                return False
            
            stats = self.rag_control.get_database_stats()
            
            print("\n📊 Database Statistics:")
            print("=" * 40)
            print(f"📄 Total documents: {stats['unique_documents']}")
            print(f"📝 Total chunks: {stats['total_chunks']}")
            print(f"💾 Total content size: {stats['total_content_size_bytes']:,} bytes")
            print(f"🗂️  Database path: {stats['database_path']}")
            print(f"📚 Collection name: {stats['collection_name']}")
            
            if stats['unique_documents'] == 0:
                print("\n💡 No documents found. Add documents first:")
                print("   python ragstudy.py md alice_in_wonderland.md")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            print(f"❌ Error getting database stats: {str(e)}")
            return False
    
    def interactive_search(self):
        """Start interactive search mode."""
        try:
            print("\n🔍 Interactive Search Mode")
            print("=" * 40)
            print("Type 'quit' or 'exit' to end interactive mode")
            print("Type 'list' to see available documents")
            print("Type 'stats' to see database statistics")
            print()
            
            while True:
                try:
                    query = input("🔍 Enter your search query: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    elif query.lower() == 'list':
                        self.list_available_documents()
                        continue
                    elif query.lower() == 'stats':
                        self.get_database_stats()
                        continue
                    elif not query:
                        print("❌ Please enter a search query")
                        continue
                    
                    # Ask for document filter
                    filter_choice = input("📄 Filter by specific document? (optional, press Enter to skip): ").strip()
                    document_filter = filter_choice if filter_choice else None
                    
                    # Ask for number of results
                    try:
                        n_results_input = input("📊 Number of results (default 5): ").strip()
                        n_results = int(n_results_input) if n_results_input else 5
                    except ValueError:
                        n_results = 5
                    
                    # Perform search
                    self.search_documents(query, document_filter, n_results)
                    print("\n" + "=" * 60 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
                    
        except Exception as e:
            self.logger.error(f"Interactive search failed: {str(e)}")
            print(f"❌ Interactive search error: {str(e)}")


def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(
        description="RAG Query - Search documents stored in ChromaDB database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ragquery.py "Please tell me about Alice"
  python ragquery.py "What happened in the rabbit hole" --filter alice_in_wonderland
  python ragquery.py "Tell me about the Queen" --results 10
  python ragquery.py --list
  python ragquery.py --stats
  python ragquery.py --interactive
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Search query'
    )
    
    parser.add_argument(
        '--filter',
        dest='document_filter',
        help='Filter search by specific document ID or filename'
    )
    
    parser.add_argument(
        '--results',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
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
        '--interactive',
        action='store_true',
        help='Start interactive search mode'
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
    
    # Initialize RAGQuery
    rag_query = RAGQuery()
    
    # Initialize RAGControl
    if not rag_query.initialize_rag_control(args.db_path):
        print("❌ Failed to initialize RAG system")
        sys.exit(1)
    
    try:
        # Handle different command modes
        if args.interactive:
            rag_query.interactive_search()
        elif args.list:
            rag_query.list_available_documents()
        elif args.stats:
            rag_query.get_database_stats()
        elif args.query:
            # Perform search
            success = rag_query.search_documents(
                args.query, 
                args.document_filter, 
                args.results
            )
            if not success:
                sys.exit(1)
        else:
            print("❌ Please provide a search query, or use --list, --stats, or --interactive")
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