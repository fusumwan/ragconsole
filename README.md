# ragconsole

A comprehensive RAG (Retrieval-Augmented Generation) system for document processing, storage, and semantic search using ChromaDB.

## Features

- Document ingestion and processing (Markdown and PDF)
- Semantic search and retrieval
- Vector database storage with ChromaDB
- Command-line interface for all operations
- Comprehensive error handling and logging

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the system

## Usage

- `python ragstudy.py md <filename>` - Study and store a markdown document
- `python ragstudy.py pdf <filename>` - Study and store a PDF document
- `python ragquery.py "search query"` - Search documents
- `python ragquery.py --list` - List all stored documents
- `python ragquery.py --stats` - Show database statistics
- `python ragdelete.py <filename>` - Delete a document

## Dependencies

- chromadb
- sentence-transformers
- pdfplumber
- And other dependencies listed in requirements.txt
