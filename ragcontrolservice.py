# ragcontrolservice.py

# ------------------------------------------------------------------------------
# Environment: set ONNX Runtime provider early (pre-Chroma import) for macOS
# ------------------------------------------------------------------------------
import os
os.environ.setdefault("ONNXRUNTIME_PROVIDER_LIST", "CPUExecutionProvider")

"""
================================================================================
RAG Control Service (Pluggable Embeddings)
================================================================================

SYSTEM OVERVIEW:
RAGControlService is a Retrieval-Augmented Generation (RAG) service that manages
document storage, processing, and semantic search using ChromaDB as the vector
database backend. It supports multi-format ingestion (Markdown/PDF), chunking,
metadata management, and pluggable embedding backends.

NEW (this version):
- Pluggable embeddings via `embedding_method`:
  * "Sentence-Transformers" (local, no API needed; default model all-MiniLM-L6-v2)
  * "OpenAIEmbeddings" (cloud; default model text-embedding-3-small)
- Separate Chroma collections per embedding method to avoid vector dimension
  collisions and allow side-by-side A/B testing.
- Easy extension: add new embedding methods in `build_embedding_function`.

USAGE:
# Sentence-Transformers (local)
python ragstudy.py Sentence-Transformers md alice_in_wonderland.md

# OpenAI (requires OPENAI_API_KEY)
python ragstudy.py OpenAIEmbeddings md alice_in_wonderland.md

ARCHITECTURE COMPONENTS:
1) Document Processing Layer  : format detection, extraction, chunking
2) Vector Database Layer      : Chroma persistent storage, collections
3) Search & Retrieval Layer   : semantic similarity search w/ optional filters
4) Data Management Layer      : lifecycle ops (list/delete/stats)
5) Logging & Monitoring Layer : structured logging

DEPENDENCIES:
- chromadb
- pdfplumber
- sentence-transformers (if using Sentence-Transformers)
- OpenAI key (if using OpenAIEmbeddings)

AUTHOR: RAG System Development Team
VERSION: 2.0 (Pluggable Embeddings)
LAST UPDATED: 2025-08-14
================================================================================
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import sys
import logging
import hashlib
from typing import List, Dict, Optional, Any
from pathlib import Path
import re
from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()  # will read .env from current directory by default

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
import pdfplumber
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ------------------------------------------------------------------------------
# Supported Embeddings & Factory
# ------------------------------------------------------------------------------
SUPPORTED_EMBEDDINGS = ("Sentence-Transformers", "OpenAIEmbeddings")


def _safe_collection_suffix(name: str) -> str:
    """Sanitize a name to be safe in a Chroma collection suffix."""
    return re.sub(r'[^A-Za-z0-9_]+', '_', name.strip().replace(' ', '_'))


def build_embedding_function(
    embedding_method: str,
    *,
    sentence_model_name: str = "all-MiniLM-L6-v2",
    openai_api_key: Optional[str] = None,
    openai_model_name: str = "text-embedding-3-small",
):
    """
    Build and return a Chroma embedding function for the requested method.

    Extend by adding new `elif` branches for additional providers.
    """
    if embedding_method == "Sentence-Transformers":
        # Local, fast, 384-dim, no API key required.
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=sentence_model_name
        )

    elif embedding_method == "OpenAIEmbeddings":
        # Requires OpenAI API key; reads from parameter or env var.
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OpenAIEmbeddings selected but no OPENAI_API_KEY provided "
                "(env var or constructor arg)."
            )
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=openai_model_name,
        )

    raise ValueError(
        f"Unsupported embedding method: {embedding_method}. "
        f"Supported: {SUPPORTED_EMBEDDINGS}"
    )


class RAGControlService:
    """
    RAG Control Service class for managing document storage and retrieval using ChromaDB.
    Handles both markdown and PDF files with professional error handling and logging.

    Pluggable embedding backends (Sentence-Transformers / OpenAIEmbeddings).
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        embedding_method: str = "Sentence-Transformers",
        *,
        # Optional model overrides:
        sentence_model_name: str = "all-MiniLM-L6-v2",
        openai_model_name: str = "text-embedding-3-small",
        # Optional API keys:
        openai_api_key: Optional[str] = None,
        # Chroma settings:
        anonymized_telemetry: bool = False,
        allow_reset: bool = True,
    ):
        """
        Initialize RAGControlService with ChromaDB connection and logging setup.

        Args:
            db_path: ChromaDB persistent path.
            embedding_method: One of SUPPORTED_EMBEDDINGS.
            sentence_model_name: Model for Sentence-Transformers.
            openai_model_name: Model for OpenAI embeddings.
            openai_api_key: OpenAI key (if using OpenAIEmbeddings).
            anonymized_telemetry: Chroma telemetry flag.
            allow_reset: Allow Chroma reset operations.
        """
        self.db_path = db_path
        self.embedding_method = embedding_method
        self.sentence_model_name = sentence_model_name
        self.openai_model_name = openai_model_name

        # Setup logging before DB ops
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        try:
            os.makedirs(self.db_path, exist_ok=True)

            # Persistent Chroma client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=anonymized_telemetry,
                    allow_reset=allow_reset,
                ),
            )

            # Build embedding function
            ef = build_embedding_function(
                embedding_method=self.embedding_method,
                sentence_model_name=self.sentence_model_name,
                openai_api_key=openai_api_key,
                openai_model_name=self.openai_model_name,
            )

            # Use a per-embedding-method collection to prevent vector size collisions
            suffix = _safe_collection_suffix(self.embedding_method)
            self.collection_name = f"documents_{suffix}"
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "RAG document storage",
                    "embedding_method": self.embedding_method,
                    "sentence_model_name": self.sentence_model_name,
                    "openai_model_name": self.openai_model_name,
                },
                embedding_function=ef,
            )

            self.logger.info(
                f"RAGControlService initialized at '{self.db_path}' "
                f"using collection '{self.collection_name}' "
                f"with embedding '{self.embedding_method}'."
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize RAGControlService: {str(e)}")
            raise RuntimeError(f"Database initialization failed: {str(e)}") from e

    # --------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------
    def setup_logging(self):
        """Setup comprehensive logging configuration for the entire system."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("rag_system.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    # --------------------------------------------------------------------------
    # Utility: IDs & File IO
    # --------------------------------------------------------------------------
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID based on absolute file path hash."""
        try:
            file_path = os.path.abspath(file_path)
            file_hash = hashlib.md5(file_path.encode()).hexdigest()
            return f"doc_{file_hash}"
        except Exception as e:
            self.logger.error(f"Failed to generate document ID for {file_path}: {str(e)}")
            raise

    def _read_markdown_file(self, file_path: str) -> str:
        """Read and return markdown file content."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Markdown file not found: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.logger.info(f"Successfully read markdown file: {file_path}")
            return content
        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error reading markdown file {file_path}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to read markdown file {file_path}: {str(e)}")
            raise

    def _read_pdf_file(self, file_path: str) -> str:
        """Read and return extracted text from PDF using pdfplumber."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")

            content = ""
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to extract text from page {page_num + 1}: {str(e)}"
                        )
                        continue

            if not content.strip():
                raise ValueError(f"No text content extracted from PDF: {file_path}")

            self.logger.info(f"Successfully read PDF file: {file_path}")
            return content

        except Exception as e:
            self.logger.error(f"Failed to read PDF file {file_path}: {str(e)}")
            raise

    # --------------------------------------------------------------------------
    # Text Chunking
    # --------------------------------------------------------------------------
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.

        Attempts to break near sentence boundaries if a boundary occurs
        after ~70% of the current window.
        """
        try:
            if not text.strip():
                return []

            text = re.sub(r"\s+", " ", text.strip())  # normalize whitespace

            chunks: List[str] = []
            start = 0
            L = len(text)

            while start < L:
                end = start + chunk_size
                chunk = text[start:end]

                if end < L:
                    last_period = chunk.rfind(".")
                    last_exclamation = chunk.rfind("!")
                    last_question = chunk.rfind("?")
                    last_newline = chunk.rfind("\n")
                    break_point = max(last_period, last_exclamation, last_question, last_newline)

                    if break_point > chunk_size * 0.7:
                        chunk = chunk[: break_point + 1]
                        end = start + break_point + 1

                chunks.append(chunk.strip())
                start = end - overlap
                if start >= L:
                    break

            self.logger.info(f"Created {len(chunks)} chunks from text")
            return chunks

        except Exception as e:
            self.logger.error(f"Failed to chunk text: {str(e)}")
            raise

    # --------------------------------------------------------------------------
    # Existence Check
    # --------------------------------------------------------------------------
    def _check_document_exists(self, document_id: str) -> bool:
        """Return True if any chunk with this document_id exists in collection."""
        try:
            results = self.collection.get(where={"document_id": document_id}, limit=1)
            if not results or not isinstance(results, dict) or "ids" not in results:
                return False
            if not isinstance(results["ids"], list):
                return False
            return len(results["ids"]) > 0
        except Exception as e:
            self.logger.error(f"Failed to check document existence: {str(e)}")
            return False

    # --------------------------------------------------------------------------
    # Public API: Study (Ingest)
    # --------------------------------------------------------------------------
    def study_document(self, file_path: str, file_type: str = "md") -> Dict[str, Any]:
        """
        Study and store a document (markdown or PDF) in the ChromaDB database.

        Returns a summary with document_id and chunk count.
        """
        try:
            self.logger.info(
                f"Starting document study: {file_path} (type: {file_type}) "
                f"into collection '{self.collection_name}'"
            )

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            document_id = self._generate_document_id(file_path)

            if self._check_document_exists(document_id):
                self.logger.warning(f"Document already exists in database: {file_path}")
                return {
                    "status": "exists",
                    "document_id": document_id,
                    "file_path": file_path,
                    "message": "Document already studied and stored in database",
                    "collection_name": self.collection_name,
                    "embedding_method": self.embedding_method,
                }

            # Read content
            file_type_lower = file_type.lower()
            if file_type_lower == "md":
                content = self._read_markdown_file(file_path)
            elif file_type_lower == "pdf":
                content = self._read_pdf_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Chunk
            chunks = self._chunk_text(content)

            if not chunks:
                raise ValueError(f"No content extracted from file: {file_path}")

            # IDs & metadata
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            timestamp = datetime.now().isoformat()
            metadatas = [
                {
                    "document_id": document_id,
                    "file_path": os.path.abspath(file_path),
                    "file_type": file_type_lower,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": timestamp,
                    "embedding_method": self.embedding_method,
                    "collection_name": self.collection_name,
                }
                for i in range(len(chunks))
            ]

            # Add to Chroma (auto-embeds via collection's embedding_function)
            self.collection.add(documents=chunks, ids=chunk_ids, metadatas=metadatas)

            self.logger.info(
                f"Stored document {file_path} with {len(chunks)} chunks "
                f"in collection '{self.collection_name}'"
            )

            return {
                "status": "success",
                "document_id": document_id,
                "file_path": os.path.abspath(file_path),
                "chunks_count": len(chunks),
                "file_type": file_type_lower,
                "collection_name": self.collection_name,
                "embedding_method": self.embedding_method,
                "message": f"Document successfully studied and stored with {len(chunks)} chunks",
            }

        except Exception as e:
            self.logger.error(f"Failed to study document {file_path}: {str(e)}")
            raise

    # --------------------------------------------------------------------------
    # Public API: Search
    # --------------------------------------------------------------------------
    def search_documents(
        self,
        query: str,
        document_filter: Optional[str] = None,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Search for relevant chunks.

        `document_filter` can be a document_id (starts with "doc_") or an exact file_path.
        """
        try:
            self.logger.info(
                f"Searching '{self.collection_name}' with query: '{query}' "
                f"(embedding: {self.embedding_method})"
            )

            if not query or not query.strip():
                raise ValueError("Search query cannot be empty")

            where_clause = None
            if document_filter:
                if document_filter.startswith("doc_"):
                    where_clause = {"document_id": document_filter}
                else:
                    where_clause = {"file_path": os.path.abspath(document_filter)}

            results = self.collection.query(
                query_texts=[query], n_results=n_results, where=where_clause
            )

            formatted_results: List[Dict[str, Any]] = []
            if results and "documents" in results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "chunk_id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None,
                    }
                    formatted_results.append(result)

            self.logger.info(f"Search completed, found {len(formatted_results)} results")

            return {
                "status": "success",
                "query": query,
                "results_count": len(formatted_results),
                "results": formatted_results,
                "filter_applied": document_filter is not None,
                "collection_name": self.collection_name,
                "embedding_method": self.embedding_method,
            }

        except Exception as e:
            self.logger.error(f"Failed to search documents: {str(e)}")
            raise

    # --------------------------------------------------------------------------
    # Public API: List
    # --------------------------------------------------------------------------
    def list_documents(self) -> Dict[str, Any]:
        """
        List unique documents (grouped by document_id) in the active collection.
        """
        try:
            self.logger.info(
                f"Listing documents in collection '{self.collection_name}' "
                f"(embedding: {self.embedding_method})"
            )

            # Probe collection
            try:
                _ = self.collection.get(limit=1)
                results = self.collection.get()
            except Exception as e:
                self.logger.error(f"Error calling collection methods: {str(e)}")
                return {"status": "success", "documents_count": 0, "documents": []}

            if not results or not isinstance(results, dict):
                return {"status": "success", "documents_count": 0, "documents": []}

            if "ids" not in results or "metadatas" not in results:
                return {"status": "success", "documents_count": 0, "documents": []}

            if not isinstance(results["ids"], list) or not isinstance(results["metadatas"], list):
                return {"status": "success", "documents_count": 0, "documents": []}

            if len(results["ids"]) == 0:
                return {"status": "success", "documents_count": 0, "documents": []}

            documents: Dict[str, Dict[str, Any]] = {}

            for i, meta in enumerate(results["metadatas"]):
                if i < len(results["ids"]):
                    document_id = meta.get("document_id", "unknown")
                    if document_id not in documents:
                        documents[document_id] = {
                            "document_id": document_id,
                            "file_path": meta.get("file_path", "unknown"),
                            "file_type": meta.get("file_type", "unknown"),
                            "total_chunks": meta.get("total_chunks", 0),
                            "timestamp": meta.get("timestamp", "unknown"),
                            "embedding_method": meta.get("embedding_method", self.embedding_method),
                            "collection_name": meta.get("collection_name", self.collection_name),
                        }

            doc_list = list(documents.values())
            self.logger.info(f"Found {len(doc_list)} documents in collection")
            return {"status": "success", "documents_count": len(doc_list), "documents": doc_list}

        except Exception as e:
            self.logger.error(f"Failed to list documents: {str(e)}")
            raise

    # --------------------------------------------------------------------------
    # Public API: Delete by document_id
    # --------------------------------------------------------------------------
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document and all its chunks from the active collection by document_id.
        """
        try:
            self.logger.info(
                f"Deleting document '{document_id}' from collection '{self.collection_name}'"
            )

            results = self.collection.get(where={"document_id": document_id})
            if not results or not isinstance(results, dict) or "ids" not in results:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                    "message": "Document not found in database",
                }

            if not isinstance(results["ids"], list) or len(results["ids"]) == 0:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                    "message": "Document not found in database",
                }

            self.collection.delete(ids=results["ids"])
            self.logger.info(
                f"Successfully deleted document {document_id} with {len(results['ids'])} chunks"
            )

            return {
                "status": "success",
                "document_id": document_id,
                "chunks_deleted": len(results["ids"]),
                "message": f"Document successfully deleted with {len(results['ids'])} chunks",
            }

        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise

    # --------------------------------------------------------------------------
    # Public API: Delete by file path
    # --------------------------------------------------------------------------
    def delete_document_by_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a document from the active collection by its file path (exact path),
        with a filename fallback scan if exact match is not found.
        """
        try:
            self.logger.info(f"Attempting to delete document by file path: {file_path}")
            if not file_path or not file_path.strip():
                raise ValueError("File path cannot be empty")

            normalized_file_path = os.path.abspath(file_path)

            results = self.collection.get(where={"file_path": normalized_file_path})

            if not results or not isinstance(results, dict) or "ids" not in results:
                # Fallback: try filename match scan
                filename = os.path.basename(file_path)
                self.logger.info(f"No exact path match, trying filename match: {filename}")

                all_results = self.collection.get()
                if (
                    not all_results
                    or not isinstance(all_results, dict)
                    or "ids" not in all_results
                    or "metadatas" not in all_results
                ):
                    return {
                        "status": "not_found",
                        "file_path": file_path,
                        "message": f"No document found with file path or filename: {file_path}",
                    }

                if not isinstance(all_results["ids"], list) or not isinstance(
                    all_results["metadatas"], list
                ):
                    return {
                        "status": "not_found",
                        "file_path": file_path,
                        "message": f"No document found with file path or filename: {file_path}",
                    }

                matching_ids: List[str] = []
                for i, metadata in enumerate(all_results["metadatas"]):
                    if i < len(all_results["ids"]):
                        stored_filename = os.path.basename(metadata.get("file_path", ""))
                        if stored_filename == filename:
                            matching_ids.append(all_results["ids"][i])

                if matching_ids:
                    results = self.collection.get(ids=matching_ids)
                    self.logger.info(f"Found {len(matching_ids)} chunks by filename match")
                else:
                    return {
                        "status": "not_found",
                        "file_path": file_path,
                        "message": f"No document found with file path or filename: {file_path}",
                    }
            else:
                # validate ids
                if not isinstance(results["ids"], list) or len(results["ids"]) == 0:
                    # same fallback as above
                    filename = os.path.basename(file_path)
                    self.logger.info(f"No exact path match, trying filename match: {filename}")

                    all_results = self.collection.get()
                    if (
                        not all_results
                        or not isinstance(all_results, dict)
                        or "ids" not in all_results
                        or "metadatas" not in all_results
                    ):
                        return {
                            "status": "not_found",
                            "file_path": file_path,
                            "message": f"No document found with file path or filename: {file_path}",
                        }

                    if not isinstance(all_results["ids"], list) or not isinstance(
                        all_results["metadatas"], list
                    ):
                        return {
                            "status": "not_found",
                            "file_path": file_path,
                            "message": f"No document found with file path or filename: {file_path}",
                        }

                    matching_ids: List[str] = []
                    for i, metadata in enumerate(all_results["metadatas"]):
                        if i < len(all_results["ids"]):
                            stored_filename = os.path.basename(metadata.get("file_path", ""))
                            if stored_filename == filename:
                                matching_ids.append(all_results["ids"][i])

                    if matching_ids:
                        results = self.collection.get(ids=matching_ids)
                        self.logger.info(f"Found {len(matching_ids)} chunks by filename match")
                    else:
                        return {
                            "status": "not_found",
                            "file_path": file_path,
                            "message": f"No document found with file path or filename: {file_path}",
                        }

            if not isinstance(results, dict) or "ids" not in results or "metadatas" not in results:
                return {"status": "error", "file_path": file_path, "message": "Invalid results structure"}

            if not isinstance(results["ids"], list) or not isinstance(results["metadatas"], list):
                return {"status": "error", "file_path": file_path, "message": "Invalid data types in results"}

            document_ids = list({m.get("document_id", "unknown") for m in results["metadatas"]})

            self.collection.delete(ids=results["ids"])
            self.logger.info(
                f"Successfully deleted document {file_path} with {len(results['ids'])} chunks"
            )

            return {
                "status": "success",
                "file_path": file_path,
                "document_ids": document_ids,
                "chunks_deleted": len(results["ids"]),
                "message": f"Document successfully deleted with {len(results['ids'])} chunks",
            }

        except Exception as e:
            self.logger.error(f"Failed to delete document by file path {file_path}: {str(e)}")
            raise

    # --------------------------------------------------------------------------
    # Public API: Stats
    # --------------------------------------------------------------------------
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Return collection stats: total chunks, unique documents, total bytes, etc.
        (For the active collection only.)
        """
        try:
            try:
                _ = self.collection.get(limit=1)
                results = self.collection.get()
            except Exception as e:
                self.logger.error(f"Error calling collection methods: {str(e)}")
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_content_size_bytes": 0,
                    "database_path": self.db_path,
                    "collection_name": self.collection_name,
                    "embedding_method": self.embedding_method,
                }

            if (
                not results
                or not isinstance(results, dict)
                or "ids" not in results
                or "metadatas" not in results
                or "documents" not in results
            ):
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_content_size_bytes": 0,
                    "database_path": self.db_path,
                    "collection_name": self.collection_name,
                    "embedding_method": self.embedding_method,
                }

            if (
                not isinstance(results["ids"], list)
                or not isinstance(results["metadatas"], list)
                or not isinstance(results["documents"], list)
            ):
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_content_size_bytes": 0,
                    "database_path": self.db_path,
                    "collection_name": self.collection_name,
                    "embedding_method": self.embedding_method,
                }

            if len(results["ids"]) == 0:
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_content_size_bytes": 0,
                    "database_path": self.db_path,
                    "collection_name": self.collection_name,
                    "embedding_method": self.embedding_method,
                }

            total_chunks = len(results["ids"])
            unique_documents = len({m.get("document_id", "unknown") for m in results["metadatas"]})
            total_content_size = sum(len(doc) for doc in results["documents"])

            stats = {
                "total_chunks": total_chunks,
                "unique_documents": unique_documents,
                "total_content_size_bytes": total_content_size,
                "database_path": self.db_path,
                "collection_name": self.collection_name,
                "embedding_method": self.embedding_method,
            }
            self.logger.info(f"Database stats: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            raise


# ------------------------------------------------------------------------------
# Backward-compatibility alias (some CLIs may import RAGService)
# ------------------------------------------------------------------------------
RAGService = RAGControlService
