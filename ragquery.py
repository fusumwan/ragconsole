# ragquery.py
# ------------------------------------------------------------------------------
# CLI to query a Chroma collection with a chosen embedding method.
# Usage examples:
#   python ragquery.py "rabbit hole" --embedding Sentence-Transformers
#   python ragquery.py "rabbit hole" --embedding OpenAIEmbeddings
# ------------------------------------------------------------------------------

import argparse
import json
import logging
import os
import sys
from typing import Optional, Dict, Any, List

# Your service with pluggable embeddings (alias kept as RAGService)
from ragcontrolservice import RAGService


class RAGQuery:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rag_control: Optional[RAGService] = None

    def initialize_rag_control(
        self,
        db_path: str = "./chroma_db",
        embedding_method: str = "Sentence-Transformers",
        sentence_model_name: str = "all-MiniLM-L6-v2",
        openai_model_name: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
    ) -> bool:
        """
        Initialize the RAG service with the selected embedding method so that
        we query the correct per-embedding collection.
        """
        try:
            self.logger.info(
                f"Initializing RAGService at '{db_path}' with embedding '{embedding_method}'"
            )
            self.rag_control = RAGService(
                db_path=db_path,
                embedding_method=embedding_method,
                sentence_model_name=sentence_model_name,
                openai_model_name=openai_model_name,
                openai_api_key=openai_api_key,  # service will also read env if None
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize RAGService: {str(e)}")
            return False

    def search(
        self,
        query: str,
        document_filter: Optional[str] = None,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform a semantic search against the active collection.
        """
        assert self.rag_control is not None, "RAG service not initialized"
        return self.rag_control.search_documents(
            query=query,
            document_filter=document_filter,
            n_results=n_results,
        )


def configure_logging(verbosity: int):
    """
    Configure logging level based on -v occurrences (0: WARNING, 1: INFO, 2+: DEBUG)
    """
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG Query CLI - Query a Chroma collection with a chosen embedding method."
    )

    # Positional: the query text
    parser.add_argument(
        "query",
        help="Search query text.",
    )

    # Embedding selection (matches study-side options)
    parser.add_argument(
        "--embedding",
        choices=["Sentence-Transformers", "OpenAIEmbeddings"],
        default="Sentence-Transformers",
        help="Embedding method/collection to query (default: Sentence-Transformers).",
    )

    # Optional filters & parameters
    parser.add_argument(
        "--filter",
        dest="document_filter",
        default=None,
        help="Filter results to a specific document_id (starts with 'doc_') or an exact file path.",
    )
    parser.add_argument(
        "--n",
        dest="n_results",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )

    # DB path and embedding backend config
    parser.add_argument(
        "--db",
        dest="db_path",
        default="./chroma_db",
        help="Path to Chroma persistent directory (default: ./chroma_db).",
    )
    parser.add_argument(
        "--openai-api-key",
        dest="openai_api_key",
        default=None,
        help="OpenAI API key (optional; if omitted, will use OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--sentence-model",
        dest="sentence_model_name",
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--openai-model",
        dest="openai_model_name",
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small).",
    )

    # Output & logging
    parser.add_argument(
        "-v",
        dest="verbosity",
        action="count",
        default=1,
        help="Increase logging verbosity (-v: INFO, -vv: DEBUG).",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print raw JSON results only.",
    )

    return parser


def _print_human(results: Dict[str, Any]):
    # Pretty, human-readable output
    status = results.get("status")
    collection = results.get("collection_name")
    embedding = results.get("embedding_method")
    count = results.get("results_count", 0)
    query = results.get("query")

    print("\n=== RAG Query Results ===")
    print(f"Status     : {status}")
    print(f"Query      : {query}")
    print(f"Collection : {collection}")
    print(f"Embedding  : {embedding}")
    print(f"Matches    : {count}")

    items: List[Dict[str, Any]] = results.get("results", [])
    for idx, item in enumerate(items, start=1):
        chunk_id = item.get("chunk_id")
        distance = item.get("distance")
        content = item.get("content", "")
        snippet = content[:200].replace("\n", " ") + ("..." if len(content) > 200 else "")

        meta = item.get("metadata", {}) or {}
        doc_id = meta.get("document_id")
        file_path = meta.get("file_path")
        chunk_index = meta.get("chunk_index")
        total_chunks = meta.get("total_chunks")

        print("\n---")
        print(f"#{idx}  chunk_id   : {chunk_id}")
        if distance is not None:
            print(f"    distance  : {distance:.6f}")
        print(f"    document  : {doc_id}")
        print(f"    file_path : {file_path}")
        print(f"    chunk     : {chunk_index}/{total_chunks - 1 if isinstance(total_chunks, int) and total_chunks > 0 else '?'}")
        print(f"    snippet   : {snippet}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.verbosity)
    log = logging.getLogger("ragquery")

    # Initialize query controller
    rq = RAGQuery()
    ok = rq.initialize_rag_control(
        db_path=args.db_path,
        embedding_method=args.embedding,
        sentence_model_name=args.sentence_model_name,
        openai_model_name=args.openai_model_name,
        openai_api_key=args.openai_api_key,  # service will fall back to env if None
    )
    if not ok:
        log.error("Failed to initialize RAG service.")
        sys.exit(2)

    # Execute search
    try:
        results = rq.search(
            query=args.query,
            document_filter=args.document_filter,
            n_results=args.n_results,
        )
        if args.json_output:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            _print_human(results)
    except Exception as e:
        log.exception(f"Query failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
