# ragstudy.py
# ------------------------------------------------------------------------------
# CLI to ingest (study) a document into Chroma with a chosen embedding method.
# Usage examples:
#   python ragstudy.py Sentence-Transformers md alice_in_wonderland.md
#   python ragstudy.py OpenAIEmbeddings   md alice_in_wonderland.md
# ------------------------------------------------------------------------------

import argparse
import json
import logging
import os
import sys
from typing import Optional

# Your service with pluggable embeddings
from ragcontrolservice import RAGService


class RAGStudy:
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
        Initialize the RAG service with the selected embedding method.
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
                openai_api_key=openai_api_key,  # if None, service will read from env
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize RAGService: {str(e)}")
            return False

    def study_document(self, file_path: str, file_type: str) -> dict:
        """
        Ingest the document into Chroma (chunk -> embed -> store).
        """
        assert self.rag_control is not None, "RAG service not initialized"
        return self.rag_control.study_document(file_path=file_path, file_type=file_type)


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
        description="RAG Study CLI - Ingest a document into Chroma with a chosen embedding method."
    )

    # Positional args (match your requested command format)
    parser.add_argument(
        "embedding_method",
        choices=["Sentence-Transformers", "OpenAIEmbeddings"],
        help="Embedding method to use for indexing the document.",
    )
    parser.add_argument(
        "file_type",
        choices=["md", "pdf"],
        help="Type of file to study: 'md' or 'pdf'.",
    )
    parser.add_argument(
        "file_path",
        help="Path to the file to study.",
    )

    # Optional args
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
        help="Print raw JSON result only (no extra text).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.verbosity)
    log = logging.getLogger("ragstudy")

    # Validate file existence early
    if not os.path.exists(args.file_path):
        log.error(f"File not found: {args.file_path}")
        sys.exit(1)

    # Initialize study controller
    rag_study = RAGStudy()
    ok = rag_study.initialize_rag_control(
        db_path=args.db_path,
        embedding_method=args.embedding_method,
        sentence_model_name=args.sentence_model_name,
        openai_model_name=args.openai_model_name,
        openai_api_key=args.openai_api_key,  # service will fall back to env if None
    )
    if not ok:
        log.error("Failed to initialize RAG service.")
        sys.exit(2)

    # Run study (ingest)
    try:
        result = rag_study.study_document(args.file_path, args.file_type)
        if args.json_output:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("\n=== Study Result ===")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("\nTip: use the same embedding when querying this collection.")
            print(f"Collection: {result.get('collection_name')} | Embedding: {result.get('embedding_method')}")
    except Exception as e:
        log.exception(f"Study failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
