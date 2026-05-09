#!/usr/bin/env python3
"""
Query script for RAG system.

Command-line interface for querying the knowledge base using HybridSearch + Reranker.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import Settings
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import Reranker
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.core.trace import TraceContext


def format_result(idx: int, result, verbose: bool = False) -> str:
    """Format a single retrieval result for display"""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Result #{idx + 1} (Score: {result.score:.4f})")
    lines.append(f"{'-'*80}")

    # Text preview (first 200 chars)
    text_preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
    lines.append(f"Text: {text_preview}")

    # Metadata
    if result.metadata:
        lines.append(f"\nMetadata:")
        for key, value in result.metadata.items():
            lines.append(f"  {key}: {value}")

    if verbose:
        lines.append(f"\nChunk ID: {result.chunk_id}")

    return "\n".join(lines)


def format_stage_results(stage_name: str, results, max_display: int = 5) -> str:
    """Format intermediate stage results for verbose mode"""
    lines = []
    lines.append(f"\n{'#'*80}")
    lines.append(f"# {stage_name}")
    lines.append(f"{'#'*80}")
    lines.append(f"Total results: {len(results)}")

    if results:
        lines.append(f"\nTop {min(max_display, len(results))} results:")
        for idx, result in enumerate(results[:max_display]):
            lines.append(f"\n  [{idx + 1}] Score: {result.score:.4f}, ID: {result.chunk_id}")
            text_preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
            lines.append(f"      Text: {text_preview}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/query.py --query "如何配置 Azure？"
  python scripts/query.py --query "配置文档" --top-k 5 --verbose
  python scripts/query.py --query "API 文档" --collection docs --no-rerank
        """
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text (required)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        help="Limit search to specific collection"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show intermediate results from each stage"
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking stage"
    )

    args = parser.parse_args()

    try:
        # Load settings
        print("Loading configuration...")
        settings = Settings.from_yaml("config/settings.yaml")

        # Initialize components
        print("Initializing components...")

        # Embedding client
        embedding_client = EmbeddingFactory.create(settings.__dict__)

        # Vector store
        vector_store = VectorStoreFactory.create(settings.__dict__)

        # BM25 indexer
        bm25_indexer = BM25Indexer(settings)

        # Query processor
        query_processor = QueryProcessor(settings)

        # Dense retriever
        dense_retriever = DenseRetriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store
        )

        # Sparse retriever
        sparse_retriever = SparseRetriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store
        )

        # Fusion
        fusion = RRFFusion(settings)

        # Hybrid search
        hybrid_search = HybridSearch(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )

        # Reranker (if not disabled)
        reranker = None
        if not args.no_rerank:
            reranker = Reranker(settings)

        # Build filters
        filters = {}
        if args.collection:
            filters["collection"] = args.collection

        # Create trace context for verbose mode
        trace = TraceContext() if args.verbose else None

        # Execute search
        print(f"\nSearching for: '{args.query}'")
        if filters:
            print(f"Filters: {filters}")
        print(f"Top-K: {args.top_k}")
        print()

        search_results = hybrid_search.search(
            query=args.query,
            top_k=args.top_k * 2 if not args.no_rerank else args.top_k,  # Get more candidates for reranking
            filters=filters if filters else None,
            trace=trace
        )

        # Check if any results found
        if not search_results:
            print("❌ No results found.")
            print("\nPossible reasons:")
            print("  1. No documents have been ingested yet")
            print("  2. Query doesn't match any indexed content")
            print("  3. Filters are too restrictive")
            print("\nTry running: python scripts/ingest.py --source <your_data>")
            return

        # Verbose: show search results
        if args.verbose:
            print(format_stage_results("HybridSearch Results", search_results))

        # Apply reranking
        final_results = search_results
        rerank_fallback = False

        if reranker:
            print("\nApplying reranking...")
            rerank_result = reranker.rerank(
                query=args.query,
                candidates=search_results,
                trace=trace
            )
            final_results = rerank_result.results[:args.top_k]
            rerank_fallback = rerank_result.fallback

            if rerank_fallback:
                print(f"⚠️  Reranking failed (fallback to fusion ranking): {rerank_result.error}")

            # Verbose: show rerank results
            if args.verbose:
                print(format_stage_results("Reranker Results", rerank_result.results))
        else:
            final_results = search_results[:args.top_k]
            print("\nReranking skipped (--no-rerank)")

        # Display final results
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS (Top {len(final_results)})")
        print(f"{'='*80}")

        for idx, result in enumerate(final_results):
            print(format_result(idx, result, verbose=args.verbose))

        # Summary
        print(f"\n{'='*80}")
        print(f"Summary: Found {len(final_results)} results")
        if rerank_fallback:
            print("Note: Reranking failed, results are fusion-ranked")
        print(f"{'='*80}\n")

        # Verbose: show trace
        if args.verbose and trace:
            print("\n" + "="*80)
            print("TRACE INFORMATION")
            print("="*80)
            print(trace.to_dict())

    except FileNotFoundError as e:
        print(f"❌ Error: Configuration file not found: {e}")
        print("\nMake sure config/settings.yaml exists.")
        sys.exit(1)

    except ValueError as e:
        print(f"❌ Error: Invalid input: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
