#!/usr/bin/env python
"""
Ingestion script for processing documents into the RAG system.

Usage:
    python scripts/ingest.py --path <file_or_directory> [--collection <name>] [--force]

Examples:
    # Ingest a single PDF file
    python scripts/ingest.py --path docs/sample.pdf

    # Ingest all PDFs in a directory
    python scripts/ingest.py --path docs/ --collection my_docs

    # Force reprocess already ingested files
    python scripts/ingest.py --path docs/ --force
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.pipeline import IngestionPipeline, PipelineConfig
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_collector import TraceCollector


def create_pipeline(settings) -> IngestionPipeline:
    """
    Create and configure the ingestion pipeline.

    Args:
        settings: Application settings

    Returns:
        Configured IngestionPipeline instance
    """
    # Initialize components
    integrity_checker = SQLiteIntegrityChecker()
    loader = PdfLoader()
    chunker = DocumentChunker(settings=settings)

    # Create pipeline with progress callback
    def on_progress(stage: str, current: int, total: int):
        if stage == "integrity_check":
            print(f"  [CHECK] Checking file integrity...")
        elif stage == "load":
            print(f"  [LOAD] Loading document...")
        elif stage == "split":
            print(f"  [SPLIT] Splitting into chunks...")
        elif stage == "transform":
            print(f"  [TRANSFORM] Applying transformations ({current}/{total})...")
        elif stage == "encode":
            if total > 0:
                print(f"  [ENCODE] Encoding chunks ({current}/{total})...")
        elif stage == "upsert":
            if total > 0:
                print(f"  [STORE] Storing vectors ({current}/{total})...")
        elif stage == "bm25_index":
            if total > 0:
                print(f"  [INDEX] Building BM25 index ({current}/{total})...")
        elif stage == "completed":
            print(f"  [OK] Completed successfully")
        elif stage == "failed":
            print(f"  [ERROR] Processing failed")

    pipeline = IngestionPipeline(
        integrity_checker=integrity_checker,
        loader=loader,
        chunker=chunker,
        on_progress=on_progress
    )

    return pipeline


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--path",
        required=True,
        help="Path to file or directory to ingest"
    )

    parser.add_argument(
        "--collection",
        default="default",
        help="Collection name (default: default)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess already ingested files"
    )

    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="File pattern for directory ingestion (default: *.pdf)"
    )

    args = parser.parse_args()

    # Validate path
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Load settings
    try:
        settings = load_settings("config/settings.yaml")
    except Exception as e:
        print(f"Error loading settings: {e}", file=sys.stderr)
        sys.exit(1)

    # Create pipeline
    pipeline = create_pipeline(settings)

    # Create pipeline config
    config = PipelineConfig(
        collection=args.collection,
        force_reprocess=args.force
    )

    # Create trace collector
    trace_collector = TraceCollector()

    # Process files
    print(f"\n{'='*60}")
    print(f"Ingestion started")
    print(f"Collection: {args.collection}")
    print(f"Force reprocess: {args.force}")
    print(f"{'='*60}\n")

    try:
        if path.is_file():
            # Single file ingestion
            print(f"Processing file: {path}")

            # Create trace context
            trace = TraceContext(trace_type="ingestion")
            trace.metadata["file_path"] = str(path)
            trace.metadata["collection"] = args.collection

            result = pipeline.ingest_file(str(path), config=config, trace=trace)

            # Finish and collect trace
            trace.finish()
            trace_collector.collect(trace)

            if result.get("error"):
                print(f"\n[ERROR] Ingestion failed: {result['error']}", file=sys.stderr)
                sys.exit(1)

            print(f"\n{'='*60}")
            print(f"Ingestion completed successfully")
            print(f"File hash: {result['file_hash']}")
            print(f"Chunks: {result['chunk_count']}")
            print(f"Images: {result['image_count']}")
            print(f"{'='*60}\n")

        else:
            # Directory ingestion
            print(f"Processing directory: {path}")
            print(f"Pattern: {args.pattern}\n")

            results = pipeline.ingest_directory(
                str(path),
                pattern=args.pattern,
                config=config
            )

            # Summary
            total = len(results)
            successful = sum(1 for r in results if not r.get("error"))
            skipped = sum(1 for r in results if r.get("skipped"))
            failed = sum(1 for r in results if r.get("error"))
            total_chunks = sum(r.get("chunk_count", 0) for r in results)

            print(f"\n{'='*60}")
            print(f"Ingestion completed")
            print(f"Total files: {total}")
            print(f"Successful: {successful}")
            print(f"Skipped: {skipped}")
            print(f"Failed: {failed}")
            print(f"Total chunks: {total_chunks}")
            print(f"{'='*60}\n")

            if failed > 0:
                print("Failed files:")
                for r in results:
                    if r.get("error"):
                        print(f"  - {r['file_path']}: {r['error']}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
