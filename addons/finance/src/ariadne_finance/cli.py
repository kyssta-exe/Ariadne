"""CLI commands for the finance addon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def cmd_finance(args: argparse.Namespace) -> None:
    """Finance research subcommand dispatcher."""
    if not hasattr(args, "finance_action") or not args.finance_action:
        print("Usage: ariadne finance <ingest|search|tickers> [options]")
        sys.exit(1)

    match args.finance_action:
        case "ingest":
            _cmd_ingest(args)
        case "search":
            _cmd_search(args)
        case "tickers":
            _cmd_tickers(args)
        case _:
            print(f"Unknown finance action: {args.finance_action}")
            sys.exit(1)


def _cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a financial document into Ariadne."""
    from arriadne.config import AriadneConfig
    from arriadne.interface import AriadneMemory
    from arriadne_finance.extractors import (
        CSVExtractor,
        ExcelExtractor,
        PDFExtractor,
    )

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Pick extractor
    ext = file_path.suffix.lower()
    extractor = None
    if ext == ".pdf":
        try:
            extractor = PDFExtractor()
        except Exception:
            print("PDF extraction not available. Install with: pip install 'ariadne-finance[pdf]'", file=sys.stderr)
            sys.exit(1)
    elif ext in (".xlsx", ".xls"):
        extractor = ExcelExtractor()
    elif ext in (".csv", ".tsv"):
        extractor = CSVExtractor()
    else:
        print(f"Unsupported file type: {ext}", file=sys.stderr)
        sys.exit(1)

    # Extract
    print(f"Extracting from {file_path}...")
    result = extractor.extract(file_path)
    content = result["content"]
    entities = result.get("entities", [])
    tables = result.get("tables", [])

    if not content.strip():
        print("No content extracted from file.", file=sys.stderr)
        sys.exit(1)

    # Chunk if very long
    chunks = _chunk_content(content, max_chars=2000)
    print(f"Extracted {len(content)} chars, {len(chunks)} chunks, {len(entities)} entities")

    # Store in Ariadne
    config = AriadneConfig(db_path=args.db_path)
    mem = AriadneMemory(config=config)

    # Build entity list for the memory
    entity_names = [e["value"] for e in entities if e.get("type") in ("ticker", "sector")]

    stored = 0
    for i, chunk in enumerate(chunks):
        result = mem.remember(
            content=chunk,
            memory_type="episodic",
            importance=args.importance if hasattr(args, "importance") else 0.5,
            entities=entity_names if entity_names else None,
            metadata={
                "source": str(file_path),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_type": ext,
            },
        )
        if result["status"] == "created":
            stored += 1

    mem.close()
    print(f"Stored {stored}/{len(chunks)} chunks in Ariadne")
    if entity_names:
        print(f"Entities: {', '.join(entity_names)}")


def _cmd_search(args: argparse.Namespace) -> None:
    """Search finance memories."""
    from arriadne.config import AriadneConfig
    from arriadne.interface import AriadneMemory

    config = AriadneConfig(db_path=args.db_path)
    mem = AriadneMemory(config=config)

    results = mem.recall(
        query=args.query,
        k=args.k if hasattr(args, "k") else 10,
    )
    mem.close()

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        print(f"[{i}] (id={r['id']}, score={score:.4f}, type={r['memory_type']})")
        content = r["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"    {content}")
        meta = r.get("metadata", {})
        if meta:
            source = meta.get("source", "")
            if source:
                print(f"    source: {source}")
        print()


def _cmd_tickers(args: argparse.Namespace) -> None:
    """Recognize tickers in a file."""
    from arriadne_finance.extractors import recognize_tickers

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    text = file_path.read_text(errors="replace")
    tickers = recognize_tickers(text)

    if not tickers:
        print("No tickers found.")
        return

    print(f"Found {len(tickers)} ticker(s):\n")
    for t in tickers:
        print(f"  {t['value']} (confidence={t['confidence']:.1f}, pos={t['start']}-{t['end']})")


def _chunk_content(content: str, max_chars: int = 2000) -> list[str]:
    """Split content into chunks, trying to break at paragraph boundaries."""
    if len(content) <= max_chars:
        return [content]

    chunks = []
    paragraphs = content.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def register_finance_args(subparsers: Any) -> None:
    """Register finance subcommand args with the main CLI parser."""
    finance_parser = subparsers.add_parser("finance", help="Finance research tools")
    finance_sub = finance_parser.add_subparsers(dest="finance_action")

    # ingest
    ingest = finance_sub.add_parser("ingest", help="Ingest a financial document")
    ingest.add_argument("file", help="Path to PDF, Excel, or CSV file")
    ingest.add_argument("--importance", type=float, default=0.5, help="Importance (0-1)")

    # search
    search = finance_sub.add_parser("search", help="Search finance memories")
    search.add_argument("query", help="Search query")
    search.add_argument("-k", type=int, default=10, help="Number of results")

    # tickers
    tickers = finance_sub.add_parser("tickers", help="Recognize tickers in a file")
    tickers.add_argument("file", help="Path to text file")
