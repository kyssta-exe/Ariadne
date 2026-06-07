"""Command-line interface for Ariadne memory system.

Provides init, add, search, stats, and migrate commands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from arriadne.config import AriadneConfig
from arriadne.interface import AriadneMemory


def _setup_logging(verbose: bool) -> None:
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new Ariadne database.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    try:
        config = AriadneConfig(
            db_path=args.db_path,
            embedding_dim=args.dim,
        )
        mem = AriadneMemory(config=config)
        stats = mem.stats()
        mem.close()

        print(f"Initialized Ariadne database at {args.db_path}")
        print(f"  Embedding dimension: {args.dim}")
        print(f"  Total memories: {stats.get('active_memories', 0)}")
        return 0
    except Exception as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        return 1


def cmd_add(args: argparse.Namespace) -> int:
    """Add a memory to the database.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    try:
        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)

        metadata = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError:
                print(f"Warning: Invalid metadata JSON: {args.metadata}", file=sys.stderr)

        result = mem.remember(
            content=args.content,
            memory_type=args.type,
            importance=args.importance,
            entities=args.entities.split(",") if args.entities else None,
            metadata=metadata,
        )
        mem.close()

        if result["status"] == "duplicate":
            print(f"Duplicate memory detected (similar to {result.get('duplicate_of', 'unknown')})")
        elif result["status"] == "created":
            print(f"Memory added: id={result['memory_id']}")
        else:
            print(f"Error: {result.get('error', 'unknown error')}", file=sys.stderr)
            return 1

        return 0
    except Exception as e:
        print(f"Error adding memory: {e}", file=sys.stderr)
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search memories.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    try:
        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)

        results = mem.recall(
            query=args.query,
            k=args.k,
            type_filter=args.type,
            importance_min=args.min_importance,
        )
        mem.close()

        if not results:
            print("No results found.")
            return 0

        print(f"Found {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            print(f"[{i}] (id={r['id']}, score={score:.4f}, type={r['memory_type']})")
            content = r["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"    {content}")
            print(f"    importance={r['importance']:.2f} created={r['created_at']:.0f}")
            print()

        return 0
    except Exception as e:
        print(f"Error searching: {e}", file=sys.stderr)
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Show database statistics.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    try:
        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)
        stats = mem.stats()
        mem.close()

        print("Ariadne Database Statistics")
        print("=" * 40)
        print(f"  Active memories:     {stats.get('active_memories', 0)}")
        print(f"  Deleted memories:    {stats.get('deleted_memories', 0)}")
        print(f"  Total memories:      {stats.get('total_memories', 0)}")
        print(f"  Total entities:      {stats.get('total_entities', 0)}")
        print(f"  Total edges:         {stats.get('total_edges', 0)}")
        print(f"  Total memory links:  {stats.get('total_memory_links', 0)}")
        print(f"  Consolidations:      {stats.get('total_consolidations', 0)}")
        print(f"  FAISS vectors:       {stats.get('faiss_vectors', 0)}")
        print(f"  FAISS type:          {stats.get('faiss_type', 'none')}")
        print(f"  FAISS dimension:     {stats.get('faiss_dimension', 0)}")
        print(f"  Avg importance:      {stats.get('avg_importance', 0):.4f}")
        print(f"  Dedup index size:    {stats.get('dedup_index_size', 0)}")
        db_size = stats.get("db_size_bytes", 0)
        if db_size > 1024 * 1024:
            size_str = f"{db_size / (1024 * 1024):.2f} MB"
        elif db_size > 1024:
            size_str = f"{db_size / 1024:.2f} KB"
        else:
            size_str = f"{db_size} B"
        print(f"  DB size:             {size_str}")

        by_type = stats.get("by_type", {})
        if by_type:
            print("\n  By type:")
            for t, count in sorted(by_type.items()):
                print(f"    {t}: {count}")

        return 0
    except Exception as e:
        print(f"Error getting stats: {e}", file=sys.stderr)
        return 1


def cmd_migrate(args: argparse.Namespace) -> int:
    """Migrate memories from Mnemosyne JSON export.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    try:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"File not found: {args.source}", file=sys.stderr)
            return 1

        with open(source_path) as f:
            data = json.load(f)

        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)

        # Handle Mnemosyne export format
        cards = data if isinstance(data, list) else data.get("cards", [])
        if not cards:
            print("No cards found in export file.", file=sys.stderr)
            return 1

        imported = 0
        skipped = 0
        for card in cards:
            if isinstance(card, dict):
                content = card.get("question", "") or card.get("answer", "")
                if not content:
                    skipped += 1
                    continue

                importance = card.get("importance", 0.5)
                if isinstance(importance, (int, float)):
                    importance = max(0.0, min(1.0, importance / 10.0))
                else:
                    importance = 0.5

                tags = card.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",")]

                result = mem.remember(
                    content=content,
                    memory_type="semantic",
                    importance=importance,
                    entities=tags if tags else None,
                )
                if result["status"] == "created":
                    imported += 1
                else:
                    skipped += 1

        mem.close()
        print(f"Migration complete: {imported} imported, {skipped} skipped")
        return 0

    except json.JSONDecodeError as e:
        print(f"Invalid JSON file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Migration error: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="ariadne",
        description="Ariadne - Production-ready memory system",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--db-path",
        default="arriadne.db",
        help="Path to database file (default: arriadne.db)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize a new database")
    init_parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")

    # add
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("content", help="Memory content")
    add_parser.add_argument("--type", default="semantic", help="Memory type")
    add_parser.add_argument("--importance", type=float, default=0.5, help="Importance (0-1)")
    add_parser.add_argument("--entities", help="Comma-separated entity names")
    add_parser.add_argument("--metadata", help="JSON metadata string")

    # search
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-k", type=int, default=10, help="Number of results")
    search_parser.add_argument("--type", help="Filter by memory type")
    search_parser.add_argument("--min-importance", type=float, help="Minimum importance")

    # stats
    subparsers.add_parser("stats", help="Show database statistics")

    # migrate
    migrate_parser = subparsers.add_parser("migrate", help="Import from Mnemosyne JSON")
    migrate_parser.add_argument("source", help="Path to Mnemosyne JSON export")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    _setup_logging(args.verbose)

    match args.command:
        case "init":
            return cmd_init(args)
        case "add":
            return cmd_add(args)
        case "search":
            return cmd_search(args)
        case "stats":
            return cmd_stats(args)
        case "migrate":
            return cmd_migrate(args)
        case _:
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(main())
