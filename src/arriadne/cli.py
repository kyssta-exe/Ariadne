"""Command-line interface for Ariadne memory system.

Provides init, add, search, stats, and migrate commands.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

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


def cmd_backup(args: argparse.Namespace) -> int:
    """Create a consistent SQLite backup of the Ariadne database.

    Performs a WAL checkpoint, then copies .db, .wal, and .shm files.
    Default output name: arriadne-backup-YYYYMMDDTHHMMSS.db
    """
    try:
        db_path = Path(args.db_path)
        if not db_path.exists():
            print(f"Database not found: {db_path}", file=sys.stderr)
            return 1

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            output_path = Path(f"arriadne-backup-{ts}.db")

        # WAL checkpoint for consistency
        print("Checkpointing WAL...")
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()

        # Copy main database file
        shutil.copy2(str(db_path), str(output_path))
        print(f"Backed up database to {output_path}")

        # Copy WAL and SHM if present
        wal_path = Path(str(db_path) + "-wal")
        shm_path = Path(str(db_path) + "-shm")
        if wal_path.exists():
            shutil.copy2(str(wal_path), str(output_path) + "-wal")
            print(f"  Copied WAL: {wal_path}")
        if shm_path.exists():
            shutil.copy2(str(shm_path), str(output_path) + "-shm")
            print(f"  Copied SHM: {shm_path}")

        return 0
    except Exception as e:
        print(f"Error creating backup: {e}", file=sys.stderr)
        return 1


def cmd_restore(args: argparse.Namespace) -> int:
    """Restore the Ariadne database from a backup file.

    Optionally creates a safety backup of the current DB first,
    then verifies the restored DB by counting memories.
    """
    try:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Backup file not found: {source_path}", file=sys.stderr)
            return 1

        db_path = Path(args.db_path)

        # Safety backup of current DB (unless --no-safety-backup)
        if not args.no_safety_backup and db_path.exists():
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            safety_path = Path(f"arriadne-safety-{ts}.db")
            shutil.copy2(str(db_path), str(safety_path))
            # Copy WAL/SHM if present
            wal = Path(str(db_path) + "-wal")
            shm = Path(str(db_path) + "-shm")
            if wal.exists():
                shutil.copy2(str(wal), str(safety_path) + "-wal")
            if shm.exists():
                shutil.copy2(str(shm), str(safety_path) + "-shm")
            print(f"Safety backup created: {safety_path}")

        # Stop any existing connections by closing current memory
        if db_path.exists():
            try:
                config = AriadneConfig(db_path=args.db_path)
                mem = AriadneMemory(config=config)
                mem.close()
            except Exception:
                pass  # Best effort close

        # Remove current DB files
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()

        # Copy backup files to target location
        shutil.copy2(str(source_path), str(db_path))
        for ext in ("-wal", "-shm"):
            src = Path(str(source_path) + ext)
            dst = Path(str(db_path) + ext)
            if src.exists():
                shutil.copy2(str(src), str(dst))

        print(f"Restored database from {source_path}")

        # Verify restored DB by counting memories
        try:
            config = AriadneConfig(db_path=args.db_path)
            mem = AriadneMemory(config=config)
            stats = mem.stats()
            count = stats.get("active_memories", 0)
            mem.close()
            print(f"Verified: restored database contains {count} active memories")
        except Exception as e:
            print(f"Warning: Could not verify restored database: {e}", file=sys.stderr)

        return 0
    except Exception as e:
        print(f"Error restoring backup: {e}", file=sys.stderr)
        return 1


def cmd_export(args: argparse.Namespace) -> int:
    """Export all memories as JSON."""
    try:
        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)
        data = mem.export_json()
        output = args.output
        if output:
            import json

            with open(output, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Exported {len(data.get('memories', []))} memories to {output}")
        else:
            import json

            print(json.dumps(data, indent=2))
        mem.close()
        return 0
    except Exception as e:
        print(f"Error exporting: {e}", file=sys.stderr)
        return 1


def cmd_import(args: argparse.Namespace) -> int:
    """Import memories from JSON file."""
    try:
        import json

        with open(args.source, "r") as f:
            data = json.load(f)
        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)
        count = mem.import_json(data)
        mem.close()
        print(f"Imported {count} memories from {args.source}")
        return 0
    except Exception as e:
        print(f"Error importing: {e}", file=sys.stderr)
        return 1


def cmd_maintain(args: argparse.Namespace) -> int:
    """Run scheduled maintenance (consolidate + evict + prune)."""
    try:
        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)
        result = mem.maintenance()
        print(
            f"Maintenance complete: consolidated={result.get('consolidated', 0)}, "
            f"evicted={result.get('evicted', 0)}, "
            f"pruned={result.get('access_log_pruned', 0)}, "
            f"purged={result.get('purged', 0)}"
        )
        mem.close()
        return 0
    except Exception as e:
        print(f"Error during maintenance: {e}", file=sys.stderr)
        return 1


def cmd_curate(args: argparse.Namespace) -> int:
    """Run the memory curation cycle (decay + conflict resolution + consolidate)."""
    try:
        from arriadne.curator import MemoryCurator

        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)
        curator = MemoryCurator(
            mem,
            decay_ttl_seconds=args.decay_ttl,
            decay_importance_threshold=args.decay_importance,
        )
        report = curator.curate()
        mem.close()
        print(
            f"Curation complete: decayed={report.decayed}, "
            f"contradictions={report.contradictions_resolved}, "
            f"consolidated={report.consolidated}"
        )
        return 0
    except Exception as e:
        print(f"Error during curation: {e}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List recent memories.

    Shows the most recently created active memories, optionally filtered by
    type or namespace. Useful for quick inspection without opening the
    dashboard.
    """
    try:
        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)

        db = mem._db
        assert db.conn is not None
        where_parts: list[str] = ["is_deleted = 0"]
        params: list[Any] = []

        if args.type:
            where_parts.append("memory_type = ?")
            params.append(args.type)
        if args.namespace:
            where_parts.append("namespace = ?")
            params.append(args.namespace)
        where_sql = " AND ".join(where_parts)

        # --limit is cast to int at parse time; safe to interpolate into LIMIT.
        rows = db.conn.execute(
            f"""SELECT id, memory_type, namespace, importance, created_at,
                       substr(content, 1, 80) AS preview
                FROM memories
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT ?""",
            (*params, args.limit),
        ).fetchall()

        mem.close()

        print(f"Memories (showing {len(rows)}/{args.limit}):")
        for row in rows:
            created = row["created_at"]
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))
            print(
                f"  id={row['id']:>6}  type={row['memory_type']:<10}  "
                f"namespace={row['namespace']:<12}  importance={row['importance']:.2f}"
            )
            print(f"         {ts}    {row['preview']!r}")
        return 0
    except Exception as e:
        print(f"Error listing memories: {e}", file=sys.stderr)
        return 1


def cmd_purge(args: argparse.Namespace) -> int:
    """Purge (permanently delete) soft-deleted memories.

    By default only rows soft-deleted more than ``--older`` seconds ago are
    purged, keeping recently deleted rows recoverable. Pass ``--older 0`` to
    purge everything currently soft-deleted.
    """
    try:
        config = AriadneConfig(db_path=args.db_path)
        mem = AriadneMemory(config=config)
        purged = mem.purge_deleted(older_than_seconds=args.older)
        mem.close()
        print(f"Purged {purged} soft-deleted memory(es) (older than {args.older}s).")
        return 0
    except Exception as e:
        print(f"Error purging deleted memories: {e}", file=sys.stderr)
        return 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the Ariadne web dashboard.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    try:
        import uvicorn  # noqa: F401  — lazy import
    except ImportError:
        print(
            "uvicorn is required for the dashboard.  "
            "Install it with:  pip install 'arriadne[dashboard]'",
            file=sys.stderr,
        )
        return 1

    try:
        from arriadne.dashboard.server import create_app
    except ImportError as e:
        print(f"Dashboard module not available: {e}", file=sys.stderr)
        return 1

    app = create_app(db_path=args.db_path)

    # Auto-open browser unless --no-browser
    if not args.no_browser:
        import threading
        import webbrowser

        def _open() -> None:
            import time as _time

            _time.sleep(1.0)
            webbrowser.open(f"http://{args.host}:{args.port}")

        threading.Thread(target=_open, daemon=True).start()

    print(f"Ariadne Dashboard running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


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
        "-v",
        "--verbose",
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

    # export
    export_parser = subparsers.add_parser("export", help="Export memories as JSON")
    export_parser.add_argument("-o", "--output", help="Output file (default: stdout)")

    # import
    import_parser = subparsers.add_parser("import", help="Import memories from JSON")
    import_parser.add_argument("source", help="Path to JSON export file")

    # maintain
    subparsers.add_parser("maintain", help="Run maintenance (consolidate + evict + prune)")

    # curate
    curate_parser = subparsers.add_parser(
        "curate", help="Run memory curation (decay + conflict + consolidate)"
    )
    curate_parser.add_argument(
        "--decay-ttl",
        type=float,
        default=86400.0 * 30,
        help="Decay TTL in seconds (default: 30 days); memories older & low-importance are removed",
    )
    curate_parser.add_argument(
        "--decay-importance",
        type=float,
        default=0.4,
        help="Importance below which stale memories are decayed (default: 0.4)",
    )

    # list
    list_parser = subparsers.add_parser("list", help="List recent memories")
    list_parser.add_argument(
        "-n", "--limit", type=int, default=20, help="Max memories to list (default: 20)"
    )
    list_parser.add_argument("--type", help="Filter by memory type")
    list_parser.add_argument("--namespace", help="Filter by namespace")

    # purge
    purge_parser = subparsers.add_parser("purge", help="Permanently purge soft-deleted memories")
    purge_parser.add_argument(
        "--older",
        type=float,
        default=604800.0,
        help="Purge only rows deleted this many seconds ago (default: 7 days); use 0 for all",
    )

    # backup
    backup_parser = subparsers.add_parser("backup", help="Create a consistent database backup")
    backup_parser.add_argument(
        "-o", "--output", help="Output backup file path (default: arriadne-backup-TIMESTAMP.db)"
    )

    # restore
    restore_parser = subparsers.add_parser("restore", help="Restore database from a backup")
    restore_parser.add_argument("source", help="Path to backup file")
    restore_parser.add_argument(
        "--no-safety-backup",
        action="store_true",
        help="Skip creating a safety backup before restoring",
    )

    # dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dash_parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    dash_parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    dash_parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

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
        case "export":
            return cmd_export(args)
        case "import":
            return cmd_import(args)
        case "maintain":
            return cmd_maintain(args)
        case "curate":
            return cmd_curate(args)
        case "list":
            return cmd_list(args)
        case "purge":
            return cmd_purge(args)
        case "backup":
            return cmd_backup(args)
        case "restore":
            return cmd_restore(args)
        case "dashboard":
            return cmd_dashboard(args)
        case _:
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(main())
