---
title: "Backup & Restore — Ariadne"
description: "Back up and restore your Ariadne database. CLI commands, dashboard UI, Python API patterns, safety backups, cron scheduling, and machine migration."
---

Ariadne stores everything in a single SQLite database file (default `ariadne.db`) with optional WAL sidecar files (`-wal`, `-shm`). Backup and restore are straightforward file-copy operations with WAL checkpointing for consistency.

## Quick Reference

| Method | Backup Command | Restore Command |
|--------|---------------|-----------------|
| CLI | `ariadne backup` | `ariadne restore SOURCE` |
| Dashboard | Settings → Backup Database | Settings → Restore Database |
| Python API | `shutil.copy2()` + WAL checkpoint | `shutil.copy2()` + reinit |

---

## CLI Commands

### `backup`

Create a consistent snapshot of the Ariadne database.

```bash
ariadne backup [-o OUTPUT]
```

**How it works:**

1. Performs a `PRAGMA wal_checkpoint(TRUNCATE)` to flush all pending WAL writes into the main `.db` file
2. Copies the `.db` file to the output path
3. Copies `-wal` and `-shm` sidecar files if present

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `arriadne-backup-TIMESTAMP.db` | Output file path |

**Examples:**

```bash
# Default: creates arriadne-backup-20260610T143000.db
ariadne backup

# Custom output path
ariadne backup -o ~/backups/ariadne-june.db

# Backup a non-default database
ariadne --db-path ~/.ariadne/memory.db backup -o ~/backups/memory-june.db
```

**Output:**

```
Checkpointing WAL...
Backed up database to arriadne-backup-20260610T143000.db
  Copied WAL: ariadne.db-wal
  Copied SHM: ariadne.db-shm
```

### `restore`

Restore the database from a backup file.

```bash
ariadne restore SOURCE [--no-safety-backup]
```

**How it works:**

1. Creates a safety backup of the current database (unless `--no-safety-backup`)
2. Closes any open connections to the current database
3. Removes the current `.db`, `-wal`, and `-shm` files
4. Copies the backup files into place
5. Reopens the database and verifies by counting active memories

**Options:**

| Option | Description |
|--------|-------------|
| `source` | Path to the backup `.db` file |
| `--no-safety-backup` | Skip creating a safety backup before restoring |

**Examples:**

```bash
# Restore from a backup (safety backup created automatically)
ariadne restore arriadne-backup-20260610T143000.db

# Restore without safety backup
ariadne restore arriadne-backup-20260610T143000.db --no-safety-backup

# Restore to a non-default database location
ariadne --db-path ~/.ariadne/memory.db restore ~/backups/ariadne-june.db
```

**Output:**

```
Safety backup created: arriadne-safety-20260610T143500.db
Restored database from arriadne-backup-20260610T143000.db
Verified: restored database contains 1523 active memories
```

---

## Dashboard Backup & Restore

The web dashboard provides one-click backup and restore through the UI.

### Accessing Backup/Restore

1. Open the dashboard: `ariadne dashboard`
2. Navigate to **Settings → Data Management**
3. Click **Backup Database** or **Restore Database**

### Backup (Download)

- Clicking **Backup Database** triggers a WAL checkpoint on the server and downloads the `.db` file to your browser
- The file is named `arriadne.db` (or your custom `--db-path` filename)

### Restore (Upload)

1. Click **Restore Database** and select a `.db` file
2. A confirmation dialog warns that current data will be overwritten
3. The server creates a safety backup in `<db-dir>/backups/` before applying
4. The memory system reinitializes with the restored database

### Dashboard REST API

For programmatic access without the Python package:

```bash
# Backup — downloads the .db file
curl -o backup.db http://localhost:8765/api/backup

# Restore — uploads a .db file
curl -X POST http://localhost:8765/api/restore \
  -F "file=@backup.db"
```

!!! tip
    The dashboard restore endpoint always creates a safety backup in `<db-dir>/backups/ariadne_backup_TIMESTAMP.db` before overwriting. There is no flag to disable this server-side safety backup.

---

## Python API Patterns

Ariadne does not expose dedicated `backup_db()` / `restore_db()` methods, but backup and restore are trivial to implement programmatically since the database is a single SQLite file.

### Backup with Python

```python
import shutil
import sqlite3
import datetime
from pathlib import Path

def backup_ariadne(db_path: str = "arriadne.db", output: str | None = None) -> str:
    """Create a consistent backup of the Ariadne database.

    Performs a WAL checkpoint to flush pending writes, then copies
    the .db file (and WAL/SHM sidecars if present).

    Returns the path to the backup file.
    """
    src = Path(db_path)
    if not src.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if output is None:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        output = f"arriadne-backup-{ts}.db"
    dst = Path(output)

    # WAL checkpoint for consistency
    conn = sqlite3.connect(str(src))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()

    # Copy main database
    shutil.copy2(str(src), str(dst))

    # Copy WAL and SHM sidecars if present
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(src) + suffix)
        if sidecar.exists():
            shutil.copy2(str(sidecar), str(dst) + suffix)

    return str(dst)


# Usage
backup_path = backup_ariadne()
print(f"Backup saved to: {backup_path}")
```

### Restore with Python

```python
import shutil
import sqlite3
from pathlib import Path

def restore_ariadne(source: str, db_path: str = "arriadne.db",
                     safety_backup: bool = True) -> None:
    """Restore the Ariadne database from a backup file.

    Optionally creates a safety backup of the current database first,
    then replaces the database files and verifies the result.
    """
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Backup file not found: {source}")

    dst = Path(db_path)

    # Safety backup
    if safety_backup and dst.exists():
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        safety = Path(f"arriadne-safety-{ts}.db")
        shutil.copy2(str(dst), str(safety))
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(dst) + suffix)
            if sidecar.exists():
                shutil.copy2(str(sidecar), str(safety) + suffix)
        print(f"Safety backup: {safety}")

    # Close any open connections (best effort)
    try:
        conn = sqlite3.connect(str(dst))
        conn.close()
    except Exception:
        pass

    # Remove current database files
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(dst) + suffix)
        if p.exists():
            p.unlink()

    # Copy backup into place
    shutil.copy2(str(src), str(dst))
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(src) + suffix)
        if sidecar.exists():
            shutil.copy2(str(sidecar), str(dst) + suffix)

    # Verify
    from arriadne import AriadneMemory, AriadneConfig
    config = AriadneConfig(db_path=db_path)
    mem = AriadneMemory(config=config)
    stats = mem.stats()
    print(f"Verified: {stats['active_memories']} active memories")
    mem.close()


# Usage
restore_ariadne("arriadne-backup-20260610T143000.db")
```

### Inline Backup (Minimal)

For a quick backup without the helper functions:

```python
import shutil, sqlite3
from pathlib import Path

db = Path("arriadne.db")
conn = sqlite3.connect(str(db))
conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
conn.close()
shutil.copy2(str(db), "backup.db")
```

!!! note
    The FAISS index is embedded in the SQLite database and is automatically rebuilt when the database is opened. You do not need to back up any separate index files.

---

## Safety Backups

Ariadne creates safety backups automatically before destructive operations to prevent data loss.

### What Is a Safety Backup?

A snapshot of the current database taken immediately before a restore operation. If the restore fails or produces unexpected results, you can recover from the safety backup.

### Where Safety Backups Are Stored

| Method | Location | Filename Pattern |
|--------|----------|-----------------|
| CLI | Current working directory | `arriadne-safety-TIMESTAMP.db` |
| Dashboard | `<db-dir>/backups/` | `<stem>_backup_TIMESTAMP.db` |
| Python (manual) | Current working directory | `arriadne-safety-TIMESTAMP.db` |

### Disabling Safety Backups

```bash
# CLI only — skips safety backup
ariadne restore backup.db --no-safety-backup
```

!!! warning
    Disabling safety backups means there is no recovery path if the restore fails. Only use `--no-safety-backup` when you are certain the backup file is valid and you have copies elsewhere.

### Recovering from a Safety Backup

If a restore went wrong:

```bash
# From CLI safety backup
ariadne restore arriadne-safety-20260610T143500.db

# From dashboard safety backup
ariadne restore backups/ariadne_backup_20260610_143500.db
```

---

## Scheduling Backups with Cron

Automate regular backups using cron (Linux/macOS) or Task Scheduler (Windows).

### Basic Cron Setup

```bash
# Edit your crontab
crontab -e
```

### Daily Backup at 2 AM

```bash
0 2 * * * cd /path/to/project && ariadne backup -o /backups/ariadne-$(date +\%Y\%m\%d).db 2>> /var/log/ariadne-backup.log
```

### Weekly Backup with Rotation

```bash
# Full backup every Sunday at 3 AM
0 3 * * 0 cd /path/to/project && ariadne backup -o /backups/ariadne-$(date +\%Y\%m\%d).db 2>> /var/log/ariadne-backup.log

# Delete backups older than 30 days (runs daily at 4 AM)
0 4 * * * find /backups -name "ariadne-*.db" -mtime +30 -delete 2>> /var/log/ariadne-backup.log
```

### Using Python in Cron

```bash
# crontab entry
0 2 * * * cd /path/to/project && python3 -c "
import shutil, sqlite3
from pathlib import Path
from datetime import datetime
db = Path('arriadne.db')
conn = sqlite3.connect(str(db))
conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
conn.close()
ts = datetime.now().strftime('%Y%m%d')
shutil.copy2(str(db), f'/backups/ariadne-{ts}.db')
" 2>> /var/log/ariadne-backup.log
```

### Cron Best Practices

- **Use full paths** — cron runs in a minimal environment; specify the full path to `ariadne` and the database
- **Redirect output** — use `>>` to log errors for debugging
- **Test first** — run the cron command manually before adding it to crontab
- **Rotate old backups** — combine with `find -mtime +N -delete` to avoid disk exhaustion

---

## Migrating Between Machines

Moving your Ariadne database to a new machine is a file copy operation.

### Step 1: Backup on the Source Machine

```bash
# On the source machine
ariadne backup -o ariadne-migration.db
```

This produces `ariadne-migration.db` (plus `-wal` and `-shm` if present).

### Step 2: Transfer the Files

```bash
# Copy all backup files (db + sidecars)
scp ariadne-migration.db ariadne-migration.db-wal ariadne-migration.db-shm \
    user@new-machine:/path/to/project/

# Or use rsync
rsync -av ariadne-migration.db* user@new-machine:/path/to/project/
```

!!! tip
    Always transfer the `-wal` and `-shm` sidecar files alongside the `.db` file. They may contain recent writes not yet flushed to the main database.

### Step 3: Restore on the Target Machine

```bash
# On the target machine
cd /path/to/project
ariadne restore ariadne-migration.db
```

### Step 4: Verify

```bash
ariadne stats
ariadne search "test query" -k 3
```

### Migration Checklist

- [ ] Install `ariadne-memory` on the target machine (`pip install ariadne-memory`)
- [ ] Install the same embedding model used on the source (`pip install sentence-transformers`)
- [ ] Backup on source with `ariadne backup`
- [ ] Copy `.db`, `-wal`, and `.shm` files
- [ ] Restore on target with `ariadne restore`
- [ ] Run `ariadne stats` to verify memory count matches
- [ ] Test a few searches to confirm embedding compatibility

!!! warning
    The FAISS index is rebuilt from stored embeddings when the database opens. If you change the embedding model (different dimension or architecture) between machines, you must re-embed all memories. Use `ariadne init --dim NEW_DIM` on the target and re-add memories via `ariadne migrate` from a JSON export.

### Migrating with Different Embedding Dimensions

If the target machine uses a different embedding dimension:

```bash
# On source: export memories as JSON
ariadne export -o memories.json

# On target: initialize with new dimension
ariadne init --db-path new_memory.db --dim 768

# Import memories (embeddings will be regenerated)
ariadne migrate memories.json --db-path new_memory.db
```

---

## Troubleshooting

### "Database not found" during backup

The database file doesn't exist at the expected path. Check your `--db-path` or current working directory.

### WAL file is large after backup

A `PRAGMA wal_checkpoint(TRUNCATE)` is supposed to truncate the WAL, but if another process has an open read transaction, the checkpoint may be incomplete. Close all other connections and retry.

### Restore verification shows 0 memories

The backup file may be corrupted or from a different embedding dimension. Check the source database with `ariadne stats --db-path backup.db`.

### Safety backup not created

The `--no-safety-backup` flag was used, or the current database file doesn't exist yet (first-time restore into a new location).

### Dashboard restore fails with "reinitialization failed"

The uploaded file may be incompatible (wrong format, corrupted, or wrong embedding dimension). The server automatically attempts to roll back to the safety backup on failure.
