# Phase 1 SQLite Data Layer

## Environment

```bash
conda activate shreyas-ipl
python -c "import sys, sqlite3; print(sys.executable); print(sqlite3.sqlite_version)"
sqlite3 --version
```

## One-time DB setup

```bash
python -m ipl.storage.cli verify-env
python -m ipl.storage.cli init-db
```

## Historical load

```bash
python -m ipl.storage.cli load-historical --csv data/raw/IPL.csv
```

Run again anytime; writes are idempotent via upsert on:
- `ball_by_ball.ball_id`
- `match_list.match_id`

## Schedule refresh (one shot)

```bash
python -m ipl.storage.cli refresh-schedule --days-back 7 --days-forward 60
```

## Optional schema print

```bash
python -m ipl.storage.cli discover-schemas --csv data/raw/IPL.csv
```

## Storage locations

- SQLite DB: `$DB_PATH` when set, otherwise `data/sqlite/ipl.db`
- Raw schedule payloads: `data/raw_api/`
- Logs table: `api_fetch_log` in SQLite (`payload_path` points to raw JSON files)
- Optional log directory reserved for future file logs: `data/logs/`
