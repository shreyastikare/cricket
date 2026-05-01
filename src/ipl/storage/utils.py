"""Utility helpers for storage workflows."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import DATA_DIR, LOG_DIR, RAW_API_DIR, SQLITE_DIR


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_id(prefix: str | None = None) -> str:
    suffix = uuid.uuid4().hex
    if prefix:
        return f"{prefix}_{suffix}"
    return suffix


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_phase1_dirs() -> None:
    ensure_dir(DATA_DIR)
    ensure_dir(SQLITE_DIR)
    ensure_dir(RAW_API_DIR)
    ensure_dir(LOG_DIR)


def save_json(payload: Any, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(input_path: Path) -> Any:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_dates(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        return []
    span = (end_date - start_date).days
    return [start_date + timedelta(days=i) for i in range(span + 1)]


def to_yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def build_ball_id(match_id: Any, row_id: Any, innings: Any, over: Any, ball_no: Any) -> str:
    if match_id is not None and row_id is not None:
        return f"{match_id}:{row_id}"

    raw = "|".join(
        [
            str(match_id),
            str(row_id),
            str(innings),
            str(over),
            str(ball_no),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"fallback:{digest}"
