"""Canonical records and atomic outputs. Never normalize text implicitly."""
from __future__ import annotations
import gzip
import io
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "pyproject.toml").exists():
    ROOT = Path.cwd()

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()

def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, data, *, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":") if compact else None,
                      indent=None if compact else 2) + "\n").encode("utf-8")
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(raw)
    os.replace(temp, path)

def write_jsonl(path, rows: Iterable[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as raw:
        stream = gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) if path.suffix==".gz" else raw
        with io.TextIOWrapper(stream,encoding="utf-8",newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)

def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from exc

def validate_text_record(row):
    raw = row["text"].encode("utf-8", errors="strict")
    if row.get("sha256") and row["sha256"] != sha256(raw):
        raise ValueError(f"Text hash mismatch: {row.get('id')}")
    if row.get("utf8_bytes") is not None and row["utf8_bytes"] != len(raw):
        raise ValueError(f"Byte count mismatch: {row.get('id')}")
    return raw
