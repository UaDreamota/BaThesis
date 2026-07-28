from __future__ import annotations

"""Small, deterministic provenance helpers for resumable NLP artifacts."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROVENANCE_VERSION = 2
SAMPLE_BYTES = 1 << 20


def canonical_json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sampled_file_content_fingerprint(
    path: Path,
    sample_bytes: int = SAMPLE_BYTES,
) -> str:
    """Hash all small files and three deterministic windows of large files."""
    path = path.resolve()
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        if size <= sample_bytes * 3:
            while block := handle.read(sample_bytes):
                digest.update(block)
        else:
            offsets = (0, max((size - sample_bytes) // 2, 0), size - sample_bytes)
            for offset in offsets:
                handle.seek(offset)
                digest.update(str(offset).encode("ascii"))
                digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sampled_content_sha256": sampled_file_content_fingerprint(path),
    }


def artifact_fingerprint(path: Path) -> str:
    """Fingerprint a file or directory without depending on its absolute path."""
    path = path.resolve()
    if path.is_file():
        payload = {
            "kind": "file",
            "size": path.stat().st_size,
            "sampled_content_sha256": sampled_file_content_fingerprint(path),
        }
        return canonical_json_fingerprint(payload)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": child.relative_to(path).as_posix(),
                "size": child.stat().st_size,
                "sampled_content_sha256": sampled_file_content_fingerprint(child),
            }
        )
    if not files:
        raise ValueError(f"Artifact directory contains no files: {path}")
    return canonical_json_fingerprint({"kind": "directory", "files": files})


def dataframe_fingerprint(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = list(columns) if columns is not None else list(frame.columns)
    missing = sorted(set(selected) - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot fingerprint missing dataframe columns: {missing}")
    data = frame[selected]
    row_hashes = pd.util.hash_pandas_object(data, index=True).to_numpy(
        dtype="uint64",
        copy=False,
    )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(str(len(data)).encode("ascii"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def records_fingerprint(
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(fields), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    count = 0
    for row in rows:
        values = [row.get(field) for field in fields]
        digest.update(
            json.dumps(
                values,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\n")
        count += 1
    digest.update(str(count).encode("ascii"))
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
