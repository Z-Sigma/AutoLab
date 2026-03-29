"""Filesystem helpers for dataset and artifact I/O."""

from __future__ import annotations

from pathlib import Path


def read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    p = Path(path)
    data = p.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def list_dir(path: Path, glob: str = "*") -> list[str]:
    p = Path(path)
    if not p.is_dir():
        return []
    return sorted(str(x.relative_to(p)) for x in p.glob(glob) if x.is_file())
