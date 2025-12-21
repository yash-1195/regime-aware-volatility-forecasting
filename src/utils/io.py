"""
Lightweight filesystem and I/O helpers used across the project.

"""
from __future__ import annotations
from pathlib import Path

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p