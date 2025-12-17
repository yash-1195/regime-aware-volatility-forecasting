"""
Plotting helpers.

Constraints:
- matplotlib only (no seaborn)
- do not set custom colors unless necessary
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(fig, outpath: str | Path, dpi: int = 200) -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def set_mpl_defaults() -> None:
    # Keep defaults mostly intact. Only enforce readability.
    plt.rcParams["figure.figsize"] = (10, 4)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
