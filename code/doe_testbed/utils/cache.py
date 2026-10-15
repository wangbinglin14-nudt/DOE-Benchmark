"""
doe_testbed.utils.cache
==========================
Local cache management for reference solutions of expensive cases.

Cache files are stored at:
    Linux/Mac:  ~/.doe_testbed/cache/
    Windows:    %USERPROFILE%\\.doe_testbed\\cache\\

Set DOE_BENCHMARK_CACHE_DIR to override the default cache directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _cache_dir() -> Path:
    env = os.environ.get("DOE_BENCHMARK_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".doe_testbed" / "cache"


class CacheManager:
    """Reference-solution cache manager (npz format)."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self._dir = Path(cache_dir) if cache_dir else _cache_dir()

    def _path(self, case_id: str, n: int, seed: int, fidelity: str) -> Path:
        fname = f"{case_id}__ref_n{n}_s{seed}_{fidelity}.npz"
        return self._dir / fname

    def get(
        self,
        case_id: str,
        n: int,
        seed: int,
        fidelity: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Read from cache; return None if not found."""
        p = self._path(case_id, n, seed, fidelity)
        if not p.exists():
            return None
        try:
            data = np.load(p)
            return data["X"], data["Y"]
        except Exception:
            return None

    def store(
        self,
        case_id: str,
        n: int,
        seed: int,
        fidelity: str,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        """Store a reference solution to cache."""
        p = self._path(case_id, n, seed, fidelity)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, X=X, Y=Y)

    def clear(self, case_id: Optional[str] = None) -> int:
        """Delete cache files. Returns the number of files deleted."""
        if not self._dir.exists():
            return 0
        pattern = f"{case_id}__*.npz" if case_id else "*.npz"
        count = 0
        for f in self._dir.glob(pattern):
            f.unlink()
            count += 1
        return count

    def list_cached(self) -> list:
        """List basic information for all cached files."""
        if not self._dir.exists():
            return []
        result = []
        for f in sorted(self._dir.glob("*.npz")):
            stat = f.stat()
            result.append(
                {
                    "file": f.name,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                }
            )
        return result
