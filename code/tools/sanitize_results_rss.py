#!/usr/bin/env python
"""修复 results.csv 中 rss_gb_peak < rss_gb_start/end 的测量竞态坏行。

benchmark/core.py 的 RSS 采样器在并发下偶发写出 peak<start 的行，这类行会让
``--skip-existing`` 的 ``_validate_resume_grid`` 以 rc=1 拒绝整份 CSV，导致看门狗
陷入 rc=1 崩溃循环。本脚本在每次 sweep 启动前调用，把 peak 抬到 >= max(start,end)。

rss_* 不在 PROTOCOL_DIGEST_FIELDS 中，改它们不影响 protocol_digest 自洽性。
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def sanitize(csv: Path) -> int:
    df = pd.read_csv(csv, low_memory=False)
    for c in ("rss_gb_start", "rss_gb_end", "rss_gb_peak"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    hi = df[["rss_gb_start", "rss_gb_end"]].max(axis=1)
    bad = df["rss_gb_peak"] < hi
    n = int(bad.sum())
    if n:
        df.loc[bad, "rss_gb_peak"] = hi[bad]
        df.to_csv(csv, index=False)
    return n


if __name__ == "__main__":
    n = sanitize(Path(sys.argv[1]))
    print(f"[sanitize] 修复 rss_gb_peak 坏行: {n}")
