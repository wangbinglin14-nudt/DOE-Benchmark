#!/usr/bin/env python
"""Generate Sobol S_T bar charts for all 34 cases → docs/img/sobol/."""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import doe_testbed as db

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "img" / "sobol"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cases = sorted(db.list_cases(), key=lambda m: (m.domain, m.case_id))

for meta in cases:
    st = meta.sobol_ST
    if not st:
        print(f"  SKIP {meta.case_id}: no Sobol ST data")
        continue

    # Collect per-output Sobol ST (dict of output_name -> dict of var -> val)
    # or flat dict of var -> val
    if isinstance(next(iter(st.values())), dict):
        # Multi-output: use first output
        first_key = next(iter(st.keys()))
        vals = st[first_key]
        title_suffix = f" (output: {first_key})"
    else:
        vals = st
        title_suffix = ""

    labels = list(vals.keys())
    values = np.array([vals[k] for k in labels], dtype=float)

    # Sort by descending ST
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = values[order]

    fig, ax = plt.subplots(figsize=(max(3.5, 0.6 * len(labels)), 2.8))
    colors = plt.cm.Blues(0.3 + 0.6 * values / max(values.max(), 1e-9))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Sobol $S_T$", fontsize=9)
    ax.set_title(f"{meta.case_id}{title_suffix}", fontsize=10, fontweight="bold")
    ax.set_ylim(0, min(1.05, values.max() * 1.3))
    ax.axhline(0.05, color="red", ls="--", lw=0.7, alpha=0.6)
    ax.text(
        len(labels) - 0.5,
        0.055,
        "0.05 threshold",
        fontsize=6,
        color="red",
        ha="right",
        va="bottom",
        alpha=0.7,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = OUT_DIR / f"{meta.case_id}_sobol_st.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {meta.case_id}: {out_path.name}")

print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} charts in {OUT_DIR}")
