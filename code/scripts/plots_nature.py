"""
scripts/plots_nature.py
=========================
从 ../manuscripts/figdata/*.csv 生成论文全部图表（Nature 格式）。

Usage::

    python scripts/plots_nature.py
    python scripts/plots_nature.py --figdata ../manuscripts/figdata --out-dir ../manuscripts/figures --fig 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scripts.project_layout import FIGDATA_DIR, FIGURES_DIR  # noqa: E402

from paper_names import domain_label, family_label, method_label

import matplotlib as mpl
import matplotlib.patheffects as _pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Nature 格式设置 ───────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlepad": 5.0,
    "axes.labelsize": 9,
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#525252",
    "text.color": "#222222",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": "#525252",
    "ytick.color": "#525252",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Keep text selectable/editable in vector output (Nature figure guide).
    "svg.fonttype": "none",
})
# Static final widths in millimetres (Nature single/double column).  Declared as
# explicit ``*_width_mm`` values so the figure preflight can verify the target
# journal geometry from source rather than guessing.
SINGLE_COL_WIDTH_MM = 89.0
DOUBLE_COL_WIDTH_MM = 183.0
SINGLE_COL_IN = SINGLE_COL_WIDTH_MM / 25.4
DOUBLE_COL_IN = DOUBLE_COL_WIDTH_MM / 25.4
# Consistent display-item family: every figure uses the double-column width so
# all render at the same width under ``\includegraphics[width=\linewidth]`` and
# none is up-/down-scaled by LaTeX (the previous single-column Fig 4 and
# portrait Fig 2 were the two odd ones out).  Heights are drawn from a small
# controlled set so the figures read as one family.
FIG_H_WIDE = 3.10   # two-panel landscape rows (Figs 1, 3, 6)
FIG_H_GRID = 4.05   # 2x3 small-multiples grid (Fig 2)
FIG_H_TALL = 4.20   # many-row lollipop (Fig 4)

# ── Colour-blind-safe palette (Nature Methods, GAP-F4) ──────────────────
# Wong (2011) "Points of view: Color blindness" Nature Methods 8:441 lists
# eight categorical hues that remain distinguishable under deuteranopia,
# protanopia and tritanopia.  We use them as the *categorical* palette
# (Fig 3 learning curves, Fig 6 budget tiers).  For continuous heatmaps
# (Fig 5) we use ``cividis`` which is perceptually uniform and CVD-safe
# by construction (Nuñez et al. 2018, PLOS ONE).  The existing red
# ``#d73027`` vs grey ``#969696`` significance pair stays — red/grey is
# Wong-cited as the canonical high-contrast warning combination.
WONG_PALETTE = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

# Continuous (sequential) colormap used by Fig 5 mechanism heatmap.
# ``cividis`` is the matplotlib-bundled CVD-safe perceptually uniform
# colormap; reverse it (``_r``) so lower rank = darker (better methods
# read as bolder visually).
CVD_SEQUENTIAL_CMAP = "cividis_r"

COLOR_SIG = "#d73027"
COLOR_NSIG = "#969696"
COLOR_PRIMARY = "#0072B2"
COLOR_SECONDARY = "#56B4E9"
COLOR_TEXT_OK = COLOR_PRIMARY
# Four-tier budget palette — reuse the Wong blue / sky-blue / vermillion /
# reddish-purple slots so the four budgets remain mutually distinct under
# all three CVD types.  Order chosen to maximise lightness contrast.
PALETTE_B = ("#0072B2", "#56B4E9", "#D55E00", "#CC79A7")


def cvd_categorical_colors(n: int) -> list[str]:
    """Return *n* CVD-safe categorical hues (hex strings).

    For ``n <= 8`` we use Wong's 8-colour palette directly.  For ``n > 8``
    we fall back to the matplotlib ``cividis`` colormap evenly sampled
    across [0, 1]; ``cividis`` is perceptually uniform and CVD-safe so
    even with 20 categories adjacent hues stay distinguishable for
    readers with protan/deutan/tritan vision.

    The deliberate avoidance of matplotlib's default ``tab10`` / ``tab20``
    closes GAP-F4 of the Nature-sub-journal gap analysis.
    """
    if n <= 0:
        return []
    if n <= len(WONG_PALETTE):
        return list(WONG_PALETTE[:n])
    cmap = mpl.colormaps[CVD_SEQUENTIAL_CMAP]
    return [mpl.colors.to_hex(cmap(i / (n - 1))) for i in range(n)]


# ── Surrogate-family grouping (Fig 4) ─────────────────────────────────────────────
# The D3 finding is that calibration error tracks the *surrogate family*
# rather than the acquisition strategy, so Fig 4 colour-codes methods by
# surrogate family using Wong hues.
SURROGATE_FAMILY_COLORS = {
    "Gaussian process": "#0072B2",
    "RBF": "#56B4E9",
    "Polynomial": "#009E73",
    "Tree ensemble": "#E69F00",
    "SVR / kNN / MLP": "#CC79A7",
    "Linear": "#D55E00",
}

# Redundant shape encoding so families stay distinguishable under
# colour-vision deficiency and in greyscale print (GAP-F4 follow-up).
SURROGATE_FAMILY_MARKERS = {
    "Gaussian process": "o",
    "RBF": "^",
    "Polynomial": "s",
    "Tree ensemble": "D",
    "SVR / kNN / MLP": "v",
    "Linear": "P",
}


def _surrogate_family(method: str) -> str:
    """Map a method id (e.g. ``lhs_opt+gb``) to its surrogate family."""
    m = str(method).lower()
    surrogate = m.split("+", 1)[1] if "+" in m else m
    if surrogate in {"rf", "gb"}:
        return "Tree ensemble"
    if surrogate in {"svr", "knn", "mlp"}:
        return "SVR / kNN / MLP"
    if surrogate == "linear":
        return "Linear"
    if surrogate == "rbf":
        return "RBF"
    if surrogate.startswith("poly"):
        return "Polynomial"
    # gp, gp_rbf and every sequential/BO method (GP-backed) fall through here.
    return "Gaussian process"


# ── Human-readable method labels ──────────────────────────────────────────
# Figures show the same publication display names as the manuscript tables
# (single source of truth in scripts/paper_names.py).
_method_label = method_label


def _despine(ax, *, left: bool = False, bottom: bool = False) -> None:
    """Remove chartjunk spines for a clean, modern look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def _panel_label(ax, letter: str, *, dx: float = -0.12, dy: float = 1.04) -> None:
    """Bold lower-case panel tag (Nature house style) at the axes corner."""
    ax.text(
        dx, dy, letter, transform=ax.transAxes,
        fontsize=10.5, fontweight="bold", va="bottom", ha="right",
    )


FIGDATA_SOURCES = {
    1: ["fig1_case_catalog.csv", "fig1_method_catalog.csv"],
    2: ["fig2_d1_spearman.csv", "fig2_d1_ranks.csv"],
    3: ["fig3_d2_learning.csv", "fig3_d2_rule.csv"],
    4: ["fig4_d3_bias.csv"],
    5: ["fig5_mechanism.csv"],
    6: ["fig6_pareto.csv"],
}
# Combination line+halftone artwork needs 600 dpi; the PNG preview stays at 300
# dpi for on-screen inspection.  Both values are explicit for the preflight.
PREVIEW_DPI = 300
SUBMISSION_DPI = 600

ALL_FIGURES = tuple(sorted(FIGDATA_SOURCES))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flatten_tiff_to_rgb(path: Path) -> None:
    """Rewrite a submission TIFF as 8-bit RGB at its existing DPI.

    Matplotlib writes RGB TIFFs with a fully opaque alpha channel.  Nature
    Portfolio asks for RGB raster art, and an opaque alpha plane is redundant
    weight, so flatten it away while preserving resolution metadata.
    """
    from PIL import Image

    with Image.open(path) as im:
        dpi = im.info.get("dpi", (SUBMISSION_DPI, SUBMISSION_DPI))
        rgb = im.convert("RGB") if im.mode != "RGB" else im.copy()
    rgb.save(path, format="TIFF", compression="tiff_lzw",
             dpi=dpi, resolution_unit=2)


def _save_fig(fig, out: Path, name: str) -> None:
    """Write the full Nature export bundle for one figure.

    Vector PDF and SVG are the primary submission formats (text stays editable
    via ``pdf.fonttype=42`` and ``svg.fonttype='none'``); the 600 dpi TIFF is
    the raster submission copy and the 300 dpi PNG is the on-screen preview.
    """
    for sub in ("pdf", "svg", "png", "tiff"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pdf" / f"{name}.pdf", dpi=PREVIEW_DPI)
    fig.savefig(out / "svg" / f"{name}.svg")
    fig.savefig(out / "png" / f"{name}.png", dpi=PREVIEW_DPI)
    # Submission TIFF: RGB (drop the fully opaque alpha channel) with LZW
    # compression, matching the Nature/Communications Engineering request for
    # RGB raster art and keeping each file small enough to upload.
    tiff_path = out / "tiff" / f"{name}.tiff"
    fig.savefig(
        tiff_path,
        dpi=SUBMISSION_DPI,
        facecolor="white",
        pil_kwargs={
            "compression": "tiff_lzw",
            "resolution_unit": 2,  # inches, so the DPI tag is written
        },
    )
    plt.close(fig)
    _flatten_tiff_to_rgb(tiff_path)
    print(f"  [ok] {name}.pdf + {name}.svg + {name}.png + {name}.tiff")


def _manifest_path(path: Path, paper_root: Path) -> str:
    """Return a portable path for paper/figure_manifest.json."""
    try:
        return path.resolve().relative_to(paper_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _ts_slug() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace("+", "Z")


def _replace_figure_package_atomically(
    src_figures: Path,
    dst_figures: Path,
    src_manifest: Path,
    dst_manifest: Path,
) -> None:
    """Replace figures and manifest as one derived publication bundle."""
    src_figures = src_figures.resolve()
    dst_figures = dst_figures.resolve()
    src_manifest = src_manifest.resolve()
    dst_manifest = dst_manifest.resolve()
    dst_figures.parent.mkdir(parents=True, exist_ok=True)
    dst_manifest.parent.mkdir(parents=True, exist_ok=True)

    suffix = _ts_slug()
    figures_backup: Path | None = None
    manifest_backup: Path | None = None
    if dst_figures.exists():
        figures_backup = dst_figures.with_name(f".{dst_figures.name}.backup.{suffix}")
        if figures_backup.exists():
            shutil.rmtree(figures_backup)
        dst_figures.rename(figures_backup)
    if dst_manifest.exists():
        manifest_backup = dst_manifest.with_name(f".{dst_manifest.name}.backup.{suffix}")
        if manifest_backup.exists():
            if manifest_backup.is_dir():
                shutil.rmtree(manifest_backup)
            else:
                manifest_backup.unlink()
        dst_manifest.rename(manifest_backup)

    try:
        shutil.move(str(src_figures), str(dst_figures))
        shutil.move(str(src_manifest), str(dst_manifest))
    except Exception:
        if dst_figures.exists():
            shutil.rmtree(dst_figures)
        if dst_manifest.exists():
            if dst_manifest.is_dir():
                shutil.rmtree(dst_manifest)
            else:
                dst_manifest.unlink()
        if figures_backup is not None and figures_backup.exists():
            figures_backup.rename(dst_figures)
        if manifest_backup is not None and manifest_backup.exists():
            manifest_backup.rename(dst_manifest)
        raise

    if figures_backup is not None and figures_backup.exists():
        shutil.rmtree(figures_backup)
    if manifest_backup is not None and manifest_backup.exists():
        manifest_backup.unlink()


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _manifest_payload(
    *,
    figdata: Path,
    out_dir: Path,
    paper_root: Path,
    manifest: list[dict],
    errors: list[str],
    status: str = "complete",
) -> dict:
    """Assemble the figure_manifest.json payload: figure/figdata hashes plus the copied analysis-policy contract."""
    figdata_manifest = figdata / "figdata_manifest.json"
    figdata_manifest_entry = None
    analysis_policy = None
    if figdata_manifest.exists():
        figdata_manifest_entry = {
            "path": _manifest_path(figdata_manifest, paper_root),
            "sha256": _sha256(figdata_manifest),
        }
        try:
            linked_manifest = json.loads(figdata_manifest.read_text(encoding="utf-8"))
        except Exception:
            linked_manifest = {}
        if isinstance(linked_manifest, dict):
            policy = linked_manifest.get("analysis_policy")
            if isinstance(policy, dict):
                analysis_policy = policy
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/plots_nature.py",
        "status": status,
        "figdata_dir": _manifest_path(figdata, paper_root),
        "out_dir": _manifest_path(out_dir, paper_root),
        "figdata_manifest": figdata_manifest_entry,
        "analysis_policy": analysis_policy,
        "figures": manifest,
        "errors": errors,
    }


def _build_figures(
    *,
    figdata: Path,
    build_out_dir: Path,
    manifest_out_dir: Path,
    paper_root: Path,
    figures: list[int],
    dispatch: dict[int, object],
) -> tuple[list[dict], list[str]]:
    """Render the requested figures and collect their output paths for the manifest."""
    manifest = []
    errors = []
    for i in figures:
        print(f"\n[Fig {i}] generating...")
        try:
            dispatch[i](figdata, build_out_dir)
        except Exception as e:
            print(f"  [ERROR] fig{i}: {e}")
            errors.append(f"fig{i}: {e}")
            continue
        pdf = build_out_dir / "pdf" / f"fig{i}.pdf"
        svg = build_out_dir / "svg" / f"fig{i}.svg"
        png = build_out_dir / "png" / f"fig{i}.png"
        tiff = build_out_dir / "tiff" / f"fig{i}.tiff"
        missing = [p.name for p in (pdf, svg, png, tiff) if not p.exists()]
        if missing:
            errors.append(f"fig{i}: missing output file(s): {missing}")
            continue
        manifest.append({
            "figure": f"fig{i}",
            "pdf": _manifest_path(manifest_out_dir / "pdf" / f"fig{i}.pdf", paper_root),
            "svg": _manifest_path(manifest_out_dir / "svg" / f"fig{i}.svg", paper_root),
            "png": _manifest_path(manifest_out_dir / "png" / f"fig{i}.png", paper_root),
            "tiff": _manifest_path(manifest_out_dir / "tiff" / f"fig{i}.tiff", paper_root),
            "pdf_sha256": _sha256(pdf),
            "svg_sha256": _sha256(svg),
            "png_sha256": _sha256(png),
            "tiff_sha256": _sha256(tiff),
            "source_figdata": [
                {
                    "path": _manifest_path(figdata / f, paper_root),
                    "sha256": _sha256(figdata / f) if (figdata / f).exists() else None,
                }
                for f in FIGDATA_SOURCES.get(i, [])
            ],
        })
    return manifest, errors


def _filter_by_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if "pairing_mode" in df.columns:
        return df[df["pairing_mode"] == mode]
    return df


def _distinct_values(series: pd.Series) -> list:
    """Return the distinct non-missing values used to lay out a panel.

    This is a *category enumeration* for axis layout, not a row-level data
    exclusion: it never removes rows from a statistic, it only asks which
    budgets / pairing modes / analysis types are present in the frozen figdata.
    Row-level completeness is enforced upstream by
    ``scripts/prepare_figdata.py`` (exact row-count check against the
    registered grid), so no observation is silently dropped at plotting time.
    """
    return sorted(series.dropna().unique())


def _truthy_series(series: pd.Series) -> pd.Series:
    """Return a bool Series for CSV booleans such as True/False/1/0."""
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


# ─── Fig 1 ──────────────────────────────────────────────────────────────
def plot_fig1_scope(figdata: Path, out: Path) -> None:
    """Fig. 1: benchmark scope - case catalogue by domain/dimension (a) and method-pool composition (b)."""
    cases = pd.read_csv(figdata / "fig1_case_catalog.csv")
    methods = pd.read_csv(figdata / "fig1_method_catalog.csv")
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_IN, FIG_H_WIDE),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )
    # (a) case library as count-bubbles over domain × dimensionality.
    ax = axes[0]
    domains = _distinct_values(cases["domain"])
    dom_map = {d: i for i, d in enumerate(domains)}
    palette = cvd_categorical_colors(max(len(domains), 1))
    dom_color = {d: palette[i] for i, d in enumerate(domains)}
    grp = cases.groupby(["domain", "dim"]).size().reset_index(name="n")
    dimensions = sorted(cases["dim"].unique())
    dim_map = {d: i for i, d in enumerate(dimensions)}
    ax.scatter(
        grp["dim"].map(dim_map), grp["domain"].map(dom_map),
        s=grp["n"] * 95, alpha=0.9,
        c=[dom_color[d] for d in grp["domain"]],
        edgecolors="white", linewidths=0.7, zorder=3,
    )
    for _, r in grp.iterrows():
        if int(r["n"]) > 1:
            ax.text(
                dim_map[r["dim"]], dom_map[r["domain"]], str(int(r["n"])),
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=("black" if dom_color[r["domain"]] in {"#F0E442", "#56B4E9", "#E69F00"} else "white"), zorder=4,
            )
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([domain_label(d) for d in domains])
    ax.set_ylim(-0.6, len(domains) - 0.4)
    ax.set_xticks(range(len(dimensions)), [str(int(d)) for d in dimensions])
    ax.set_xlabel("Input dimensionality $d$ (distinct values)")
    ax.margins(x=0.08)
    ax.grid(axis="x", zorder=0)
    _despine(ax)
    _panel_label(ax, "a", dx=-0.30)
    # (b) method-pool composition by sampler family.
    ax = axes[1]
    fam = methods["family"].value_counts().sort_values()
    fam.index = [family_label(f) for f in fam.index]
    fam_palette = cvd_categorical_colors(max(len(fam), 1))
    # ``fam`` is sorted ascending; with the ``cividis_r`` fallback the palette
    # runs light-to-dark, so ascending order gives the largest family the
    # darkest, most salient hue.  With the Wong palette reverse so the largest
    # bar takes the first (black) slot.
    if len(fam) <= len(WONG_PALETTE):
        bar_cols = list(reversed(fam_palette[: len(fam)]))
    else:
        bar_cols = fam_palette[: len(fam)]
    ax.barh(
        fam.index, fam.values, height=0.66,
        color=bar_cols, edgecolor="white", linewidth=0.6, zorder=3,
    )
    for v, lab in zip(fam.values, fam.index):
        ax.text(v + 0.12, lab, str(int(v)), va="center", fontsize=8)
    ax.set_xlabel("Number of methods")
    ax.margins(x=0.16)
    ax.grid(axis="x", zorder=0)
    _despine(ax)
    _panel_label(ax, "b", dx=-0.22)
    fig.tight_layout(w_pad=2.2)
    _save_fig(fig, out, "fig1")


# ─── Fig 2 ──────────────────────────────────────────────────────────────
def plot_fig2_d1(figdata: Path, out: Path) -> None:
    """Fig. 2: D1 rank transfer - analytical vs physics method ranks per pairing and budget."""
    spearman = pd.read_csv(figdata / "fig2_d1_spearman.csv")
    ranks = pd.read_csv(figdata / "fig2_d1_ranks.csv")
    # Pick the two budgets that bracket the protocol's main endpoints
    # (DOE-Benchmark §4.1: D1 primary endpoint = budget 10·d; secondary
    # endpoint = budget 20·d).  Fall back to whatever budgets are
    # actually present, capped to two columns to match the paper layout.
    available_budgets = _distinct_values(spearman["budget_mult"])
    preferred = [b for b in (10, 20) if b in available_budgets]
    if len(preferred) < 2:
        preferred = available_budgets[:2]
    budgets = preferred or available_budgets[:1]
    if not budgets:
        print("  [skip] fig2: no budget_mult values in figdata/fig2_d1_spearman.csv")
        return
    # Pairing modes are taken from the data so a partial run (e.g. only
    # ``--pairing-mode dimension``) does not produce empty subplots.
    expected_modes = ["dimension", "topology", "random"]
    available_modes: list[str] = []
    if "pairing_mode" in spearman.columns:
        present = set(map(str, _distinct_values(spearman["pairing_mode"])))
        available_modes = [m for m in expected_modes if m in present]
        # Append any non-canonical modes the user may have introduced.
        available_modes += sorted(present - set(expected_modes))
    modes = available_modes or ["dimension"]
    # Landscape 2x3 small-multiples (rows = budget, cols = pairing mode) so the
    # figure stays within the double-column width instead of ballooning into a
    # full-page portrait block when set to \linewidth.
    n_r, n_c = len(budgets), len(modes)
    fig, axes = plt.subplots(
        n_r, n_c,
        figsize=(DOUBLE_COL_IN, FIG_H_GRID),
        sharex=True, sharey=True,
        squeeze=False,
    )
    nmax = 20
    if {"rank_analytical", "rank_physical"}.issubset(ranks.columns):
        try:
            nmax = int(np.nanmax([
                ranks["rank_analytical"].max(), ranks["rank_physical"].max(),
            ]))
        except (ValueError, TypeError):
            nmax = 20
    lim = (0.3, nmax + 0.7)
    for i, bm in enumerate(budgets):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            sub = _filter_by_mode(ranks, mode)
            sub = sub[sub["budget_mult"] == bm]
            # Identity line is the meaningful reference for rank agreement
            # (perfect transfer ⇒ points on y = x).  The previous 0.70
            # guide lines were a correlation threshold drawn on rank axes
            # and are therefore meaningless here.
            ax.plot(lim, lim, color="#BBBBBB", lw=0.9, ls="--", zorder=1)
            ax.scatter(
                sub["rank_analytical"], sub["rank_physical"],
                s=28, alpha=0.85, color=COLOR_PRIMARY,
                edgecolors="white", linewidths=0.5, zorder=3,
            )
            row = _filter_by_mode(spearman, mode)
            row = row[row["budget_mult"] == bm]
            if not row.empty:
                r = row.iloc[0]
                rho = r.get("spearman_rho", float("nan"))
                ci_lo = r.get("spearman_rho_ci_lo", float("nan"))
                ci_hi = r.get("spearman_rho_ci_hi", float("nan"))
                label = f"ρ = {rho:.2f}\n[{ci_lo:.2f}, {ci_hi:.2f}]"
                color = COLOR_SIG if np.isfinite(ci_lo) and ci_lo < 0.70 else COLOR_TEXT_OK
                ax.text(
                    0.05, 0.95, label, transform=ax.transAxes,
                    va="top", ha="left", fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#D0D0D0", lw=0.5, alpha=0.9),
                )
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            # Panels fill their grid cells (no forced ``equal`` aspect) so the
            # figure crops cleanly without large top/bottom whitespace; the
            # dashed y=x guide still reads correctly because every panel shares
            # identical rank limits.
            ax.set_title(f"{mode} · {int(bm)}$d$", fontsize=8.5)
            ax.grid(True, lw=0.4, alpha=0.6, zorder=0)
            _despine(ax)
    fig.supxlabel("Rank on analytical problem (1 = best)", fontsize=9.5)
    fig.supylabel("Rank on physical case (1 = best)", fontsize=9.5)
    fig.tight_layout()
    _save_fig(fig, out, "fig2")


# ─── Fig 3 ──────────────────────────────────────────────────────────────
def plot_fig3_d2(figdata: Path, out: Path) -> None:
    """Fig. 3: D2 sample complexity - learning curves by topology (a) and 10d pass fractions (b)."""
    lc = pd.read_csv(figdata / "fig3_d2_learning.csv")
    rule = pd.read_csv(figdata / "fig3_d2_rule.csv")
    topos = rule.sort_values("frac_pass_at_10d", ascending=False)["topology"].tolist()
    # GAP-F4: tab20 mixes red/green hues that collide under deuteranopia.
    # Use Wong's 8-colour palette (extended via cividis if > 8 topologies).
    colors = cvd_categorical_colors(max(len(topos), 1))
    topo_color = {t: colors[i] for i, t in enumerate(topos)}
    fig, (ax_lc, ax_bar) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_IN, FIG_H_WIDE),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )
    # (a) learning curves
    topology_markers = dict(zip(topos, ("o", "s", "^", "D", "v", "P", "X")))
    for topo in topos:
        if "topology" not in lc.columns:
            break
        sub = lc[lc["topology"] == topo]
        if sub.empty:
            continue
        agg = (
            sub.groupby("budget_mult")
            .agg(median=("median", "median"), lo=("ci_lo", "median"), hi=("ci_hi", "median"))
            .reset_index()
        )
        ax_lc.plot(
            agg["budget_mult"], agg["median"], color=topo_color[topo],
            lw=1.3, marker=topology_markers[topo], ms=3.5, label=topo.replace("_", " "),
        )
        ax_lc.fill_between(
            agg["budget_mult"], agg["lo"], agg["hi"],
            alpha=0.14, color=topo_color[topo], lw=0,
        )
    ax_lc.axhline(0.20, color=COLOR_SIG, lw=1.0, ls="--", zorder=1)
    ax_lc.text(
        ax_lc.get_xlim()[1], 0.205, "target NRMSE = 0.20",
        color=COLOR_SIG, fontsize=8, va="bottom", ha="right",
    )
    ax_lc.set_xlabel("Sampling budget (multiples of $d$)")
    ax_lc.set_ylabel("Median NRMSE")
    ax_lc.legend(fontsize=8, ncol=2, loc="upper right",
                 handlelength=1.3, columnspacing=1.0, labelspacing=0.3)
    ax_lc.grid(True, lw=0.4, alpha=0.6)
    _despine(ax_lc)
    _panel_label(ax_lc, "a", dx=-0.16)
    # (b) 10·d-rule pass rate (sorted; passing topologies highlighted)
    if "exploratory" in rule.columns:
        exploratory = _truthy_series(rule["exploratory"])
    else:
        exploratory = pd.Series(False, index=rule.index)
    rsorted = rule.assign(_explo=exploratory.values).sort_values(
        "frac_pass_at_10d", ascending=True
    ).reset_index(drop=True)
    ypos = list(range(len(rsorted)))
    bar_colors = [
        (COLOR_SECONDARY if e else (COLOR_PRIMARY if f >= 0.5 else COLOR_NSIG))
        for f, e in zip(rsorted["frac_pass_at_10d"], rsorted["_explo"])
    ]
    ax_bar.barh(
        ypos, rsorted["frac_pass_at_10d"], height=0.66,
        color=bar_colors, edgecolor="white", linewidth=0.6, zorder=3,
    )
    ax_bar.axvline(0.5, color=COLOR_SIG, lw=1.0, ls="--", zorder=2)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels([t.replace("_", " ") for t in rsorted["topology"]])
    if "n_cases" in rsorted.columns:
        for k, (f, ncs) in enumerate(zip(rsorted["frac_pass_at_10d"], rsorted["n_cases"])):
            ax_bar.text(f + 0.02, k, f"{f:.0%} (n={int(ncs)})",
                        va="center", fontsize=8)
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("Fraction of methods reaching target at 10$d$")
    ax_bar.grid(axis="x", lw=0.4, alpha=0.6)
    _despine(ax_bar)
    _panel_label(ax_bar, "b", dx=-0.34)
    fig.tight_layout(w_pad=2.2)
    _save_fig(fig, out, "fig3")


# ─── Fig 4 ──────────────────────────────────────────────────────────────
def plot_fig4_d3(figdata: Path, out: Path) -> None:
    """Fig. 4: D3 calibration - per-method signed probability-of-failure bias, coloured by surrogate family."""
    bias = pd.read_csv(figdata / "fig4_d3_bias.csv")
    if "signed_bias" in bias.columns:
        bias = bias.sort_values("signed_bias")
    bias = bias.reset_index(drop=True)
    n = len(bias)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, FIG_H_TALL))
    y = list(range(n))
    p_holm = bias.get("p_holm", pd.Series([1.0] * n)).fillna(1.0).tolist()
    # Colour encodes the surrogate family (the mechanistic driver of the
    # bias, see main-text Finding 3); non-significant methods fall back to
    # grey so significance is never over-claimed visually.
    families = [
        _surrogate_family(m)
        for m in (bias["method"] if "method" in bias.columns else [""] * n)
    ]
    pt_colors = [
        SURROGATE_FAMILY_COLORS.get(fam, COLOR_PRIMARY) if p < 0.05 else COLOR_NSIG
        for fam, p in zip(families, p_holm)
    ]
    values = bias.get("signed_bias", bias.iloc[:, 1]).tolist()
    has_ci = {"ci_lo", "ci_hi"}.issubset(bias.columns)
    ax.axvline(0, color="#888888", lw=0.8, zorder=1)
    # connector stems from zero make the magnitude pre-attentively obvious
    for yi, v, c in zip(y, values, pt_colors):
        ax.plot([0, v], [yi, yi], color=c, lw=1.1, alpha=0.30, zorder=2)
    if has_ci:
        for yi, lo, hi in zip(y, bias["ci_lo"], bias["ci_hi"]):
            ax.plot([lo, hi], [yi, yi], color="#A8A8A8", lw=0.8, zorder=3)
    for yi, v, c, fam in zip(y, values, pt_colors, families):
        ax.scatter([v], [yi], s=28, c=[c],
                   marker=SURROGATE_FAMILY_MARKERS.get(fam, "o"),
                   edgecolors="white", linewidths=0.5, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [_method_label(m) for m in bias["method"]]
        if "method" in bias.columns else y,
        fontsize=8,
    )
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Signed $P_f$ bias (predicted − true)")
    # Significance is already encoded by marker colour, so the previous
    # per-row asterisk column was redundant (and pure chartjunk when every
    # method is significant).  When the result is mixed we add a compact
    # red/grey colour key; when every method is significant a key would have
    # only one entry, so we fold the conclusion into the finding note instead.
    n_sig = sum(1 for p in p_holm if p < 0.05)
    legend_handles = [
        plt.Line2D([0], [0], marker=SURROGATE_FAMILY_MARKERS.get(fam, "o"),
                   ls="none", ms=5, mec="white",
                   mew=0.5, mfc=col, label=fam)
        for fam, col in SURROGATE_FAMILY_COLORS.items()
        if fam in set(families)
    ]
    n_cross_zero = int(((bias["ci_lo"] <= 0) & (bias["ci_hi"] >= 0)).sum())
    finding = f"{n_cross_zero}/{n} case-cluster intervals include zero"
    ax.legend(handles=legend_handles, loc="center right",
              bbox_to_anchor=(1.0, 0.45), fontsize=8, title="Surrogate family",
              title_fontsize=8, frameon=False, handletextpad=0.3,
              borderaxespad=0.4, labelspacing=0.4)
    ax.annotate(
        finding,
        xy=(0.97, 0.02), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8, color="#666666", style="italic",
    )
    ax.grid(axis="x", lw=0.4, alpha=0.6)
    _despine(ax)
    fig.tight_layout()
    _save_fig(fig, out, "fig4")


# ─── Fig 5 ──────────────────────────────────────────────────────────────
def plot_fig5_mechanism(figdata: Path, out: Path) -> None:
    """Fig. 5: mechanism strata - mean method rank by response topology at the 50d budget."""
    mech = pd.read_csv(figdata / "fig5_mechanism.csv")
    rank_col = next(
        (c for c in ["mean_rank", "rank", "nrmse_rank"] if c in mech.columns),
        None,
    )
    if rank_col is None or not {"topology", "method"}.issubset(mech.columns):
        print("  [skip] fig5: missing required columns in fig5_mechanism.csv")
        return
    if "budget_mult" in mech.columns:
        bmax = mech["budget_mult"].max()
        mech = mech[mech["budget_mult"] == bmax]
    pivot = mech.pivot_table(
        index="method", columns="topology", values=rank_col, aggfunc="mean",
    )
    exploratory = set()
    if "n_cases" in mech.columns:
        nc = mech.groupby("topology")["n_cases"].first().sort_values(ascending=False)
        pivot = pivot.reindex(columns=nc.index)
        exploratory = set(nc[nc < 3].index)
    # order methods by overall mean rank (best at top) for readability
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values().index)
    n_methods, n_topos = pivot.shape
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_IN, max(4.2, n_methods * 0.16 + 1.0)),
    )
    # GAP-F4: RdYlBu_r is unsafe under deuteranopia (red↔green axis).
    # Reverse the categorical fallback map for the rank heatmap:
    # cividis runs dark-to-light, so lower rank is darker and better.
    im = ax.imshow(
        pivot.values, cmap=mpl.colormaps[CVD_SEQUENTIAL_CMAP].reversed(),
        aspect="auto", vmin=1, vmax=max(n_methods, 2),
    )
    # annotate every cell with its mean rank, contrast-aware text colour
    for ii in range(n_methods):
        for jj in range(n_topos):
            v = pivot.values[ii, jj]
            if np.isfinite(v):
                norm = (v - 1) / max(n_methods - 1, 1)
                tc = "white" if norm < 0.45 else "#1A1A1A"
                ax.text(jj, ii, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color=tc)
    nc_lookup = (
        mech.groupby("topology")["n_cases"].first()
        if "n_cases" in mech.columns else {}
    )
    xlabels = []
    for t in pivot.columns:
        lbl = str(t).replace("_", " ")
        try:
            lbl += f"\n(n={int(nc_lookup.get(t, 0))})"
        except (TypeError, ValueError):
            pass
        xlabels.append(lbl)
    ax.set_xticks(range(n_topos))
    ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([_method_label(m) for m in pivot.index], fontsize=8)
    # white minor grid cleanly separates the cells
    ax.set_xticks(np.arange(-0.5, n_topos, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_methods, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    cbar = plt.colorbar(
        im, ax=ax, label="Mean rank (1 = best)",
        shrink=0.55, fraction=0.035, pad=0.02,
    )
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    _save_fig(fig, out, "fig5")


# ─── Fig 6 ──────────────────────────────────────────────────────────────
def _median_tradeoff_front(points: np.ndarray) -> np.ndarray:
    """Non-dominated rows when both displayed coordinates are minimised."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Trade-off points must have two columns: error and time.")
    if not np.isfinite(points).all():
        raise ValueError("Trade-off points must be finite.")
    return np.array([
        not np.any(np.all(points <= point, axis=1)
                   & np.any(points < point, axis=1))
        for point in points
    ], dtype=bool)


def plot_fig6_practice(figdata: Path, out: Path) -> None:
    """Plot every method-budget median; calculate fronts in those coordinates."""
    from matplotlib.lines import Line2D

    pareto = pd.read_csv(figdata / "fig6_pareto.csv")
    if pareto.empty:
        raise ValueError("The accuracy-time source table is empty.")
    budgets = _distinct_values(pareto["budget_mult"])
    time_col = "time_median_s" if "time_median_s" in pareto.columns else "wall_time_median"
    types = ["complete_case", "penalized_failed_in_denominator"]
    if set(pareto["analysis_type"]) != set(types):
        raise ValueError("Both successful-run and failure-inclusive summaries are required.")
    if len(budgets) != 4:
        raise ValueError("The trade-off figure requires all four registered budgets.")
    coordinates = pareto[["nrmse_median", time_col]].to_numpy(dtype=float)
    if (not np.isfinite(coordinates).all()
            or np.any(coordinates[:, 0] < 0)
            or np.any(coordinates[:, 1] <= 0)):
        raise ValueError("Median errors must be nonnegative and log-axis times positive.")
    if pareto.duplicated(["analysis_type", "budget_mult", "method"]).any():
        raise ValueError("Each policy, budget and method must identify one point.")
    title_map = {
        "complete_case": "Successful runs only",
        "penalized_failed_in_denominator": "Including failed runs",
    }
    markers = ("o", "s", "^", "D")
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_IN, 3.65), sharex=True, sharey=True,
    )
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.30, top=0.86, wspace=0.10)
    for ax, atype, letter in zip(axes, types, ("a", "b")):
        for bm, col, marker in zip(budgets, PALETTE_B, markers):
            sub = pareto[(pareto["budget_mult"] == bm)
                         & (pareto["analysis_type"] == atype)]
            if len(sub) != 20:
                raise ValueError(f"{atype}, budget {bm}: expected all 20 methods.")
            # The archived flags were derived using means. They do not define
            # the frontier of the median coordinates displayed in this figure.
            mask = _median_tradeoff_front(sub[["nrmse_median", time_col]].to_numpy())
            non_p, on_p = sub[~mask], sub[mask]
            ax.scatter(non_p["nrmse_median"], non_p[time_col],
                       marker=marker, s=24, facecolors="none",
                       edgecolors=col, linewidths=0.8, alpha=0.55, zorder=2)
            ax.scatter(on_p["nrmse_median"], on_p[time_col],
                       marker=marker, s=32, facecolors=col,
                       edgecolors="#222222", linewidths=0.45, zorder=4)
            front = on_p.sort_values("nrmse_median")
            ax.step(front["nrmse_median"], front[time_col],
                    color=col, lw=0.9, where="post", alpha=0.8, zorder=3)
        ax.set_yscale("log")
        ax.set_xlim(0, float(coordinates[:, 0].max()) * 1.06)
        ax.set_ylim(float(coordinates[:, 1].min()) / 1.8,
                    float(coordinates[:, 1].max()) * 1.8)
        ax.set_xlabel("Median NRMSE", labelpad=5)
        ax.set_title(title_map[atype], pad=10)
        ax.text(0, 1.12, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", ha="left", va="bottom")
        ax.grid(True, which="major", lw=0.4, alpha=0.45)
        _despine(ax)
    axes[0].set_ylabel("Median elapsed time (s)", labelpad=6)
    legend_handles = [
        Line2D([], [], marker=m, color=c, linestyle="none",
               markersize=5, label=f"{int(b)}$d$")
        for b, c, m in zip(budgets, PALETTE_B, markers)
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.54, 0.10), ncol=4,
               title="Training budget", fontsize=8, title_fontsize=8,
               columnspacing=1.8, handletextpad=0.4)
    fig.text(0.54, 0.065,
             "Filled + line: non-dominated    Open: dominated",
             ha="center", fontsize=8, color="#333333")
    fig.text(0.54, 0.012,
             "One point = one method at one budget; lower and left are better",
             ha="center", fontsize=8, color="#555555")
    _save_fig(fig, out, "fig6")


# ─── CLI ─────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate Nature-quality figures from figdata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--figdata", type=Path,
        default=FIGDATA_DIR,
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=FIGURES_DIR,
    )
    p.add_argument(
        "--fig", type=int, nargs="+", default=list(range(1, 7)),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paper_root = args.out_dir.resolve().parent
    dispatch = {
        1: plot_fig1_scope, 2: plot_fig2_d1,
        3: plot_fig3_d2, 4: plot_fig4_d3,
        5: plot_fig5_mechanism, 6: plot_fig6_practice,
    }

    selection_errors = []
    for i in args.fig:
        if i not in dispatch:
            selection_errors.append(f"fig{i}: not implemented")
    if len(set(args.fig)) != len(args.fig):
        selection_errors.append("duplicate figure ids are not allowed")
    if selection_errors:
        print("[ERROR] invalid figure selection:")
        for e in selection_errors:
            print(f"  - {e}")
        return 2

    manifest_path = args.out_dir.parent / "figure_manifest.json"
    full_package = tuple(sorted(args.fig)) == ALL_FIGURES

    if full_package:
        tmp_parent = Path(tempfile.mkdtemp(
            prefix=f".{args.out_dir.name}.tmp.",
            dir=str(args.out_dir.resolve().parent),
        ))
        tmp_out = tmp_parent / args.out_dir.name
        tmp_manifest = tmp_parent / "figure_manifest.json"
        try:
            manifest, errors = _build_figures(
                figdata=args.figdata,
                build_out_dir=tmp_out,
                manifest_out_dir=args.out_dir,
                paper_root=paper_root,
                figures=args.fig,
                dispatch=dispatch,
            )
            payload = _manifest_payload(
                figdata=args.figdata,
                out_dir=args.out_dir,
                paper_root=paper_root,
                manifest=manifest,
                errors=errors,
                status="failed" if errors or len(manifest) != len(args.fig) else "complete",
            )
            if errors or len(manifest) != len(args.fig):
                print("[ERROR] full figure package failed; existing figures unchanged.")
                _write_manifest(tmp_manifest, payload)
                return 1
            _write_manifest(tmp_manifest, payload)
            _replace_figure_package_atomically(
                tmp_out,
                args.out_dir,
                tmp_manifest,
                manifest_path,
            )
        finally:
            if tmp_parent.exists():
                shutil.rmtree(tmp_parent)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(
            "\n[warn] partial --fig selection; writing selected figures directly "
            "without full-package replacement."
        )
        manifest, errors = _build_figures(
            figdata=args.figdata,
            build_out_dir=args.out_dir,
            manifest_out_dir=args.out_dir,
            paper_root=paper_root,
            figures=args.fig,
            dispatch=dispatch,
        )
        _write_manifest(
            manifest_path,
            _manifest_payload(
                figdata=args.figdata,
                out_dir=args.out_dir,
                paper_root=paper_root,
                manifest=manifest,
                errors=[
                    *errors,
                    "partial figure package produced with --fig; rerun all figures before submission",
                ],
                status="partial_fig_selection",
            ),
        )

    print(f"\n[ok] wrote {manifest_path}")
    if errors:
        print("[ERROR] some figures failed:")
        for e in errors:
            print(f"  - {e}")
        return 1
    if len(manifest) != len(args.fig):
        print(f"[ERROR] expected {len(args.fig)} figures, wrote {len(manifest)}")
        return 1
    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
