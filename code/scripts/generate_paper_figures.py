#!/usr/bin/env python
"""Generate publication-quality figures for paper/fig/.

Figures produced
----------------
1. sobol_4panel.pdf   – Sobol ST bar charts for 4 representative cases
2. response_cantilever.pdf – 2-D response surface (cantilever_beam_5d)
3. response_airfoil.pdf    – 2-D response surface (airfoil_stall_2d)
4. response_pressure.pdf   – 2-D response surface (pressure_vessel_4d)
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure doe_testbed is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import doe_testbed as db

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "fig")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# 1. Sobol ST 4-panel bar chart
# ──────────────────────────────────────────────────────────────────
CASES_4 = [
    "cantilever_beam_5d",
    "airfoil_stall_2d",
    "pressure_vessel_4d",
    "ten_bar_truss_10d",
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.ravel()

for ax, cid in zip(axes, CASES_4, strict=False):
    case = db.get_case(cid)
    st = case.meta.sobol_ST
    if isinstance(next(iter(st.values())), dict):
        first_out = next(iter(st.keys()))
        names = list(st[first_out].keys())
        vals = np.array([st[first_out][n] for n in names])
    else:
        names = list(st.keys())
        vals = np.array(list(st.values()))

    order = np.argsort(vals)[::-1]
    names_sorted = [names[i] for i in order]
    vals_sorted = vals[order]

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(vals_sorted)))
    bars = ax.barh(
        range(len(vals_sorted)), vals_sorted, color=colors, edgecolor="0.3", linewidth=0.5
    )
    ax.set_yticks(range(len(vals_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("$S_T$", fontsize=10)
    ax.set_title(cid.replace("_", r"\_"), fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.axvline(0.05, color="gray", ls="--", lw=0.8, alpha=0.6)

fig.tight_layout()
path1 = os.path.join(OUT_DIR, "sobol_4panel.pdf")
fig.savefig(path1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {path1}")


# ──────────────────────────────────────────────────────────────────
# 2. Response surface heatmaps
# ──────────────────────────────────────────────────────────────────
def plot_response_surface(
    case_id, var_x, var_y, output_idx, output_label, n_grid=80, title_extra="", fname=None
):
    """Plot 2-D response surface slice, fixing other vars at midpoint."""
    case = db.get_case(case_id)
    d = case.meta.dim_input
    var_names = [v.name for v in case.meta.input_variables]
    ix = var_names.index(var_x)
    iy = var_names.index(var_y)

    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    xx, yy = np.meshgrid(x, y)
    X_grid = np.full((n_grid * n_grid, d), 0.5)
    X_grid[:, ix] = xx.ravel()
    X_grid[:, iy] = yy.ravel()

    Y = case.evaluate(X_grid)
    Z = Y[:, output_idx].reshape(n_grid, n_grid)

    # Physical labels
    vx = case.meta.input_variables[ix]
    vy = case.meta.input_variables[iy]
    x_phys = np.linspace(vx.lower, vx.upper, n_grid)
    y_phys = np.linspace(vy.lower, vy.upper, n_grid)
    if vx.log_scale:
        x_phys = np.logspace(np.log10(vx.lower), np.log10(vx.upper), n_grid)
    if vy.log_scale:
        y_phys = np.logspace(np.log10(vy.lower), np.log10(vy.upper), n_grid)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    cf = ax.contourf(x_phys, y_phys, Z, levels=30, cmap="RdYlBu_r")
    ax.contour(x_phys, y_phys, Z, levels=10, colors="k", linewidths=0.4, alpha=0.5)

    # Failure threshold contour if present
    ov = case.meta.output_variables[output_idx]
    if ov.failure_threshold is not None:
        ax.contour(
            x_phys,
            y_phys,
            Z,
            levels=[ov.failure_threshold],
            colors="red",
            linewidths=2.0,
            linestyles="--",
        )

    cbar = fig.colorbar(cf, ax=ax, shrink=0.9)
    cbar.set_label(output_label, fontsize=10)

    xlabel = f"${vx.symbol}$"
    ylabel = f"${vy.symbol}$"
    if vx.unit and vx.unit != "-":
        xlabel += f" ({vx.unit})"
    if vy.unit and vy.unit != "-":
        ylabel += f" ({vy.unit})"
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    if vx.log_scale:
        ax.set_xscale("log")
    if vy.log_scale:
        ax.set_yscale("log")

    ax.set_title(
        f"{case_id.replace('_', chr(95))} — {output_label}{title_extra}",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, fname or f"response_{case_id}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# Cantilever beam: sigma_max vs (F, h) — shows interaction
plot_response_surface(
    "cantilever_beam_5d",
    "F",
    "h",
    output_idx=0,
    output_label=r"$\sigma_{\max}$ (MPa)",
    title_extra=" (other vars at midpoint)",
    fname="response_cantilever.pdf",
)

# Airfoil stall: CL vs (alpha, Re) — shows stall discontinuity
plot_response_surface(
    "airfoil_stall_2d",
    "alpha",
    "Re",
    output_idx=0,
    output_label=r"$C_L$",
    title_extra=" (Ma, t/c at midpoint)",
    fname="response_airfoil.pdf",
)

# Pressure vessel: FS vs (P, t) — shows failure boundary
plot_response_surface(
    "pressure_vessel_4d",
    "P",
    "t",
    output_idx=1,
    output_label="Safety Factor (FS)",
    title_extra=r" ($R$, $\sigma_y$ at midpoint)",
    fname="response_pressure.pdf",
)

print("All paper figures generated successfully.")
