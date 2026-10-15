"""
doe_testbed.utils.visualization
===================================
Response-surface visualisation utilities for individual cases.

Functions
---------
plot_response_surface  — Fix remaining variables and plot a 2-D heatmap
save_figure            — Unified figure-saving interface (auto-creates directories)
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np


def _get_plt():
    """Lazy-import matplotlib to avoid errors in headless environments."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 120,
        }
    )
    return plt


def plot_response_surface(
    case,
    dim_x: int = 0,
    dim_y: int = 1,
    fixed_values: Optional[Dict[int, float]] = None,
    output_idx: int = 0,
    n_grid: int = 50,
    figsize: Tuple[float, float] = (6, 5),
    cmap: str = "viridis",
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot a 2-D response-surface heatmap with remaining variables fixed.

    Parameters
    ----------
    case : BaseCase
        Any simulation case object.
    dim_x, dim_y : int
        Input dimensions for the x/y axes (0-indexed, normalised space).
    fixed_values : dict, optional
        Fixed values for other dimensions (normalised [0, 1]); None fixes all at 0.5.
    output_idx : int
        Which output to plot, default 0.
    n_grid : int
        Grid points per axis; total evaluations = n_grid * n_grid.
    cmap : str
        Colormap name.
    title : str
        Plot title; empty string auto-generates one.
    save_path : str, optional
        If specified, save the figure to this path.
    show : bool
        Whether to call ``plt.show()``, default True.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from doe_testbed import get_case
    >>> from doe_testbed.utils import plot_response_surface
    >>> case = get_case("cantilever_beam_5d")
    >>> plot_response_surface(case, dim_x=0, dim_y=3)
    """
    plt = _get_plt()

    d = case.meta.dim_input
    fixed = {i: 0.5 for i in range(d) if i not in (dim_x, dim_y)}
    if fixed_values:
        fixed.update(fixed_values)

    xi = np.linspace(0.0, 1.0, n_grid)
    yi = np.linspace(0.0, 1.0, n_grid)
    Xg, Yg = np.meshgrid(xi, yi)

    X_in = np.full((n_grid * n_grid, d), 0.5)
    X_in[:, dim_x] = Xg.ravel()
    X_in[:, dim_y] = Yg.ravel()
    for dim_i, val in fixed.items():
        X_in[:, dim_i] = val

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Y_out = case.evaluate(X_in)[:, output_idx].reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(Xg, Yg, Y_out, cmap=cmap, shading="gouraud")

    out_vars = case.meta.output_variables
    out_label = out_vars[output_idx].name if out_vars else f"y[{output_idx}]"
    plt.colorbar(pcm, ax=ax, label=out_label)

    in_vars = case.meta.input_variables
    xlabel = in_vars[dim_x].symbol if in_vars else f"$x_{{{dim_x+1}}}$"
    ylabel = in_vars[dim_y].symbol if in_vars else f"$x_{{{dim_y+1}}}$"
    ax.set_xlabel(f"{xlabel} (normalized)", fontsize=11)
    ax.set_ylabel(f"{ylabel} (normalized)", fontsize=11)

    if not title:
        title = f"{case.meta.case_id} — {out_label}"
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def save_figure(fig, path: str, dpi: int = 150, **kwargs) -> None:
    """Unified figure-saving interface; auto-creates parent directories.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str
        Save path (extension determines format: .png / .pdf / .svg ...).
    dpi : int
        Resolution, default 150.
    """
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
