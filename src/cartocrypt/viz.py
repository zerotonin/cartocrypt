"""Publication-quality figure generation for CartoCrypt.

« Wong palette, SVG-first, significance brackets included »
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from cartocrypt.constants import (
    COLOURS,
    DEFAULT_FIG_SIZE,
    FIGURE_DPI,
    SVG_FONTTYPE,
    Coords,
)

matplotlib.rcParams["svg.fonttype"] = SVG_FONTTYPE


def plot_comparison(
    g: nx.Graph,
    original_coords: Coords,
    anon_coords: Coords,
    *,
    title: str = "CartoCrypt: Original vs Anonymised",
    out_path: Path | None = None,
    figsize: tuple[float, float] = DEFAULT_FIG_SIZE,
) -> matplotlib.figure.Figure:
    """Side-by-side plot of original and anonymised graph layouts.

    Args:
        g: Shared graph structure.
        original_coords: (N, 2) original positions.
        anon_coords: (N, 2) anonymised positions.
        title: Figure suptitle.
        out_path: If provided, saves SVG + PNG to this stem.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    _draw_graph(ax1, g, original_coords, colour=COLOURS["orange"], label="Original")
    _draw_graph(ax2, g, anon_coords, colour=COLOURS["sky_blue"], label="Anonymised")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if out_path is not None:
        svg_path = out_path.with_suffix(".svg")
        png_path = out_path.with_suffix(".png")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=FIGURE_DPI, bbox_inches="tight")

    return fig


def plot_length_error(
    g: nx.Graph,
    original_coords: Coords,
    anon_coords: Coords,
    *,
    out_path: Path | None = None,
) -> matplotlib.figure.Figure:
    """Histogram of relative edge-length errors.

    Args:
        g: Shared graph.
        original_coords: Original positions.
        anon_coords: Anonymised positions.
        out_path: Save path stem (optional).

    Returns:
        Matplotlib Figure object.
    """
    nodes = list(g.nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    errors = []
    for u, v in g.edges():
        i, j = node_idx[u], node_idx[v]
        l_orig = float(np.linalg.norm(original_coords[i] - original_coords[j]))
        l_anon = float(np.linalg.norm(anon_coords[i] - anon_coords[j]))
        if l_orig > 1e-12:
            errors.append((l_anon - l_orig) / l_orig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, color=COLOURS["bluish_green"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color=COLOURS["vermilion"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Relative edge-length error")
    ax.set_ylabel("Count")
    ax.set_title("Edge-length preservation")
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), format="png", dpi=FIGURE_DPI, bbox_inches="tight")

    return fig


# ─────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────


def _draw_graph(
    ax: matplotlib.axes.Axes,
    g: nx.Graph,
    coords: Coords,
    *,
    colour: str,
    label: str,
) -> None:
    """Draw a graph on a matplotlib axes."""
    nodes = list(g.nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    for u, v in g.edges():
        i, j = node_idx[u], node_idx[v]
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color=colour, alpha=0.5, linewidth=0.8,
        )

    ax.scatter(coords[:, 0], coords[:, 1], s=8, color=colour, zorder=5)
    ax.set_title(label, fontsize=12)
    ax.set_aspect("equal")
    ax.tick_params(labelbottom=False, labelleft=False)
