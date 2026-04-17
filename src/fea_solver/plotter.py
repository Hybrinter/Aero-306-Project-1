"""Matplotlib-based plots for FEA results.

All axis labels are derived from model.unit_system so that empirical and SI
problems display the correct units without any hard-coded strings.

Max and min values are shown as distinct markers (circle / square) with their
numeric values recorded in the legend, not as floating text annotations.

For single-solution plots, fill_between shading is used to emphasise sign.
For multi-solution overlay plots (more than one SolutionSeries), fill_between
is suppressed so overlapping fills do not obscure one another; each series is
drawn in a distinct color from _SERIES_COLORS.

Provides:
  - plot_shear_force_diagram:       V(x) vs x
  - plot_bending_moment_diagram:    M(x) vs x (inverted y-axis, sagging positive down)
  - plot_transverse_displacement:   v(x) with physical units on both axes
  - plot_axial_displacement:        u(x) with physical units on both axes
  - plot_rotation:                  theta(x) in radians
  - plot_truss_axial_forces:        2D wireframe with color-coded member forces (tension/compression)
  - show_all_plots:                 plt.show() wrapper

_SERIES_COLORS:         List of hex color strings cycled across multiple series.
_SERIES_LINESTYLES:     List of line style strings cycled across multiple series
                        (solid, dashed, dash-dot, dotted) for greyscale legibility.
_concatenate_diagrams:  Sorts and concatenates x, V, M, v, u, theta across elements.
_unit_labels:           Returns UNIT_LABELS dict for the model's unit system.
_plot_extremes:         Adds max/min markers with legend entries to an Axes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np

from fea_solver.models import DOFType, ElementResult, FEAModel, SolutionSeries
from fea_solver.units import UNIT_LABELS

logger = logging.getLogger(__name__)

# Tableau palette hex codes -- used in order for successive series in overlay plots.
# Cycles if more than 6 solutions are present (rare for this project).
_SERIES_COLORS: list[str] = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
]

# Line styles cycled across successive series so that solutions remain distinguishable
# in greyscale or when printed without color. First series is always solid.
_SERIES_LINESTYLES: list[str] = ["-", "--", "-.", ":"]


def _unit_labels(model: FEAModel) -> dict[str, str]:
    """Return the UNIT_LABELS dict for the canonical system of model.

    Args:
        model (FEAModel): FEA model whose unit_system is queried.

    Returns:
        dict[str, str]: Mapping of quantity names to unit label strings.
            Keys: length, displacement, rotation, force, moment, distributed.

    Notes:
        Thin wrapper around UNIT_LABELS so callers do not need to import units directly.
    """
    return UNIT_LABELS[model.unit_system]


def _concatenate_diagrams(
    element_results: list[ElementResult],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Concatenate x, V, M, v, u, theta arrays across all elements in spatial order.

    Args:
        element_results (list[ElementResult]): List of post-processed element results.

    Returns:
        tuple of six NDArrays (x, V, M, v, u, theta), each concatenated across
        all elements sorted by x_stations[0] (ascending spatial order).

    Notes:
        Elements are sorted by their first x_station to ensure spatial continuity.
        Suitable for plotting continuous diagrams across the entire structure.
    """
    sorted_results = sorted(element_results, key=lambda er: er.x_stations[0])
    x = np.concatenate([er.x_stations for er in sorted_results])
    V = np.concatenate([er.shear_forces for er in sorted_results])
    M = np.concatenate([er.bending_moments for er in sorted_results])
    v = np.concatenate([er.transverse_displacements for er in sorted_results])
    u_ax = np.concatenate([er.axial_displacements for er in sorted_results])
    theta = np.concatenate([er.rotations for er in sorted_results])
    return x, V, M, v, u_ax, theta


def _plot_extremes(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str = "black",
    fmt: str = ".4g",
    series_label: str = "",
) -> None:
    """Plot max and min of y as distinct markers; record values in the legend.

    Args:
        ax (plt.Axes): Axes to annotate.
        x (np.ndarray): x-coordinate array parallel to y.
        y (np.ndarray): y-value array.
        color (str): Marker fill colour. Default "black".
        fmt (str): Format string for legend value text. Default ".4g".
        series_label (str): If non-empty, prepended to legend text so multiple
            series can be distinguished. E.g. "coarse max = 5.0" vs "fine max = 4.9".
            Default "" (produces "max = 5.0" without a prefix).

    Returns:
        None

    Notes:
        Maximum is drawn as a filled circle (marker "o").
        Minimum is drawn as a filled square (marker "s").
        If max and min are at the same index (e.g. all-zero array), only the
        max marker is plotted.
        Both markers are added as legend handles so the caller must call
        ax.legend() after this function.
    """
    max_idx = int(np.argmax(y))
    min_idx = int(np.argmin(y))
    prefix = f"{series_label} " if series_label else ""

    ax.plot(
        x[max_idx], y[max_idx],
        marker="o", color=color, markersize=7, linestyle="none", zorder=5,
        label=f"{prefix}max = {y[max_idx]:{fmt}}",
    )
    if max_idx != min_idx:
        ax.plot(
            x[min_idx], y[min_idx],
            marker="s", color=color, markersize=7, linestyle="none", zorder=5,
            label=f"{prefix}min = {y[min_idx]:{fmt}}",
        )


def _truss_colormap_norm(values: list[float]) -> tuple[Colormap, TwoSlopeNorm]:
    """Return a coolwarm Colormap and a TwoSlopeNorm centered at zero.

    The norm is symmetric: vmin = -max_abs, vcenter = 0, vmax = max_abs.
    Falls back to vmin=-1, vmax=1 when all values are zero to avoid a
    degenerate norm.

    Args:
        values (list[float]): Scalar values to be represented (e.g. axial forces
            or stresses). May contain positive, negative, or zero entries.

    Returns:
        tuple[Colormap, TwoSlopeNorm]: The coolwarm colormap and the norm.

    Notes:
        Tension (positive) maps to the red end; compression (negative) maps to
        the blue end of coolwarm. The symmetric range ensures zero always maps
        to the neutral white midpoint regardless of force sign distribution.
    """
    max_abs = max((abs(v) for v in values), default=0.0)
    if max_abs == 0.0:
        max_abs = 1.0
    cmap: Colormap = plt.cm.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    return cmap, norm


def _truss_node_displacements(sol: SolutionSeries) -> dict[int, tuple[float, float]]:
    """Extract per-node (U, V) global displacements from a SolutionSeries.

    Args:
        sol (SolutionSeries): Solution bundle whose result.displacements and
            result.dof_map are used to recover nodal translations.

    Returns:
        dict[int, tuple[float, float]]: Maps node_id to (U, V) displacement tuple
            in the model's canonical length units. All nodes in sol.model.mesh.nodes
            are present as keys.

    Notes:
        TRUSS elements always carry both U and V DOFs at every node, so this
        function is safe for all truss meshes. Using it on non-truss models
        (BAR, BEAM) that lack U or V DOFs will raise KeyError.
    """
    u_vec = sol.result.displacements
    dof_map = sol.result.dof_map
    disps: dict[int, tuple[float, float]] = {}
    for node in sol.model.mesh.nodes:
        U = float(u_vec[dof_map.index(node.id, DOFType.U)])
        V = float(u_vec[dof_map.index(node.id, DOFType.V)])
        disps[node.id] = (U, V)
    return disps


def plot_shear_force_diagram(
    series: list[SolutionSeries],
    title: str = "Shear",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot V(x) vs x (shear force diagram) for one or more solutions on shared axes.

    For a single solution, preserves fill_between shading (positive shear: blue,
    negative shear: red). For multiple solutions, fill_between is suppressed to
    avoid visual clutter; each series is drawn in a distinct color from
    _SERIES_COLORS. Max and min markers are shown per series with the series
    label included in the legend entry when multiple series are present.

    Args:
        series (list[SolutionSeries]): One or more solution bundles to overlay.
            Must be non-empty. Unit labels are taken from series[0].model.
        title (str): Plot title. Default "Shear".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Raises:
        ValueError: If series is empty.

    Notes:
        If output_path is provided, saves figure at 150 dpi.
    """
    if not series:
        raise ValueError("series must be non-empty")

    lbl = _unit_labels(series[0].model)
    multi = len(series) > 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    for i, sol in enumerate(series):
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        linestyle = _SERIES_LINESTYLES[i % len(_SERIES_LINESTYLES)]
        x, V, *_ = _concatenate_diagrams(list(sol.element_results))
        ax.plot(x, V, color=color, linestyle=linestyle, linewidth=1.5, label=f"V(x) [{sol.label}]")
        if not multi:
            ax.fill_between(x, V, 0, where=(V >= 0), alpha=0.3, color="blue", label="V > 0")
            ax.fill_between(x, V, 0, where=(V < 0), alpha=0.3, color="red", label="V < 0")
        _plot_extremes(ax, x, V, color=color, series_label=sol.label if multi else "")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"V [{lbl['force']}]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Shear plot saved to %s", output_path)

    return fig


def plot_bending_moment_diagram(
    series: list[SolutionSeries],
    title: str = "Moments",
    output_path: Path | None = None,
    invert_y: bool = True,
) -> plt.Figure:
    """Plot M(x) vs x (bending moment diagram) for one or more solutions on shared axes.

    By default, the y-axis is inverted so that sagging (positive moment)
    plots downward, following structural engineering convention. For a single
    solution, fill_between shading is used (sagging: green, hogging: orange).
    For multiple solutions, fill_between is suppressed; each series uses a
    distinct color from _SERIES_COLORS.

    Args:
        series (list[SolutionSeries]): One or more solution bundles to overlay.
            Must be non-empty. Unit labels taken from series[0].model.
        title (str): Plot title. Default "Moments".
        output_path (Path | None): If provided, save figure to this path as PNG.
        invert_y (bool): If True, invert y-axis (sagging positive = downward).

    Returns:
        plt.Figure: The matplotlib Figure.

    Raises:
        ValueError: If series is empty.

    Notes:
        If output_path is provided, saves figure at 150 dpi.
    """
    if not series:
        raise ValueError("series must be non-empty")

    lbl = _unit_labels(series[0].model)
    multi = len(series) > 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    for i, sol in enumerate(series):
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        linestyle = _SERIES_LINESTYLES[i % len(_SERIES_LINESTYLES)]
        x, _, M, *_ = _concatenate_diagrams(list(sol.element_results))
        ax.plot(x, M, color=color, linestyle=linestyle, linewidth=1.5, label=f"M(x) [{sol.label}]")
        if not multi:
            ax.fill_between(x, M, 0, where=(M >= 0), alpha=0.3,
                            color="green", label="M > 0 (sagging)")
            ax.fill_between(x, M, 0, where=(M < 0), alpha=0.3,
                            color="orange", label="M < 0 (hogging)")
        _plot_extremes(ax, x, M, color=color, series_label=sol.label if multi else "")

    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"M [{lbl['moment']}]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Moments plot saved to %s", output_path)

    return fig


def plot_transverse_displacement(
    series: list[SolutionSeries],
    title: str = "Vertical Displacement",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot transverse displacement v(x) for one or more solutions on shared axes.

    No scale factor is applied; the y-axis shows actual displacement values
    with the correct units for the problem. For a single solution, a light blue
    fill_between is shown. For multiple solutions, fill_between is suppressed;
    each series uses a distinct color.

    Args:
        series (list[SolutionSeries]): One or more solution bundles to overlay.
            Must be non-empty. Unit labels taken from series[0].model.
        title (str): Plot title. Default "Vertical Displacement".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Raises:
        ValueError: If series is empty.

    Notes:
        v(x) is recovered via Hermite cubic shape functions in the postprocessor.
        If output_path is provided, saves figure at 150 dpi.
    """
    if not series:
        raise ValueError("series must be non-empty")

    lbl = _unit_labels(series[0].model)
    multi = len(series) > 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    for i, sol in enumerate(series):
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        linestyle = _SERIES_LINESTYLES[i % len(_SERIES_LINESTYLES)]
        x, _, _, v, *_ = _concatenate_diagrams(list(sol.element_results))
        ax.plot(x, v, color=color, linestyle=linestyle, linewidth=1.5, label=f"v(x) [{sol.label}]")
        if not multi:
            ax.fill_between(x, v, 0, alpha=0.15, color="blue")
        _plot_extremes(ax, x, v, color=color, series_label=sol.label if multi else "")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"v(x) [{lbl['displacement']}]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Vertical displacement plot saved to %s", output_path)

    return fig


def plot_axial_displacement(
    series: list[SolutionSeries],
    title: str = "Axial Displacement",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot axial displacement u(x) for one or more solutions on shared axes.

    For BAR and FRAME elements, u(x) is linearly interpolated from nodal axial
    DOFs. For pure BEAM elements, u(x) is identically zero. For a single
    solution, a light red fill_between is shown. For multiple solutions,
    fill_between is suppressed; each series uses a distinct color.

    Args:
        series (list[SolutionSeries]): One or more solution bundles to overlay.
            Must be non-empty. Unit labels taken from series[0].model.
        title (str): Plot title. Default "Axial Displacement".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Raises:
        ValueError: If series is empty.

    Notes:
        If output_path is provided, saves figure at 150 dpi.
    """
    if not series:
        raise ValueError("series must be non-empty")

    lbl = _unit_labels(series[0].model)
    multi = len(series) > 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    for i, sol in enumerate(series):
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        linestyle = _SERIES_LINESTYLES[i % len(_SERIES_LINESTYLES)]
        x, _, _, _, u_ax, _ = _concatenate_diagrams(list(sol.element_results))
        ax.plot(x, u_ax, color=color, linestyle=linestyle, linewidth=1.5, label=f"u(x) [{sol.label}]")
        if not multi:
            ax.fill_between(x, u_ax, 0, alpha=0.15, color="red")
        _plot_extremes(ax, x, u_ax, color=color, series_label=sol.label if multi else "")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"u(x) [{lbl['displacement']}]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Axial displacement plot saved to %s", output_path)

    return fig


def plot_rotation(
    series: list[SolutionSeries],
    title: str = "Rotation",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot cross-section rotation theta(x) for one or more solutions on shared axes.

    theta(x) = dv/dx, recovered from Hermite shape function first derivatives.
    For BAR elements, rotation is identically zero. For a single solution, a
    light green fill_between is shown. For multiple solutions, fill_between is
    suppressed; each series uses a distinct color.

    Args:
        series (list[SolutionSeries]): One or more solution bundles to overlay.
            Must be non-empty. Unit labels taken from series[0].model.
        title (str): Plot title. Default "Rotation".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Raises:
        ValueError: If series is empty.

    Notes:
        Rotation is always in radians regardless of unit system.
        If output_path is provided, saves figure at 150 dpi.
    """
    if not series:
        raise ValueError("series must be non-empty")

    lbl = _unit_labels(series[0].model)
    multi = len(series) > 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    for i, sol in enumerate(series):
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        linestyle = _SERIES_LINESTYLES[i % len(_SERIES_LINESTYLES)]
        x, _, _, _, _, theta = _concatenate_diagrams(list(sol.element_results))
        ax.plot(x, theta, color=color, linestyle=linestyle, linewidth=1.5, label=f"theta(x) [{sol.label}]")
        if not multi:
            ax.fill_between(x, theta, 0, alpha=0.15, color="green")
        _plot_extremes(ax, x, theta, color=color, series_label=sol.label if multi else "")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"theta(x) [{lbl['rotation']}]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Rotation plot saved to %s", output_path)

    return fig


def plot_truss_axial_forces(
    sol: SolutionSeries,
    title: str = "Truss Member Forces",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot 2D truss geometry with color-coded member axial forces.

    Members in tension are drawn in blue; members in compression are drawn in red.
    Each member is annotated with its axial force value at midpoint.

    Args:
        sol (SolutionSeries): Single solution bundle containing element results and model.
        title (str): Plot title. Default "Truss Member Forces".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        Node coordinates are taken from model.mesh nodes. Positive axial force (tension)
        produces blue members; negative (compression) produces red members.
        If output_path is provided, saves figure at 150 dpi.
    """
    model = sol.model
    lbl = _unit_labels(model)

    # Build element id -> ElementResult lookup
    result_by_id = {er.element_id: er for er in sol.element_results}
    nodes_by_id = {n.id: n for n in model.mesh.nodes}

    fig, ax = plt.subplots(figsize=(10, 6))

    for element in model.mesh.elements:
        n_i = nodes_by_id[element.node_i.id]
        n_j = nodes_by_id[element.node_j.id]
        er = result_by_id.get(element.id)
        N = er.axial_force if er is not None else 0.0

        color = "#1f77b4" if N >= 0.0 else "#d62728"  # blue=tension, red=compression
        ax.plot([n_i.x, n_j.x], [n_i.y, n_j.y], color=color, linewidth=2.0)

        # Annotate midpoint with force value
        mid_x = (n_i.x + n_j.x) / 2.0
        mid_y = (n_i.y + n_j.y) / 2.0
        ax.text(mid_x, mid_y, f"{N:.3g}", fontsize=7, ha="center", va="bottom",
                color=color)

    # Draw nodes
    for node in model.mesh.nodes:
        ax.plot(node.x, node.y, "ko", markersize=4, zorder=5)
        ax.text(node.x, node.y, f" {node.id}", fontsize=8, va="bottom")

    # Legend entries for tension/compression
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", linewidth=2, label="Tension (N > 0)"),
        Line2D([0], [0], color="#d62728", linewidth=2, label="Compression (N < 0)"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"y [{lbl['length']}]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Truss plot saved to %s", output_path)

    return fig


def show_all_plots(figures: list[plt.Figure]) -> None:
    """Display all figures. Call plt.show() once after all figures are ready.

    Args:
        figures (list[plt.Figure]): List of matplotlib Figure objects to display.

    Returns:
        None

    Notes:
        Blocks execution until all plot windows are closed.
        Only use when not saving plots to disk.
    """
    plt.show()
