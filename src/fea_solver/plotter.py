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
  - plot_truss_forces:              2D undeformed wireframe colored by axial force gradient
  - plot_truss_deformed:            2D deformed wireframe with auto-scale, colored by axial force;
                                    optional buckling overlay draws a half-sine bow on failed members.
  - plot_truss_stress:              2D undeformed wireframe colored by axial stress (N/A) gradient
  - show_all_plots:                 plt.show() wrapper

_SERIES_COLORS:             List of hex color strings cycled across multiple series.
_SERIES_LINESTYLES:         List of line style strings cycled across multiple series
                            (solid, dashed, dash-dot, dotted) for greyscale legibility.
_concatenate_diagrams:      Sorts and concatenates x, V, M, v, u, theta across elements.
_unit_labels:               Returns UNIT_LABELS dict for the model's unit system.
_plot_extremes:             Adds max/min markers with legend entries to an Axes.
_truss_colormap_norm:       Returns coolwarm Colormap + TwoSlopeNorm centered at zero.
_truss_node_displacements:  Extracts per-node (U, V) from SolutionSeries.result.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np

from fea_solver.models import DOFType, ElementResult, FEAModel, MemberBuckling, SolutionSeries
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
    cmap: Colormap = plt.get_cmap("coolwarm")
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


def plot_truss_forces(
    sol: SolutionSeries,
    title: str = "Truss Member Forces",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot 2D truss geometry (undeformed) with coolwarm gradient coloring by axial force.

    Members are colored on a continuous diverging scale: blue for compression
    (N < 0), white for zero force, red for tension (N > 0). A colorbar shows
    the force scale. Each member is annotated at its midpoint with the numeric
    force value.

    Args:
        sol (SolutionSeries): Solution bundle containing element results and model.
        title (str): Plot title. Default "Truss Member Forces".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        Node coordinates are the original (undeformed) positions.
        If output_path is provided, saves figure at 150 dpi.
    """
    model = sol.model
    lbl = _unit_labels(model)
    result_by_id = {er.element_id: er for er in sol.element_results}
    nodes_by_id = {n.id: n for n in model.mesh.nodes}

    forces = [
        result_by_id[e.id].axial_force if e.id in result_by_id else 0.0
        for e in model.mesh.elements
    ]
    cmap, norm = _truss_colormap_norm(forces)

    fig, ax = plt.subplots(figsize=(10, 6))

    for element, N in zip(model.mesh.elements, forces):
        n_i = nodes_by_id[element.node_i.id]
        n_j = nodes_by_id[element.node_j.id]
        color = cmap(norm(N))
        ax.plot([n_i.x, n_j.x], [n_i.y, n_j.y], color=color, linewidth=2.5)
        mid_x = (n_i.x + n_j.x) / 2.0
        mid_y = (n_i.y + n_j.y) / 2.0
        ax.text(mid_x, mid_y, f"{N:.3g}", fontsize=7, ha="center", va="bottom",
                color="black")

    for node in model.mesh.nodes:
        ax.plot(node.x, node.y, "ko", markersize=4, zorder=5)
        ax.text(node.x, node.y, f" {node.id}", fontsize=8, va="bottom")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"N [{lbl['force']}]")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"y [{lbl['length']}]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Truss forces plot saved to %s", output_path)

    return fig


def plot_truss_deformed(
    sol: SolutionSeries,
    title: str = "Truss Deformed Shape",
    output_path: Path | None = None,
    buckling: tuple[MemberBuckling, ...] | None = None,
) -> plt.Figure:
    """Plot 2D truss geometry in its deformed state with coolwarm gradient by axial force.

    Node positions are shifted by (scale * U, scale * V) where scale is chosen
    automatically so the largest displacement equals 10% of the bounding-box
    diagonal. The scale factor is appended to the plot title.

    When buckling is provided, each entry whose is_buckled == True draws a
    half-sine lateral bow on top of the deformed member line. The bow has
    amplitude 0.1 * element.length (original undeformed length) along the unit
    perpendicular to the deformed-member axis; it is drawn as a black dashed
    line so it reads as a buckling-mode indicator rather than additional
    geometry.

    Args:
        sol (SolutionSeries): Solution bundle. sol.result.displacements provides
            nodal translations; sol.element_results provides axial forces for color.
        title (str): Base plot title. Scale factor is appended automatically.
            Default "Truss Deformed Shape".
        output_path (Path | None): If provided, save figure to this path as PNG.
        buckling (tuple[MemberBuckling, ...] | None): Optional per-element
            buckling results. None (default) preserves the prior behaviour
            exactly. When provided, members with is_buckled=True receive a
            half-sine overlay.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        Scale factor formula: scale = 0.1 * bbox_diagonal / max_abs_displacement.
        Falls back to scale = 1.0 when all displacements are zero.
        bbox_diagonal = hypot(max_x - min_x, max_y - min_y) over original node coords.
        If output_path is provided, saves figure at 150 dpi.
    """
    model = sol.model
    lbl = _unit_labels(model)
    result_by_id = {er.element_id: er for er in sol.element_results}
    nodes_by_id = {n.id: n for n in model.mesh.nodes}
    node_disps = _truss_node_displacements(sol)
    buckled_ids: set[int] = (
        {mb.element_id for mb in buckling if mb.is_buckled}
        if buckling is not None else set()
    )

    xs = np.array([n.x for n in model.mesh.nodes])
    ys = np.array([n.y for n in model.mesh.nodes])
    bbox_diag = float(np.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min())))
    if bbox_diag == 0.0:
        bbox_diag = 1.0
    all_disp_mags = [abs(d) for U, V in node_disps.values() for d in (U, V)]
    max_disp = max(all_disp_mags) if all_disp_mags else 0.0
    scale = 0.1 * bbox_diag / max_disp if max_disp > 0.0 else 1.0

    forces = [
        result_by_id[e.id].axial_force if e.id in result_by_id else 0.0
        for e in model.mesh.elements
    ]
    cmap, norm = _truss_colormap_norm(forces)

    fig, ax = plt.subplots(figsize=(10, 6))

    for element, N in zip(model.mesh.elements, forces):
        n_i = nodes_by_id[element.node_i.id]
        n_j = nodes_by_id[element.node_j.id]
        U_i, V_i = node_disps[n_i.id]
        U_j, V_j = node_disps[n_j.id]
        x_i_def = n_i.x + scale * U_i
        y_i_def = n_i.y + scale * V_i
        x_j_def = n_j.x + scale * U_j
        y_j_def = n_j.y + scale * V_j
        color = cmap(norm(N))
        ax.plot([x_i_def, x_j_def], [y_i_def, y_j_def], color=color, linewidth=2.5)
        mid_x = (x_i_def + x_j_def) / 2.0
        mid_y = (y_i_def + y_j_def) / 2.0
        ax.text(mid_x, mid_y, f"{N:.3g}", fontsize=7, ha="center", va="bottom",
                color="black")

        if element.id in buckled_ids:
            dx = x_j_def - x_i_def
            dy = y_j_def - y_i_def
            chord = float(np.hypot(dx, dy))
            if chord > 0.0:
                cos_a = dx / chord
                sin_a = dy / chord
                # Amplitude tied to undeformed length (not deformed chord) so
                # the overlay size does not scale with the displacement scale
                # factor. Perpendicular direction (-sin_a, cos_a) bulges to the
                # left of the chord by convention.
                amp = 0.1 * element.length
                xi = np.linspace(0.0, 1.0, 30)
                bow = amp * np.sin(np.pi * xi)
                bx = x_i_def + xi * dx + bow * (-sin_a)
                by = y_i_def + xi * dy + bow * (cos_a)
                ax.plot(bx, by, color="black", linestyle="--",
                        linewidth=1.5, zorder=4)

    for node in model.mesh.nodes:
        U, V = node_disps[node.id]
        x_def = node.x + scale * U
        y_def = node.y + scale * V
        ax.plot(x_def, y_def, "ko", markersize=4, zorder=5)
        ax.text(x_def, y_def, f" {node.id}", fontsize=8, va="bottom")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"N [{lbl['force']}]")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"y [{lbl['length']}]")
    ax.set_title(f"{title} (scale {scale:.2g}x)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Truss deformed plot saved to %s", output_path)

    return fig


def plot_truss_stress(
    sol: SolutionSeries,
    title: str = "Truss Member Stresses",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot 2D truss geometry (undeformed) with coolwarm gradient coloring by axial stress.

    Axial stress per member: sigma = N / A, where N is the axial force from
    ElementResult and A is the element's cross-sectional area. Members are
    colored on a continuous diverging scale (blue = compression, red = tension)
    with a colorbar showing the stress magnitude. Each member is annotated at
    its midpoint with the numeric stress value.

    Args:
        sol (SolutionSeries): Solution bundle containing element results and model.
        title (str): Plot title. Default "Truss Member Stresses".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        Stress unit label is composed as f"{force_unit}/{length_unit}^2"
        (e.g. "N/m^2" for SI, "lb/in^2" for empirical). No changes to units.py needed.
        Node coordinates are the original (undeformed) positions.
        If output_path is provided, saves figure at 150 dpi.
    """
    model = sol.model
    lbl = _unit_labels(model)
    result_by_id = {er.element_id: er for er in sol.element_results}
    nodes_by_id = {n.id: n for n in model.mesh.nodes}

    stresses = [
        (result_by_id[e.id].axial_force if e.id in result_by_id else 0.0) / e.material.A
        for e in model.mesh.elements
    ]
    cmap, norm = _truss_colormap_norm(stresses)
    stress_unit = f"{lbl['force']}/{lbl['length']}^2"

    fig, ax = plt.subplots(figsize=(10, 6))

    for element, sigma in zip(model.mesh.elements, stresses):
        n_i = nodes_by_id[element.node_i.id]
        n_j = nodes_by_id[element.node_j.id]
        color = cmap(norm(sigma))
        ax.plot([n_i.x, n_j.x], [n_i.y, n_j.y], color=color, linewidth=2.5)
        mid_x = (n_i.x + n_j.x) / 2.0
        mid_y = (n_i.y + n_j.y) / 2.0
        ax.text(mid_x, mid_y, f"{sigma:.3g}", fontsize=7, ha="center", va="bottom",
                color="black")

    for node in model.mesh.nodes:
        ax.plot(node.x, node.y, "ko", markersize=4, zorder=5)
        ax.text(node.x, node.y, f" {node.id}", fontsize=8, va="bottom")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"sigma [{stress_unit}]")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"y [{lbl['length']}]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Truss stress plot saved to %s", output_path)

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
