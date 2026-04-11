"""Matplotlib-based plots for FEA results.

All axis labels are derived from model.unit_system so that empirical and SI
problems display the correct units without any hard-coded strings.

Max and min values are shown as distinct markers (circle / square) with their
numeric values recorded in the legend, not as floating text annotations.

Provides:
  - plot_shear_force_diagram:       V(x) vs x
  - plot_bending_moment_diagram:    M(x) vs x (inverted y-axis, sagging positive down)
  - plot_transverse_displacement:   v(x) with physical units on both axes
  - plot_axial_displacement:        u(x) with physical units on both axes
  - plot_rotation:                  theta(x) in radians
  - show_all_plots:                 plt.show() wrapper

_concatenate_diagrams:  Sorts and concatenates x, V, M, v, u, theta across elements.
_unit_labels:           Returns UNIT_LABELS dict for the model's unit system.
_plot_extremes:         Adds max/min markers with legend entries to an Axes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fea_solver.models import ElementResult, FEAModel
from fea_solver.units import UNIT_LABELS

logger = logging.getLogger(__name__)


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
) -> None:
    """Plot max and min of y as distinct markers; record values in the legend.

    Args:
        ax (plt.Axes): Axes to annotate.
        x (np.ndarray): x-coordinate array parallel to y.
        y (np.ndarray): y-value array.
        color (str): Marker fill colour. Default "black".
        fmt (str): Format string for legend value text. Default ".4g".

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

    ax.plot(
        x[max_idx], y[max_idx],
        marker="o", color=color, markersize=7, linestyle="none", zorder=5,
        label=f"max = {y[max_idx]:{fmt}}",
    )
    if max_idx != min_idx:
        ax.plot(
            x[min_idx], y[min_idx],
            marker="s", color=color, markersize=7, linestyle="none", zorder=5,
            label=f"min = {y[min_idx]:{fmt}}",
        )


def plot_shear_force_diagram(
    element_results: list[ElementResult],
    model: FEAModel,
    title: str = "Shear",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot V(x) vs x (shear force diagram) with unit-aware axis labels.

    Positive shear is plotted above the baseline (blue fill),
    negative shear below (red fill). Max and min are shown as markers
    whose values appear in the legend.

    Args:
        element_results (list[ElementResult]): Post-processed element results.
        model (FEAModel): FEA model; used to determine axis unit labels.
        title (str): Plot title. Default "Shear".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        If output_path is provided, saves figure at 150 dpi.
    """
    lbl = _unit_labels(model)
    x, V, *_ = _concatenate_diagrams(element_results)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, V, "k-", linewidth=1.5, label="V(x)")
    ax.fill_between(x, V, 0, where=(V >= 0), alpha=0.3, color="blue", label="V > 0")
    ax.fill_between(x, V, 0, where=(V < 0), alpha=0.3, color="red", label="V < 0")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    _plot_extremes(ax, x, V)

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
    element_results: list[ElementResult],
    model: FEAModel,
    title: str = "Moments",
    output_path: Path | None = None,
    invert_y: bool = True,
) -> plt.Figure:
    """Plot M(x) vs x (bending moment diagram) with unit-aware axis labels.

    By default, the y-axis is inverted so that sagging (positive moment)
    plots downward, following structural engineering convention. Max and min
    are shown as markers whose values appear in the legend.

    Args:
        element_results (list[ElementResult]): Post-processed element results.
        model (FEAModel): FEA model; used to determine axis unit labels.
        title (str): Plot title. Default "Moments".
        output_path (Path | None): If provided, save figure to this path as PNG.
        invert_y (bool): If True, invert y-axis (sagging positive = downward).

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        Sagging (positive) moments shown in green; hogging (negative) in orange.
        If output_path is provided, saves figure at 150 dpi.
    """
    lbl = _unit_labels(model)
    x, _, M, *__ = _concatenate_diagrams(element_results)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, M, "k-", linewidth=1.5, label="M(x)")
    ax.fill_between(x, M, 0, where=(M >= 0), alpha=0.3, color="green", label="M > 0 (sagging)")
    ax.fill_between(x, M, 0, where=(M < 0), alpha=0.3, color="orange", label="M < 0 (hogging)")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    _plot_extremes(ax, x, M)

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
    element_results: list[ElementResult],
    model: FEAModel,
    title: str = "Vertical Displacement",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot transverse displacement v(x) in physical units.

    No scale factor is applied; the y-axis shows actual displacement values
    with the correct units for the problem. Max and min are shown as markers
    whose values appear in the legend.

    Args:
        element_results (list[ElementResult]): Post-processed element results.
        model (FEAModel): FEA model; used to determine axis unit labels.
        title (str): Plot title. Default "Vertical Displacement".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        v(x) is recovered via Hermite cubic shape functions in the postprocessor.
        If output_path is provided, saves figure at 150 dpi.
    """
    lbl = _unit_labels(model)
    x, _, _, v, *__ = _concatenate_diagrams(element_results)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, v, "b-", linewidth=1.5, label="v(x)")
    ax.fill_between(x, v, 0, alpha=0.15, color="blue")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    _plot_extremes(ax, x, v)

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
    element_results: list[ElementResult],
    model: FEAModel,
    title: str = "Axial Displacement",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot axial displacement u(x) in physical units.

    For BAR and FRAME elements, u(x) is linearly interpolated from nodal
    axial DOFs. For pure BEAM elements, u(x) is identically zero. Max and
    min are shown as markers whose values appear in the legend.

    Args:
        element_results (list[ElementResult]): Post-processed element results.
        model (FEAModel): FEA model; used to determine axis unit labels.
        title (str): Plot title. Default "Axial Displacement".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        If output_path is provided, saves figure at 150 dpi.
    """
    lbl = _unit_labels(model)
    x, _, _, _, u_ax, _ = _concatenate_diagrams(element_results)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, u_ax, "r-", linewidth=1.5, label="u(x)")
    ax.fill_between(x, u_ax, 0, alpha=0.15, color="red")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    _plot_extremes(ax, x, u_ax)

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
    element_results: list[ElementResult],
    model: FEAModel,
    title: str = "Rotation",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot cross-section rotation theta(x) in radians.

    theta(x) = dv/dx, recovered from Hermite shape function first derivatives.
    For BAR elements, rotation is identically zero. Max and min are shown as
    markers whose values appear in the legend.

    Args:
        element_results (list[ElementResult]): Post-processed element results.
        model (FEAModel): FEA model; used to determine axis unit labels.
        title (str): Plot title. Default "Rotation".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        Rotation is always in radians regardless of unit system.
        If output_path is provided, saves figure at 150 dpi.
    """
    lbl = _unit_labels(model)
    x, _, _, _, _, theta = _concatenate_diagrams(element_results)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, theta, "g-", linewidth=1.5, label="theta(x)")
    ax.fill_between(x, theta, 0, alpha=0.15, color="green")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    _plot_extremes(ax, x, theta)

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
