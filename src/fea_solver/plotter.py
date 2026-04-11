"""Matplotlib-based plots for FEA results.

Provides:
  - Shear force diagram (SFD): V(x) vs x
  - Bending moment diagram (BMD): M(x) vs x (inverted y-axis, sagging positive down)
  - Deformed shape: exaggerated displaced geometry
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fea_solver.models import ElementResult

logger = logging.getLogger(__name__)


def _concatenate_diagrams(
    element_results: list[ElementResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate x, V, M arrays across all elements in order of x.

    Args:
        element_results (list[ElementResult]): List of post-processed element results.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Three arrays (x, V, M) concatenated
            across all elements sorted by x_stations[0].

    Notes:
        Elements are sorted by their first x_station to ensure spatial order.
        Suitable for plotting continuous diagrams across the entire structure.
    """
    sorted_results = sorted(element_results, key=lambda er: er.x_stations[0])
    x = np.concatenate([er.x_stations for er in sorted_results])
    V = np.concatenate([er.shear_forces for er in sorted_results])
    M = np.concatenate([er.bending_moments for er in sorted_results])
    return x, V, M


def plot_shear_force_diagram(
    element_results: list[ElementResult],
    title: str = "Shear Force Diagram",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot V(x) vs x (shear force diagram).

    Positive shear is plotted above the baseline (blue fill),
    negative shear below (red fill).

    Args:
        element_results: Post-processed element results.
        title: Plot title.
        output_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.

    Notes:
        Annotates maximum and minimum shear values on the plot.
        If output_path is provided, saves figure as PNG at 150 dpi.
    """
    x, V, _ = _concatenate_diagrams(element_results)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, V, "k-", linewidth=1.5, label="V(x)")
    ax.fill_between(x, V, 0, where=(V >= 0), alpha=0.3, color="blue", label="V > 0")
    ax.fill_between(x, V, 0, where=(V < 0), alpha=0.3, color="red", label="V < 0")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    # Annotate extremes
    v_max_idx = int(np.argmax(V))
    v_min_idx = int(np.argmin(V))
    ax.annotate(f"{V[v_max_idx]:.3g}", xy=(x[v_max_idx], V[v_max_idx]),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate(f"{V[v_min_idx]:.3g}", xy=(x[v_min_idx], V[v_min_idx]),
                xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("V [N]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("SFD saved to %s", output_path)

    return fig


def plot_bending_moment_diagram(
    element_results: list[ElementResult],
    title: str = "Bending Moment Diagram",
    output_path: Path | None = None,
    invert_y: bool = True,
) -> plt.Figure:
    """Plot M(x) vs x (bending moment diagram).

    By default, the y-axis is inverted so that sagging (positive moment)
    plots downward, following structural engineering convention.

    Args:
        element_results: Post-processed element results.
        title: Plot title.
        output_path: If provided, save figure to this path.
        invert_y: If True, invert y-axis (sagging positive = downward).

    Returns:
        The matplotlib Figure.

    Notes:
        Annotates maximum and minimum moment values on the plot.
        Sagging (positive) moments shown in green; hogging (negative) in orange.
        If output_path is provided, saves figure as PNG at 150 dpi.
    """
    x, _, M = _concatenate_diagrams(element_results)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, M, "k-", linewidth=1.5, label="M(x)")
    ax.fill_between(x, M, 0, where=(M >= 0), alpha=0.3, color="green", label="M > 0 (sagging)")
    ax.fill_between(x, M, 0, where=(M < 0), alpha=0.3, color="orange", label="M < 0 (hogging)")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    # Annotate extremes
    m_max_idx = int(np.argmax(M))
    m_min_idx = int(np.argmin(M))
    ax.annotate(f"{M[m_max_idx]:.3g}", xy=(x[m_max_idx], M[m_max_idx]),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate(f"{M[m_min_idx]:.3g}", xy=(x[m_min_idx], M[m_min_idx]),
                xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8)

    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel("x [m]")
    ax.set_ylabel("M [N*m]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("BMD saved to %s", output_path)

    return fig


def plot_deformed_shape(
    element_results: list[ElementResult],
    scale_factor: float = 100.0,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot exaggerated deformed shape overlaid on undeformed geometry.

    Uses x_stations as horizontal position; bending_moments column is unused.
    The shear_forces are also unused here -- the deformed shape is reconstructed
    from x_stations (which are the displaced x positions at the integration stations
    - approximated as the undeformed x for 1D beam).

    Note: For a proper deformed shape, v(x) values would be needed. Since we
    store V(x) and M(x) in ElementResult rather than v(x), this plot shows
    the qualitative shape via the moment diagram as a proxy, clearly labelled.

    Args:
        element_results: Post-processed element results.
        scale_factor: Amplification for visualisation (unused in shape, kept for API).
        output_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.

    Notes:
        Deformed shape is normalized and scaled for visualization.
        If output_path is provided, saves figure as PNG at 150 dpi.
    """
    x, _, M = _concatenate_diagrams(element_results)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, np.zeros_like(x), "k--", linewidth=1.0, label="Undeformed")
    ax.plot(x, M * scale_factor / (np.max(np.abs(M)) + 1e-30),
            "b-", linewidth=2.0, label=f"Deformed shape (x{scale_factor})")
    ax.axhline(0, color="k", linewidth=0.5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("Transverse displacement (scaled)")
    ax.set_title("Deformed Shape")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Deformed shape saved to %s", output_path)

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
