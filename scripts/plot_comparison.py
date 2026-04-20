"""Side-by-side comparison plot: baseline vs. optimized truss deformation.

Solves both YAML cases on the same FE pipeline, then renders the deformed
shapes in two adjacent panels using:

  - the same displacement amplification scale (so deformations are visually
    comparable, not auto-scaled per-panel),
  - the same axis limits (so the two panels share the same physical extent),
  - the same diverging force colormap norm (so red and blue mean the same
    axial-force magnitude in both panels).

Members that fail Euler buckling are overlaid with a dashed half-sine bow,
matching the convention used by plotter.plot_truss_deformed. Each panel's
title shows the case stiffness K = |F_load| / |v_load_node| and the count
of buckled members so the comparison reads at a glance.

Run:
    uv run python scripts/plot_comparison.py
    uv run python scripts/plot_comparison.py \\
        --baseline config/problem_7.yaml \\
        --optimized optimization_runs/2026-04-17_heavy/best_design.yaml \\
        --output optimization_runs/2026-04-17_heavy/plots/baseline_vs_best.png

Helpers:
    parse_args:      CLI argument parser.
    _Case:           Bundle of (label, model, result, element_results, buckling,
                     node_disps, forces) used as the input to _draw_panel.
    _solve_case:     Load YAML, run the assemble -> solve -> postprocess -> buckling
                     pipeline, return a populated _Case.
    _tip_node_id:    Locate the node carrying the headline POINT_FORCE_Y load.
    _shared_axes:    Compute the shared (xlim, ylim, scale, max_abs_force) tuple
                     across all input cases so both panels share rendering state.
    _draw_panel:     Render one deformed truss into an Axes using shared scale,
                     shared norm, shared limits, and the buckling overlay.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede any other matplotlib import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.buckling import compute_truss_buckling
from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.models import (
    DOFType,
    ElementResult,
    FEAModel,
    LoadType,
    MemberBuckling,
    SolutionResult,
)
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.solver import run_solve_pipeline
from fea_solver.units import UNIT_LABELS

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison plot script.

    Returns:
        argparse.Namespace: Parsed arguments with the following fields:
            baseline (Path): YAML for the baseline (un-optimized) geometry.
            optimized (Path): YAML for the optimized geometry.
            output (Path): Destination PNG path; parent dir is created on save.
            n_stations (int): Sampling stations forwarded to postprocess_all_elements.
            dpi (int): Output figure resolution.

    Notes:
        Defaults point at the heavy-run output committed under
        optimization_runs/2026-04-17_heavy/, so the script can be invoked with
        no arguments after a fresh clone.
    """
    p = argparse.ArgumentParser(
        description="Side-by-side baseline vs optimized truss deformation plot."
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("config/problem_7.yaml"),
        help="Baseline YAML case file.",
    )
    p.add_argument(
        "--optimized",
        type=Path,
        default=Path("optimization_runs/2026-04-17_heavy/best_design.yaml"),
        help="Optimized YAML case file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("optimization_runs/2026-04-17_heavy/plots/baseline_vs_best.png"),
        help="Output PNG path. Parent directory is created if needed.",
    )
    p.add_argument(
        "--n-stations",
        type=int,
        default=2,
        help="Sampling stations per element forwarded to postprocess_all_elements.",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


@dataclass(frozen=True, slots=True)
class _Case:
    """Solved truss case packaged for comparison plotting.

    Fields:
        label (str): Short panel title (e.g. "Baseline", "Optimized").
        model (FEAModel): FE model (geometry, materials, loads, BCs).
        result (SolutionResult): Forward-solve displacements + reactions + DOFMap.
        element_results (tuple[ElementResult, ...]): Per-element internals.
        buckling (tuple[MemberBuckling, ...]): Per-element Euler buckling outcomes.
        node_disps (dict[int, tuple[float, float]]): node_id -> (U, V) translation
            in canonical model length units.
        forces (tuple[float, ...]): Axial force per element in mesh order; aligned
            with model.mesh.elements.

    Notes:
        Frozen and slotted for consistency with the rest of the project.
        Defensive default: elements without a matching ElementResult contribute
        a zero axial force so the plot remains drawable.
    """

    label: str
    model: FEAModel
    result: SolutionResult
    element_results: tuple[ElementResult, ...]
    buckling: tuple[MemberBuckling, ...]
    node_disps: dict[int, tuple[float, float]]
    forces: tuple[float, ...]


def _tip_node_id(model: FEAModel) -> int:
    """Return the node id carrying the dominant POINT_FORCE_Y load.

    Args:
        model (FEAModel): Solved or unsolved FE model.

    Returns:
        int: id of the nodal load with the largest |magnitude| among all
            POINT_FORCE_Y entries.

    Raises:
        ValueError: If the model has no POINT_FORCE_Y nodal load.

    Notes:
        Used to compute K = |F| / |v_tip|, the headline stiffness metric in
        each panel title.
    """
    candidates = [
        nl for nl in model.nodal_loads if nl.load_type == LoadType.POINT_FORCE_Y
    ]
    if not candidates:
        raise ValueError("Model has no POINT_FORCE_Y nodal load")
    tip = max(candidates, key=lambda nl: abs(nl.magnitude))
    return tip.node_id


def _solve_case(label: str, yaml_path: Path, n_stations: int) -> _Case:
    """Load a YAML case, run the FE pipeline, and bundle into a _Case.

    Args:
        label (str): Panel title to assign to the resulting _Case.
        yaml_path (Path): Path to a YAML case file.
        n_stations (int): Sampling stations per element passed to
            postprocess_all_elements.

    Returns:
        _Case: Fully populated case ready to render.

    Notes:
        Pipeline: load_models_from_yaml -> build_dof_map -> assemble K, F ->
        run_solve_pipeline -> postprocess_all_elements -> compute_truss_buckling.
        Element axial forces fall back to 0.0 for any element id missing from
        the postprocessor output (defensive; shouldn't happen for a well-formed
        truss model).
    """
    model = load_models_from_yaml(yaml_path)[0]
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    elem_results = postprocess_all_elements(model, result, n_stations=n_stations)
    buckling = compute_truss_buckling(model, elem_results)

    node_disps: dict[int, tuple[float, float]] = {}
    for node in model.mesh.nodes:
        U = float(result.displacements[dof_map.index(node.id, DOFType.U)])
        V = float(result.displacements[dof_map.index(node.id, DOFType.V)])
        node_disps[node.id] = (U, V)

    result_by_id = {er.element_id: er for er in elem_results}
    forces = tuple(
        result_by_id[e.id].axial_force if e.id in result_by_id else 0.0
        for e in model.mesh.elements
    )

    return _Case(
        label=label,
        model=model,
        result=result,
        element_results=tuple(elem_results),
        buckling=tuple(buckling),
        node_disps=node_disps,
        forces=forces,
    )


def _shared_axes(
    cases: tuple[_Case, ...],
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    """Compute shared rendering parameters across all panels.

    Args:
        cases (tuple[_Case, ...]): Cases to be rendered together.

    Returns:
        tuple containing:
            scale (float): Displacement amplification factor; targets a max
                deformation of 10% of the combined undeformed bbox diagonal.
            max_abs_force (float): Largest |axial force| across all cases;
                used to define the symmetric TwoSlopeNorm color range.
            xlim (tuple[float, float]): Shared x-axis limits computed on the
                deformed coordinates of all cases (with 5% padding).
            ylim (tuple[float, float]): Shared y-axis limits, same convention.

    Notes:
        bbox_diagonal is computed on undeformed node coordinates so the scale
        formula matches plotter.plot_truss_deformed but combines both cases.
        Falls back to bbox_diag = 1.0 and max_disp -> scale = 1.0 in degenerate
        all-coincident-nodes / zero-displacement edge cases.
    """
    all_xs = np.array(
        [n.x for case in cases for n in case.model.mesh.nodes], dtype=np.float64
    )
    all_ys = np.array(
        [n.y for case in cases for n in case.model.mesh.nodes], dtype=np.float64
    )
    bbox_diag = float(
        np.hypot(
            float(all_xs.max() - all_xs.min()),
            float(all_ys.max() - all_ys.min()),
        )
    )
    if bbox_diag == 0.0:
        bbox_diag = 1.0

    max_disp = max(
        (
            abs(d)
            for case in cases
            for U, V in case.node_disps.values()
            for d in (U, V)
        ),
        default=0.0,
    )
    scale = 0.1 * bbox_diag / max_disp if max_disp > 0.0 else 1.0

    max_abs_force = max(
        (abs(N) for case in cases for N in case.forces), default=0.0
    )
    if max_abs_force == 0.0:
        max_abs_force = 1.0

    deformed_xs: list[float] = []
    deformed_ys: list[float] = []
    for case in cases:
        for node in case.model.mesh.nodes:
            U, V = case.node_disps[node.id]
            deformed_xs.append(node.x + scale * U)
            deformed_ys.append(node.y + scale * V)
    xs_arr = np.asarray(deformed_xs, dtype=np.float64)
    ys_arr = np.asarray(deformed_ys, dtype=np.float64)
    pad_x = 0.05 * float(xs_arr.max() - xs_arr.min() or 1.0)
    pad_y = 0.05 * float(ys_arr.max() - ys_arr.min() or 1.0)
    xlim = (float(xs_arr.min()) - pad_x, float(xs_arr.max()) + pad_x)
    ylim = (float(ys_arr.min()) - pad_y, float(ys_arr.max()) + pad_y)

    return scale, max_abs_force, xlim, ylim


def _draw_panel(
    ax: plt.Axes,
    case: _Case,
    scale: float,
    cmap: Colormap,
    norm: TwoSlopeNorm,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    tip_node_id: int,
) -> None:
    """Render one case into the supplied Axes using shared rendering state.

    Args:
        ax (plt.Axes): Target subplot.
        case (_Case): Case to render.
        scale (float): Shared displacement amplification factor.
        cmap (Colormap): Shared diverging colormap (coolwarm).
        norm (TwoSlopeNorm): Shared symmetric color norm centered at zero.
        xlim (tuple[float, float]): Shared x-axis limits.
        ylim (tuple[float, float]): Shared y-axis limits.
        tip_node_id (int): Node carrying the headline POINT_FORCE_Y load;
            highlighted with a red-edged marker and a "load" label so the
            reader can locate it without cluttering every node with an id.

    Returns:
        None

    Notes:
        Buckled members receive a dashed half-sine bow drawn perpendicular to
        the deformed chord (amplitude = 0.1 * undeformed_length). Convention
        matches plotter.plot_truss_deformed so the two scripts read identically.
        Only the tip/load node is labeled; all other nodes are drawn as plain
        black dots to keep the plot uncluttered on dense meshes.
    """
    nodes_by_id = {n.id: n for n in case.model.mesh.nodes}
    buckled_ids = {mb.element_id for mb in case.buckling if mb.is_buckled}

    for element, N in zip(case.model.mesh.elements, case.forces):
        n_i = nodes_by_id[element.node_i.id]
        n_j = nodes_by_id[element.node_j.id]
        U_i, V_i = case.node_disps[n_i.id]
        U_j, V_j = case.node_disps[n_j.id]
        x_i = n_i.x + scale * U_i
        y_i = n_i.y + scale * V_i
        x_j = n_j.x + scale * U_j
        y_j = n_j.y + scale * V_j
        color = cmap(norm(N))
        ax.plot([x_i, x_j], [y_i, y_j], color=color, linewidth=2.5)

        if element.id in buckled_ids:
            dx = x_j - x_i
            dy = y_j - y_i
            chord = float(np.hypot(dx, dy))
            if chord > 0.0:
                cos_a = dx / chord
                sin_a = dy / chord
                amp = 0.1 * element.length
                xi = np.linspace(0.0, 1.0, 30)
                bow = amp * np.sin(np.pi * xi)
                bx = x_i + xi * dx + bow * (-sin_a)
                by = y_i + xi * dy + bow * cos_a
                ax.plot(
                    bx, by,
                    color="black", linestyle="--", linewidth=1.5, zorder=4,
                )

    for node in case.model.mesh.nodes:
        U, V = case.node_disps[node.id]
        x_def = node.x + scale * U
        y_def = node.y + scale * V
        if node.id == tip_node_id:
            ax.plot(
                x_def, y_def, "o",
                markersize=7, markerfacecolor="white",
                markeredgecolor="red", markeredgewidth=1.5, zorder=6,
            )
            ax.annotate(
                "load", xy=(x_def, y_def), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color="red",
            )
        else:
            ax.plot(x_def, y_def, "ko", markersize=4, zorder=5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def main() -> None:
    """Render the side-by-side comparison plot and save to disk.

    Returns:
        None

    Notes:
        Logs at INFO level via the module logger; also prints a one-line
        confirmation to stdout for interactive use.
        Closes the figure after saving so repeated invocations don't leak
        Matplotlib state.
        Uses constrained_layout so the colorbar matches the axes height
        without manual positioning. The displacement scale factor is baked
        into the colorbar label rather than a separate suptitle.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    baseline = _solve_case("Baseline", args.baseline, args.n_stations)
    optimized = _solve_case("Optimized", args.optimized, args.n_stations)
    cases = (baseline, optimized)

    scale, max_abs_force, xlim, ylim = _shared_axes(cases)
    cmap: Colormap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-max_abs_force, vcenter=0.0, vmax=max_abs_force)

    lbl = UNIT_LABELS[baseline.model.unit_system]

    tip_ids: list[int] = [_tip_node_id(case.model) for case in cases]
    tip_disps: list[float] = [
        abs(case.node_disps[tip_id][1]) for case, tip_id in zip(cases, tip_ids)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, case, tip_id, tip_disp in zip(axes, cases, tip_ids, tip_disps):
        _draw_panel(ax, case, scale, cmap, norm, xlim, ylim, tip_id)
        title = (
            f"{case.label}: |v_tip| = {tip_disp:.3g} {lbl['displacement']}"
        )
        if case.label == "Optimized" and tip_disps[0] > 0.0:
            delta_pct = 100.0 * (tip_disp - tip_disps[0]) / tip_disps[0]
            title += f"  ({delta_pct:+.1f}% vs baseline)"
        ax.set_title(title)
        ax.set_xlabel(f"x [{lbl['length']}]")
    axes[0].set_ylabel(f"y [{lbl['length']}]")

    axes[0].legend(
        handles=[Line2D([0], [0], color="black", linestyle="--", linewidth=1.5)],
        labels=["buckled member"],
        loc="lower left", fontsize=8, framealpha=0.8,
    )

    data_aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig_w, fig_h = fig.get_size_inches()
    cell_w = fig_w / 2.0
    drawn_h = min(fig_h, cell_w * data_aspect)
    cbar_shrink = float(np.clip(drawn_h / fig_h, 0.25, 1.0))

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes.tolist(), shrink=cbar_shrink, pad=0.02, fraction=0.04
    )
    cbar.set_label(
        f"Axial force N [{lbl['force']}]   (disp scaled {scale:.2g}x)"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot saved to %s", args.output)
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
