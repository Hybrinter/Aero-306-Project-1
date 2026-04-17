"""CLI for the geometry optimizer.

Reads a base FEAModel YAML, builds a GeometryOptimizationProblem, runs the
DE + CMA-ES + SLSQP-polish ensemble, and writes:
  - best_design.yaml (drop-in replacement YAML)
  - report.md (one-pager)
  - plots/*.png (deformed truss, force gradient, stress, buckling overlay)

Run:
    uv run python scripts/optimize_geometry.py --base config/problem_7.yaml --run-id smoke --smoke
    uv run python scripts/optimize_geometry.py --base config/problem_7.yaml \\
        --de-seeds 16 --cmaes-seeds 16 --de-maxiter 600 --cmaes-maxiter 800 \\
        --run-id 2026-04-17_heavy
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be called before any other matplotlib import
import yaml

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.buckling import compute_truss_buckling
from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.logging_config import configure_logging
from fea_solver.models import DOFType, SolutionSeries
from fea_solver.optimization.ensemble import EnsembleConfig, run_ensemble
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    apply_x_to_model,
    baseline_x,
)
from fea_solver.optimization.report import write_report
from fea_solver.plotter import (
    plot_truss_deformed,
    plot_truss_forces,
    plot_truss_stress,
)
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.solver import run_solve_pipeline

import numpy as np

logger = logging.getLogger(__name__)


FREE_NODE_IDS = (2, 3, 5, 6, 7, 8)
FROZEN_NODE_IDS = (1, 4, 9)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the geometry optimizer CLI.

    Returns:
        argparse.Namespace: Parsed arguments with the following fields:
            base (Path): Path to the baseline YAML case file.
            F (float): Applied load magnitude in kN (default 15.0).
            sigma_max (float): Maximum allowable stress in ksi (default 72.0).
            L_min (float): Minimum member length in inches (default 5.0).
            de_seeds (int): Number of DE random seeds (default 16).
            cmaes_seeds (int): Number of CMA-ES random seeds (default 16).
            de_popsize (int): DE population size per seed (default 30).
            de_maxiter (int): DE maximum iterations per seed (default 600).
            cmaes_popsize (int): CMA-ES population size per seed (default 20).
            cmaes_maxiter (int): CMA-ES maximum iterations per seed (default 800).
            cmaes_sigma0 (float): CMA-ES initial step size (default 5.0).
            cmaes_restarts (int): CMA-ES restart count (default 5).
            top_k (int): Number of top candidates to polish (default 5).
            polish_max_iter (int): SLSQP polish iteration limit (default 200).
            workers (str): Parallelism level, 'auto' or an int string (default 'auto').
            run_id (str): Unique identifier for this run (required).
            output_dir (Path): Root output directory (default 'optimization_runs').
            smoke (bool): If True, collapse budgets to ~5% for a fast smoke test.
            no_plot (bool): If True, skip all plot generation.

    Notes:
        When --smoke is passed, _apply_smoke further overrides budget arguments
        to minimal values suitable for a quick correctness check.
    """
    p = argparse.ArgumentParser(
        description="Geometry optimizer for the AERO 306 bonus problem."
    )
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--F", type=float, default=15.0)
    p.add_argument("--sigma-max", type=float, default=72.0)
    p.add_argument("--L-min", type=float, default=5.0)
    p.add_argument("--de-seeds", type=int, default=16)
    p.add_argument("--cmaes-seeds", type=int, default=16)
    p.add_argument("--de-popsize", type=int, default=30)
    p.add_argument("--de-maxiter", type=int, default=600)
    p.add_argument("--cmaes-popsize", type=int, default=20)
    p.add_argument("--cmaes-maxiter", type=int, default=800)
    p.add_argument("--cmaes-sigma0", type=float, default=5.0)
    p.add_argument("--cmaes-restarts", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--polish-max-iter", type=int, default=200)
    p.add_argument("--workers", default="auto")
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("optimization_runs"))
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Collapse every budget knob to ~5%% of full for a fast smoke run.",
    )
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def _apply_smoke(args: argparse.Namespace) -> None:
    """Override budget arguments to minimal values for a fast smoke test.

    Modifies args in-place. No-op when args.smoke is False.

    Args:
        args (argparse.Namespace): Parsed CLI arguments. Modified in-place
            when args.smoke is True.

    Returns:
        None

    Notes:
        Smoke values are chosen so the entire run completes in under 30 seconds
        on a laptop: 1-2 DE seeds, 1-2 CMA-ES seeds, 5 iterations each,
        no restarts, top_k capped at 1, polish limited to 30 iterations.
    """
    if not args.smoke:
        return
    args.de_seeds = max(1, args.de_seeds // 8)
    args.cmaes_seeds = max(1, args.cmaes_seeds // 8)
    args.de_popsize = 5
    args.de_maxiter = 5
    args.cmaes_popsize = 6
    args.cmaes_maxiter = 5
    args.cmaes_restarts = 0
    args.top_k = max(1, args.top_k // 5)
    args.polish_max_iter = 30


def _resolve_workers(arg: str) -> int:
    """Resolve the --workers argument to a concrete integer worker count.

    Args:
        arg (str): Either 'auto' or a string representation of an integer.

    Returns:
        int: Number of parallel workers. 'auto' resolves to os.cpu_count()
            (minimum 1 when cpu_count returns None).

    Notes:
        On single-core machines or when os.cpu_count() is None, returns 1
        to avoid spawning zero workers.
    """
    if arg == "auto":
        return max(1, os.cpu_count() or 1)
    return int(arg)


def _baseline_tip_disp(problem: GeometryOptimizationProblem) -> float:
    """Compute the tip displacement magnitude for the baseline geometry.

    Runs a single forward FE solve on the unoptimized design vector so that
    the report can express improvement as a relative percentage.

    Args:
        problem (GeometryOptimizationProblem): Problem definition supplying
            the baseline model and load node id.

    Returns:
        float: Absolute tip displacement |v_load_node| in model length units.

    Notes:
        Uses baseline_x to obtain the un-perturbed design vector, then runs
        the standard assemble -> solve pipeline. The DOF queried is (V) at
        problem.load_node_id.
    """
    model = apply_x_to_model(baseline_x(problem), problem)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    res = run_solve_pipeline(model, dof_map, K, F)
    return abs(
        float(res.displacements[dof_map.index(problem.load_node_id, DOFType.V)])
    )


def _emit_best_design_yaml(
    base_yaml: Path,
    problem: GeometryOptimizationProblem,
    x: np.ndarray,
    out_path: Path,
) -> None:
    """Write a YAML case file with the winning optimized node positions.

    Reads the baseline YAML as a plain dict, overwrites free-node coordinates
    with the values encoded in x, adjusts the applied load magnitude to match
    problem.F_magnitude, then serialises back to YAML at out_path.

    Args:
        base_yaml (Path): Path to the baseline YAML (e.g. config/problem_7.yaml).
        problem (GeometryOptimizationProblem): Problem definition supplying
            free_node_ids and load_node_id.
        x (numpy.ndarray): Winning design vector of length 2 * len(free_node_ids),
            laid out as [x0, y0, x1, y1, ...] for free_node_ids in order.
        out_path (Path): Destination YAML path (created or overwritten).

    Returns:
        None

    Notes:
        The output YAML preserves key insertion order (sort_keys=False).
        Only nodes whose id is in problem.free_node_ids are modified.
        The nodal load at problem.load_node_id is set to -problem.F_magnitude
        (negative because the load convention is downward in the Y direction).
    """
    data = yaml.safe_load(base_yaml.read_text())
    free_lookup = {
        node_id: (float(x[2 * i]), float(x[2 * i + 1]))
        for i, node_id in enumerate(problem.free_node_ids)
    }
    for node in data["mesh"]["nodes"]:
        if node["id"] in free_lookup:
            node["x"], node["y"] = free_lookup[node["id"]]
    for nl in data.get("loads", {}).get("nodal", []):
        if nl["node_id"] == problem.load_node_id:
            nl["magnitude"] = -problem.F_magnitude
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))


def _render_plots(
    problem: GeometryOptimizationProblem,
    x: np.ndarray,
    out_dir: Path,
) -> None:
    """Run a final FE solve on the winner and render the three truss plots.

    Builds a SolutionSeries from the winning design, then calls:
      - plot_truss_deformed (with buckling overlay)
      - plot_truss_forces
      - plot_truss_stress

    Args:
        problem (GeometryOptimizationProblem): Problem definition used to
            rebuild the FEA model from x.
        x (numpy.ndarray): Winning design vector.
        out_dir (Path): Directory where PNG files are saved. Created if absent.

    Returns:
        None

    Notes:
        plot_truss_deformed, plot_truss_forces, and plot_truss_stress all
        accept a SolutionSeries object (not separate model/result arguments).
        The SolutionSeries wraps (label, element_results, model, result).
        Plots are saved at 150 dpi as PNG files to out_dir.
    """
    model = apply_x_to_model(x, problem)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F_vec = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F_vec)
    elem_results = postprocess_all_elements(model, result, n_stations=2)
    buckling = compute_truss_buckling(model, elem_results)

    sol = SolutionSeries(
        label="winner",
        element_results=tuple(elem_results),
        model=model,
        result=result,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_truss_deformed(
        sol,
        output_path=out_dir / "truss_deformed.png",
        buckling=buckling,
    )
    plot_truss_forces(sol, output_path=out_dir / "truss_forces.png")
    plot_truss_stress(sol, output_path=out_dir / "truss_stress.png")


def main() -> None:
    """Entry point for the geometry optimizer CLI.

    Parses arguments, applies smoke overrides, builds the optimization problem,
    runs the DE + CMA-ES + SLSQP-polish ensemble, then writes:
      - optimization_runs/<run_id>/config.json   (resolved arg provenance)
      - optimization_runs/<run_id>/best_design.yaml
      - optimization_runs/<run_id>/report.md
      - optimization_runs/<run_id>/plots/*.png   (unless --no-plot)

    Returns:
        None

    Notes:
        Logs a summary line with the winner's stiffness (F / tip_disp),
        feasibility status, and report path.
        The if __name__ == '__main__' guard is required on Windows for
        multiprocessing to work correctly.
    """
    args = parse_args()
    _apply_smoke(args)
    workers = _resolve_workers(args.workers)
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir=run_dir, case_label="optimize_run")

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )

    base = Path(args.base)
    model = load_models_from_yaml(base)[0]
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=args.F,
        sigma_max=args.sigma_max,
        L_min=args.L_min,
    )

    config = EnsembleConfig(
        de_seeds=args.de_seeds,
        cmaes_seeds=args.cmaes_seeds,
        de_popsize=args.de_popsize,
        de_maxiter=args.de_maxiter,
        cmaes_popsize=args.cmaes_popsize,
        cmaes_maxiter=args.cmaes_maxiter,
        cmaes_sigma0=args.cmaes_sigma0,
        cmaes_restarts=args.cmaes_restarts,
        top_k=args.top_k,
        polish_max_iter=args.polish_max_iter,
        workers=workers,
        run_dir=run_dir,
    )

    er = run_ensemble(problem, config)

    _emit_best_design_yaml(base, problem, er.winner_x, run_dir / "best_design.yaml")
    write_report(
        er,
        problem,
        run_dir / "report.md",
        run_id=args.run_id,
        baseline_tip_disp=_baseline_tip_disp(problem),
    )
    if not args.no_plot:
        _render_plots(problem, er.winner_x, run_dir / "plots")

    logger.info(
        "Done. Winner K = %.4f N/mm; feasible=%s; report at %s",
        args.F / max(er.winner_eval.tip_disp, 1e-30),
        er.feasible,
        run_dir / "report.md",
    )


if __name__ == "__main__":
    main()
