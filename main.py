"""AERO 306 FEA Solver -- CLI entrypoint.

Usage:
    uv run python main.py                                            # interactive menu
    uv run python main.py config/example_case_02_cantilever_beam.yaml
    uv run python main.py config/example_case_01_bar_axial.yaml --no-plot
    uv run python main.py config/problem_1.yaml --save-plots outputs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for CLI and CI

from rich.console import Console as _Console

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.logging_config import configure_logging
from fea_solver.models import SolutionSeries
from fea_solver.models import ElementType
from fea_solver.plotter import (
    plot_axial_displacement,
    plot_bending_moment_diagram,
    plot_rotation,
    plot_shear_force_diagram,
    plot_transverse_displacement,
    plot_truss_deformed,
    plot_truss_forces,
    plot_truss_stress,
    show_all_plots,
)
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.reporter import (
    generate_report,
    print_dof_table,
    print_element_forces,
    print_nodal_results,
    print_reaction_forces,
)
from fea_solver.solver import run_solve_pipeline


_CONFIG_DIR = Path(__file__).parent / "config"


def _sanitize_label(label: str) -> str:
    """Return a filesystem-safe version of label by replacing path separators with '_'.

    Args:
        label (str): Raw model label, potentially containing '/' separators
            (e.g. "problem_1/coarse" for multi-solution composite labels).

    Returns:
        str: Label with '/' and '\\' replaced by '_'.

    Notes:
        Multi-solution composite labels use '/' as a separator. Using such a label
        directly in a filename would create subdirectories on most operating systems.
        This helper produces a flat filename-safe string instead.
    """
    return label.replace("/", "_").replace("\\", "_")


def _interactive_select_config() -> Path | None:
    """Display a numbered menu of available YAML config files and return the chosen path.

    Args:
        None

    Returns:
        Path | None: Path to the selected YAML file, or None if no files are found.

    Notes:
        Scans _CONFIG_DIR (project root / config/) for *.yaml files.
        Re-prompts on invalid input until a valid integer in range is entered.
        Returns None (causing a graceful exit in main) if config/ is empty or missing.
    """
    console = _Console()
    yaml_files = sorted(_CONFIG_DIR.glob("*.yaml"))
    if not yaml_files:
        console.print("[bold red]No YAML files found in config/[/bold red]")
        return None

    console.print("\n[bold cyan]AERO 306 FEA Solver[/bold cyan] -- select a case:\n")
    for i, f in enumerate(yaml_files, start=1):
        console.print(f"  [bold]{i:>2}.[/bold] {f.stem}")

    console.print()
    while True:
        raw = input(f"Case [1-{len(yaml_files)}]: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(yaml_files):
                return yaml_files[idx - 1]
        except ValueError:
            pass
        console.print(f"[yellow]Enter a number between 1 and {len(yaml_files)}.[/yellow]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the FEA solver CLI.

    Args:
        argv (list[str] | None): Argument list to parse. Uses sys.argv[1:] if None.

    Returns:
        argparse.Namespace: Parsed arguments with fields:
            config (Path | None): Path to YAML case file, or None to trigger
                interactive selection in main().
            no_plot (bool): Suppress all plot windows if True.
            save_plots (Path): Directory to save plot images. Default: Path("outputs").
            n_stations (int): Number of evaluation points per element for post-processing.
            log_dir (Path): Directory for log files.

    Notes:
        Default n_stations is 100.
    """
    parser = argparse.ArgumentParser(
        description="AERO 306 1D FEA Bar/Beam Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config", type=Path, nargs="?", default=None,
        help="Path to YAML config file (omit for interactive menu)",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip rendering plots"
    )
    parser.add_argument(
        "--save-plots", type=Path, default=Path("outputs"),
        help="Directory to save plot images (PNG) -- default: outputs/"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"),
        help="Directory for log files (default: logs/)"
    )
    parser.add_argument(
        "--n-stations", type=int, default=100,
        help="Number of evaluation stations per element for internal forces (default: 100)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the FEA solver pipeline: load config, assemble, solve, post-process, report, plot.

    Supports both single-solution YAML files (legacy format with top-level 'mesh' key)
    and multi-solution YAML files (top-level 'solutions' list). When multiple solutions
    are present, all are solved and their results are overlaid on a single set of plots.

    Args:
        argv (list[str] | None): Command-line arguments. Uses sys.argv[1:] if None.

    Returns:
        int: Exit code (0 for success, 1 for file/config error, 2 for solve error).

    Notes:
        If no config path is provided via argv, an interactive numbered menu is shown
        listing all *.yaml files in config/ for the user to pick from.
        For multi-solution files, report functions are called once per solution while
        plot functions receive all solutions at once for overlaid rendering.
        All output logged to file and console with configurable levels.
    """
    args = parse_args(argv)

    # --- Interactive config selection (no-arg run) ---
    if args.config is None:
        selected = _interactive_select_config()
        if selected is None:
            return 1
        args.config = selected

    # --- Load models (single or multi-solution) ---
    try:
        models = load_models_from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    top_label = models[0].label.split("/")[0] if models else args.config.stem

    # --- Configure logging once (using top-level label) ---
    logger = configure_logging(args.log_dir, top_label)
    logger.info("=== FEA Solver started: %s ===", top_label)
    logger.info("Config: %s", args.config)
    logger.info("Solutions: %d", len(models))

    # --- Solve each model; collect SolutionSeries for overlay plotting ---
    all_series: list[SolutionSeries] = []

    for model in models:
        logger.info("--- Solving: %s ---", model.label)

        # Build DOF map and display
        dof_map = build_dof_map(model)
        print_dof_table(model, dof_map)

        # Assemble
        logger.info("Assembling global stiffness matrix (%dx%d)...",
                    dof_map.total_dofs, dof_map.total_dofs)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)

        # Solve
        logger.info("Solving displacement system...")
        try:
            result = run_solve_pipeline(model, dof_map, K, F)
        except Exception as e:
            logger.error("Solve failed for '%s': %s", model.label, e)
            print(f"ERROR: Solve failed for '{model.label}' -- {e}", file=sys.stderr)
            return 2

        # Post-process
        logger.info("Post-processing element internal forces...")
        element_results = postprocess_all_elements(model, result, n_stations=args.n_stations)

        # Report (per solution)
        print_nodal_results(result)
        print_reaction_forces(result)
        print_element_forces(element_results, model)
        generate_report(model, dof_map, result, element_results)

        # Collect series for overlay plots
        all_series.append(SolutionSeries(
            label=model.label.split("/")[-1],   # "coarse" not "problem_1/coarse"
            element_results=tuple(element_results),
            model=model,
            result=result,
        ))

    # --- Plots (all solutions overlaid on shared axes) ---
    if not args.no_plot:
        save_dir = args.save_plots
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        safe = _sanitize_label(top_label)
        figures = []

        # Detect pure-truss problems: all elements in every solution are TRUSS
        all_truss = all(
            all(e.element_type == ElementType.TRUSS for e in m.mesh.elements)
            for m in models
        )

        if all_truss:
            # 2D truss: three gradient plots per solution (overlay not meaningful)
            for sol in all_series:
                safe_sol = _sanitize_label(sol.label)

                forces_path = (save_dir / f"{safe_sol}_truss_forces.png") if save_dir else None
                figures.append(
                    plot_truss_forces(sol,
                                      title=f"Truss Forces: {sol.label}",
                                      output_path=forces_path)
                )

                deformed_path = (save_dir / f"{safe_sol}_truss_deformed.png") if save_dir else None
                figures.append(
                    plot_truss_deformed(sol,
                                        title=f"Truss Deformed: {sol.label}",
                                        output_path=deformed_path)
                )

                stress_path = (save_dir / f"{safe_sol}_truss_stress.png") if save_dir else None
                figures.append(
                    plot_truss_stress(sol,
                                      title=f"Truss Stress: {sol.label}",
                                      output_path=stress_path)
                )
        else:
            # 1D bar/beam/frame: overlay SFD, BMD, displacements, rotation plots
            shear_path = (save_dir / f"{safe}_shear.png") if save_dir else None
            figures.append(
                plot_shear_force_diagram(all_series,
                                         title=f"Shear: {top_label}",
                                         output_path=shear_path)
            )

            moment_path = (save_dir / f"{safe}_moment.png") if save_dir else None
            figures.append(
                plot_bending_moment_diagram(all_series,
                                            title=f"Moments: {top_label}",
                                            output_path=moment_path)
            )

            vert_disp_path = (save_dir / f"{safe}_vertical_disp.png") if save_dir else None
            figures.append(
                plot_transverse_displacement(all_series,
                                             title=f"Vertical Displacement: {top_label}",
                                             output_path=vert_disp_path)
            )

            axial_disp_path = (save_dir / f"{safe}_axial_disp.png") if save_dir else None
            figures.append(
                plot_axial_displacement(all_series,
                                        title=f"Axial Displacement: {top_label}",
                                        output_path=axial_disp_path)
            )

            theta_path = (save_dir / f"{safe}_rotation.png") if save_dir else None
            figures.append(
                plot_rotation(all_series,
                              title=f"Rotation: {top_label}",
                              output_path=theta_path)
            )

        if save_dir:
            logger.info("Plots saved to %s", save_dir)
        else:
            show_all_plots(figures)

    logger.info("=== FEA Solver complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
