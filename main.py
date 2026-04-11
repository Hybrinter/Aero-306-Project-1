"""AERO 306 FEA Solver -- CLI entrypoint.

Usage:
    uv run python main.py                                            # interactive menu
    uv run python main.py config/case_02_cantilever_beam.yaml
    uv run python main.py config/case_01_bar_axial.yaml --no-plot
    uv run python main.py config/case_06_distributed_load.yaml --save-plots outputs/
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
from fea_solver.io_yaml import load_model_from_yaml
from fea_solver.logging_config import configure_logging
from fea_solver.plotter import (
    plot_bending_moment_diagram,
    plot_deformed_shape,
    plot_shear_force_diagram,
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
            save_plots (str | None): Directory to save plot images, or None.
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
        "--save-plots", type=Path, default=None,
        help="Directory to save plot images (PNG)"
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

    Args:
        argv (list[str] | None): Command-line arguments. Uses sys.argv[1:] if None.

    Returns:
        int: Exit code (0 for success, 1 for file/config error, 2 for solve error).

    Notes:
        If no config path is provided via argv, an interactive numbered menu is shown
        listing all *.yaml files in config/ for the user to pick from.
        Prints DOF table, nodal results, reactions, and element forces to console.
        Generates optional plots (SFD, BMD, deformed shape) if not suppressed.
        All output logged to file and console with configurable levels.
    """
    args = parse_args(argv)

    # --- Interactive config selection (no-arg run) ---
    if args.config is None:
        selected = _interactive_select_config()
        if selected is None:
            return 1
        args.config = selected

    # --- Load model ---
    try:
        model = load_model_from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # --- Configure logging ---
    logger = configure_logging(args.log_dir, model.label)
    logger.info("=== FEA Solver started: %s ===", model.label)
    logger.info("Config: %s", args.config)

    # --- Build DOF map and display ---
    dof_map = build_dof_map(model)
    print_dof_table(model, dof_map)  # Show DOF layout BEFORE solving

    # --- Assemble ---
    logger.info("Assembling global stiffness matrix (%dx%d)...",
                dof_map.total_dofs, dof_map.total_dofs)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)

    # --- Solve ---
    logger.info("Solving displacement system...")
    try:
        result = run_solve_pipeline(model, dof_map, K, F)
    except Exception as e:
        logger.error("Solve failed: %s", e)
        print(f"ERROR: Solve failed -- {e}", file=sys.stderr)
        return 2

    # --- Post-process ---
    logger.info("Post-processing element internal forces...")
    element_results = postprocess_all_elements(model, result, n_stations=args.n_stations)

    # --- Report ---
    print_nodal_results(result)
    print_reaction_forces(result)
    print_element_forces(element_results)
    generate_report(model, dof_map, result, element_results)

    # --- Plots ---
    if not args.no_plot:
        save_dir = args.save_plots
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        figures = []

        sfd_path = (save_dir / f"{model.label}_sfd.png") if save_dir else None
        figures.append(
            plot_shear_force_diagram(element_results, title=f"SFD -- {model.label}",
                                     output_path=sfd_path)
        )

        bmd_path = (save_dir / f"{model.label}_bmd.png") if save_dir else None
        figures.append(
            plot_bending_moment_diagram(element_results, title=f"BMD -- {model.label}",
                                        output_path=bmd_path)
        )

        def_path = (save_dir / f"{model.label}_deformed.png") if save_dir else None
        figures.append(
            plot_deformed_shape(element_results, output_path=def_path)
        )

        if save_dir:
            logger.info("Plots saved to %s", save_dir)
        else:
            show_all_plots(figures)

    logger.info("=== FEA Solver complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
