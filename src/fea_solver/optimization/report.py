"""Markdown one-pager generator for an EnsembleResult.

Self-contained: writes a markdown file directly. Does not import the
existing reporter.py (per the spec's boundary discipline).

_format_wallclock:   Format a wall-clock duration in seconds as 'Hh MMm SSs'.
write_report:        Render an EnsembleResult into a markdown report file.
"""
from __future__ import annotations

from pathlib import Path
from statistics import median

from fea_solver.optimization.checkpoint import EnsembleResult
from fea_solver.optimization.problem import GeometryOptimizationProblem


def _format_wallclock(s: float) -> str:
    """Format wall-clock seconds as 'Hh MMm SSs'.

    Args:
        s (float): Duration in seconds.

    Returns:
        str: Formatted string like '3h 11m 15s'.
    """
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}h {m:02d}m {sec:02d}s"


def write_report(
    er: EnsembleResult,
    problem: GeometryOptimizationProblem,
    out_path: Path,
    run_id: str,
    baseline_tip_disp: float,
) -> None:
    """Write a markdown report for an EnsembleResult.

    Args:
        er (EnsembleResult): Aggregate result.
        problem (GeometryOptimizationProblem): Problem definition.
        out_path (Path): Destination .md file.
        run_id (str): Run identifier (used in title and headers).
        baseline_tip_disp (float): Tip displacement of the baseline geometry,
            used to compute the relative improvement row.

    Returns:
        None

    Notes:
        Does not import plotter.py or reporter.py (boundary discipline rule).
        The markdown table uses |stress| notation for the max member stress row.
    """
    F = problem.F_magnitude
    K_winner = F / max(er.winner_eval.tip_disp, 1.0e-30)
    K_baseline = F / max(baseline_tip_disp, 1.0e-30)
    improvement_pct = (K_winner / K_baseline - 1.0) * 100.0 if K_baseline > 0 else 0.0

    # Per-algorithm stats
    de_seeds = [s for s in er.all_seeds if s.algorithm == "DE" and s.best_eval.solve_ok]
    cm_seeds = [s for s in er.all_seeds if s.algorithm == "CMA-ES" and s.best_eval.solve_ok]

    def stats(seeds: list) -> tuple[str, str, str, int]:
        """Compute best/median/worst-feasible stiffness stats for a seed list.

        Args:
            seeds (list): List of SeedResult objects to analyze.

        Returns:
            tuple[str, str, str, int]: (best_K, median_K, worst_feasible_K, count)
                where each K string is formatted to 3 decimal places or 'n/a'.
        """
        if not seeds:
            return ("n/a", "n/a", "n/a", 0)
        ks = [F / max(s.best_eval.tip_disp, 1.0e-30) for s in seeds]
        feas_ks = [
            F / max(s.best_eval.tip_disp, 1.0e-30) for s in seeds if s.best_eval.feasible
        ]
        worst_feas = f"{min(feas_ks):.3f}" if feas_ks else "n/a"
        return (f"{max(ks):.3f}", f"{median(ks):.3f}", worst_feas, len(seeds))

    de_stats = stats(de_seeds)
    cm_stats = stats(cm_seeds)

    # Active constraints at optimum
    e = er.winner_eval
    active_stress = [i + 1 for i, v in enumerate(e.stress_violations) if v >= -1e-3 and v <= 1e-3 and abs(v) < 1e-3]
    active_buck = [i + 1 for i, v in enumerate(e.buckling_violations) if abs(v) < 1e-3 and v == 0.0 and e.max_buckling_ratio > 0.99]
    active_len = [i + 1 for i, v in enumerate(e.length_violations) if abs(v) < 1e-3 and e.min_length < 1.001 * problem.L_min]

    # Build node table (frozen + free)
    nodes_by_id = {n.id: n for n in problem.baseline_model.mesh.nodes}
    free_lookup = {
        node_id: (float(er.winner_x[2 * i]), float(er.winner_x[2 * i + 1]))
        for i, node_id in enumerate(problem.free_node_ids)
    }
    sorted_ids = sorted(nodes_by_id.keys())
    node_lines = []
    for nid in sorted_ids:
        if nid in free_lookup:
            x, y = free_lookup[nid]
            status = "free"
        else:
            x, y = nodes_by_id[nid].pos
            status = "frozen"
        node_lines.append(f"|  {nid}   | {x:7.3f} | {y:7.3f} | {status:6s} |")

    feasible_str = "yes" if er.feasible else "no"
    wallclock = _format_wallclock(er.wall_clock_s)

    md = f"""# Geometry Optimization Report -- {run_id}

**Wall-clock**: {wallclock}
**Origin**: {er.winner_origin[0]} seed {er.winner_origin[1]}, polished
**Feasible**: {feasible_str}

## Objective
| Quantity              | Value         | Target / Limit  |
|-----------------------|--------------:|----------------:|
| Tip displacement      | {er.winner_eval.tip_disp:.6f} mm | (minimised) |
| Stiffness K = F/|v|   | {K_winner:.3f} N/mm | (maximised) |
| Baseline K            | {K_baseline:.3f} N/mm | -- |
| Improvement           | {improvement_pct:+.1f} % | -- |

## Constraints
| Quantity              | Value         | Limit           | Slack    |
|-----------------------|--------------:|----------------:|---------:|
| Max member |stress|   | {er.winner_eval.max_stress:.3f} MPa | {problem.sigma_max:.3f} MPa | {problem.sigma_max - er.winner_eval.max_stress:+.3f} MPa |
| Max buckling ratio    | {er.winner_eval.max_buckling_ratio:.4f} | < 1.0000 | {1.0 - er.winner_eval.max_buckling_ratio:+.4f} |
| Min element length    | {er.winner_eval.min_length:.4f} mm | >= {problem.L_min:.4f} mm | {er.winner_eval.min_length - problem.L_min:+.4f} mm |

## Best design
| Node | x [mm] | y [mm] | Status |
|------|-------:|-------:|--------|
""" + "\n".join(node_lines) + f"""

## Ensemble summary
| Algorithm | Seeds | Best K | Median K | Worst feasible K |
|-----------|------:|-------:|---------:|-----------------:|
| DE        | {de_stats[3]:5d} | {de_stats[0]:>6} | {de_stats[1]:>8} | {de_stats[2]:>16} |
| CMA-ES    | {cm_stats[3]:5d} | {cm_stats[0]:>6} | {cm_stats[1]:>8} | {cm_stats[2]:>16} |

Active constraints at optimum: stress members {active_stress or 'none'}; buckling members {active_buck or 'none'}; length members {active_len or 'none'}.
"""
    out_path.write_text(md)
