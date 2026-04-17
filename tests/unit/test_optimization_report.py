"""Unit tests for the markdown report generator."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.checkpoint import EnsembleResult
from fea_solver.optimization.objective import EvalResult
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    baseline_x,
)
from fea_solver.optimization.report import write_report

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem() -> GeometryOptimizationProblem:
    """Build a GeometryOptimizationProblem from problem_7.yaml for testing."""
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=(2, 3, 5, 6, 7, 8),
        frozen_node_ids=(1, 4, 9),
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )


def test_write_report_creates_markdown_with_required_sections(tmp_path: Path) -> None:
    """Test that write_report creates a markdown file with all required sections."""
    problem = _problem()
    z = tuple([0.0] * 16)
    er = EnsembleResult(
        winner_x=baseline_x(problem),
        winner_eval=EvalResult(
            tip_disp=0.0184, max_stress=71.5, max_buckling_ratio=0.92,
            min_length=5.001,
            stress_violations=z, buckling_violations=z, length_violations=z,
            feasible=True, solve_ok=True,
        ),
        winner_origin=("CMA-ES", 7),
        all_seeds=(),
        all_polish=(),
        wall_clock_s=11500.0,
        feasible=True,
    )
    out = tmp_path / "report.md"
    write_report(er, problem, out, run_id="test_run", baseline_tip_disp=0.05)
    text = out.read_text()
    assert "# Geometry Optimization Report" in text
    assert "test_run" in text
    assert "CMA-ES" in text
    assert "Tip displacement" in text
    assert "Stiffness" in text
    assert "Max member |stress|" in text
    assert "Max buckling ratio" in text
    assert "Min element length" in text
    # All 9 nodes printed
    for nid in range(1, 10):
        assert f"|  {nid}  " in text or f"| {nid}  " in text or f"|  {nid} " in text
