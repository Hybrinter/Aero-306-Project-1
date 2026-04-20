"""Unit tests for evaluate(x, problem) and EvalResult."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dataclasses import replace

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.models import DOFType, NodalLoad
from fea_solver.optimization.objective import EvalResult, evaluate
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    apply_x_to_model,
    baseline_x,
)
from fea_solver.solver import run_solve_pipeline

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"

FREE_NODE_IDS = (2, 3, 5, 6, 7, 8)
FROZEN_NODE_IDS = (1, 4, 9)


def _baseline_problem(F: float = 15.0) -> GeometryOptimizationProblem:
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=F,
        sigma_max=72.0,
        L_min=5.0,
    )


def _independent_baseline_tip_disp(problem: GeometryOptimizationProblem) -> float:
    """Solve the baseline model with F=15 N independently for cross-check."""
    model = apply_x_to_model(baseline_x(problem), problem)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    res = run_solve_pipeline(model, dof_map, K, F)
    return abs(float(res.displacements[dof_map.index(9, DOFType.V)]))


def test_evaluate_baseline_reproduces_independent_solve() -> None:
    problem = _baseline_problem()
    er = evaluate(baseline_x(problem), problem)
    expected = _independent_baseline_tip_disp(problem)
    assert er.solve_ok is True
    assert er.tip_disp == pytest.approx(expected, rel=1e-9)


def test_evaluate_returns_full_metrics() -> None:
    problem = _baseline_problem()
    er = evaluate(baseline_x(problem), problem)
    assert isinstance(er, EvalResult)
    assert er.max_stress >= 0.0
    assert er.max_buckling_ratio >= 0.0
    assert er.min_length > 0.0
    assert len(er.stress_violations) == 16
    assert len(er.buckling_violations) == 16
    assert len(er.length_violations) == 16


def test_evaluate_coincident_nodes_is_safely_infeasible() -> None:
    problem = _baseline_problem()
    x = baseline_x(problem).copy()
    # Move node 2 onto node 3
    x[0] = x[2]
    x[1] = x[3]
    er = evaluate(x, problem)
    assert er.solve_ok is False
    assert er.feasible is False
    # Tip disp must be finite (sentinel) so penalty is finite
    assert np.isfinite(er.tip_disp)


def test_evaluate_flags_stress_violation() -> None:
    """A baseline reshape that drives stress above 72 MPa should violate."""
    problem = _baseline_problem(F=10000.0)  # absurd load -> guaranteed overstress
    er = evaluate(baseline_x(problem), problem)
    assert er.solve_ok is True
    assert er.max_stress > problem.sigma_max
    assert any(v > 0.0 for v in er.stress_violations)
    assert er.feasible is False


def test_evaluate_min_length_below_L_min_marks_infeasible() -> None:
    """Place two free nodes very close so the connecting edge < L_min."""
    problem = _baseline_problem()
    x = baseline_x(problem).copy()
    # Move node 7 (index 8,9) to (4.99, -10) -- distance to node 4 (0,-10) is 4.99 < 5
    x[8] = 4.99
    x[9] = -10.0
    er = evaluate(x, problem)
    assert er.min_length < problem.L_min
    assert any(v > 0.0 for v in er.length_violations)
    assert er.feasible is False
