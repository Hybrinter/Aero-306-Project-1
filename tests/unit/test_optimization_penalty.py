"""Unit tests for the penalized objective."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.objective import SENTINEL_TIP_DISP, evaluate
from fea_solver.optimization.penalty import (
    DEFAULT_WEIGHTS,
    PenaltyWeights,
    penalized_objective,
)
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    baseline_x,
)

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem(F: float = 15.0) -> GeometryOptimizationProblem:
    """Build a GeometryOptimizationProblem from problem_7.yaml."""
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=(2, 3, 5, 6, 7, 8),
        frozen_node_ids=(1, 4, 9),
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=F,
        sigma_max=72.0,
        L_min=5.0,
    )


def test_feasible_penalty_equals_objective() -> None:
    """Penalty equals tip_disp when no violations are present."""
    problem = _problem()
    x = baseline_x(problem)
    er = evaluate(x, problem)
    assert er.solve_ok is True
    p = penalized_objective(x, problem, DEFAULT_WEIGHTS)
    # Penalty = tip_disp + weighted sum of squared violations
    s_pen = DEFAULT_WEIGHTS.stress * float(np.sum(np.asarray(er.stress_violations) ** 2))
    b_pen = DEFAULT_WEIGHTS.buckling * float(np.sum(np.asarray(er.buckling_violations) ** 2))
    l_pen = DEFAULT_WEIGHTS.length * float(np.sum(np.asarray(er.length_violations) ** 2))
    assert p == pytest.approx(er.tip_disp + s_pen + b_pen + l_pen, rel=1e-12)


def test_overstress_penalty_dominates_objective() -> None:
    """Penalty exceeds tip_disp alone when structure is overstressed."""
    problem = _problem(F=10000.0)
    x = baseline_x(problem)
    er = evaluate(x, problem)
    assert er.feasible is False
    assert er.max_stress > problem.sigma_max
    p = penalized_objective(x, problem, DEFAULT_WEIGHTS)
    # Penalty should be much larger than tip_disp alone
    assert p > er.tip_disp + 1e-3


def test_solve_failure_returns_finite_sentinel() -> None:
    """Penalized objective is finite (sentinel) for a degenerate design vector."""
    problem = _problem()
    x = baseline_x(problem).copy()
    x[0] = x[2]
    x[1] = x[3]  # coincident node 2 and node 3
    p = penalized_objective(x, problem, DEFAULT_WEIGHTS)
    assert np.isfinite(p)
    assert p >= SENTINEL_TIP_DISP


def test_weights_can_be_overridden() -> None:
    """Stronger weights produce a larger penalized objective for infeasible designs."""
    problem = _problem(F=10000.0)
    x = baseline_x(problem)
    weak = PenaltyWeights(stress=1.0, buckling=1.0, length=1.0)
    strong = PenaltyWeights(stress=1.0e6, buckling=1.0e6, length=1.0e6)
    p_weak = penalized_objective(x, problem, weak)
    p_strong = penalized_objective(x, problem, strong)
    assert p_strong > p_weak
