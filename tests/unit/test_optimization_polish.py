"""Unit tests for SLSQP polish."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.polish import slsqp_polish
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    baseline_x,
)

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem(F: float = 15.0) -> GeometryOptimizationProblem:
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


def test_polish_baseline_returns_feasible() -> None:
    problem = _problem()
    x0 = baseline_x(problem)
    result = slsqp_polish(x0, problem, source="test_baseline", max_iter=50)
    assert result.eval_polished.solve_ok is True
    # Polish must not break feasibility from a feasible start
    assert result.eval_polished.feasible is True


def test_polish_does_not_crash_on_degenerate_start() -> None:
    problem = _problem()
    x0 = baseline_x(problem).copy()
    # Move every free node to the same location
    for i in range(0, 12, 2):
        x0[i] = 5.0
        x0[i + 1] = -5.0
    result = slsqp_polish(x0, problem, source="test_degenerate", max_iter=10)
    # Either succeeded into something feasible, or recorded failure cleanly.
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
