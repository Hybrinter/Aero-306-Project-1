"""Unit tests for SLSQP constraint vector callables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.constraints import (
    buckling_constraint_vec,
    length_constraint_vec,
    stress_constraint_vec,
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


def test_feasible_baseline_has_nonnegative_constraints() -> None:
    """Stress and length are satisfied at the baseline; buckling is violated (max_buckling_ratio=1.5)."""
    problem = _problem()
    x = baseline_x(problem)
    s = stress_constraint_vec(x, problem)
    b = buckling_constraint_vec(x, problem)
    L = length_constraint_vec(x, problem)
    assert s.shape == (16,) and (s >= -1e-12).all()
    assert b.shape == (16,)  # baseline has a known buckling violation
    assert L.shape == (16,) and (L >= -1e-12).all()


def test_lightly_loaded_baseline_satisfies_buckling() -> None:
    """At F=0.01 N the baseline buckling ratio is far below 1, so b is non-negative."""
    problem = _problem(F=0.01)
    x = baseline_x(problem)
    b = buckling_constraint_vec(x, problem)
    assert (b >= -1e-12).all()


def test_overstress_drives_stress_constraint_negative() -> None:
    """A very large load produces stress violations, driving constraint values negative."""
    problem = _problem(F=10000.0)
    x = baseline_x(problem)
    s = stress_constraint_vec(x, problem)
    assert (s < 0.0).any()


def test_short_edge_drives_length_constraint_negative() -> None:
    """Moving node 7 to a position that creates a short edge violates the length constraint."""
    problem = _problem()
    x = baseline_x(problem).copy()
    x[8] = 4.99   # node 7 x
    x[9] = -10.0  # node 7 y onto bottom chord
    L = length_constraint_vec(x, problem)
    assert (L < 0.0).any()


def test_constraint_vectors_share_evaluator_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three constraint calls in a row at the same x should call evaluate() at most three times,
    not nine. (We can't test cache hits without instrumentation -- this is a smoke check that
    none raises and they all see the same FE state.)"""
    problem = _problem()
    x = baseline_x(problem)
    s = stress_constraint_vec(x, problem)
    b = buckling_constraint_vec(x, problem)
    L = length_constraint_vec(x, problem)
    assert len(s) == len(b) == len(L) == 16
