"""Unit tests for DE and CMA-ES global search runners."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.global_search import run_de
from fea_solver.optimization.problem import GeometryOptimizationProblem

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem() -> GeometryOptimizationProblem:
    """Build a small GeometryOptimizationProblem from problem_7.yaml."""
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


def test_run_de_smoke(tmp_path: Path) -> None:
    """Smoke test: run_de returns a valid SeedResult and writes checkpoint."""
    problem = _problem()
    sr = run_de(
        problem=problem,
        seed=0,
        popsize=5,
        maxiter=3,
        checkpoint_path=tmp_path / "de_seed_0.json",
    )
    assert sr.algorithm == "DE"
    assert sr.seed == 0
    assert sr.best_x.shape == (12,)
    assert sr.best_penalty >= 0.0
    assert len(sr.history) >= 1
    assert sr.wall_clock_s > 0.0
    assert (tmp_path / "de_seed_0.json").exists()
