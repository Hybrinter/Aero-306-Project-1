"""Unit tests for ensemble orchestration and selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.optimization.checkpoint import (
    EnsembleResult,
    PolishResult,
    SeedResult,
)
from fea_solver.optimization.ensemble import (
    EnsembleConfig,
    run_ensemble,
    select_best,
)
from fea_solver.optimization.objective import EvalResult
from fea_solver.optimization.problem import GeometryOptimizationProblem
from fea_solver.io_yaml import load_models_from_yaml

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem() -> GeometryOptimizationProblem:
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


def _eval(tip: float, feas: bool = True, ok: bool = True) -> EvalResult:
    z = tuple([0.0] * 16)
    nz = tuple([0.5] * 16)
    return EvalResult(
        tip_disp=tip, max_stress=10.0, max_buckling_ratio=0.1, min_length=10.0,
        stress_violations=z if feas else nz,
        buckling_violations=z if feas else nz,
        length_violations=z if feas else nz,
        feasible=feas, solve_ok=ok,
    )


def test_select_best_chooses_min_tip_disp_among_feasible() -> None:
    polishes = (
        PolishResult(source="a", x_polished=np.zeros(12), eval_polished=_eval(0.05),
                     success=True, n_iter=10, message="ok"),
        PolishResult(source="b", x_polished=np.ones(12), eval_polished=_eval(0.02),
                     success=True, n_iter=10, message="ok"),
        PolishResult(source="c", x_polished=np.ones(12) * 2,
                     eval_polished=_eval(0.01, feas=False),
                     success=True, n_iter=10, message="ok"),
    )
    seeds = ()
    er = select_best(seeds, polishes)
    assert er.feasible is True
    assert er.winner_eval.tip_disp == pytest.approx(0.02)


def test_select_best_falls_back_when_none_feasible() -> None:
    polishes = (
        PolishResult(source="a", x_polished=np.zeros(12),
                     eval_polished=_eval(0.05, feas=False),
                     success=True, n_iter=10, message="ok"),
        PolishResult(source="b", x_polished=np.ones(12),
                     eval_polished=_eval(0.02, feas=False),
                     success=True, n_iter=10, message="ok"),
    )
    er = select_best((), polishes)
    assert er.feasible is False
    # Falls back to combined-score winner
    assert er.winner_eval.tip_disp in (0.02, 0.05)


def test_run_ensemble_smoke(tmp_path: Path) -> None:
    problem = _problem()
    config = EnsembleConfig(
        de_seeds=2, cmaes_seeds=2,
        de_popsize=5, de_maxiter=2,
        cmaes_popsize=6, cmaes_maxiter=2, cmaes_sigma0=2.0, cmaes_restarts=0,
        top_k=2, polish_max_iter=20,
        workers=2, run_dir=tmp_path,
    )
    er = run_ensemble(problem, config)
    assert isinstance(er, EnsembleResult)
    assert len(er.all_seeds) == 4
    # Polish runs: top_k from each seed
    assert len(er.all_polish) == 4 * config.top_k
    assert (tmp_path / "ensemble_result.json").exists()
