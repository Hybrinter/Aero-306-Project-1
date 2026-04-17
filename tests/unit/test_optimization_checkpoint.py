"""Unit tests for checkpoint serializers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fea_solver.optimization.checkpoint import (
    HistoryPoint,
    PolishResult,
    SeedResult,
    load_seed_result,
    save_seed_result,
)
from fea_solver.optimization.objective import EvalResult


def _sample_eval() -> EvalResult:
    return EvalResult(
        tip_disp=0.0184,
        max_stress=71.5,
        max_buckling_ratio=0.92,
        min_length=5.001,
        stress_violations=tuple([0.0] * 16),
        buckling_violations=tuple([0.0] * 16),
        length_violations=tuple([0.0] * 16),
        feasible=True,
        solve_ok=True,
    )


def _sample_seed() -> SeedResult:
    return SeedResult(
        algorithm="DE",
        seed=42,
        best_x=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        best_eval=_sample_eval(),
        best_penalty=0.0184,
        history=(
            HistoryPoint(generation=0, best_penalty=10.0, mean_penalty=20.0, n_feasible=0),
            HistoryPoint(generation=10, best_penalty=0.5, mean_penalty=1.0, n_feasible=12),
        ),
        wall_clock_s=12.5,
        checkpoint_path=Path("ignored.json"),
    )


def test_seed_result_round_trip(tmp_path: Path) -> None:
    """Round-trip save/load for SeedResult."""
    sr = _sample_seed()
    path = tmp_path / "de_seed_42.json"
    save_seed_result(sr, path)
    loaded = load_seed_result(path)
    assert loaded.algorithm == sr.algorithm
    assert loaded.seed == sr.seed
    np.testing.assert_array_equal(loaded.best_x, sr.best_x)
    assert loaded.best_eval == sr.best_eval
    assert loaded.best_penalty == sr.best_penalty
    assert loaded.history == sr.history
    assert loaded.wall_clock_s == sr.wall_clock_s


def test_load_corrupt_seed_result_raises(tmp_path: Path) -> None:
    """Corrupt JSON raises ValueError with 'checkpoint' in message."""
    path = tmp_path / "corrupt.json"
    path.write_text("{ not valid json")
    with pytest.raises(ValueError, match="checkpoint"):
        load_seed_result(path)
