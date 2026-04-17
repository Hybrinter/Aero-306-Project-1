"""Unit tests for GeometryOptimizationProblem and apply_x_to_model."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.models import FEAModel
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    apply_x_to_model,
    baseline_x,
)

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"

FREE_NODE_IDS = (2, 3, 5, 6, 7, 8)
FROZEN_NODE_IDS = (1, 4, 9)


def _baseline_model() -> FEAModel:
    return load_models_from_yaml(PROBLEM_7)[0]


def test_baseline_problem_constructs() -> None:
    model = _baseline_model()
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )
    assert problem.n_vars == 12
    assert len(problem.box_bounds) == 12
    assert all(lo == -10.0 and hi == 30.0 for lo, hi in problem.box_bounds[0::2])
    assert all(lo == -25.0 and hi == 15.0 for lo, hi in problem.box_bounds[1::2])


def test_rejects_overlapping_free_and_frozen() -> None:
    model = _baseline_model()
    with pytest.raises(ValueError, match="overlap"):
        GeometryOptimizationProblem.from_baseline(
            model=model,
            free_node_ids=(2, 3),
            frozen_node_ids=(2, 9),
            x_bounds=(-10.0, 30.0),
            y_bounds=(-25.0, 15.0),
            F_magnitude=15.0,
            sigma_max=72.0,
            L_min=5.0,
        )


def test_rejects_inverted_bounds() -> None:
    model = _baseline_model()
    with pytest.raises(ValueError, match="bounds"):
        GeometryOptimizationProblem.from_baseline(
            model=model,
            free_node_ids=FREE_NODE_IDS,
            frozen_node_ids=FROZEN_NODE_IDS,
            x_bounds=(30.0, -10.0),
            y_bounds=(-25.0, 15.0),
            F_magnitude=15.0,
            sigma_max=72.0,
            L_min=5.0,
        )


def test_rejects_inverted_y_bounds() -> None:
    """Inverted y_bounds should raise ValueError with message matching 'bounds'."""
    model = _baseline_model()
    with pytest.raises(ValueError, match="bounds"):
        GeometryOptimizationProblem.from_baseline(
            model=model,
            free_node_ids=FREE_NODE_IDS,
            frozen_node_ids=FROZEN_NODE_IDS,
            x_bounds=(-10.0, 30.0),
            y_bounds=(15.0, -25.0),
            F_magnitude=15.0,
            sigma_max=72.0,
            L_min=5.0,
        )


def test_baseline_x_extracts_free_node_positions() -> None:
    model = _baseline_model()
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )
    x0 = baseline_x(problem)
    # Node 2 is at (10, 0) in problem_7.yaml
    assert x0[0] == pytest.approx(10.0)
    assert x0[1] == pytest.approx(0.0)
    # Node 8 is at (15, -5)
    assert x0[10] == pytest.approx(15.0)
    assert x0[11] == pytest.approx(-5.0)


def test_apply_x_overwrites_only_free_nodes() -> None:
    model = _baseline_model()
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )
    x = np.array([
        11.0, 1.0,    # node 2
        21.0, 0.5,    # node 3
        9.0, -11.0,   # node 5
        22.0, -9.5,   # node 6
        4.0, -6.0,    # node 7
        16.0, -4.0,   # node 8
    ])
    new_model = apply_x_to_model(x, problem)
    nodes_by_id = {n.id: n for n in new_model.mesh.nodes}
    # Free nodes overwritten
    assert nodes_by_id[2].pos == pytest.approx((11.0, 1.0))
    assert nodes_by_id[8].pos == pytest.approx((16.0, -4.0))
    # Frozen nodes preserved
    assert nodes_by_id[1].pos == pytest.approx((0.0, 0.0))
    assert nodes_by_id[4].pos == pytest.approx((0.0, -10.0))
    assert nodes_by_id[9].pos == pytest.approx((25.0, -5.0))
    # Connectivity preserved
    assert len(new_model.mesh.elements) == 16


def test_apply_x_overrides_F_magnitude() -> None:
    model = _baseline_model()
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )
    x = baseline_x(problem)
    new_model = apply_x_to_model(x, problem)
    nodal_loads = new_model.nodal_loads
    assert len(nodal_loads) == 1
    assert nodal_loads[0].node_id == 9
    assert nodal_loads[0].magnitude == pytest.approx(-15.0)
