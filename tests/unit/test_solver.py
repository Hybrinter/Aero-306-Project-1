"""Tests for the penalty-method solver pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.models import (
    DOFType,
    Element,
    ElementType,
    FEAModel,
    LinearConstraint,
    LoadType,
    MaterialProperties,
    Mesh,
    Node,
    NodalLoad,
    SolutionResult,
)
from fea_solver.solver import compute_penalty_parameter, run_solve_pipeline, solve_system


def _make_cantilever_bar_model() -> FEAModel:
    """Single bar element: E=A=L=1, fixed U at node 1, P=1 at node 2."""
    mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
    n1, n2 = Node(1, (0.0, 0.0)), Node(2, (1.0, 0.0))
    elem = Element(id=1, node_i=n1, node_j=n2,
                   element_type=ElementType.BAR, material=mat)
    c = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
    load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_X, magnitude=1.0)
    return FEAModel(
        mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
        boundary_conditions=(c,),
        nodal_loads=(load,),
        distributed_loads=(),
        label="cantilever_bar",
        penalty_alpha=1e10,
    )


class TestComputePenaltyParameter:
    """Tests for compute_penalty_parameter."""

    def test_scales_with_max_diagonal(self) -> None:
        """k_penalty = penalty_alpha * max(abs(diag(K)))."""
        K = np.diag([2.0, 5.0, 3.0])
        k = compute_penalty_parameter(K, penalty_alpha=1e8)
        assert k == pytest.approx(5.0e8)

    def test_uses_absolute_value_of_diagonal(self) -> None:
        """Uses abs of diagonal entries."""
        K = np.diag([-4.0, 1.0])
        k = compute_penalty_parameter(K, penalty_alpha=1e8)
        assert k == pytest.approx(4.0e8)


class TestSolveSystem:
    """Tests for solve_system."""

    def test_trivial_system(self) -> None:
        """Solves identity system correctly."""
        K = np.eye(3)
        F = np.array([1.0, 2.0, 3.0])
        u = solve_system(K, F)
        np.testing.assert_allclose(u, F)

    def test_singular_raises(self) -> None:
        """Singular matrix raises LinAlgError."""
        K = np.zeros((2, 2))
        F = np.array([1.0, 1.0])
        with pytest.raises(np.linalg.LinAlgError):
            solve_system(K, F)

    def test_penalty_aware_threshold_suppresses_expected_cond(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Well-posed model with large penalty_alpha must not warn.

        Builds K_mod with cond = 1e15 and penalty_alpha = 1e8, so
        cond / penalty_alpha = 1e7. Float64 still has ~9 digits of margin,
        below the 1e8 * penalty_alpha = 1e16 precision-breakdown threshold.
        The old fixed 1e14 threshold would (spuriously) fire here.
        """
        penalty_alpha = 1.0e8
        K = np.diag([1.0e15, 1.0])
        F = np.array([1.0, 1.0])
        with caplog.at_level("WARNING", logger="fea_solver.solver"):
            solve_system(K, F, penalty_alpha=penalty_alpha)
        assert not any("nearly singular" in r.message for r in caplog.records)

    def test_penalty_aware_threshold_fires_on_genuine_near_mechanism(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn when cond(K_mod) > 1e8 * penalty_alpha (float64 breakdown).

        Builds K_mod whose cond = 1e17 with penalty_alpha = 1e8, i.e.,
        cond / penalty_alpha = 1e9 -- decisively past the 1e8 threshold.
        """
        penalty_alpha = 1.0e8
        K = np.diag([1.0e17, 1.0])
        F = np.array([1.0, 1.0])
        with caplog.at_level("WARNING", logger="fea_solver.solver"):
            solve_system(K, F, penalty_alpha=penalty_alpha)
        assert any("nearly singular" in r.message for r in caplog.records)

    def test_default_threshold_preserved_without_penalty_alpha(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Legacy callers that do not pass penalty_alpha keep the 1e14 threshold."""
        K = np.diag([1.0e15, 1.0])
        F = np.array([1.0, 1.0])
        with caplog.at_level("WARNING", logger="fea_solver.solver"):
            solve_system(K, F)
        assert any("nearly singular" in r.message for r in caplog.records)


class TestRunSolvePipeline:
    """Tests for run_solve_pipeline."""

    def test_cantilever_bar_tip_displacement(self) -> None:
        """Tip displacement equals P*L/(E*A) = 1.0."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        u_tip = result.displacements[dof_map.index(2, DOFType.U)]
        assert u_tip == pytest.approx(1.0, rel=1e-6)

    def test_returns_solution_result_type(self) -> None:
        """run_solve_pipeline returns SolutionResult."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        assert isinstance(result, SolutionResult)

    def test_reactions_shape_matches_constraints(self) -> None:
        """reactions array length equals number of constraints."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        assert result.reactions.shape == (len(model.boundary_conditions),)

    def test_reaction_equals_applied_load(self) -> None:
        """For a single-constraint bar, reaction magnitude equals applied load."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        # The pin reaction at node 1 must balance the applied load P=1
        assert abs(result.reactions[0]) == pytest.approx(1.0, rel=1e-4)

    def test_fixed_node_displacement_near_zero(self) -> None:
        """Constrained node displacement is near zero (within penalty tolerance)."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        u_fixed = result.displacements[dof_map.index(1, DOFType.U)]
        assert abs(u_fixed) <= 1.0 / model.penalty_alpha
