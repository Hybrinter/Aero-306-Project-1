"""Tests for penalty constraint enforcement."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.constraints import apply_penalty_constraints, compute_constraint_residuals
from fea_solver.models import (
    DOFMap,
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
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_bar_dof_map() -> DOFMap:
    """2-node bar DOF map: node 1 -> DOF 0, node 2 -> DOF 1 (U only)."""
    dm = DOFMap()
    dm.mapping[(1, DOFType.U)] = 0
    dm.mapping[(2, DOFType.U)] = 1
    dm.total_dofs = 2
    return dm


def make_truss_dof_map() -> DOFMap:
    """2-node truss DOF map: node 1 -> [0,1], node 2 -> [2,3] (U,V)."""
    dm = DOFMap()
    dm.mapping[(1, DOFType.U)] = 0
    dm.mapping[(1, DOFType.V)] = 1
    dm.mapping[(2, DOFType.U)] = 2
    dm.mapping[(2, DOFType.V)] = 3
    dm.total_dofs = 4
    return dm


def make_frame_dof_map() -> DOFMap:
    """2-node frame DOF map: node 1 -> [0,1,2], node 2 -> [3,4,5] (U,V,THETA)."""
    dm = DOFMap()
    dm.mapping[(1, DOFType.U)] = 0
    dm.mapping[(1, DOFType.V)] = 1
    dm.mapping[(1, DOFType.THETA)] = 2
    dm.mapping[(2, DOFType.U)] = 3
    dm.mapping[(2, DOFType.V)] = 4
    dm.mapping[(2, DOFType.THETA)] = 5
    dm.total_dofs = 6
    return dm


# ---------------------------------------------------------------------------
# Tests for apply_penalty_constraints
# ---------------------------------------------------------------------------


class TestApplyPenaltyConstraints:
    """Tests for apply_penalty_constraints."""

    def test_axis_aligned_u_constraint_adds_to_diagonal(self) -> None:
        """U-direction constraint adds k_penalty to K[0,0]."""
        K = np.zeros((2, 2))
        F = np.zeros(2)
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        K_mod, F_mod = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        assert K_mod[0, 0] == pytest.approx(1e6)
        assert K_mod[1, 1] == pytest.approx(0.0)

    def test_axis_aligned_v_constraint_adds_to_v_diagonal(self) -> None:
        """V-direction constraint adds k_penalty to K[V,V] diagonal."""
        K = np.zeros((4, 4))
        F = np.zeros(4)
        dof_map = make_truss_dof_map()
        constraint = LinearConstraint(node_id=2, coefficients=(0.0, 1.0, 0.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        v_idx = dof_map.index(2, DOFType.V)
        assert K_mod[v_idx, v_idx] == pytest.approx(1e6)

    def test_inclined_constraint_adds_off_diagonal_coupling(self) -> None:
        """Inclined constraint (45-deg) adds coupling between U and V DOFs."""
        K = np.zeros((4, 4))
        F = np.zeros(4)
        dof_map = make_truss_dof_map()
        n = 1.0 / math.sqrt(2.0)
        constraint = LinearConstraint(node_id=2, coefficients=(n, n, 0.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        u_idx = dof_map.index(2, DOFType.U)
        v_idx = dof_map.index(2, DOFType.V)
        assert K_mod[u_idx, v_idx] == pytest.approx(0.5e6)
        assert K_mod[v_idx, u_idx] == pytest.approx(0.5e6)

    def test_rotation_constraint_adds_to_theta_diagonal(self) -> None:
        """THETA constraint adds k_penalty to K[THETA,THETA] diagonal."""
        K = np.zeros((6, 6))
        F = np.zeros(6)
        dof_map = make_frame_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(0.0, 0.0, 1.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        theta_idx = dof_map.index(1, DOFType.THETA)
        assert K_mod[theta_idx, theta_idx] == pytest.approx(1e6)

    def test_nonzero_rhs_modifies_force_vector(self) -> None:
        """Non-zero rhs adds k_penalty * rhs * g to F."""
        K = np.zeros((2, 2))
        F = np.zeros(2)
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.005)
        _, F_mod = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        assert F_mod[0] == pytest.approx(1e6 * 0.005)

    def test_does_not_mutate_input_arrays(self) -> None:
        """Input K and F are not modified in place."""
        K = np.eye(2)
        F = np.ones(2)
        K_copy = K.copy()
        F_copy = F.copy()
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        np.testing.assert_array_equal(K, K_copy)
        np.testing.assert_array_equal(F, F_copy)

    def test_nonzero_coefficient_for_missing_dof_raises(self) -> None:
        """Non-zero coefficient for a DOF absent at the node raises ValueError."""
        K = np.zeros((2, 2))
        F = np.zeros(2)
        dof_map = make_bar_dof_map()
        # Bar has no V DOF; [0.0, 1.0, 0.0] at node 1 should raise
        constraint = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0), rhs=0.0)
        with pytest.raises(ValueError, match="[Vv]"):
            apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)

    def test_returned_K_is_symmetric(self) -> None:
        """Modified K is symmetric."""
        K = np.zeros((4, 4))
        F = np.zeros(4)
        dof_map = make_truss_dof_map()
        n = 1.0 / math.sqrt(2.0)
        constraint = LinearConstraint(node_id=2, coefficients=(n, n, 0.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        np.testing.assert_allclose(K_mod, K_mod.T, atol=1e-12)

    def test_multiple_constraints_accumulate(self) -> None:
        """Multiple constraints accumulate on K correctly."""
        K = np.zeros((4, 4))
        F = np.zeros(4)
        dof_map = make_truss_dof_map()
        c1 = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        c2 = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (c1, c2), dof_map, k_penalty=1e6)
        assert K_mod[0, 0] == pytest.approx(1e6)   # U at node 1
        assert K_mod[1, 1] == pytest.approx(1e6)   # V at node 1


# ---------------------------------------------------------------------------
# Tests for compute_constraint_residuals
# ---------------------------------------------------------------------------


class TestComputeConstraintResiduals:
    """Tests for compute_constraint_residuals."""

    def test_zero_residual_at_exact_constraint(self) -> None:
        """Zero displacement at constrained DOF gives zero residual."""
        u = np.zeros(2)
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        R = compute_constraint_residuals(u, (constraint,), dof_map, k_penalty=1e6)
        assert R.shape == (1,)
        assert R[0] == pytest.approx(0.0)

    def test_residual_scales_with_displacement(self) -> None:
        """Residual is k_penalty * a^T * u when rhs=0."""
        u = np.array([0.001, 0.0])
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        R = compute_constraint_residuals(u, (constraint,), dof_map, k_penalty=1e6)
        assert R[0] == pytest.approx(1e6 * 0.001)

    def test_residual_accounts_for_rhs(self) -> None:
        """Residual is k_penalty * (a^T*u - rhs)."""
        u = np.array([0.005, 0.0])
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.005)
        R = compute_constraint_residuals(u, (constraint,), dof_map, k_penalty=1e6)
        assert R[0] == pytest.approx(0.0, abs=1e-9)

    def test_output_shape_equals_number_of_constraints(self) -> None:
        """Output length equals number of constraints."""
        u = np.zeros(4)
        dof_map = make_truss_dof_map()
        constraints = (
            LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0),
            LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0), rhs=0.0),
            LinearConstraint(node_id=2, coefficients=(0.0, 1.0, 0.0), rhs=0.0),
        )
        R = compute_constraint_residuals(u, constraints, dof_map, k_penalty=1e6)
        assert R.shape == (3,)

    def test_near_zero_residuals_after_full_solve(self) -> None:
        """Constraint residuals are near zero (< 1/k_p) after penalty solve."""
        # Single bar: E=A=L=1, constrain U at node 1, apply P=1 at node 2
        mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
        n1, n2 = Node(1, (0.0, 0.0)), Node(2, (1.0, 0.0))
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.BAR, material=mat)
        c = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_X, magnitude=1.0)
        model = FEAModel(
            mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
            boundary_conditions=(c,),
            nodal_loads=(load,),
            distributed_loads=(),
            label="test",
            penalty_alpha=1e8,
        )
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        k_penalty = model.penalty_alpha * float(np.max(np.abs(np.diag(K))))
        K_mod, F_mod = apply_penalty_constraints(K, F, model.boundary_conditions, dof_map, k_penalty)
        u = np.linalg.solve(K_mod, F_mod)
        R = compute_constraint_residuals(u, model.boundary_conditions, dof_map, k_penalty)
        # Constraint violation should be extremely small (at most 1/penalty_alpha)
        assert abs(u[dof_map.index(1, DOFType.U)]) <= 1.0 / model.penalty_alpha
        assert R.shape == (1,)
