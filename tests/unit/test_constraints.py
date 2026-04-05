"""Tests for kinematic constraint application — TDD first."""
from __future__ import annotations
import numpy as np
import pytest
from fea_solver.models import (
    BoundaryCondition, BoundaryConditionType, DOFMap, DOFType,
    Element, ElementType, FEAModel, MaterialProperties, Mesh, Node, NodalLoad, LoadType
)
from fea_solver.constraints import (
    get_constrained_dof_indices,
    get_free_dof_indices,
    apply_constraints_reduction,
)


def make_bar_dof_map(n_nodes: int = 2) -> DOFMap:
    """Bar DOF map: node i -> DOF index i-1 (U only)."""
    dm = DOFMap()
    for i in range(n_nodes):
        dm.mapping[(i + 1, DOFType.U)] = i
    dm.total_dofs = n_nodes
    return dm


def make_beam_dof_map(n_nodes: int = 2) -> DOFMap:
    """Beam DOF map: node i -> DOF indices [2i-2, 2i-1] (V, THETA)."""
    dm = DOFMap()
    for i in range(n_nodes):
        dm.mapping[(i + 1, DOFType.V)] = 2 * i
        dm.mapping[(i + 1, DOFType.THETA)] = 2 * i + 1
    dm.total_dofs = 2 * n_nodes
    return dm


def make_bar_model_with_bc(bc_type: BoundaryConditionType, n_nodes: int = 2) -> FEAModel:
    mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
    nodes = tuple(Node(id=i+1, x=float(i)) for i in range(n_nodes))
    elements = tuple(
        Element(id=i+1, node_i=nodes[i], node_j=nodes[i+1],
                element_type=ElementType.BAR, material=mat)
        for i in range(n_nodes - 1)
    )
    bc = BoundaryCondition(node_id=1, bc_type=bc_type)
    return FEAModel(mesh=Mesh(nodes=nodes, elements=elements),
                    boundary_conditions=(bc,), nodal_loads=(), distributed_loads=(),
                    label="test")


def make_beam_model_with_bc(bc_type: BoundaryConditionType) -> FEAModel:
    mat = MaterialProperties(E=1.0, A=1.0, I=1.0)
    n1, n2 = Node(1, 0.0), Node(2, 1.0)
    elem = Element(id=1, node_i=n1, node_j=n2, element_type=ElementType.BEAM, material=mat)
    bc = BoundaryCondition(node_id=1, bc_type=bc_type)
    load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_Y, magnitude=-1.0)
    return FEAModel(mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
                    boundary_conditions=(bc,), nodal_loads=(load,), distributed_loads=(),
                    label="test_beam")


class TestGetConstrainedDofIndices:
    def test_fixed_u_on_bar_returns_one_index(self):
        model = make_bar_model_with_bc(BoundaryConditionType.FIXED_U)
        dof_map = make_bar_dof_map(2)
        constrained = get_constrained_dof_indices(model, dof_map)
        assert len(constrained) == 1
        assert dof_map.index(1, DOFType.U) in constrained

    def test_fixed_all_on_beam_returns_two_indices(self):
        model = make_beam_model_with_bc(BoundaryConditionType.FIXED_ALL)
        dof_map = make_beam_dof_map(2)
        constrained = get_constrained_dof_indices(model, dof_map)
        assert len(constrained) == 2
        assert dof_map.index(1, DOFType.V) in constrained
        assert dof_map.index(1, DOFType.THETA) in constrained

    def test_pin_on_beam_returns_v_only(self):
        # BEAM has no U DOF, so PIN (which targets u+v) should only constrain v
        model = make_beam_model_with_bc(BoundaryConditionType.PIN)
        dof_map = make_beam_dof_map(2)
        constrained = get_constrained_dof_indices(model, dof_map)
        assert dof_map.index(1, DOFType.V) in constrained
        # THETA should NOT be constrained by PIN
        assert dof_map.index(1, DOFType.THETA) not in constrained

    def test_roller_constrains_v_only(self):
        model = make_beam_model_with_bc(BoundaryConditionType.ROLLER)
        dof_map = make_beam_dof_map(2)
        constrained = get_constrained_dof_indices(model, dof_map)
        assert len(constrained) == 1
        assert dof_map.index(1, DOFType.V) in constrained

    def test_constrained_indices_are_sorted(self):
        model = make_beam_model_with_bc(BoundaryConditionType.FIXED_ALL)
        dof_map = make_beam_dof_map(2)
        constrained = get_constrained_dof_indices(model, dof_map)
        assert constrained == sorted(constrained)


class TestGetFreeDofIndices:
    def test_free_plus_constrained_equals_all(self):
        model = make_bar_model_with_bc(BoundaryConditionType.FIXED_U)
        dof_map = make_bar_dof_map(2)
        free = get_free_dof_indices(model, dof_map)
        constrained = get_constrained_dof_indices(model, dof_map)
        assert sorted(free + constrained) == list(range(dof_map.total_dofs))

    def test_no_overlap_between_free_and_constrained(self):
        model = make_beam_model_with_bc(BoundaryConditionType.FIXED_ALL)
        dof_map = make_beam_dof_map(2)
        free = set(get_free_dof_indices(model, dof_map))
        constrained = set(get_constrained_dof_indices(model, dof_map))
        assert free.isdisjoint(constrained)

    def test_free_indices_are_sorted(self):
        model = make_bar_model_with_bc(BoundaryConditionType.FIXED_U)
        dof_map = make_bar_dof_map(2)
        free = get_free_dof_indices(model, dof_map)
        assert free == sorted(free)


class TestApplyConstraintsReduction:
    def test_reduced_K_shape(self):
        # 2-DOF bar, 1 constrained -> 1x1 K_ff
        K = np.array([[1.0, -1.0], [-1.0, 1.0]])
        F = np.array([0.0, 1.0])
        K_ff, F_f = apply_constraints_reduction(K, F, constrained_dofs=[0])
        assert K_ff.shape == (1, 1)
        assert F_f.shape == (1,)

    def test_reduced_K_correct_values(self):
        # 3-DOF bar: K = [[1,-1,0],[-1,2,-1],[0,-1,1]], constrain DOF 0
        K = np.array([[1,-1,0],[-1,2,-1],[0,-1,1]], dtype=float)
        F = np.array([0.0, 0.0, 1.0])
        K_ff, F_f = apply_constraints_reduction(K, F, constrained_dofs=[0])
        expected_K_ff = np.array([[2,-1],[-1,1]], dtype=float)
        np.testing.assert_allclose(K_ff, expected_K_ff)

    def test_reduced_F_correct_values(self):
        K = np.array([[1,-1,0],[-1,2,-1],[0,-1,1]], dtype=float)
        F = np.array([0.0, 0.0, 5.0])
        K_ff, F_f = apply_constraints_reduction(K, F, constrained_dofs=[0])
        np.testing.assert_allclose(F_f, np.array([0.0, 5.0]))

    def test_reduced_K_is_symmetric(self):
        K = np.array([[12,6,-12,6],[6,4,-6,2],[-12,-6,12,-6],[6,2,-6,4]], dtype=float)
        F = np.zeros(4)
        K_ff, _ = apply_constraints_reduction(K, F, constrained_dofs=[0, 1])
        np.testing.assert_allclose(K_ff, K_ff.T, atol=1e-12)
