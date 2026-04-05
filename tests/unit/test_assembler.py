"""Tests for DOF map construction and global matrix assembly — TDD first."""
from __future__ import annotations
import numpy as np
import pytest
from fea_solver.models import (
    BoundaryCondition, BoundaryConditionType, DistributedLoad, DOFType,
    Element, ElementType, FEAModel, LoadType, MaterialProperties, Mesh, Node, NodalLoad
)
from fea_solver.assembler import (
    build_dof_map, get_element_dof_indices,
    assemble_global_stiffness, assemble_global_force_vector,
)

def make_bar_model(n_nodes: int = 2) -> FEAModel:
    mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
    nodes = tuple(Node(id=i+1, x=float(i)) for i in range(n_nodes))
    elements = tuple(
        Element(id=i+1, node_i=nodes[i], node_j=nodes[i+1],
                element_type=ElementType.BAR, material=mat)
        for i in range(n_nodes - 1)
    )
    mesh = Mesh(nodes=nodes, elements=elements)
    bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_U)
    load = NodalLoad(node_id=n_nodes, load_type=LoadType.POINT_FORCE_X, magnitude=1.0)
    return FEAModel(mesh=mesh, boundary_conditions=(bc,), nodal_loads=(load,),
                    distributed_loads=(), label=f"bar_{n_nodes}_nodes")

def make_beam_model(n_nodes: int = 2) -> FEAModel:
    mat = MaterialProperties(E=1.0, A=1.0, I=1.0)
    nodes = tuple(Node(id=i+1, x=float(i)) for i in range(n_nodes))
    elements = tuple(
        Element(id=i+1, node_i=nodes[i], node_j=nodes[i+1],
                element_type=ElementType.BEAM, material=mat)
        for i in range(n_nodes - 1)
    )
    mesh = Mesh(nodes=nodes, elements=elements)
    bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_ALL)
    load = NodalLoad(node_id=n_nodes, load_type=LoadType.POINT_FORCE_Y, magnitude=-1.0)
    return FEAModel(mesh=mesh, boundary_conditions=(bc,), nodal_loads=(load,),
                    distributed_loads=(), label=f"beam_{n_nodes}_nodes")


class TestBuildDofMap:
    def test_bar_two_nodes_gives_two_dofs(self):
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        assert dof_map.total_dofs == 2

    def test_bar_three_nodes_gives_three_dofs(self):
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        assert dof_map.total_dofs == 3

    def test_beam_two_nodes_gives_four_dofs(self):
        model = make_beam_model(2)
        dof_map = build_dof_map(model)
        assert dof_map.total_dofs == 4

    def test_bar_dof_types_are_u_only(self):
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        assert dof_map.has_dof(1, DOFType.U)
        assert not dof_map.has_dof(1, DOFType.V)
        assert not dof_map.has_dof(1, DOFType.THETA)

    def test_beam_dof_types_are_v_and_theta(self):
        model = make_beam_model(2)
        dof_map = build_dof_map(model)
        assert not dof_map.has_dof(1, DOFType.U)
        assert dof_map.has_dof(1, DOFType.V)
        assert dof_map.has_dof(1, DOFType.THETA)

    def test_dof_indices_start_at_zero(self):
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        assert dof_map.index(1, DOFType.U) == 0

    def test_dof_indices_are_contiguous(self):
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        indices = sorted(dof_map.mapping.values())
        assert indices == list(range(3))


class TestGetElementDofIndices:
    def test_bar_returns_two_indices(self):
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        idx = get_element_dof_indices(1, model, dof_map)
        assert len(idx) == 2

    def test_beam_returns_four_indices(self):
        model = make_beam_model(2)
        dof_map = build_dof_map(model)
        idx = get_element_dof_indices(1, model, dof_map)
        assert len(idx) == 4

    def test_two_bar_elements_share_middle_node_dof(self):
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        idx1 = get_element_dof_indices(1, model, dof_map)
        idx2 = get_element_dof_indices(2, model, dof_map)
        # Element 1 ends at node 2, element 2 starts at node 2
        assert idx1[1] == idx2[0]


class TestAssembleGlobalStiffness:
    def test_shape_matches_total_dofs(self):
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        n = dof_map.total_dofs
        assert K.shape == (n, n)

    def test_symmetry(self):
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_two_bar_element_known_K(self):
        # 3-node bar, each element EA/L=1
        # Global K should be [[1,-1,0],[-1,2,-1],[0,-1,1]]
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        expected = np.array([[1,-1,0],[-1,2,-1],[0,-1,1]], dtype=float)
        np.testing.assert_allclose(K, expected, atol=1e-12)

    def test_beam_global_K_shape_4x4(self):
        model = make_beam_model(2)
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        assert K.shape == (4, 4)


class TestAssembleGlobalForceVector:
    def test_shape_matches_total_dofs(self):
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        F = assemble_global_force_vector(model, dof_map)
        assert F.shape == (dof_map.total_dofs,)

    def test_nodal_load_placed_correctly(self):
        # Bar model: P=1 at node 2 (global DOF index 1)
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        F = assemble_global_force_vector(model, dof_map)
        assert F[dof_map.index(2, DOFType.U)] == pytest.approx(1.0)

    def test_no_loads_gives_zero_vector(self):
        mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
        n1, n2 = Node(1, 0.0), Node(2, 1.0)
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.BAR, material=mat)
        model = FEAModel(
            mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
            boundary_conditions=(BoundaryCondition(1, BoundaryConditionType.FIXED_U),),
            nodal_loads=(), distributed_loads=(), label="no_loads"
        )
        dof_map = build_dof_map(model)
        F = assemble_global_force_vector(model, dof_map)
        np.testing.assert_allclose(F, np.zeros(dof_map.total_dofs))

    def test_distributed_load_adds_to_force_vector(self):
        # Beam with UDL: consistent nodal forces must be non-zero
        mat = MaterialProperties(E=1.0, A=1.0, I=1.0)
        n1, n2 = Node(1, 0.0), Node(2, 1.0)
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.BEAM, material=mat)
        dist_load = DistributedLoad(element_id=1, load_type=LoadType.DISTRIBUTED_Y,
                                    w_i=-1.0, w_j=-1.0)
        model = FEAModel(
            mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
            boundary_conditions=(BoundaryCondition(1, BoundaryConditionType.FIXED_ALL),),
            nodal_loads=(), distributed_loads=(dist_load,), label="beam_udl"
        )
        dof_map = build_dof_map(model)
        F = assemble_global_force_vector(model, dof_map)
        assert not np.allclose(F, 0.0)
