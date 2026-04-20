"""Tests for DOF map construction and global matrix assembly -- TDD first."""
from __future__ import annotations
import numpy as np
import pytest
from fea_solver.models import (
    DistributedLoad, DOFType,
    Element, ElementType, FEAModel, LinearConstraint, LoadType, MaterialProperties, Mesh, Node, NodalLoad
)
from fea_solver.assembler import (
    build_dof_map, get_element_dof_indices,
    assemble_global_stiffness, assemble_global_force_vector,
)

def make_bar_model(n_nodes: int = 2) -> FEAModel:
    """Create a bar model with n_nodes nodes."""
    mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
    nodes = tuple(Node(id=i+1, pos=(float(i), 0.0)) for i in range(n_nodes))
    elements = tuple(
        Element(id=i+1, node_i=nodes[i], node_j=nodes[i+1],
                element_type=ElementType.BAR, material=mat)
        for i in range(n_nodes - 1)
    )
    mesh = Mesh(nodes=nodes, elements=elements)
    c = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0))
    load = NodalLoad(node_id=n_nodes, load_type=LoadType.POINT_FORCE_X, magnitude=1.0)
    return FEAModel(mesh=mesh, boundary_conditions=(c,), nodal_loads=(load,),
                    distributed_loads=(), label=f"bar_{n_nodes}_nodes")

def make_beam_model(n_nodes: int = 2) -> FEAModel:
    """Create a beam model with n_nodes nodes."""
    mat = MaterialProperties(E=1.0, A=1.0, I=1.0)
    nodes = tuple(Node(id=i+1, pos=(float(i), 0.0)) for i in range(n_nodes))
    elements = tuple(
        Element(id=i+1, node_i=nodes[i], node_j=nodes[i+1],
                element_type=ElementType.BEAM, material=mat)
        for i in range(n_nodes - 1)
    )
    mesh = Mesh(nodes=nodes, elements=elements)
    c_v = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0))
    c_t = LinearConstraint(node_id=1, coefficients=(0.0, 0.0, 1.0))
    load = NodalLoad(node_id=n_nodes, load_type=LoadType.POINT_FORCE_Y, magnitude=-1.0)
    return FEAModel(mesh=mesh, boundary_conditions=(c_v, c_t), nodal_loads=(load,),
                    distributed_loads=(), label=f"beam_{n_nodes}_nodes")


class TestBuildDofMap:
    """Tests for build_dof_map DOF count and ordering."""
    def test_bar_two_nodes_gives_two_dofs(self) -> None:
        """Two-node bar model has exactly 2 total DOFs."""
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        assert dof_map.total_dofs == 2

    def test_bar_three_nodes_gives_three_dofs(self) -> None:
        """Three-node bar model has exactly 3 total DOFs."""
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        assert dof_map.total_dofs == 3

    def test_beam_two_nodes_gives_four_dofs(self) -> None:
        """Two-node beam model has exactly 4 total DOFs."""
        model = make_beam_model(2)
        dof_map = build_dof_map(model)
        assert dof_map.total_dofs == 4

    def test_bar_dof_types_are_u_only(self) -> None:
        """Bar DOF types are U only, no V or THETA."""
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        assert dof_map.has_dof(1, DOFType.U)
        assert not dof_map.has_dof(1, DOFType.V)
        assert not dof_map.has_dof(1, DOFType.THETA)

    def test_beam_dof_types_are_v_and_theta(self) -> None:
        """Beam DOF types are V and THETA, no U."""
        model = make_beam_model(2)
        dof_map = build_dof_map(model)
        assert not dof_map.has_dof(1, DOFType.U)
        assert dof_map.has_dof(1, DOFType.V)
        assert dof_map.has_dof(1, DOFType.THETA)

    def test_dof_indices_start_at_zero(self) -> None:
        """DOF indices start at zero."""
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        assert dof_map.index(1, DOFType.U) == 0

    def test_dof_indices_are_contiguous(self) -> None:
        """DOF indices are contiguous from 0 to total_dofs-1."""
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        indices = sorted(dof_map.mapping.values())
        assert indices == list(range(3))


class TestGetElementDofIndices:
    """Tests for get_element_dof_indices DOF index retrieval."""
    def test_bar_returns_two_indices(self) -> None:
        """Bar element returns 2 DOF indices."""
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        idx = get_element_dof_indices(1, model, dof_map)
        assert len(idx) == 2

    def test_beam_returns_four_indices(self) -> None:
        """Beam element returns 4 DOF indices."""
        model = make_beam_model(2)
        dof_map = build_dof_map(model)
        idx = get_element_dof_indices(1, model, dof_map)
        assert len(idx) == 4

    def test_two_bar_elements_share_middle_node_dof(self) -> None:
        """Two bar elements share the middle node DOF."""
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        idx1 = get_element_dof_indices(1, model, dof_map)
        idx2 = get_element_dof_indices(2, model, dof_map)
        # Element 1 ends at node 2, element 2 starts at node 2
        assert idx1[1] == idx2[0]


class TestAssembleGlobalStiffness:
    """Tests for assemble_global_stiffness matrix assembly."""
    def test_shape_matches_total_dofs(self) -> None:
        """Global stiffness matrix shape matches total DOFs."""
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        n = dof_map.total_dofs
        assert K.shape == (n, n)

    def test_symmetry(self) -> None:
        """Global stiffness matrix is symmetric."""
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_two_bar_element_known_K(self) -> None:
        """Two-bar assembly produces known stiffness matrix."""
        # 3-node bar, each element EA/L=1
        # Global K should be [[1,-1,0],[-1,2,-1],[0,-1,1]]
        model = make_bar_model(3)
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        expected = np.array([[1,-1,0],[-1,2,-1],[0,-1,1]], dtype=float)
        np.testing.assert_allclose(K, expected, atol=1e-12)

    def test_beam_global_K_shape_4x4(self) -> None:
        """Two-node beam assembly produces 4x4 stiffness matrix."""
        model = make_beam_model(2)
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        assert K.shape == (4, 4)


class TestAssembleGlobalForceVector:
    """Tests for assemble_global_force_vector assembly."""
    def test_shape_matches_total_dofs(self) -> None:
        """Global force vector shape matches total DOFs."""
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        F = assemble_global_force_vector(model, dof_map)
        assert F.shape == (dof_map.total_dofs,)

    def test_nodal_load_placed_correctly(self) -> None:
        """Nodal load is placed at correct global DOF index."""
        # Bar model: P=1 at node 2 (global DOF index 1)
        model = make_bar_model(2)
        dof_map = build_dof_map(model)
        F = assemble_global_force_vector(model, dof_map)
        assert F[dof_map.index(2, DOFType.U)] == pytest.approx(1.0)

    def test_no_loads_gives_zero_vector(self) -> None:
        """Model with no loads produces zero force vector."""
        mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
        n1, n2 = Node(1, (0.0, 0.0)), Node(2, (1.0, 0.0))
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.BAR, material=mat)
        c = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0))
        model = FEAModel(
            mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
            boundary_conditions=(c,),
            nodal_loads=(), distributed_loads=(), label="no_loads"
        )
        dof_map = build_dof_map(model)
        F = assemble_global_force_vector(model, dof_map)
        np.testing.assert_allclose(F, np.zeros(dof_map.total_dofs))

    def test_distributed_load_adds_to_force_vector(self) -> None:
        """Distributed load produces non-zero force vector."""
        # Beam with UDL: consistent nodal forces must be non-zero
        mat = MaterialProperties(E=1.0, A=1.0, I=1.0)
        n1, n2 = Node(1, (0.0, 0.0)), Node(2, (1.0, 0.0))
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.BEAM, material=mat)
        dist_load = DistributedLoad(element_id=1, load_type=LoadType.DISTRIBUTED_Y,
                                    w_i=-1.0, w_j=-1.0)
        c_v = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0))
        c_t = LinearConstraint(node_id=1, coefficients=(0.0, 0.0, 1.0))
        model = FEAModel(
            mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
            boundary_conditions=(c_v, c_t),
            nodal_loads=(), distributed_loads=(dist_load,), label="beam_udl"
        )
        dof_map = build_dof_map(model)
        F = assemble_global_force_vector(model, dof_map)
        assert not np.allclose(F, 0.0)


class TestTrussDofMap:
    """Tests for TRUSS element DOF wiring in the assembler."""

    def test_truss_node_has_u_and_v_dofs(self) -> None:
        """TRUSS node gets U and V DOFs (not THETA)."""
        mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
        n1 = Node(1, (0.0, 0.0))
        n2 = Node(2, (1.0, 0.0))
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.TRUSS, material=mat)
        model = FEAModel(mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
                         boundary_conditions=(), nodal_loads=(), distributed_loads=(),
                         label="truss_dof_test")
        dof_map = build_dof_map(model)
        assert dof_map.has_dof(1, DOFType.U)
        assert dof_map.has_dof(1, DOFType.V)
        assert not dof_map.has_dof(1, DOFType.THETA)

    def test_truss_two_nodes_gives_four_dofs(self) -> None:
        """Two-node truss has 4 total DOFs (U and V at each node)."""
        mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
        n1 = Node(1, (0.0, 0.0))
        n2 = Node(2, (1.0, 1.0))
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.TRUSS, material=mat)
        model = FEAModel(mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
                         boundary_conditions=(), nodal_loads=(), distributed_loads=(),
                         label="truss_4dof")
        dof_map = build_dof_map(model)
        assert dof_map.total_dofs == 4

    def test_truss_element_returns_four_dof_indices(self) -> None:
        """TRUSS element get_element_dof_indices returns 4 indices."""
        mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
        n1 = Node(1, (0.0, 0.0))
        n2 = Node(2, (1.0, 0.0))
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.TRUSS, material=mat)
        model = FEAModel(mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
                         boundary_conditions=(), nodal_loads=(), distributed_loads=(),
                         label="truss_idx_test")
        dof_map = build_dof_map(model)
        indices = get_element_dof_indices(1, model, dof_map)
        assert len(indices) == 4
