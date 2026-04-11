"""Tests for element post-processing (internal forces) -- TDD first."""
from __future__ import annotations
import numpy as np
import pytest
from fea_solver.models import (
    BoundaryCondition, BoundaryConditionType, DOFMap, DOFType,
    Element, ElementType, FEAModel, LoadType, MaterialProperties,
    Mesh, Node, NodalLoad, SolutionResult
)
from fea_solver.assembler import build_dof_map, assemble_global_stiffness, assemble_global_force_vector
from fea_solver.solver import run_solve_pipeline
from fea_solver.postprocessor import (
    compute_element_axial_force,
    compute_beam_internal_forces,
    postprocess_all_elements,
)


def solve_cantilever_beam(L: float = 1.0, EI: float = 1.0, P: float = -1.0) -> tuple:
    """Helper: single-element cantilever beam, fixed at node 1, tip load P at node 2."""
    mat = MaterialProperties(E=EI, A=1.0, I=1.0)
    n1, n2 = Node(1, 0.0), Node(2, L)
    elem = Element(id=1, node_i=n1, node_j=n2, element_type=ElementType.BEAM, material=mat)
    bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_ALL)
    load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_Y, magnitude=P)
    model = FEAModel(mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
                     boundary_conditions=(bc,), nodal_loads=(load,),
                     distributed_loads=(), label="cantilever")
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    return model, dof_map, result


def solve_bar(L: float = 1.0, EA: float = 1.0, P: float = 1.0) -> tuple:
    """Helper: single bar, fixed at node 1, axial load P at node 2."""
    mat = MaterialProperties(E=EA, A=1.0, I=0.0)
    n1, n2 = Node(1, 0.0), Node(2, L)
    elem = Element(id=1, node_i=n1, node_j=n2, element_type=ElementType.BAR, material=mat)
    bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_U)
    load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_X, magnitude=P)
    model = FEAModel(mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
                     boundary_conditions=(bc,), nodal_loads=(load,),
                     distributed_loads=(), label="bar")
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    return model, dof_map, result


class TestComputeElementAxialForce:
    """Tests for compute_element_axial_force."""
    def test_bar_unit_force(self) -> None:
        """Bar with unit load gives unit axial force."""
        # E=1, A=1, L=1, P=1 -> u_tip = 1 -> N = EA/L*(u_j - u_i) = 1
        model, dof_map, result = solve_bar(L=1.0, EA=1.0, P=1.0)
        N = compute_element_axial_force(1, model, dof_map, result.displacements)
        assert N == pytest.approx(1.0)

    def test_bar_scaled_force(self) -> None:
        """Bar with scaled load gives correct axial force."""
        # E=200e9, A=0.01, L=1, P=10000 -> u_tip = P/(EA) = 10000/(2e9) = 5e-6
        # N = EA/L*(u_j - u_i) = 2e9 * 5e-6 = 10000
        model, dof_map, result = solve_bar(L=1.0, EA=200.0e9*0.01, P=10000.0)
        N = compute_element_axial_force(1, model, dof_map, result.displacements)
        assert N == pytest.approx(10000.0, rel=1e-6)


class TestComputeBeamInternalForces:
    """Tests for compute_beam_internal_forces."""
    def test_returns_five_arrays_of_same_length(self) -> None:
        """Beam internal forces returns five arrays of same length."""
        model, dof_map, result = solve_cantilever_beam()
        x, V, M, v, theta = compute_beam_internal_forces(
            1, model, dof_map, result.displacements, n_stations=20
        )
        assert x.shape == V.shape == M.shape == v.shape == theta.shape == (20,)

    def test_x_stations_span_element(self) -> None:
        """X stations span the entire element."""
        model, dof_map, result = solve_cantilever_beam(L=2.0)
        x, V, M, v, theta = compute_beam_internal_forces(
            1, model, dof_map, result.displacements, n_stations=50
        )
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(2.0)

    def test_cantilever_shear_is_constant(self) -> None:
        """Cantilever shear force is constant."""
        model, dof_map, result = solve_cantilever_beam(L=1.0, EI=1.0, P=-1.0)
        x, V, M, v, theta = compute_beam_internal_forces(
            1, model, dof_map, result.displacements, n_stations=50
        )
        assert np.std(V) == pytest.approx(0.0, abs=1e-8)

    def test_cantilever_moment_at_fixed_end(self) -> None:
        """Cantilever moment at fixed end matches formula."""
        model, dof_map, result = solve_cantilever_beam(L=1.0, EI=1.0, P=-1.0)
        x, V, M, v, theta = compute_beam_internal_forces(
            1, model, dof_map, result.displacements, n_stations=100
        )
        assert abs(M[0]) == pytest.approx(1.0, rel=0.01)

    def test_cantilever_moment_at_free_end_is_zero(self) -> None:
        """Cantilever moment at free end is zero."""
        model, dof_map, result = solve_cantilever_beam(L=1.0, EI=1.0, P=-1.0)
        x, V, M, v, theta = compute_beam_internal_forces(
            1, model, dof_map, result.displacements, n_stations=100
        )
        assert M[-1] == pytest.approx(0.0, abs=1e-8)

    def test_cantilever_displacement_at_fixed_end_is_zero(self) -> None:
        """Cantilever transverse displacement at fixed end is zero."""
        model, dof_map, result = solve_cantilever_beam(L=1.0, EI=1.0, P=-1.0)
        x, V, M, v, theta = compute_beam_internal_forces(
            1, model, dof_map, result.displacements, n_stations=100
        )
        assert v[0] == pytest.approx(0.0, abs=1e-10)

    def test_cantilever_tip_displacement_matches_formula(self) -> None:
        """Cantilever tip displacement v(L) = P*L^3 / (3*EI)."""
        # With EI=1, L=1, P=-1: v_tip = -1/(3*1) = -1/3
        model, dof_map, result = solve_cantilever_beam(L=1.0, EI=1.0, P=-1.0)
        x, V, M, v, theta = compute_beam_internal_forces(
            1, model, dof_map, result.displacements, n_stations=100
        )
        expected = -1.0 / 3.0
        assert v[-1] == pytest.approx(expected, rel=1e-6)

    def test_cantilever_rotation_at_fixed_end_is_zero(self) -> None:
        """Cantilever rotation at fixed end is zero."""
        model, dof_map, result = solve_cantilever_beam(L=1.0, EI=1.0, P=-1.0)
        x, V, M, v, theta = compute_beam_internal_forces(
            1, model, dof_map, result.displacements, n_stations=100
        )
        assert theta[0] == pytest.approx(0.0, abs=1e-10)

    def test_x_stations_are_global_coordinates(self) -> None:
        """X stations use global coordinates."""
        mat = MaterialProperties(E=1.0, A=1.0, I=1.0)
        n1, n2 = Node(1, 2.0), Node(2, 3.0)
        elem = Element(id=1, node_i=n1, node_j=n2, element_type=ElementType.BEAM, material=mat)
        bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_ALL)
        load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_Y, magnitude=-1.0)
        model = FEAModel(mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
                         boundary_conditions=(bc,), nodal_loads=(load,),
                         distributed_loads=(), label="offset_beam")
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        x, V, M, v, theta = compute_beam_internal_forces(
            1, model, dof_map, result.displacements, n_stations=10
        )
        assert x[0] == pytest.approx(2.0)
        assert x[-1] == pytest.approx(3.0)


class TestPostprocessAllElements:
    """Tests for postprocess_all_elements."""
    def test_returns_list_of_element_results(self) -> None:
        """postprocess_all_elements returns list of ElementResults."""
        from fea_solver.models import ElementResult
        model, dof_map, result = solve_cantilever_beam()
        element_results = postprocess_all_elements(model, result)
        assert len(element_results) == 1
        assert isinstance(element_results[0], ElementResult)

    def test_result_element_id_matches(self) -> None:
        """Element result ID matches element ID."""
        model, dof_map, result = solve_cantilever_beam()
        element_results = postprocess_all_elements(model, result)
        assert element_results[0].element_id == 1

    def test_bar_axial_force_in_result(self) -> None:
        """Bar axial force is present in result."""
        model, dof_map, result = solve_bar(L=1.0, EA=1.0, P=5.0)
        element_results = postprocess_all_elements(model, result)
        assert element_results[0].axial_force == pytest.approx(5.0)

    def test_element_result_has_displacement_fields(self) -> None:
        """ElementResult includes transverse_displacements, axial_displacements, rotations."""
        model, _, result = solve_cantilever_beam()
        ers = postprocess_all_elements(model, result, n_stations=50)
        er = ers[0]
        assert er.transverse_displacements.shape == (50,)
        assert er.axial_displacements.shape == (50,)
        assert er.rotations.shape == (50,)

    def test_beam_axial_displacements_are_zero(self) -> None:
        """Pure BEAM element has zero axial displacements."""
        model, _, result = solve_cantilever_beam()
        ers = postprocess_all_elements(model, result, n_stations=50)
        np.testing.assert_array_equal(ers[0].axial_displacements, np.zeros(50))

    def test_bar_transverse_displacements_are_zero(self) -> None:
        """BAR element has zero transverse displacements and rotations."""
        model, _, result = solve_bar()
        ers = postprocess_all_elements(model, result, n_stations=50)
        np.testing.assert_array_equal(ers[0].transverse_displacements, np.zeros(50))
        np.testing.assert_array_equal(ers[0].rotations, np.zeros(50))

    def test_bar_axial_displacements_match_linear_interpolation(self) -> None:
        """BAR axial displacement at tip matches PL/EA."""
        # P=1, EA=1, L=1 -> u_tip = 1.0
        model, _, result = solve_bar(L=1.0, EA=1.0, P=1.0)
        ers = postprocess_all_elements(model, result, n_stations=50)
        assert ers[0].axial_displacements[-1] == pytest.approx(1.0, rel=1e-6)
