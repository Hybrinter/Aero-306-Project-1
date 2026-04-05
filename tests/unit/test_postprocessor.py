"""Tests for element post-processing (internal forces) — TDD first."""
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


def solve_cantilever_beam(L: float = 1.0, EI: float = 1.0, P: float = -1.0):
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


def solve_bar(L: float = 1.0, EA: float = 1.0, P: float = 1.0):
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
    def test_bar_unit_force(self):
        # E=1, A=1, L=1, P=1 -> u_tip = 1 -> N = EA/L*(u_j - u_i) = 1
        model, dof_map, result = solve_bar(L=1.0, EA=1.0, P=1.0)
        N = compute_element_axial_force(1, model, dof_map, result.displacements)
        assert N == pytest.approx(1.0)

    def test_bar_scaled_force(self):
        # E=200e9, A=0.01, L=1, P=10000 -> u_tip = P/(EA) = 10000/(2e9) = 5e-6
        # N = EA/L*(u_j - u_i) = 2e9 * 5e-6 = 10000
        model, dof_map, result = solve_bar(L=1.0, EA=200.0e9*0.01, P=10000.0)
        N = compute_element_axial_force(1, model, dof_map, result.displacements)
        assert N == pytest.approx(10000.0, rel=1e-6)


class TestComputeBeamInternalForces:
    def test_returns_three_arrays_of_same_length(self):
        model, dof_map, result = solve_cantilever_beam()
        x, V, M = compute_beam_internal_forces(1, model, dof_map, result.displacements, n_stations=20)
        assert x.shape == V.shape == M.shape == (20,)

    def test_x_stations_span_element(self):
        model, dof_map, result = solve_cantilever_beam(L=2.0)
        x, V, M = compute_beam_internal_forces(1, model, dof_map, result.displacements, n_stations=50)
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(2.0)

    def test_cantilever_shear_is_constant(self):
        # Cantilever with tip load P=-1: V = P = -1 everywhere (from equilibrium)
        # In our sign convention for beam FEA: shear should be constant magnitude
        model, dof_map, result = solve_cantilever_beam(L=1.0, EI=1.0, P=-1.0)
        x, V, M = compute_beam_internal_forces(1, model, dof_map, result.displacements, n_stations=50)
        # Shear should be approximately constant (within numerical precision)
        assert np.std(V) == pytest.approx(0.0, abs=1e-8)

    def test_cantilever_moment_at_fixed_end(self):
        # Cantilever L=1, P=-1 at tip: M(x=0) = P*L = -1*1 = -1 (hogging = negative moment)
        # M(x) = P*(L-x) in beam convention
        model, dof_map, result = solve_cantilever_beam(L=1.0, EI=1.0, P=-1.0)
        x, V, M = compute_beam_internal_forces(1, model, dof_map, result.displacements, n_stations=100)
        # At x=0 (fixed end), M should equal P*L = -1
        assert abs(M[0]) == pytest.approx(1.0, rel=0.01)

    def test_cantilever_moment_at_free_end_is_zero(self):
        # M(x=L) = 0 for cantilever with tip load (no applied moment)
        model, dof_map, result = solve_cantilever_beam(L=1.0, EI=1.0, P=-1.0)
        x, V, M = compute_beam_internal_forces(1, model, dof_map, result.displacements, n_stations=100)
        assert M[-1] == pytest.approx(0.0, abs=1e-8)

    def test_x_stations_are_global_coordinates(self):
        # For element from x=2.0 to x=3.0, x_stations should start at 2.0
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
        x, V, M = compute_beam_internal_forces(1, model, dof_map, result.displacements, n_stations=10)
        assert x[0] == pytest.approx(2.0)
        assert x[-1] == pytest.approx(3.0)


class TestPostprocessAllElements:
    def test_returns_list_of_element_results(self):
        from fea_solver.models import ElementResult
        model, dof_map, result = solve_cantilever_beam()
        element_results = postprocess_all_elements(model, result)
        assert len(element_results) == 1
        assert isinstance(element_results[0], ElementResult)

    def test_result_element_id_matches(self):
        model, dof_map, result = solve_cantilever_beam()
        element_results = postprocess_all_elements(model, result)
        assert element_results[0].element_id == 1

    def test_bar_axial_force_in_result(self):
        model, dof_map, result = solve_bar(L=1.0, EA=1.0, P=5.0)
        element_results = postprocess_all_elements(model, result)
        assert element_results[0].axial_force == pytest.approx(5.0)
