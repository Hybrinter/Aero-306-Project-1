"""Tests for displacement solver and reaction computation — TDD first."""
from __future__ import annotations
import numpy as np
import pytest
from fea_solver.models import (
    BoundaryCondition, BoundaryConditionType, DOFMap, DOFType,
    Element, ElementType, FEAModel, MaterialProperties, Mesh, Node, NodalLoad, LoadType
)
from fea_solver.solver import solve_displacements, compute_reactions, run_solve_pipeline


class TestSolveDisplacements:
    def test_single_dof_bar(self):
        # K_ff = [[1.0]], F_f = [1.0] -> u_f = [1.0]
        K_ff = np.array([[1.0]])
        F_f = np.array([1.0])
        u = solve_displacements(K_ff, F_f, free_dofs=[1], constrained_dofs=[0], n_total_dofs=2)
        assert u.shape == (2,)
        assert u[0] == pytest.approx(0.0)   # constrained DOF
        assert u[1] == pytest.approx(1.0)   # free DOF

    def test_two_dof_bar_analytical(self):
        # 3-node bar, E=A=L=1, P=1 at tip
        # After fixing DOF 0: K_ff = [[2,-1],[-1,1]], F_f = [0,1]
        # Solution: u=[1, 2] (for free DOFs 1 and 2)
        K_ff = np.array([[2.0, -1.0], [-1.0, 1.0]])
        F_f = np.array([0.0, 1.0])
        u = solve_displacements(K_ff, F_f, free_dofs=[1, 2], constrained_dofs=[0], n_total_dofs=3)
        assert u.shape == (3,)
        assert u[0] == pytest.approx(0.0)
        assert u[1] == pytest.approx(1.0)
        assert u[2] == pytest.approx(2.0)

    def test_singular_matrix_raises(self):
        K_ff = np.zeros((2, 2))
        F_f = np.array([1.0, 1.0])
        with pytest.raises(np.linalg.LinAlgError):
            solve_displacements(K_ff, F_f, free_dofs=[0, 1], constrained_dofs=[], n_total_dofs=2)

    def test_constrained_dofs_are_zero_in_result(self):
        K_ff = np.array([[1.0]])
        F_f = np.array([5.0])
        u = solve_displacements(K_ff, F_f, free_dofs=[1], constrained_dofs=[0], n_total_dofs=2)
        assert u[0] == pytest.approx(0.0)


class TestComputeReactions:
    def test_single_bar_reaction_equals_applied_load(self):
        # 2-DOF bar, u=[0, 1], K = [[1,-1],[-1,1]], F=[0,1]
        # Reaction at DOF 0: K[0,:] @ u - F[0] = (1*0 + (-1)*1) - 0 = -1
        K = np.array([[1.0, -1.0], [-1.0, 1.0]])
        u = np.array([0.0, 1.0])
        F = np.array([0.0, 1.0])
        R = compute_reactions(K, u, F, constrained_dofs=[0])
        assert R.shape == (1,)
        assert R[0] == pytest.approx(-1.0)

    def test_equilibrium_reactions_sum_to_zero(self):
        # For a balanced system, sum of all external forces + reactions = 0
        # Simply supported beam: two reactions + applied load = 0
        # Here we just check |sum(R) + sum(F_applied)| is small
        K = np.array([[1.0, -1.0, 0.0],
                      [-1.0, 2.0, -1.0],
                      [0.0, -1.0, 1.0]])
        u = np.array([0.0, 1.0, 2.0])
        F = np.array([0.0, 0.0, 1.0])
        R = compute_reactions(K, u, F, constrained_dofs=[0])
        # Equilibrium: sum(reactions) + sum(applied forces at free dofs) = 0
        assert (R[0] + F[2]) == pytest.approx(0.0, abs=1e-10)


class TestRunSolvePipeline:
    def test_cantilever_bar_end_displacement(self):
        # Single element bar: E=1, A=1, L=1, P=1 -> u_tip = 1
        mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
        n1, n2 = Node(1, 0.0), Node(2, 1.0)
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.BAR, material=mat)
        bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_U)
        load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_X, magnitude=1.0)
        model = FEAModel(mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
                         boundary_conditions=(bc,), nodal_loads=(load,),
                         distributed_loads=(), label="cantilever_bar")

        from fea_solver.assembler import build_dof_map, assemble_global_stiffness, assemble_global_force_vector
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)

        u_tip = result.displacements[dof_map.index(2, DOFType.U)]
        assert u_tip == pytest.approx(1.0)

    def test_returns_solution_result_type(self):
        mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
        n1, n2 = Node(1, 0.0), Node(2, 1.0)
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.BAR, material=mat)
        bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_U)
        load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_X, magnitude=1.0)
        model = FEAModel(mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
                         boundary_conditions=(bc,), nodal_loads=(load,),
                         distributed_loads=(), label="test")
        from fea_solver.assembler import build_dof_map, assemble_global_stiffness, assemble_global_force_vector
        from fea_solver.models import SolutionResult
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        assert isinstance(result, SolutionResult)
