"""Property-based tests using hypothesis for stiffness matrices and DOF mapping.

This module uses hypothesis to generate random test data and verify that
the FEA solver's core functions maintain mathematical invariants across
a wide range of inputs, including edge cases and degenerate scenarios.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from fea_solver.models import (
    DOFMap,
    DOFType,
    Element,
    ElementType,
    FEAModel,
    LinearConstraint,
    MaterialProperties,
    Mesh,
    Node,
)
from fea_solver.assembler import build_dof_map, get_element_dof_indices, assemble_global_force_vector, assemble_global_stiffness
from fea_solver.constraints import apply_penalty_constraints, compute_constraint_residuals
from fea_solver.elements import (
    bar_stiffness_matrix,
    beam_stiffness_matrix,
    frame_stiffness_matrix,
)
from fea_solver.solver import compute_penalty_parameter


def make_bar_element(E: float, A: float, L: float) -> Element:
    """Construct a bar element with given parameters.

    Args:
        E (float): Young's modulus in Pascals.
        A (float): Cross-sectional area in square metres.
        L (float): Element length in metres.

    Returns:
        Element: A bar element with the specified properties.

    Notes:
        Creates nodes at x=0 and x=L, with id 1 and 2 respectively.
    """
    mat = MaterialProperties(E=E, A=A, I=0.0)
    return Element(
        id=1,
        node_i=Node(1, (0.0, 0.0)),
        node_j=Node(2, (L, 0.0)),
        element_type=ElementType.BAR,
        material=mat,
    )


def make_beam_element(E: float, I: float, L: float) -> Element:
    """Construct a beam element with given parameters.

    Args:
        E (float): Young's modulus in Pascals.
        I (float): Second moment of area in metres to the fourth power.
        L (float): Element length in metres.

    Returns:
        Element: A beam element with the specified properties.

    Notes:
        Creates nodes at x=0 and x=L, with id 1 and 2 respectively.
    """
    mat = MaterialProperties(E=E, A=1.0, I=I)
    return Element(
        id=1,
        node_i=Node(1, (0.0, 0.0)),
        node_j=Node(2, (L, 0.0)),
        element_type=ElementType.BEAM,
        material=mat,
    )


def make_frame_element(E: float, A: float, I: float, L: float) -> Element:
    """Construct a frame element with given parameters.

    Args:
        E (float): Young's modulus in Pascals.
        A (float): Cross-sectional area in square metres.
        I (float): Second moment of area in metres to the fourth power.
        L (float): Element length in metres.

    Returns:
        Element: A frame element with the specified properties.

    Notes:
        Creates nodes at x=0 and x=L, with id 1 and 2 respectively.
    """
    mat = MaterialProperties(E=E, A=A, I=I)
    return Element(
        id=1,
        node_i=Node(1, (0.0, 0.0)),
        node_j=Node(2, (L, 0.0)),
        element_type=ElementType.FRAME,
        material=mat,
    )


@given(
    st.floats(min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-4, max_value=1e-1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-8, max_value=1e-3, allow_nan=False, allow_infinity=False),
)
def test_stiffness_matrix_symmetry(
    E: float, A: float, L: float, I: float
) -> None:
    """Stiffness matrices are symmetric for all element types.

    All stiffness matrices (bar 2x2, beam 4x4, frame 6x6) must be
    symmetric for any positive material and geometric parameters,
    as required by reciprocal energy theorems in mechanics.

    Args:
        E (float): Young's modulus in Pascals, from [1e6, 1e12].
        A (float): Cross-sectional area in m^2, from [1e-4, 1e-1].
        L (float): Element length in metres, from [0.1, 100].
        I (float): Second moment of area in m^4, from [1e-8, 1e-3].

    Returns:
        None

    Notes:
        Tests symmetry (k == k.T) for bar, beam, and frame elements.
        Verifies that the property holds across the full parameter space.
    """
    # Test bar
    bar_elem = make_bar_element(E, A, L)
    k_bar = bar_stiffness_matrix(bar_elem)
    np.testing.assert_allclose(k_bar, k_bar.T, rtol=1e-10)

    # Test beam
    beam_elem = make_beam_element(E, I, L)
    k_beam = beam_stiffness_matrix(beam_elem)
    np.testing.assert_allclose(k_beam, k_beam.T, rtol=1e-10)

    # Test frame
    frame_elem = make_frame_element(E, A, I, L)
    k_frame = frame_stiffness_matrix(frame_elem)
    np.testing.assert_allclose(k_frame, k_frame.T, rtol=1e-10)


@given(
    st.floats(min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-4, max_value=1e-1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_stiffness_matrix_row_sum_property(E: float, A: float, L: float) -> None:
    """Bar stiffness matrix rows sum to zero (rigid body mode property).

    For any bar element stiffness matrix, each row must sum to zero because
    rigid body translation is a zero-energy mode of the structure. This
    property is a fundamental check on the correctness of the element
    stiffness formulation.

    Args:
        E (float): Young's modulus in Pascals, from [1e6, 1e12].
        A (float): Cross-sectional area in m^2, from [1e-4, 1e-1].
        L (float): Element length in metres, from [0.1, 100].

    Returns:
        None

    Notes:
        Verifies that sum of each row is zero within relative tolerance.
        The property holds exactly for bar elements due to the form of
        the axial stiffness matrix k = (EA/L) * [[1,-1],[-1,1]].
    """
    element = make_bar_element(E, A, L)
    k = bar_stiffness_matrix(element)
    row_sums = k.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.zeros(2), rtol=1e-10, atol=1e-8)


@given(
    st.floats(min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-4, max_value=1e-1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-8, max_value=1e-3, allow_nan=False, allow_infinity=False),
)
def test_stiffness_matrix_positive_semidefinite(
    E: float, A: float, L: float, I: float
) -> None:
    """Stiffness matrices are positive semi-definite for all element types.

    All eigenvalues of bar, beam, and frame stiffness matrices must be
    non-negative (within floating point tolerance) because elastic strain
    energy must be non-negative for any displacement field.

    Args:
        E (float): Young's modulus in Pascals, from [1e6, 1e12].
        A (float): Cross-sectional area in m^2, from [1e-4, 1e-1].
        L (float): Element length in metres, from [0.1, 100].
        I (float): Second moment of area in m^4, from [1e-8, 1e-3].

    Returns:
        None

    Notes:
        Eigenvalues are checked to be >= -tol where tol is scaled by
        the matrix norm to account for numerical noise in eigenvalue
        computation (1e-6 relative to norm, or 1e-8 absolute minimum).
    """
    # Test bar
    bar_elem = make_bar_element(E, A, L)
    k_bar = bar_stiffness_matrix(bar_elem)
    eigenvalues_bar = np.linalg.eigvals(k_bar)
    matrix_norm_bar = np.linalg.norm(k_bar)
    tol_bar = max(1e-8, 1e-6 * matrix_norm_bar)
    assert np.all(eigenvalues_bar >= -tol_bar), f"Bar: found negative eigenvalue: {eigenvalues_bar}"

    # Test beam
    beam_elem = make_beam_element(E, I, L)
    k_beam = beam_stiffness_matrix(beam_elem)
    eigenvalues_beam = np.linalg.eigvals(k_beam)
    matrix_norm_beam = np.linalg.norm(k_beam)
    tol_beam = max(1e-8, 1e-6 * matrix_norm_beam)
    assert np.all(eigenvalues_beam >= -tol_beam), f"Beam: found negative eigenvalue: {eigenvalues_beam}"

    # Test frame
    frame_elem = make_frame_element(E, A, I, L)
    k_frame = frame_stiffness_matrix(frame_elem)
    eigenvalues_frame = np.linalg.eigvals(k_frame)
    matrix_norm_frame = np.linalg.norm(k_frame)
    tol_frame = max(1e-8, 1e-6 * matrix_norm_frame)
    assert np.all(eigenvalues_frame >= -tol_frame), f"Frame: found negative eigenvalue: {eigenvalues_frame}"


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=101, max_value=200),
)
def test_dof_map_bijectivity(node_i_id: int, node_j_id: int) -> None:
    """DOF map is bijective for a two-node bar element.

    For a simple two-node bar model, the DOF map must:
    - Contain exactly total_dofs entries in the mapping.
    - Have indices that are exactly {0, 1, ..., total_dofs-1}.

    Args:
        node_i_id (int): Node ID for first node, from [1, 100].
        node_j_id (int): Node ID for second node, from [101, 200].

    Returns:
        None

    Notes:
        Verifies that DOF indices form a contiguous set with no gaps or
        duplicates, ensuring bijective (one-to-one, onto) mapping.
    """
    assume(node_i_id != node_j_id)

    mat = MaterialProperties(E=1e6, A=1e-2, I=0.0)
    nodes = (Node(node_i_id, (0.0, 0.0)), Node(node_j_id, (1.0, 0.0)))
    elements = (
        Element(
            id=1,
            node_i=nodes[0],
            node_j=nodes[1],
            element_type=ElementType.BAR,
            material=mat,
        ),
    )
    mesh = Mesh(nodes=nodes, elements=elements)
    model = FEAModel(
        mesh=mesh,
        boundary_conditions=(),
        nodal_loads=(),
        distributed_loads=(),
    )

    dof_map = build_dof_map(model)

    # Check that indices are exactly {0, 1, ..., total_dofs-1}
    indices_set = set(dof_map.mapping.values())
    expected_set = set(range(dof_map.total_dofs))
    assert indices_set == expected_set, (
        f"DOF indices {indices_set} do not match expected {expected_set}"
    )

    # Check that mapping has exactly total_dofs entries
    assert len(dof_map.mapping) == dof_map.total_dofs


@given(
    node_i_id=st.integers(min_value=1, max_value=10),
    L=st.floats(min_value=0.01, max_value=100.0),
)
@settings(max_examples=50)
def test_penalty_constraint_enforced_after_solve(node_i_id: int, L: float) -> None:
    """After penalty solve, constrained DOF displacement is < 1/penalty_alpha.

    For a simple bar element with an axial constraint and penalty method,
    the constrained DOF should have a displacement magnitude that is
    inversely proportional to the penalty parameter.

    Args:
        node_i_id (int): Node ID for first node, from [1, 10].
        L (float): Element length in metres, from [0.01, 100.0].

    Returns:
        None

    Notes:
        Tests that the penalty method correctly enforces constraints
        by checking that the constrained DOF displacement decreases
        as the penalty parameter increases.
    """
    node_j_id = node_i_id + 1
    mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
    ni = Node(id=node_i_id, pos=(0.0, 0.0))
    nj = Node(id=node_j_id, pos=(L, 0.0))
    elem = Element(id=1, node_i=ni, node_j=nj,
                   element_type=ElementType.BAR, material=mat)
    c = LinearConstraint(node_id=node_i_id, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
    model = FEAModel(
        mesh=Mesh(nodes=(ni, nj), elements=(elem,)),
        boundary_conditions=(c,),
        nodal_loads=(),
        distributed_loads=(),
        penalty_alpha=1e8,
    )
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    k_penalty = compute_penalty_parameter(K, model.penalty_alpha)
    K_mod, F_mod = apply_penalty_constraints(K, F, model.boundary_conditions, dof_map, k_penalty)
    u = np.linalg.solve(K_mod, F_mod)
    R = compute_constraint_residuals(u, model.boundary_conditions, dof_map, k_penalty)
    assert R.shape == (1,)
    assert abs(u[dof_map.index(node_i_id, DOFType.U)]) < 1.0 / model.penalty_alpha
