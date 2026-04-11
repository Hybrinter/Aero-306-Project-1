"""Element stiffness matrices and consistent nodal load vectors.

Implements:
  - BAR element: 2 DOFs [u_i, u_j], axial stiffness EA/L
  - BEAM element: 4 DOFs [v_i, theta_i, v_j, theta_j], Euler-Bernoulli bending EI
  - FRAME element: 6 DOFs [u_i, v_i, theta_i, u_j, v_j, theta_j], combined axial+bending
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fea_solver.models import DistributedLoad, Element, ElementType, LoadType


def bar_stiffness_matrix(element: Element) -> NDArray[np.float64]:
    """Assemble the 2x2 local stiffness matrix for a bar (axial) element.

    Args:
        element (Element): Bar element with nodes and element_type == ElementType.BAR.

    Returns:
        NDArray[np.float64]: 2x2 symmetric stiffness matrix k = (EA/L) * [[1,-1],[-1,1]].

    Notes:
        DOF ordering: [u_i, u_j]. Units: N/m. Computes k = EA/L based on Young's
        modulus E, cross-sectional area A, and element length L.
    """
    EA_over_L = element.material.E * element.material.A / element.length
    k = EA_over_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return k


def beam_stiffness_matrix(element: Element) -> NDArray[np.float64]:
    """Assemble the 4x4 local stiffness matrix for an Euler-Bernoulli beam element.

    Args:
        element (Element): Beam element with nodes and element_type == ElementType.BEAM.

    Returns:
        NDArray[np.float64]: 4x4 symmetric stiffness matrix based on bending rigidity EI.

    Notes:
        DOF ordering: [v_i, theta_i, v_j, theta_j]. Units: N/m and N*m/rad.
        Computes k = (EI/L^3) * [12, 6L, -12, 6L; ...]. Uses Young's modulus E,
        second moment of area I, and element length L.
    """
    E = element.material.E
    I = element.material.I
    L = element.length
    EI = E * I
    L2 = L * L
    L3 = L2 * L
    c = EI / L3
    k = c * np.array([
        [ 12.0,  6.0*L,  -12.0,  6.0*L],
        [  6.0*L,  4.0*L2,  -6.0*L,  2.0*L2],
        [-12.0, -6.0*L,   12.0, -6.0*L],
        [  6.0*L,  2.0*L2,  -6.0*L,  4.0*L2],
    ])
    return k


def frame_stiffness_matrix(element: Element) -> NDArray[np.float64]:
    """Assemble the 6x6 local stiffness matrix for a frame (bar+beam) element.

    Args:
        element (Element): Frame element with nodes and element_type == ElementType.FRAME.

    Returns:
        NDArray[np.float64]: 6x6 symmetric stiffness matrix combining axial and bending stiffness.

    Notes:
        DOF ordering: [u_i, v_i, theta_i, u_j, v_j, theta_j]. Block-diagonal structure:
        axial stiffness in positions [0,3], bending stiffness in positions [1,2,4,5].
        Units: N/m and N*m/rad. Combines EA/L and EI/L^3 effects.
    """
    k = np.zeros((6, 6))
    # Axial block at indices [0, 3]
    bar_elem = Element(
        id=element.id, node_i=element.node_i, node_j=element.node_j,
        element_type=ElementType.BAR, material=element.material
    )
    k_bar = bar_stiffness_matrix(bar_elem)
    axial_idx = [0, 3]
    for i, gi in enumerate(axial_idx):
        for j, gj in enumerate(axial_idx):
            k[gi, gj] += k_bar[i, j]

    # Bending block at indices [1, 2, 4, 5]
    beam_elem = Element(
        id=element.id, node_i=element.node_i, node_j=element.node_j,
        element_type=ElementType.BEAM, material=element.material
    )
    k_beam = beam_stiffness_matrix(beam_elem)
    bending_idx = [1, 2, 4, 5]
    for i, gi in enumerate(bending_idx):
        for j, gj in enumerate(bending_idx):
            k[gi, gj] += k_beam[i, j]

    return k


def element_stiffness_matrix(element: Element) -> NDArray[np.float64]:
    """Dispatch to the appropriate stiffness matrix function based on element type.

    Args:
        element (Element): Element with element_type set to BAR, BEAM, or FRAME.

    Returns:
        NDArray[np.float64]: Local stiffness matrix k of size (n_dofs x n_dofs)
            where n_dofs is 2 for BAR, 4 for BEAM, 6 for FRAME.

    Notes:
        Acts as a router that selects bar_stiffness_matrix, beam_stiffness_matrix,
        or frame_stiffness_matrix based on element.element_type. Raises ValueError
        if element_type is unrecognized.
    """
    if element.element_type == ElementType.BAR:
        return bar_stiffness_matrix(element)
    elif element.element_type == ElementType.BEAM:
        return beam_stiffness_matrix(element)
    elif element.element_type == ElementType.FRAME:
        return frame_stiffness_matrix(element)
    else:
        raise ValueError(f"Unknown element type: {element.element_type}")


def beam_consistent_load_vector(
    element: Element, load: DistributedLoad
) -> NDArray[np.float64]:
    """Assemble the 4x1 consistent nodal load vector for a transversely loaded beam element.

    Args:
        element (Element): Beam element with element_type == ElementType.BEAM.
        load (DistributedLoad): Distributed load with load_type DISTRIBUTED_Y or DISTRIBUTED_LINEAR.

    Returns:
        NDArray[np.float64]: 4x1 load vector with nodal forces and moments.

    Notes:
        DOF ordering: [f_vi, m_i, f_vj, m_j]. For uniform loads (DISTRIBUTED_Y):
        f = [wL/2, wL^2/12, wL/2, -wL^2/12]. For linearly varying loads (DISTRIBUTED_LINEAR):
        uses exact integration with Hermite shape functions H1..H4. Units: N and N*m.
    """
    L = element.length
    w_i = load.w_i
    w_j = load.w_j

    if load.load_type == LoadType.DISTRIBUTED_Y:
        # Uniform: w_i == w_j (or treat average)
        w = (w_i + w_j) / 2.0
        f = np.array([
            w * L / 2.0,
            w * L**2 / 12.0,
            w * L / 2.0,
            -w * L**2 / 12.0,
        ])
    elif load.load_type == LoadType.DISTRIBUTED_LINEAR:
        # Exact integration for linearly varying load
        # w(x) = w_i*(1 - x/L) + w_j*(x/L)
        # Integrate with Hermite shape functions H1..H4
        f = np.array([
            (7.0*w_i + 3.0*w_j) * L / 20.0,
            (3.0*w_i + 2.0*w_j) * L**2 / 60.0,
            (3.0*w_i + 7.0*w_j) * L / 20.0,
            -(2.0*w_i + 3.0*w_j) * L**2 / 60.0,
        ])
    else:
        raise ValueError(f"Unsupported load type for beam: {load.load_type}")

    return f


def element_load_vector(
    element: Element, load: DistributedLoad
) -> NDArray[np.float64]:
    """Dispatch to the appropriate consistent load vector function based on element type.

    Args:
        element (Element): Element with element_type set to BEAM or FRAME.
        load (DistributedLoad): Distributed load applied to the element.

    Returns:
        NDArray[np.float64]: Consistent nodal load vector of size (n_dofs,)
            where n_dofs is 4 for BEAM, 6 for FRAME.

    Notes:
        Acts as a router that calls beam_consistent_load_vector for BEAM and FRAME
        elements. Raises NotImplementedError for BAR elements (no distributed loads).
    """
    if element.element_type in (ElementType.BEAM, ElementType.FRAME):
        return beam_consistent_load_vector(element, load)
    else:
        raise NotImplementedError(
            f"Distributed loads not implemented for element type: {element.element_type}"
        )
