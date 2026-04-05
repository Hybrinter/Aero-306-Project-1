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
    """2x2 local stiffness matrix for a bar (axial) element.

    k = (EA/L) * [[1, -1], [-1, 1]]
    DOF ordering: [u_i, u_j]
    """
    EA_over_L = element.material.E * element.material.A / element.length
    k = EA_over_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return k


def beam_stiffness_matrix(element: Element) -> NDArray[np.float64]:
    """4x4 local stiffness matrix for an Euler-Bernoulli beam element.

    k = (EI/L^3) * [[12,  6L, -12,  6L],
                     [6L, 4L^2, -6L, 2L^2],
                     [-12, -6L, 12, -6L],
                     [6L, 2L^2, -6L, 4L^2]]
    DOF ordering: [v_i, theta_i, v_j, theta_j]
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
    """6x6 local stiffness matrix for a frame (bar+beam) element.

    Block-diagonal: axial block at [u_i, u_j], bending block at [v_i, theta_i, v_j, theta_j].
    DOF ordering: [u_i, v_i, theta_i, u_j, v_j, theta_j]
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
    """Dispatch to the appropriate stiffness matrix function."""
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
    """4x1 consistent nodal force vector for a transversely loaded beam element.

    For uniform load w (constant w_i == w_j):
      f = [wL/2, wL^2/12, wL/2, -wL^2/12]

    For linearly varying load (w_i at node i, w_j at node j):
      Uses exact integration of Hermite shape functions.
    DOF ordering: [f_vi, m_i, f_vj, m_j]
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
    """Dispatch to the appropriate consistent load vector function."""
    if element.element_type in (ElementType.BEAM, ElementType.FRAME):
        return beam_consistent_load_vector(element, load)
    else:
        raise NotImplementedError(
            f"Distributed loads not implemented for element type: {element.element_type}"
        )
