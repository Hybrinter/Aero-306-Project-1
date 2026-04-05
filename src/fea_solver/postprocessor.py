"""Element-level post-processing: internal forces via Hermite shape functions.

For beam elements, displacement field v(x) is reconstructed using Hermite cubic
shape functions, which exactly represent the FE solution within each element.
Shear and moment are computed by differentiating:
  M(x) = EI * d²v/dx²
  V(x) = -EI * d³v/dx³  (minus sign: V = dM/dx for positive V convention)

For bar elements, axial force N = EA/L * (u_j - u_i).
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fea_solver.models import (
    DOFMap,
    DOFType,
    Element,
    ElementResult,
    ElementType,
    FEAModel,
    SolutionResult,
)

logger = logging.getLogger(__name__)


def compute_element_axial_force(
    element_id: int,
    model: FEAModel,
    dof_map: DOFMap,
    u: NDArray[np.float64],
) -> float:
    """Compute axial force for a BAR element.

    N = EA/L * (u_j - u_i)

    Returns:
        Axial force [N], positive in tension.
    """
    element = next(e for e in model.mesh.elements if e.id == element_id)
    u_i = u[dof_map.index(element.node_i.id, DOFType.U)]
    u_j = u[dof_map.index(element.node_j.id, DOFType.U)]
    N = element.material.E * element.material.A / element.length * (u_j - u_i)
    logger.debug("Element %d: axial force N = %.4e N", element_id, N)
    return float(N)


def compute_beam_internal_forces(
    element_id: int,
    model: FEAModel,
    dof_map: DOFMap,
    u: NDArray[np.float64],
    n_stations: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute shear force V(x) and bending moment M(x) along a BEAM element.

    Uses Hermite cubic shape functions:
      H1(ξ) = 1 - 3ξ² + 2ξ³
      H2(ξ) = L(ξ - 2ξ² + ξ³)
      H3(ξ) = 3ξ² - 2ξ³
      H4(ξ) = L(-ξ² + ξ³)
    where ξ = (x - x_i)/L

    v(ξ) = H1*v_i + H2*θ_i + H3*v_j + H4*θ_j
    M(x) = EI * d²v/dx² = (EI/L²) * d²v/dξ²
    V(x) = dM/dx = (EI/L³) * d³v/dξ³

    Args:
        element_id: Element to evaluate.
        model: FEA model.
        dof_map: DOF index mapping.
        u: Full displacement vector.
        n_stations: Number of evaluation points along the element.

    Returns:
        (x_stations, shear_forces, bending_moments) — all shape (n_stations,),
        x_stations are global coordinates.
    """
    element = next(e for e in model.mesh.elements if e.id == element_id)
    EI = element.material.E * element.material.I
    L = element.length
    x_i = element.node_i.x

    # Extract nodal DOFs [v_i, theta_i, v_j, theta_j]
    v_i = u[dof_map.index(element.node_i.id, DOFType.V)]
    t_i = u[dof_map.index(element.node_i.id, DOFType.THETA)]
    v_j = u[dof_map.index(element.node_j.id, DOFType.V)]
    t_j = u[dof_map.index(element.node_j.id, DOFType.THETA)]

    # Parametric coordinate ξ ∈ [0, 1]
    xi = np.linspace(0.0, 1.0, n_stations)
    x_stations = x_i + xi * L

    # Second derivative of Hermite shape functions w.r.t. ξ: d²H/dξ²
    # H1''(ξ) = -6 + 12ξ
    # H2''(ξ) = L(-4 + 6ξ)
    # H3''(ξ) = 6 - 12ξ
    # H4''(ξ) = L(-2 + 6ξ)
    d2H1 = -6.0 + 12.0 * xi
    d2H2 = L * (-4.0 + 6.0 * xi)
    d2H3 = 6.0 - 12.0 * xi
    d2H4 = L * (-2.0 + 6.0 * xi)

    # d²v/dξ²
    d2v_dxi2 = d2H1 * v_i + d2H2 * t_i + d2H3 * v_j + d2H4 * t_j

    # M(x) = EI * d²v/dx² = EI/L² * d²v/dξ²
    bending_moments = (EI / L**2) * d2v_dxi2

    # Third derivative of Hermite shape functions w.r.t. ξ: d³H/dξ³
    # H1'''(ξ) = 12
    # H2'''(ξ) = 6L
    # H3'''(ξ) = -12
    # H4'''(ξ) = 6L
    d3H1 = np.full_like(xi, 12.0)
    d3H2 = np.full_like(xi, 6.0 * L)
    d3H3 = np.full_like(xi, -12.0)
    d3H4 = np.full_like(xi, 6.0 * L)

    # d³v/dξ³ (constant for cubic Hermite)
    d3v_dxi3 = d3H1 * v_i + d3H2 * t_i + d3H3 * v_j + d3H4 * t_j

    # V(x) = dM/dx = EI * d³v/dx³ = EI/L³ * d³v/dξ³
    # Note: V = dM/dx (positive V → moment increasing with x)
    shear_forces = (EI / L**3) * d3v_dxi3

    logger.debug(
        "Element %d: V_range=[%.4e, %.4e], M_range=[%.4e, %.4e]",
        element_id,
        float(np.min(shear_forces)), float(np.max(shear_forces)),
        float(np.min(bending_moments)), float(np.max(bending_moments)),
    )

    return x_stations, shear_forces, bending_moments


def postprocess_all_elements(
    model: FEAModel,
    result: SolutionResult,
    n_stations: int = 50,
) -> list[ElementResult]:
    """Post-process all elements and return a list of ElementResult.

    Dispatches to axial force computation for BAR elements and
    internal force computation for BEAM and FRAME elements.
    """
    dof_map = result.dof_map
    u = result.displacements
    element_results: list[ElementResult] = []

    for element in model.mesh.elements:
        if element.element_type == ElementType.BAR:
            N = compute_element_axial_force(element.id, model, dof_map, u)
            # Bar elements have no shear/moment
            zero = np.zeros(n_stations)
            x_st = np.linspace(element.node_i.x, element.node_j.x, n_stations)
            er = ElementResult(
                element_id=element.id,
                axial_force=N,
                shear_forces=zero,
                bending_moments=zero,
                x_stations=x_st,
            )
        elif element.element_type in (ElementType.BEAM, ElementType.FRAME):
            # FRAME also has axial force if U DOFs are present
            if element.element_type == ElementType.FRAME and dof_map.has_dof(
                element.node_i.id, DOFType.U
            ):
                N = compute_element_axial_force(element.id, model, dof_map, u)
            else:
                N = 0.0
            x_st, V, M = compute_beam_internal_forces(
                element.id, model, dof_map, u, n_stations=n_stations
            )
            er = ElementResult(
                element_id=element.id,
                axial_force=N,
                shear_forces=V,
                bending_moments=M,
                x_stations=x_st,
            )
        else:
            raise ValueError(f"Unknown element type: {element.element_type}")

        element_results.append(er)
        logger.debug("Post-processed element %d", element.id)

    return element_results
