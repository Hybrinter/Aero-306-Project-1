"""Element-level post-processing: internal forces and displacements via Hermite shape functions.

For beam elements, displacement field v(x) is reconstructed using Hermite cubic
shape functions, which exactly represent the FE solution within each element.
Shear and moment are computed by differentiating:
  M(x) = EI * d^2v/dx^2
  V(x) = dM/dx = EI * d^3v/dx^3

Rotation theta(x) = dv/dx = (1/L) * d^1v/dxi^1.

For bar elements, axial force N = EA/L * (u_j - u_i) and u(x) is linear in x.

For truss elements (2D pin-jointed), axial force is recovered via coordinate projection:
  N = (EA/L) * (c*(U_j - U_i) + s*(V_j - V_i))
where c, s are direction cosines. Shear and moment are identically zero.

compute_element_axial_force:   Axial force for BAR via stiffness formula.
compute_truss_axial_force:     Axial force for TRUSS via direction-cosine projection.
compute_bar_displacements:     Linear u(x) interpolation between nodal axial DOFs.
compute_beam_internal_forces:  V(x), M(x), v(x), theta(x) via Hermite shape functions.
postprocess_all_elements:      Dispatches per element type; returns list[ElementResult].
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fea_solver.elements import _truss_direction_cosines
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

    Args:
        element_id (int): Identifier of the BAR element.
        model (FEAModel): FEA problem containing the mesh.
        dof_map (DOFMap): Global DOF index mapping.
        u (NDArray[np.float64]): Full displacement vector.

    Returns:
        float: Axial force [N], positive in tension.
    """
    element = next(e for e in model.mesh.elements if e.id == element_id)
    u_i = u[dof_map.index(element.node_i.id, DOFType.U)]
    u_j = u[dof_map.index(element.node_j.id, DOFType.U)]
    N = element.material.E * element.material.A / element.length * (u_j - u_i)
    logger.debug("Element %d: axial force N = %.4e N", element_id, N)
    return float(N)


def compute_truss_axial_force(
    element_id: int,
    model: FEAModel,
    dof_map: DOFMap,
    u: NDArray[np.float64],
) -> float:
    """Compute axial force for a TRUSS element in global coordinates.

    N = (EA/L) * (c*(U_j - U_i) + s*(V_j - V_i))

    where c = (xj - xi)/L and s = (yj - yi)/L are the direction cosines.
    Positive N indicates tension.

    Args:
        element_id (int): Identifier of the TRUSS element.
        model (FEAModel): FEA problem containing the mesh.
        dof_map (DOFMap): Global DOF index mapping.
        u (NDArray[np.float64]): Full displacement vector.

    Returns:
        float: Axial force [force units], positive in tension.

    Notes:
        Uses direction-cosine projection of nodal displacements to recover axial
        elongation, which is then multiplied by EA/L. This is the inverse of the
        truss stiffness transformation.
    """
    element = next(e for e in model.mesh.elements if e.id == element_id)
    L = element.length
    c, s = _truss_direction_cosines(element)

    U_i = u[dof_map.index(element.node_i.id, DOFType.U)]
    V_i = u[dof_map.index(element.node_i.id, DOFType.V)]
    U_j = u[dof_map.index(element.node_j.id, DOFType.U)]
    V_j = u[dof_map.index(element.node_j.id, DOFType.V)]

    delta = c * (U_j - U_i) + s * (V_j - V_i)
    N = element.material.E * element.material.A / L * delta
    logger.debug("Truss element %d: N = %.4e (c=%.4f, s=%.4f)", element_id, N, c, s)
    return float(N)


def compute_bar_displacements(
    element_id: int,
    model: FEAModel,
    dof_map: DOFMap,
    u: NDArray[np.float64],
    n_stations: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute axial displacement u(x) along a BAR element via linear interpolation.

    u(x) = u_i + (u_j - u_i) * xi,  xi = (x - x_i) / L

    Args:
        element_id (int): BAR element identifier.
        model (FEAModel): FEA model containing the mesh.
        dof_map (DOFMap): Global DOF index mapping.
        u (NDArray[np.float64]): Full displacement vector.
        n_stations (int): Number of evaluation stations. Default 50.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            (x_stations, axial_displacements), both shape (n_stations,).

    Notes:
        Bar elements use linear (not Hermite) shape functions for axial displacement.
        The result is exact for the FE solution within the element.
    """
    element = next(e for e in model.mesh.elements if e.id == element_id)
    L = element.length
    x_i = element.node_i.x

    u_i = u[dof_map.index(element.node_i.id, DOFType.U)]
    u_j = u[dof_map.index(element.node_j.id, DOFType.U)]

    xi = np.linspace(0.0, 1.0, n_stations)
    x_stations = x_i + xi * L
    axial_displacements = u_i + (u_j - u_i) * xi

    return x_stations, axial_displacements


def compute_beam_internal_forces(
    element_id: int,
    model: FEAModel,
    dof_map: DOFMap,
    u: NDArray[np.float64],
    n_stations: int = 50,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute V(x), M(x), v(x), and theta(x) along a BEAM element.

    Uses Hermite cubic shape functions:
      H1(xi) = 1 - 3*xi^2 + 2*xi^3
      H2(xi) = L*(xi - 2*xi^2 + xi^3)
      H3(xi) = 3*xi^2 - 2*xi^3
      H4(xi) = L*(-xi^2 + xi^3)
    where xi = (x - x_i) / L in [0, 1]

    v(xi)     = H1*v_i + H2*t_i + H3*v_j + H4*t_j
    theta(x)  = dv/dx = (1/L) * dv/dxi
    M(x)      = EI * d^2v/dx^2 = (EI/L^2) * d^2v/dxi^2
    V(x)      = dM/dx = (EI/L^3) * d^3v/dxi^3

    Args:
        element_id (int): Element to evaluate.
        model (FEAModel): FEA model.
        dof_map (DOFMap): DOF index mapping.
        u (NDArray[np.float64]): Full displacement vector.
        n_stations (int): Number of evaluation points along the element.

    Returns:
        tuple of five NDArrays, each shape (n_stations,):
        (x_stations, shear_forces, bending_moments, transverse_displacements, rotations)
        x_stations are global x-coordinates.

    Notes:
        Hermite cubic shape functions exactly represent the FE solution within each
        element. Second and third derivatives recover M and V; zeroth and first
        derivatives recover v and theta directly without integration.
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

    # Parametric coordinate xi in [0, 1]
    xi = np.linspace(0.0, 1.0, n_stations)
    x_stations = x_i + xi * L

    # ---------------------------------------------------------------------------
    # Zeroth-order: v(x) = H1*v_i + H2*t_i + H3*v_j + H4*t_j
    # H1(xi) = 1 - 3*xi^2 + 2*xi^3
    # H2(xi) = L*(xi - 2*xi^2 + xi^3)
    # H3(xi) = 3*xi^2 - 2*xi^3
    # H4(xi) = L*(-xi^2 + xi^3)
    # ---------------------------------------------------------------------------
    H1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    H2 = L * (xi - 2.0 * xi**2 + xi**3)
    H3 = 3.0 * xi**2 - 2.0 * xi**3
    H4 = L * (-xi**2 + xi**3)
    transverse_displacements = H1 * v_i + H2 * t_i + H3 * v_j + H4 * t_j

    # ---------------------------------------------------------------------------
    # First-order: theta(x) = dv/dx = (1/L) * dv/dxi
    # dH1/dxi = -6*xi + 6*xi^2
    # dH2/dxi = L*(1 - 4*xi + 3*xi^2)
    # dH3/dxi = 6*xi - 6*xi^2
    # dH4/dxi = L*(-2*xi + 3*xi^2)
    # ---------------------------------------------------------------------------
    dH1 = -6.0 * xi + 6.0 * xi**2
    dH2 = L * (1.0 - 4.0 * xi + 3.0 * xi**2)
    dH3 = 6.0 * xi - 6.0 * xi**2
    dH4 = L * (-2.0 * xi + 3.0 * xi**2)
    dv_dxi = dH1 * v_i + dH2 * t_i + dH3 * v_j + dH4 * t_j
    rotations = dv_dxi / L

    # ---------------------------------------------------------------------------
    # Second-order: M(x) = EI/L^2 * d^2v/dxi^2
    # d^2H1/dxi^2 = -6 + 12*xi
    # d^2H2/dxi^2 = L*(-4 + 6*xi)
    # d^2H3/dxi^2 = 6 - 12*xi
    # d^2H4/dxi^2 = L*(-2 + 6*xi)
    # ---------------------------------------------------------------------------
    d2H1 = -6.0 + 12.0 * xi
    d2H2 = L * (-4.0 + 6.0 * xi)
    d2H3 = 6.0 - 12.0 * xi
    d2H4 = L * (-2.0 + 6.0 * xi)
    d2v_dxi2 = d2H1 * v_i + d2H2 * t_i + d2H3 * v_j + d2H4 * t_j
    bending_moments = (EI / L**2) * d2v_dxi2

    # ---------------------------------------------------------------------------
    # Third-order: V(x) = dM/dx = EI/L^3 * d^3v/dxi^3  (constant for cubic)
    # d^3H1/dxi^3 = 12
    # d^3H2/dxi^3 = 6*L
    # d^3H3/dxi^3 = -12
    # d^3H4/dxi^3 = 6*L
    # ---------------------------------------------------------------------------
    d3v_dxi3 = (12.0 * v_i + 6.0 * L * t_i - 12.0 * v_j + 6.0 * L * t_j) * np.ones_like(xi)
    shear_forces = (EI / L**3) * d3v_dxi3

    logger.debug(
        "Element %d: V_range=[%.4e, %.4e], M_range=[%.4e, %.4e], v_range=[%.4e, %.4e]",
        element_id,
        float(np.min(shear_forces)), float(np.max(shear_forces)),
        float(np.min(bending_moments)), float(np.max(bending_moments)),
        float(np.min(transverse_displacements)), float(np.max(transverse_displacements)),
    )

    return x_stations, shear_forces, bending_moments, transverse_displacements, rotations


def postprocess_all_elements(
    model: FEAModel,
    result: SolutionResult,
    n_stations: int = 50,
) -> list[ElementResult]:
    """Post-process all elements and return a list of ElementResult.

    Dispatches per element type:
      - BAR: axial force N = EA/L * (u_j - u_i); linear u(x).
      - BEAM/FRAME: shear V(x), moment M(x), and displacements via Hermite shape functions.
      - TRUSS: axial force N = (EA/L) * (c*(U_j-U_i) + s*(V_j-V_i)); shear/moment are zero.

    Args:
        model (FEAModel): FEA problem with mesh and material properties.
        result (SolutionResult): Solution result containing displacements and DOF map.
        n_stations (int): Number of evaluation points per element (default 50).

    Returns:
        list[ElementResult]: List of ElementResult objects with internal forces
            and coordinate stations for each element.
    """
    dof_map = result.dof_map
    u = result.displacements
    element_results: list[ElementResult] = []

    for element in model.mesh.elements:
        if element.element_type == ElementType.BAR:
            N = compute_element_axial_force(element.id, model, dof_map, u)
            x_st, u_stations = compute_bar_displacements(
                element.id, model, dof_map, u, n_stations=n_stations
            )
            er = ElementResult(
                element_id=element.id,
                axial_force=N,
                shear_forces=np.zeros(n_stations),
                bending_moments=np.zeros(n_stations),
                x_stations=x_st,
                transverse_displacements=np.zeros(n_stations),
                axial_displacements=u_stations,
                rotations=np.zeros(n_stations),
            )
        elif element.element_type in (ElementType.BEAM, ElementType.FRAME):
            # FRAME also has axial force and axial displacement if U DOFs are present
            has_axial = (
                element.element_type == ElementType.FRAME
                and dof_map.has_dof(element.node_i.id, DOFType.U)
            )
            if has_axial:
                N = compute_element_axial_force(element.id, model, dof_map, u)
                _, u_stations = compute_bar_displacements(
                    element.id, model, dof_map, u, n_stations=n_stations
                )
            else:
                N = 0.0
                u_stations = np.zeros(n_stations)
            x_st, V, M, v_stations, theta_stations = compute_beam_internal_forces(
                element.id, model, dof_map, u, n_stations=n_stations
            )
            er = ElementResult(
                element_id=element.id,
                axial_force=N,
                shear_forces=V,
                bending_moments=M,
                x_stations=x_st,
                transverse_displacements=v_stations,
                axial_displacements=u_stations,
                rotations=theta_stations,
            )
        elif element.element_type == ElementType.TRUSS:
            N = compute_truss_axial_force(element.id, model, dof_map, u)
            xi_vals = np.linspace(0.0, 1.0, n_stations)
            x_st = element.node_i.x + xi_vals * (element.node_j.x - element.node_i.x)
            er = ElementResult(
                element_id=element.id,
                axial_force=N,
                shear_forces=np.zeros(n_stations),
                bending_moments=np.zeros(n_stations),
                x_stations=x_st,
                transverse_displacements=np.zeros(n_stations),
                axial_displacements=np.zeros(n_stations),
                rotations=np.zeros(n_stations),
            )
        else:
            raise ValueError(f"Unknown element type: {element.element_type}")

        element_results.append(er)
        logger.debug("Post-processed element %d", element.id)

    return element_results
