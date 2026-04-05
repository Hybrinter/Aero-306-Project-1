"""Global stiffness matrix and force vector assembly.

Conventions:
  - DOF ordering within build_dof_map: nodes sorted by id,
    within each node DOFs in canonical order (U, V, THETA),
    only those applicable to the element types present at that node.
  - This ordering is FROZEN — all downstream modules depend on it.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fea_solver.elements import element_load_vector, element_stiffness_matrix
from fea_solver.models import (
    DOFMap,
    DOFType,
    Element,
    ElementType,
    FEAModel,
    LoadType,
    Node,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DOF type sets per element type
# ---------------------------------------------------------------------------

_ELEMENT_DOFS: dict[ElementType, tuple[DOFType, ...]] = {
    ElementType.BAR:   (DOFType.U,),
    ElementType.BEAM:  (DOFType.V, DOFType.THETA),
    ElementType.FRAME: (DOFType.U, DOFType.V, DOFType.THETA),
}


def _dofs_for_node(node_id: int, model: FEAModel) -> tuple[DOFType, ...]:
    """Determine which DOF types exist at a given node.

    A node inherits the union of DOF types from all elements connected to it,
    in canonical order (U before V before THETA).
    """
    dof_set: set[DOFType] = set()
    for element in model.mesh.elements:
        if element.node_i.id == node_id or element.node_j.id == node_id:
            dof_set.update(_ELEMENT_DOFS[element.element_type])

    # Return in canonical order
    canonical = [DOFType.U, DOFType.V, DOFType.THETA]
    return tuple(d for d in canonical if d in dof_set)


def build_dof_map(model: FEAModel) -> DOFMap:
    """Build the global DOF mapping from (node_id, DOFType) -> int.

    Nodes processed in ascending id order.
    Within each node, DOFs assigned in canonical order: U, V, THETA.
    Only DOFs applicable to the element types present at that node.
    """
    dof_map = DOFMap()
    idx = 0
    for node in sorted(model.mesh.nodes, key=lambda n: n.id):
        for dof_type in _dofs_for_node(node.id, model):
            dof_map.mapping[(node.id, dof_type)] = idx
            idx += 1

    dof_map.total_dofs = idx
    logger.debug("Built DOF map: %d total DOFs", dof_map.total_dofs)
    return dof_map


def get_element_dof_indices(
    element_id: int, model: FEAModel, dof_map: DOFMap
) -> list[int]:
    """Return ordered list of global DOF indices for the given element.

    Order follows element DOF ordering:
      BAR:   [u_i, u_j]
      BEAM:  [v_i, theta_i, v_j, theta_j]
      FRAME: [u_i, v_i, theta_i, u_j, v_j, theta_j]
    """
    element = next(e for e in model.mesh.elements if e.id == element_id)
    dof_types = _ELEMENT_DOFS[element.element_type]
    indices: list[int] = []
    for node_id in (element.node_i.id, element.node_j.id):
        for dof_type in dof_types:
            indices.append(dof_map.index(node_id, dof_type))
    return indices


def assemble_global_stiffness(
    model: FEAModel, dof_map: DOFMap
) -> NDArray[np.float64]:
    """Assemble the global stiffness matrix K (n_dofs x n_dofs).

    For each element, computes the local stiffness matrix and scatters
    it into K using the element DOF indices.
    """
    n = dof_map.total_dofs
    K = np.zeros((n, n))

    for element in model.mesh.elements:
        k_local = element_stiffness_matrix(element)
        dof_indices = get_element_dof_indices(element.id, model, dof_map)
        for i, gi in enumerate(dof_indices):
            for j, gj in enumerate(dof_indices):
                K[gi, gj] += k_local[i, j]

    logger.debug("Assembled global K: shape %s", K.shape)
    return K


def assemble_global_force_vector(
    model: FEAModel, dof_map: DOFMap
) -> NDArray[np.float64]:
    """Assemble the global force vector F (n_dofs,).

    Applies:
      1. Nodal (concentrated) loads directly.
      2. Distributed loads converted to consistent nodal forces.
    """
    n = dof_map.total_dofs
    F = np.zeros(n)

    # --- Nodal loads ---
    _NODAL_LOAD_DOF: dict[LoadType, DOFType] = {
        LoadType.POINT_FORCE_X: DOFType.U,
        LoadType.POINT_FORCE_Y: DOFType.V,
        LoadType.POINT_MOMENT:  DOFType.THETA,
    }
    for load in model.nodal_loads:
        dof_type = _NODAL_LOAD_DOF.get(load.load_type)
        if dof_type is None:
            raise ValueError(f"Unrecognised nodal load type: {load.load_type}")
        idx = dof_map.index(load.node_id, dof_type)
        F[idx] += load.magnitude
        logger.debug(
            "Nodal load %.3g applied at node %d DOF %s (global %d)",
            load.magnitude, load.node_id, dof_type.value, idx,
        )

    # --- Distributed loads ---
    elem_by_id: dict[int, Element] = {e.id: e for e in model.mesh.elements}
    for dist_load in model.distributed_loads:
        element = elem_by_id[dist_load.element_id]
        f_local = element_load_vector(element, dist_load)
        dof_indices = get_element_dof_indices(element.id, model, dof_map)
        for i, gi in enumerate(dof_indices):
            F[gi] += f_local[i]
        logger.debug(
            "Distributed load on element %d scattered to DOFs %s",
            element.id, dof_indices,
        )

    return F
