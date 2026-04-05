"""Kinematic constraint application via the reduction (elimination) method.

The reduction method:
  1. Identify free and constrained DOF indices.
  2. Partition K into K_ff (free-free) and K_fc (free-constrained).
  3. For homogeneous BCs (u_c = 0): reduced system is K_ff * u_f = F_f.
  4. Solve the reduced system, then recover full displacement vector.

This approach is exact and numerically stable, unlike the penalty method.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fea_solver.models import (
    BoundaryConditionType,
    DOFMap,
    DOFType,
    FEAModel,
)

logger = logging.getLogger(__name__)

# Maps each BC type to the DOF types it constrains
_BC_TO_DOF_TYPES: dict[BoundaryConditionType, tuple[DOFType, ...]] = {
    BoundaryConditionType.FIXED_U:     (DOFType.U,),
    BoundaryConditionType.FIXED_V:     (DOFType.V,),
    BoundaryConditionType.FIXED_THETA: (DOFType.THETA,),
    BoundaryConditionType.FIXED_ALL:   (DOFType.U, DOFType.V, DOFType.THETA),
    BoundaryConditionType.PIN:         (DOFType.U, DOFType.V),
    BoundaryConditionType.ROLLER:      (DOFType.V,),
}


def get_constrained_dof_indices(model: FEAModel, dof_map: DOFMap) -> list[int]:
    """Return sorted list of global DOF indices that are kinematically constrained.

    Only constrains DOF types that actually exist at the node (e.g. a PIN on
    a BEAM node only constrains V, not U, because BEAM nodes have no U DOF).
    """
    constrained: set[int] = set()
    for bc in model.boundary_conditions:
        dof_types = _BC_TO_DOF_TYPES[bc.bc_type]
        for dof_type in dof_types:
            if dof_map.has_dof(bc.node_id, dof_type):
                constrained.add(dof_map.index(bc.node_id, dof_type))
    result = sorted(constrained)
    logger.debug("Constrained DOFs: %s", result)
    return result


def get_free_dof_indices(model: FEAModel, dof_map: DOFMap) -> list[int]:
    """Return sorted list of global DOF indices that are free (unconstrained)."""
    constrained = set(get_constrained_dof_indices(model, dof_map))
    free = sorted(i for i in range(dof_map.total_dofs) if i not in constrained)
    logger.debug("Free DOFs: %s", free)
    return free


def apply_constraints_reduction(
    K: NDArray[np.float64],
    F: NDArray[np.float64],
    constrained_dofs: list[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply BCs via reduction/elimination method.

    Partitions the system into free (f) and constrained (c) sub-sets.
    For homogeneous BCs (u_c = 0), the reduced system is:
        K_ff * u_f = F_f

    Args:
        K: Global stiffness matrix (n x n).
        F: Global force vector (n,).
        constrained_dofs: Sorted list of constrained DOF global indices.

    Returns:
        (K_ff, F_f): Reduced stiffness matrix and force vector for free DOFs.
    """
    n = K.shape[0]
    all_dofs = list(range(n))
    constrained_set = set(constrained_dofs)
    free_dofs = [i for i in all_dofs if i not in constrained_set]

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    logger.debug("Reduced system: %d free DOFs, %d constrained DOFs",
                 len(free_dofs), len(constrained_dofs))
    return K_ff, F_f
