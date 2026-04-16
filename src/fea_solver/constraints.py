"""Kinematic constraint application via the penalty method.

The penalty method:
  1. For each LinearConstraint, build a global coefficient vector g from
     the constraint's (node_id, coefficients) and the DOF map.
  2. Add k_penalty * outer(g, g) to K and k_penalty * rhs * g to F.
  3. Solve the full (unpartitioned) modified system K_mod * u = F_mod.
  4. Recover per-constraint reaction forces as k_penalty * (a^T * u - rhs).

The penalty parameter is computed as penalty_alpha * max(abs(diag(K_natural)))
so that it scales with the problem stiffness without needing manual tuning.

apply_penalty_constraints: add penalty terms to K and F for all constraints.
compute_constraint_residuals: compute per-constraint reaction magnitudes post-solve.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fea_solver.models import DOFMap, DOFType, LinearConstraint

logger = logging.getLogger(__name__)

# Canonical mapping from coefficient position to DOFType
_COEFF_INDEX_TO_DOF: tuple[DOFType, DOFType, DOFType] = (
    DOFType.U,
    DOFType.V,
    DOFType.THETA,
)


def apply_penalty_constraints(
    K: NDArray[np.float64],
    F: NDArray[np.float64],
    constraints: tuple[LinearConstraint, ...],
    dof_map: DOFMap,
    k_penalty: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply kinematic constraints via the penalty method.

    For each constraint with coefficient vector a = [a_U, a_V, a_THETA]:
      1. Build global coefficient vector g (length n_dofs) by placing a[i]
         at the global DOF index for (node_id, DOFType_i) for each non-zero a[i].
      2. K_mod += k_penalty * outer(g, g)
      3. F_mod += k_penalty * rhs * g

    Args:
        K (NDArray[np.float64]): Global stiffness matrix, shape (n, n).
        F (NDArray[np.float64]): Global force vector, shape (n,).
        constraints (tuple[LinearConstraint, ...]): All constraints to apply.
        dof_map (DOFMap): DOF index mapping for (node_id, DOFType) lookups.
        k_penalty (float): Penalty stiffness parameter (problem-scale-dependent).

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: New (K_mod, F_mod)
            arrays. The input K and F are not mutated.

    Raises:
        ValueError: If a constraint has a non-zero coefficient for a DOF that
            does not exist at the node (e.g., U component on a BEAM-only node).

    Notes:
        Returns copies; input arrays are never mutated.
        The penalty parameter k_penalty is typically computed as
        model.penalty_alpha * max(abs(diag(K_natural))) by the caller.
    """
    K_mod = K.copy()
    F_mod = F.copy()
    n = K_mod.shape[0]

    for constraint in constraints:
        g = np.zeros(n)
        for coeff_idx, dof_type in enumerate(_COEFF_INDEX_TO_DOF):
            a_i = constraint.coefficients[coeff_idx]
            if a_i == 0.0:
                continue
            if not dof_map.has_dof(constraint.node_id, dof_type):
                raise ValueError(
                    f"Constraint at node {constraint.node_id} has non-zero "
                    f"coefficient for {dof_type.value} ({a_i}), but that DOF "
                    f"does not exist at this node."
                )
            g[dof_map.index(constraint.node_id, dof_type)] = a_i

        K_mod += k_penalty * np.outer(g, g)
        F_mod += k_penalty * constraint.rhs * g

    logger.debug(
        "Penalty constraints applied: %d constraints, k_penalty=%.3e",
        len(constraints), k_penalty,
    )
    return K_mod, F_mod


def compute_constraint_residuals(
    u: NDArray[np.float64],
    constraints: tuple[LinearConstraint, ...],
    dof_map: DOFMap,
    k_penalty: float,
) -> NDArray[np.float64]:
    """Compute per-constraint reaction force magnitudes after solving.

    For each constraint i with coefficient vector a_i and prescribed value rhs_i:
        reactions[i] = k_penalty * (a_i^T * u_node_i - rhs_i)

    This is the constraint force: the force the support applies to the structure.

    Args:
        u (NDArray[np.float64]): Full displacement vector, shape (n_dofs,).
        constraints (tuple[LinearConstraint, ...]): All constraints.
        dof_map (DOFMap): DOF index mapping for (node_id, DOFType) lookups.
        k_penalty (float): Penalty stiffness parameter used during assembly.

    Returns:
        NDArray[np.float64]: Reaction magnitudes, shape (n_constraints,).
            reactions[i] corresponds to constraints[i].

    Notes:
        For axis-aligned constraints with unit coefficients, the reaction equals
        the physical support force (e.g., k_p * v_1 is the vertical reaction).
        Residuals are near zero when the constraint is well-enforced.
    """
    reactions = np.empty(len(constraints))
    for i, constraint in enumerate(constraints):
        a_dot_u = 0.0
        for coeff_idx, dof_type in enumerate(_COEFF_INDEX_TO_DOF):
            a_i = constraint.coefficients[coeff_idx]
            if a_i != 0.0 and dof_map.has_dof(constraint.node_id, dof_type):
                a_dot_u += a_i * u[dof_map.index(constraint.node_id, dof_type)]
        reactions[i] = k_penalty * (a_dot_u - constraint.rhs)
    logger.debug("Constraint residuals: %s", reactions)
    return reactions
