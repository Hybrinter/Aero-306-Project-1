"""Displacement solver and reaction force computation.

Solves the partitioned linear system K_ff * u_f = F_f using numpy.linalg.solve.
Computes reaction forces at constrained DOFs via R = K[c,:] @ u - F[c].
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fea_solver.constraints import (
    apply_constraints_reduction,
    get_constrained_dof_indices,
    get_free_dof_indices,
)
from fea_solver.models import DOFMap, FEAModel, SolutionResult

logger = logging.getLogger(__name__)


def solve_displacements(
    K_ff: NDArray[np.float64],
    F_f: NDArray[np.float64],
    free_dofs: list[int],
    constrained_dofs: list[int],
    n_total_dofs: int,
) -> NDArray[np.float64]:
    """Solve K_ff * u_f = F_f and return the full displacement vector.

    Args:
        K_ff: Reduced (free-free) stiffness matrix.
        F_f: Reduced force vector for free DOFs.
        free_dofs: Global indices of free DOFs.
        constrained_dofs: Global indices of constrained DOFs (u_c = 0).
        n_total_dofs: Total number of DOFs in the full system.

    Returns:
        Full displacement vector u of shape (n_total_dofs,),
        with u[constrained] = 0.

    Raises:
        np.linalg.LinAlgError: If K_ff is singular (structure is unstable).

    Notes:
        Condition number threshold of 1e14 is used to detect ill-conditioning.
        Values above 1e14 indicate nearly singular stiffness matrix;
        typically due to missing or improperly applied boundary conditions.
    """
    cond = float(np.linalg.cond(K_ff))
    logger.debug("K_ff condition number: %.3e", cond)
    if cond > 1e14:
        logger.warning("K_ff is nearly singular (cond=%.3e) -- check BCs", cond)

    try:
        u_f = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Stiffness matrix is singular -- check boundary conditions. "
            f"Original error: {exc}"
        ) from exc

    # Reconstruct full displacement vector (constrained DOFs remain 0)
    u = np.zeros(n_total_dofs)
    for local_idx, global_idx in enumerate(free_dofs):
        u[global_idx] = u_f[local_idx]

    logger.debug("Max displacement: %.6e", float(np.max(np.abs(u))))
    return u


def compute_reactions(
    K: NDArray[np.float64],
    u: NDArray[np.float64],
    F: NDArray[np.float64],
    constrained_dofs: list[int],
) -> NDArray[np.float64]:
    """Compute reaction forces at constrained DOFs.

    R = K[constrained, :] @ u - F[constrained]

    Args:
        K (NDArray[np.float64]): Full global stiffness matrix of shape (n_dofs, n_dofs).
        u (NDArray[np.float64]): Full displacement vector of shape (n_dofs,).
        F (NDArray[np.float64]): Full global force vector of shape (n_dofs,).
        constrained_dofs (list[int]): Global indices of constrained DOFs.

    Returns:
        NDArray[np.float64]: Reaction vector of shape (n_constrained,),
            same order as constrained_dofs.
    """
    R = K[constrained_dofs, :] @ u - F[constrained_dofs]
    logger.debug("Reactions: %s", R)
    return R


def run_solve_pipeline(
    model: FEAModel,
    dof_map: DOFMap,
    K: NDArray[np.float64],
    F: NDArray[np.float64],
) -> SolutionResult:
    """Orchestrate the full solve pipeline.

    Steps:
      1. get_constrained_dof_indices
      2. get_free_dof_indices
      3. apply_constraints_reduction -> K_ff, F_f
      4. solve_displacements -> u (full)
      5. compute_reactions -> R
      6. Return SolutionResult

    Args:
        model: The FEA model (provides BCs).
        dof_map: DOF index mapping.
        K: Full global stiffness matrix.
        F: Full global force vector.

    Returns:
        SolutionResult with displacements, reactions, dof_map, and model.

    Notes:
        Pipeline applies constraint reduction method: partition K and F into
        free (f) and constrained (c) DOFs, solve the reduced system K_ff*u_f = F_f,
        then recover reactions using full stiffness and all DOFs.
    """
    constrained = get_constrained_dof_indices(model, dof_map)
    free = get_free_dof_indices(model, dof_map)

    K_ff, F_f = apply_constraints_reduction(K, F, constrained)
    u = solve_displacements(K_ff, F_f, free, constrained, dof_map.total_dofs)
    R = compute_reactions(K, u, F, constrained)

    logger.info("Solve complete: max|u|=%.4e, max|R|=%.4e",
                float(np.max(np.abs(u))), float(np.max(np.abs(R))))

    return SolutionResult(
        displacements=u,
        reactions=R,
        dof_map=dof_map,
        model=model,
    )
