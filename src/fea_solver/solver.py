"""Displacement solver using the penalty-modified full stiffness system.

Solves K_mod * u = F_mod where K_mod and F_mod include penalty constraint
terms. Reactions are computed as penalty residuals per constraint.

compute_penalty_parameter: compute k_penalty from penalty_alpha and K diagonal.
solve_system:              np.linalg.solve with condition number check.
run_solve_pipeline:        orchestrate full solve (penalty apply -> solve -> reactions).
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fea_solver.constraints import apply_penalty_constraints, compute_constraint_residuals
from fea_solver.models import DOFMap, FEAModel, SolutionResult

logger = logging.getLogger(__name__)


def compute_penalty_parameter(K: NDArray[np.float64], penalty_alpha: float) -> float:
    """Compute the penalty stiffness parameter from the natural stiffness matrix.

    k_penalty = penalty_alpha * max(abs(diag(K)))

    Args:
        K (NDArray[np.float64]): Natural (pre-penalty) global stiffness matrix.
        penalty_alpha (float): Scale factor from FEAModel.penalty_alpha.

    Returns:
        float: Penalty stiffness parameter.

    Notes:
        Scaling by max(diag(K)) makes the penalty parameter dimensionally
        consistent across problem scales, so penalty_alpha=1e8 works for both
        steel aerospace structures and soft-material unit-stiffness problems.
    """
    return penalty_alpha * float(np.max(np.abs(np.diag(K))))


def solve_system(
    K_mod: NDArray[np.float64],
    F_mod: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the penalty-modified system K_mod * u = F_mod.

    Args:
        K_mod (NDArray[np.float64]): Penalty-modified stiffness matrix, shape (n, n).
        F_mod (NDArray[np.float64]): Penalty-modified force vector, shape (n,).

    Returns:
        NDArray[np.float64]: Full displacement vector u, shape (n,).

    Raises:
        np.linalg.LinAlgError: If K_mod is singular.

    Notes:
        Condition number is logged. Penalty-modified matrices have condition
        numbers proportional to penalty_alpha, so warnings above 1e14 are
        expected and do not indicate a problem for well-posed models.
    """
    cond = float(np.linalg.cond(K_mod))
    logger.debug("K_mod condition number: %.3e", cond)
    if cond > 1e14:
        logger.warning(
            "K_mod is nearly singular (cond=%.3e). For penalty-method models "
            "this is expected when penalty_alpha is large. Check that all DOFs "
            "have at least one constraint or stiffness contribution.", cond
        )

    try:
        u = np.linalg.solve(K_mod, F_mod)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Stiffness matrix is singular -- check boundary conditions. "
            f"Original error: {exc}"
        ) from exc

    logger.debug("Max displacement: %.6e", float(np.max(np.abs(u))))
    return u


def run_solve_pipeline(
    model: FEAModel,
    dof_map: DOFMap,
    K: NDArray[np.float64],
    F: NDArray[np.float64],
) -> SolutionResult:
    """Orchestrate the full penalty-method solve pipeline.

    Steps:
      1. compute_penalty_parameter -> k_penalty
      2. apply_penalty_constraints -> K_mod, F_mod
      3. solve_system -> u (full displacement vector)
      4. compute_constraint_residuals -> reactions (per constraint)
      5. Return SolutionResult

    Args:
        model (FEAModel): FEA model supplying constraints and penalty_alpha.
        dof_map (DOFMap): DOF index mapping.
        K (NDArray[np.float64]): Natural global stiffness matrix.
        F (NDArray[np.float64]): Global force vector.

    Returns:
        SolutionResult: Displacements, per-constraint reactions, dof_map, model.

    Notes:
        reactions[i] in the returned SolutionResult corresponds to
        model.boundary_conditions[i]. The penalty parameter is stored
        implicitly in model.penalty_alpha and reconstructed from K's diagonal.
    """
    k_penalty = compute_penalty_parameter(K, model.penalty_alpha)
    K_mod, F_mod = apply_penalty_constraints(
        K, F, model.boundary_conditions, dof_map, k_penalty
    )
    u = solve_system(K_mod, F_mod)
    reactions = compute_constraint_residuals(
        u, model.boundary_conditions, dof_map, k_penalty
    )

    logger.info(
        "Solve complete: max|u|=%.4e, max|R|=%.4e",
        float(np.max(np.abs(u))),
        float(np.max(np.abs(reactions))) if len(reactions) > 0 else 0.0,
    )

    return SolutionResult(
        displacements=u,
        reactions=reactions,
        dof_map=dof_map,
        model=model,
    )
