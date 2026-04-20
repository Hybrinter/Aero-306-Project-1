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
    *,
    penalty_alpha: float | None = None,
) -> NDArray[np.float64]:
    """Solve the penalty-modified system K_mod * u = F_mod.

    Args:
        K_mod (NDArray[np.float64]): Penalty-modified stiffness matrix, shape (n, n).
        F_mod (NDArray[np.float64]): Penalty-modified force vector, shape (n,).
        penalty_alpha (float | None): Penalty-scaling factor used to build K_mod,
            forwarded from FEAModel.penalty_alpha. When supplied, the near-
            singularity warning is penalty-aware and fires only when
            cond(K_mod) > 1e8 * penalty_alpha -- the point at which float64
            precision is genuinely exhausted. When None, a legacy fixed
            threshold of 1e14 is used.

    Returns:
        NDArray[np.float64]: Full displacement vector u, shape (n,).

    Raises:
        np.linalg.LinAlgError: If K_mod is singular.

    Notes:
        cond(K_mod) is proportional to penalty_alpha by construction, so a
        fixed absolute threshold trips routinely for well-posed models when
        penalty_alpha is large. The penalty-aware threshold
        (1e8 * penalty_alpha) is calibrated to float64 precision: loss of
        digits is cond * 2e-16, so cond > 1e16 is the point where double
        precision is actually exhausted.
    """
    cond = float(np.linalg.cond(K_mod))
    logger.debug("K_mod condition number: %.3e", cond)
    threshold = 1.0e8 * penalty_alpha if penalty_alpha is not None else 1.0e14
    if cond > threshold:
        normalized = cond / penalty_alpha if penalty_alpha is not None else cond
        logger.warning(
            "K_mod is nearly singular (cond=%.3e, cond/penalty_alpha=%.3e). "
            "The natural free-partition stiffness is rank-deficient -- check "
            "for mechanisms, coincident nodes, or DOFs without any stiffness "
            "contribution.", cond, normalized,
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
    u = solve_system(K_mod, F_mod, penalty_alpha=model.penalty_alpha)
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
