"""Quadratic penalty wrapper around evaluate() for global solvers.

P(x) = J(x) + w_s * sum(stress_violations**2)
            + w_b * sum(buckling_violations**2)
            + w_l * sum(length_violations**2)

Quadratic in the relative-violation domain. For solve_ok=False candidates,
J(x) == SENTINEL_TIP_DISP (1e12) so DE/CMA-ES distributions don't blow up.

PenaltyWeights:        Frozen container for the three penalty multipliers.
DEFAULT_WEIGHTS:       w_s = w_b = w_l = 100.0 (typical |J| ~ 1e-3..1e-1).
penalized_objective:   Wrapper that returns one float per call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.problem import GeometryOptimizationProblem


@dataclass(frozen=True, slots=True)
class PenaltyWeights:
    """Multipliers for the three quadratic penalty terms.

    Fields:
        stress (float): w_s for stress_violations squared.
        buckling (float): w_b for buckling_violations squared.
        length (float): w_l for length_violations squared.
    """

    stress: float
    buckling: float
    length: float


DEFAULT_WEIGHTS: PenaltyWeights = PenaltyWeights(
    stress=100.0, buckling=100.0, length=100.0
)


def penalized_objective(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
    weights: PenaltyWeights,
) -> float:
    """Compute penalized fitness for one candidate.

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.
        weights (PenaltyWeights): Quadratic penalty multipliers.

    Returns:
        float: J(x) + sum of weighted squared violations. Finite (sentinel
            when solve_ok is False) so global solvers stay numerically sound.
    """
    er = evaluate(x, problem)
    s_pen = weights.stress * float(np.sum(np.asarray(er.stress_violations) ** 2))
    b_pen = weights.buckling * float(np.sum(np.asarray(er.buckling_violations) ** 2))
    l_pen = weights.length * float(np.sum(np.asarray(er.length_violations) ** 2))
    return float(er.tip_disp + s_pen + b_pen + l_pen)
