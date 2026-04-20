"""SLSQP polish stage for the optimization ensemble.

Takes a candidate from the global stage and runs SciPy's SLSQP under the
real nonlinear constraints, pushing it from "good penalty-feasible" toward
the active constraint boundary. SLSQP failures are captured, not raised.

slsqp_polish:    Run one polish job; return a PolishResult.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from fea_solver.optimization.checkpoint import PolishResult
from fea_solver.optimization.constraints import (
    buckling_constraint_vec,
    clear_constraint_cache,
    length_constraint_vec,
    stress_constraint_vec,
)
from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.problem import GeometryOptimizationProblem

logger = logging.getLogger(__name__)


def slsqp_polish(
    x0: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
    source: str,
    max_iter: int = 200,
    ftol: float = 1.0e-9,
    eps: float = 1.0e-5,
) -> PolishResult:
    """Run one SLSQP polish from x0; return a PolishResult.

    Args:
        x0 (NDArray[np.float64]): Starting design vector.
        problem (GeometryOptimizationProblem): Problem definition.
        source (str): Identifier carried into PolishResult.source for traceability.
        max_iter (int): SLSQP maxiter. Default 200.
        ftol (float): SLSQP function-value tolerance. Default 1e-9.
        eps (float): SLSQP finite-difference step. Default 1e-5.

    Returns:
        PolishResult: success flag, final x, evaluate(x_polished),
            iteration count, and SLSQP exit message. Always returns; never raises.

    Notes:
        Cache is cleared at entry so polish jobs do not see stale entries
        from previous invocations within the same process.
    """
    clear_constraint_cache()

    def fun(x: NDArray[np.float64]) -> float:
        return float(evaluate(x, problem).tip_disp)

    constraints = [
        {"type": "ineq", "fun": lambda x, p=problem: stress_constraint_vec(x, p)},
        {"type": "ineq", "fun": lambda x, p=problem: buckling_constraint_vec(x, p)},
        {"type": "ineq", "fun": lambda x, p=problem: length_constraint_vec(x, p)},
    ]

    try:
        result = minimize(
            fun=fun,
            x0=np.asarray(x0, dtype=np.float64).copy(),
            method="SLSQP",
            bounds=problem.box_bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": ftol, "eps": eps, "disp": False},
        )
        x_final = np.asarray(result.x, dtype=np.float64)
        eval_final = evaluate(x_final, problem)
        return PolishResult(
            source=source,
            x_polished=x_final,
            eval_polished=eval_final,
            success=bool(result.success),
            n_iter=int(getattr(result, "nit", 0)),
            message=str(result.message),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("SLSQP polish %s raised: %s", source, exc)
        eval_x0 = evaluate(np.asarray(x0, dtype=np.float64), problem)
        return PolishResult(
            source=source,
            x_polished=np.asarray(x0, dtype=np.float64),
            eval_polished=eval_x0,
            success=False,
            n_iter=0,
            message=f"raised: {exc}",
        )
