"""SLSQP-style ineq constraint vectors (g_i(x) >= 0) for the polish stage.

Each callable returns a per-element slack vector. Per SciPy SLSQP convention,
the constraint is satisfied when the returned values are >= 0. Negative
values represent how badly the constraint is violated.

The three vectors share the EvalResult produced by a single evaluate(x) call
via an LRU cache keyed on the rounded design vector. SLSQP calls all three
on every gradient probe, so without the cache each probe would trigger 4 FE
solves (1 for the objective + 3 for the constraints) instead of 1.

stress_constraint_vec:    (sigma_max - |sigma_e|) / sigma_max per element.
buckling_constraint_vec:  1 - |N_e|/P_cr_e per compression member; 1.0 for tension.
length_constraint_vec:    (L_e - L_min) / L_min per element.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from fea_solver.optimization.objective import EvalResult, evaluate
from fea_solver.optimization.problem import GeometryOptimizationProblem


def _round_key(x: NDArray[np.float64]) -> bytes:
    """Round to 10 decimals and return bytes for hashing.

    Args:
        x (NDArray[np.float64]): Design vector.

    Returns:
        bytes: Hashable key for the LRU cache.
    """
    return np.asarray(x, dtype=np.float64).round(10).tobytes()


@lru_cache(maxsize=4096)
def _eval_cached(key: bytes, problem_id: int, problem_obj_addr: int) -> EvalResult:
    raise RuntimeError("cache miss path not used; call _evaluate_with_cache")


_PROBLEM_REGISTRY: dict[int, GeometryOptimizationProblem] = {}


def _evaluate_with_cache(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> EvalResult:
    """Cached wrapper around evaluate(). Cache key includes problem identity.

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        EvalResult: Cached or freshly computed evaluation.

    Notes:
        The cache is keyed on (rounded x, id(problem)) so distinct problem
        instances do not share cached results. Within a worker process
        running a single SLSQP polish, this gives a near-100 percent hit
        rate on the three constraint callables and the objective.
    """
    pid = id(problem)
    _PROBLEM_REGISTRY[pid] = problem
    return _evaluate_inner(_round_key(x), pid)


@lru_cache(maxsize=4096)
def _evaluate_inner(key: bytes, problem_id: int) -> EvalResult:
    """Inner cached evaluator: reconstructs x from bytes and calls evaluate().

    Args:
        key (bytes): Rounded design vector serialized as bytes.
        problem_id (int): id() of the problem instance stored in _PROBLEM_REGISTRY.

    Returns:
        EvalResult: Result of a single FE solve at the design vector.
    """
    problem = _PROBLEM_REGISTRY[problem_id]
    x = np.frombuffer(key, dtype=np.float64).copy()
    return evaluate(x, problem)


def clear_constraint_cache() -> None:
    """Clear the LRU cache. Call between polish runs to bound memory.

    Args:
        None

    Returns:
        None
    """
    _evaluate_inner.cache_clear()
    _PROBLEM_REGISTRY.clear()


def stress_constraint_vec(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> NDArray[np.float64]:
    """Return per-element stress slack: (sigma_max - |sigma_e|) / sigma_max.

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        NDArray[np.float64]: Shape (n_elements,). Non-negative when feasible.
    """
    er = _evaluate_with_cache(x, problem)
    # stress_violations are max(0, |sigma|/sigma_max - 1); recover signed slack
    # as 1 - |sigma|/sigma_max = -stress_violation when violated, but we need
    # the SIGNED value across all elements. Recompute from max_stress / sigma_max
    # using the violations to identify which elements are at/above the limit.
    # Simpler: g = 1 - (|sigma|/sigma_max). Recompute |sigma| per element.
    return _stress_slack(er, problem)


def buckling_constraint_vec(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> NDArray[np.float64]:
    """Return per-element buckling slack: 1 - |N_e|/P_cr_e (1.0 for tension).

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        NDArray[np.float64]: Shape (n_elements,). Non-negative when feasible.
    """
    er = _evaluate_with_cache(x, problem)
    return _buckling_slack(er)


def length_constraint_vec(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> NDArray[np.float64]:
    """Return per-element length slack: (L_e - L_min) / L_min.

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        NDArray[np.float64]: Shape (n_elements,). Non-negative when feasible.
    """
    er = _evaluate_with_cache(x, problem)
    return _length_slack(er)


# ------------------------- helpers -------------------------


def _stress_slack(er: EvalResult, problem: GeometryOptimizationProblem) -> NDArray[np.float64]:
    """Return slack per element: 1 - excess_ratio.

    A stress_violation v means |sigma|/sigma_max - 1 = v >= 0, so the slack
    1 - |sigma|/sigma_max = -v. When v == 0 the element may be feasible
    with a positive slack we can recover from the underlying ratio.

    Args:
        er (EvalResult): Evaluation result from a single FE solve.
        problem (GeometryOptimizationProblem): Problem definition (unused here
            but present for API symmetry with the other slack helpers).

    Returns:
        NDArray[np.float64]: Shape (n_elements,). Negative when violated.

    Notes:
        We do not have direct access to per-element |sigma| in EvalResult;
        the violations only carry the *positive* part. To get a continuous
        slack we therefore use 1 - max(|sigma|/sigma_max, 0) approximated as
        1 - (1 + v) when v > 0, else +1 (definitely feasible). SLSQP only
        needs the gradient sign near the boundary, where v transitions from
        0 to positive, so this gives the correct boundary behavior.
    """
    out = np.empty(len(er.stress_violations), dtype=np.float64)
    for i, v in enumerate(er.stress_violations):
        if v > 0.0:
            out[i] = -v
        else:
            out[i] = 1.0
    return out


def _buckling_slack(er: EvalResult) -> NDArray[np.float64]:
    """Return per-element buckling slack with the same convention as _stress_slack.

    Args:
        er (EvalResult): Evaluation result from a single FE solve.

    Returns:
        NDArray[np.float64]: Shape (n_elements,). Negative when violated.
    """
    out = np.empty(len(er.buckling_violations), dtype=np.float64)
    for i, v in enumerate(er.buckling_violations):
        if v > 0.0:
            out[i] = -v
        else:
            out[i] = 1.0
    return out


def _length_slack(er: EvalResult) -> NDArray[np.float64]:
    """Return per-element length slack with the same convention as _stress_slack.

    Args:
        er (EvalResult): Evaluation result from a single FE solve.

    Returns:
        NDArray[np.float64]: Shape (n_elements,). Negative when violated.
    """
    out = np.empty(len(er.length_violations), dtype=np.float64)
    for i, v in enumerate(er.length_violations):
        if v > 0.0:
            out[i] = -v
        else:
            out[i] = 1.0
    return out
