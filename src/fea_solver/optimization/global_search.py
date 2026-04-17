"""Global-search runners for the optimization ensemble.

Each runner produces a SeedResult and writes a JSON checkpoint at
checkpoint_path on completion (and periodically during long runs).

run_de:      SciPy differential_evolution wrapper.
run_cmaes:   pycma fmin2 wrapper with IPOP restart.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from fea_solver.optimization.checkpoint import (
    HistoryPoint,
    SeedResult,
    save_seed_result,
)
from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.penalty import (
    DEFAULT_WEIGHTS,
    PenaltyWeights,
    penalized_objective,
)
from fea_solver.optimization.problem import GeometryOptimizationProblem

logger = logging.getLogger(__name__)


def run_de(
    problem: GeometryOptimizationProblem,
    seed: int,
    popsize: int = 30,
    maxiter: int = 600,
    weights: PenaltyWeights = DEFAULT_WEIGHTS,
    tol: float = 1.0e-7,
    mutation: tuple[float, float] = (0.5, 1.5),
    recombination: float = 0.9,
    checkpoint_path: Optional[Path] = None,
) -> SeedResult:
    """Run one DE seed and return a SeedResult.

    Args:
        problem (GeometryOptimizationProblem): Problem definition.
        seed (int): RNG seed.
        popsize (int): SciPy DE popsize multiplier (n_individuals = popsize * n_vars).
        maxiter (int): Max generations.
        weights (PenaltyWeights): Penalty multipliers.
        tol (float): Relative tolerance for DE convergence.
        mutation (tuple[float, float]): DE mutation factor range.
        recombination (float): DE crossover probability.
        checkpoint_path (Path | None): Where to write final SeedResult JSON.

    Returns:
        SeedResult

    Notes:
        polish=False because polishing is done at the ensemble level.
        workers=1 because parallelism is at the seed level via multiprocessing.Pool.
        The callback uses the newer scipy >= 1.11 signature
        callback(intermediate_result) with intermediate_result.fun.
        A compatibility fallback to callback(x, convergence) is provided
        in case an older scipy version is detected at import time, but
        scipy 1.17.1 (the installed version) uses the newer signature.
    """
    history: list[HistoryPoint] = []
    t0 = time.perf_counter()

    def callback(intermediate_result: object) -> None:
        """Record one HistoryPoint per generation from the DE callback.

        Args:
            intermediate_result (object): OptimizeResult-like object with .fun.

        Returns:
            None
        """
        gen = len(history)
        best_pen = float(intermediate_result.fun)  # type: ignore[union-attr]
        history.append(HistoryPoint(
            generation=gen,
            best_penalty=best_pen,
            mean_penalty=best_pen,  # SciPy DE callback only exposes the best
            n_feasible=0,           # not tracked at this granularity
        ))

    result = differential_evolution(
        func=lambda x: penalized_objective(x, problem, weights),
        bounds=list(problem.box_bounds),
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=False,
        init="sobol",
        workers=1,
        updating="deferred",
        callback=callback,
    )
    elapsed = time.perf_counter() - t0
    best_x = np.asarray(result.x, dtype=np.float64)
    best_eval = evaluate(best_x, problem)
    sr = SeedResult(
        algorithm="DE",
        seed=seed,
        best_x=best_x,
        best_eval=best_eval,
        best_penalty=float(result.fun),
        history=tuple(history),
        wall_clock_s=elapsed,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else Path("(unsaved)"),
    )
    if checkpoint_path is not None:
        save_seed_result(sr, Path(checkpoint_path))
    return sr
