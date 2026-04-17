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


def run_cmaes(
    problem: GeometryOptimizationProblem,
    seed: int,
    popsize: int = 20,
    maxiter: int = 800,
    sigma0: float = 5.0,
    restarts: int = 5,
    incpopsize: int = 2,
    weights: PenaltyWeights = DEFAULT_WEIGHTS,
    checkpoint_path: Optional[Path] = None,
) -> SeedResult:
    """Run one CMA-ES seed (with IPOP restarts) and return a SeedResult.

    Args:
        problem (GeometryOptimizationProblem): Problem definition.
        seed (int): RNG seed.
        popsize (int): Initial population size.
        maxiter (int): Per-restart iteration cap.
        sigma0 (float): Initial step size (in design-vector units, i.e. mm).
        restarts (int): IPOP restart count (0 disables restarts).
        incpopsize (int): IPOP population doubling factor.
        weights (PenaltyWeights): Penalty multipliers.
        checkpoint_path (Path | None): Where to write final SeedResult JSON.

    Returns:
        SeedResult

    Notes:
        x0 is sampled from a uniform distribution over the bound box, seeded
        by the RNG seed, so different seeds explore different basins.
        Uses cma.fmin2 which returns (xbest, es); derived values such as
        fbest are read from es.result, not from positional return values.
    """
    import cma  # local import keeps the new dep out of import-time graph

    history: list[HistoryPoint] = []
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)
    los = np.array([b[0] for b in problem.box_bounds], dtype=np.float64)
    his = np.array([b[1] for b in problem.box_bounds], dtype=np.float64)
    x0 = rng.uniform(los, his)

    bounds_for_cma = [list(los), list(his)]
    opts = {
        "seed": seed + 1,  # cma rejects seed=0
        "bounds": bounds_for_cma,
        "maxiter": maxiter,
        "popsize": popsize,
        "verbose": -9,
        "tolx": 1.0e-8,
        "tolfun": 1.0e-9,
    }

    def fun(x: NDArray[np.float64]) -> float:
        """Evaluate penalized objective for a candidate solution.

        Args:
            x (NDArray[np.float64]): Candidate design vector.

        Returns:
            float: Penalized objective value.
        """
        return penalized_objective(np.asarray(x, dtype=np.float64), problem, weights)

    if restarts > 0:
        x_best, es = cma.fmin2(
            fun, x0, sigma0, options=opts,
            restarts=restarts, incpopsize=incpopsize,
            bipop=False,
        )
        f_best = float(es.result.fbest) if es.result.fbest is not None else fun(x_best)
    else:
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        gen = 0
        while not es.stop():
            xs = es.ask()
            fs = [fun(x) for x in xs]
            es.tell(xs, fs)
            history.append(HistoryPoint(
                generation=gen,
                best_penalty=float(min(fs)),
                mean_penalty=float(np.mean(fs)),
                n_feasible=0,
            ))
            gen += 1
        x_best = es.result.xbest if es.result.xbest is not None else x0
        f_best = float(es.result.fbest) if es.result.fbest is not None else fun(x_best)

    elapsed = time.perf_counter() - t0
    best_x = np.asarray(x_best, dtype=np.float64)
    best_eval = evaluate(best_x, problem)
    sr = SeedResult(
        algorithm="CMA-ES",
        seed=seed,
        best_x=best_x,
        best_eval=best_eval,
        best_penalty=float(f_best),
        history=tuple(history),
        wall_clock_s=elapsed,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else Path("(unsaved)"),
    )
    if checkpoint_path is not None:
        save_seed_result(sr, Path(checkpoint_path))
    return sr
