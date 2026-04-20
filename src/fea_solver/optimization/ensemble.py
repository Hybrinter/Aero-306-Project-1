"""Ensemble orchestrator for the geometry optimizer.

Layout:
  Phase 1 (global): launch DE seeds and CMA-ES seeds in parallel via
    multiprocessing.Pool. Each worker writes a SeedResult JSON.
  Phase 2 (polish): take top-K candidates per seed (by best_penalty),
    run SLSQP polish on each in parallel. Each worker writes a
    PolishResult JSON.
  Phase 3 (selection): apply hard rules to choose the single winner.

EnsembleConfig:    All knobs that the CLI exposes.
run_ensemble:      Top-level entry point.
select_best:       Hard-rule winner selection over polish results
                   (and seed bests as a final fallback).
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from fea_solver.optimization.checkpoint import (
    EnsembleResult,
    PolishResult,
    SeedResult,
    save_ensemble_result,
    save_polish_result,
)
from fea_solver.optimization.global_search import run_cmaes, run_de
from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.polish import slsqp_polish
from fea_solver.optimization.problem import GeometryOptimizationProblem

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """All knobs the CLI exposes for one ensemble run.

    Fields:
        de_seeds (int): Number of DE seeds.
        cmaes_seeds (int): Number of CMA-ES seeds.
        de_popsize (int): SciPy DE popsize multiplier.
        de_maxiter (int): SciPy DE maxiter.
        cmaes_popsize (int): CMA-ES initial popsize.
        cmaes_maxiter (int): CMA-ES per-restart iteration cap.
        cmaes_sigma0 (float): CMA-ES initial step size (mm).
        cmaes_restarts (int): IPOP restarts (0 disables).
        top_k (int): Per-seed candidate count promoted to polish.
        polish_max_iter (int): SLSQP maxiter.
        workers (int): multiprocessing.Pool worker count.
        run_dir (Path): Directory for all checkpoints + final artifacts.
    """

    de_seeds: int
    cmaes_seeds: int
    de_popsize: int
    de_maxiter: int
    cmaes_popsize: int
    cmaes_maxiter: int
    cmaes_sigma0: float
    cmaes_restarts: int
    top_k: int
    polish_max_iter: int
    workers: int
    run_dir: Path


def _de_worker(args: tuple) -> SeedResult:
    """Worker function for DE global search; unpacks args tuple and delegates to run_de.

    Args:
        args (tuple): (problem, seed, popsize, maxiter, ckpt).

    Returns:
        SeedResult: Result of the DE run.
    """
    problem, seed, popsize, maxiter, ckpt = args
    return run_de(
        problem=problem, seed=seed, popsize=popsize, maxiter=maxiter,
        checkpoint_path=ckpt,
    )


def _cmaes_worker(args: tuple) -> SeedResult:
    """Worker function for CMA-ES global search; unpacks args tuple and delegates to run_cmaes.

    Args:
        args (tuple): (problem, seed, popsize, maxiter, sigma0, restarts, ckpt).

    Returns:
        SeedResult: Result of the CMA-ES run.
    """
    problem, seed, popsize, maxiter, sigma0, restarts, ckpt = args
    return run_cmaes(
        problem=problem, seed=seed, popsize=popsize, maxiter=maxiter,
        sigma0=sigma0, restarts=restarts, checkpoint_path=ckpt,
    )


def _polish_worker(args: tuple) -> PolishResult:
    """Worker function for SLSQP polish; unpacks args tuple and delegates to slsqp_polish.

    Args:
        args (tuple): (x0, problem, source, max_iter).

    Returns:
        PolishResult: Result of the polish run.
    """
    x0, problem, source, max_iter = args
    return slsqp_polish(x0, problem, source=source, max_iter=max_iter)


def run_ensemble(
    problem: GeometryOptimizationProblem,
    config: EnsembleConfig,
) -> EnsembleResult:
    """Run the full DE + CMA-ES ensemble with SLSQP polish.

    Args:
        problem (GeometryOptimizationProblem): Problem definition.
        config (EnsembleConfig): All optimization knobs.

    Returns:
        EnsembleResult

    Notes:
        Creates run_dir/{checkpoints, seed_results, polish_results} sub-dirs.
        Writes ensemble_result.json at the end.
    """
    run_dir = Path(config.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "seed_results").mkdir(parents=True, exist_ok=True)
    (run_dir / "polish_results").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Phase 1: global search
    de_jobs = [
        (problem, s, config.de_popsize, config.de_maxiter,
         run_dir / "seed_results" / f"de_seed_{s:02d}.json")
        for s in range(config.de_seeds)
    ]
    cmaes_jobs = [
        (problem, s, config.cmaes_popsize, config.cmaes_maxiter,
         config.cmaes_sigma0, config.cmaes_restarts,
         run_dir / "seed_results" / f"cmaes_seed_{s:02d}.json")
        for s in range(config.cmaes_seeds)
    ]
    seed_results: list[SeedResult] = []
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = []
        for job in de_jobs:
            futures.append(pool.submit(_de_worker, job))
        for job in cmaes_jobs:
            futures.append(pool.submit(_cmaes_worker, job))
        for fut in as_completed(futures):
            try:
                seed_results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning("global-search seed raised: %s", exc)

    # Phase 2: top-K polish per seed
    polish_jobs = []
    for sr in seed_results:
        # We only have best_x per SeedResult (not the full population). Top-K is
        # therefore degenerate at K=1; for K>1 we tile best_x with small jitters.
        for k in range(config.top_k):
            if k == 0:
                x0 = sr.best_x.copy()
            else:
                rng = np.random.default_rng((sr.seed + 1) * 7919 + k)
                x0 = sr.best_x + rng.normal(0.0, 0.5, size=sr.best_x.shape)
            source = f"{sr.algorithm}_seed_{sr.seed:02d}_rank_{k}"
            polish_jobs.append((x0, problem, source, config.polish_max_iter))

    polish_results: list[PolishResult] = []
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(_polish_worker, job) for job in polish_jobs]
        for fut in as_completed(futures):
            try:
                pr = fut.result()
                polish_results.append(pr)
                out = run_dir / "polish_results" / f"{pr.source}.json"
                save_polish_result(pr, out)
            except Exception as exc:  # noqa: BLE001
                logger.warning("polish job raised: %s", exc)

    # Phase 3: selection
    er = select_best(tuple(seed_results), tuple(polish_results))
    er = _with_wallclock(er, time.perf_counter() - t0)
    save_ensemble_result(er, run_dir / "ensemble_result.json")
    return er


def _with_wallclock(er: EnsembleResult, wall_clock_s: float) -> EnsembleResult:
    """Return a new EnsembleResult with wall_clock_s set.

    Args:
        er (EnsembleResult): Original result.
        wall_clock_s (float): Wall-clock seconds to set.

    Returns:
        EnsembleResult: New instance with wall_clock_s replaced.
    """
    from dataclasses import replace
    return replace(er, wall_clock_s=wall_clock_s)


def select_best(
    seeds: tuple[SeedResult, ...],
    polishes: tuple[PolishResult, ...],
    feasibility_tol: float = 1.0e-6,
) -> EnsembleResult:
    """Pick the best feasible candidate from seeds + polishes.

    Hard rules (in order):
      1. Drop solve_ok = False candidates.
      2. Drop infeasible candidates (any violation > feasibility_tol).
      3. Among the rest, return argmin(tip_disp).

    Fallback (zero feasible candidates):
      Return argmin(tip_disp + sum_of_all_violations); flag feasible = False.

    Args:
        seeds (tuple[SeedResult, ...]): All seed results.
        polishes (tuple[PolishResult, ...]): All polish results.
        feasibility_tol (float): Slack to absorb numerical noise.

    Returns:
        EnsembleResult: Aggregate with winner picked.
    """
    from fea_solver.optimization.objective import EvalResult as _EvalResult

    candidates: list[tuple[str, int, np.ndarray, _EvalResult]] = []
    for sr in seeds:
        candidates.append((sr.algorithm, sr.seed, sr.best_x, sr.best_eval))
    for pr in polishes:
        algo, seed = _origin_from_source(pr.source)
        candidates.append((algo, seed, pr.x_polished, pr.eval_polished))

    feasible = []
    for algo, seed, x, ev in candidates:
        if not ev.solve_ok:
            continue
        max_v = max(
            max(ev.stress_violations or (0.0,)),
            max(ev.buckling_violations or (0.0,)),
            max(ev.length_violations or (0.0,)),
        )
        if max_v <= feasibility_tol:
            feasible.append((algo, seed, x, ev))

    if feasible:
        algo, seed, x, ev = min(feasible, key=lambda c: c[3].tip_disp)
        return EnsembleResult(
            winner_x=np.asarray(x, dtype=np.float64),
            winner_eval=ev,
            winner_origin=(algo, seed),
            all_seeds=seeds,
            all_polish=polishes,
            wall_clock_s=0.0,  # filled in by run_ensemble
            feasible=True,
        )

    def score(c: tuple[str, int, np.ndarray, _EvalResult]) -> float:
        """Compute combined penalty score for fallback ranking."""
        ev = c[3]
        v = (
            sum(ev.stress_violations) + sum(ev.buckling_violations)
            + sum(ev.length_violations)
        )
        return ev.tip_disp + v

    algo, seed, x, ev = min(candidates, key=score)
    return EnsembleResult(
        winner_x=np.asarray(x, dtype=np.float64),
        winner_eval=ev,
        winner_origin=(algo, seed),
        all_seeds=seeds,
        all_polish=polishes,
        wall_clock_s=0.0,
        feasible=False,
    )


def _origin_from_source(source: str) -> tuple[str, int]:
    """Parse 'DE_seed_07_rank_0' or 'CMA-ES_seed_03_rank_2' -> ("DE", 7).

    Falls back to (source, 0) when the source string does not match the
    expected '<algo>_seed_<nn>_rank_<k>' format (e.g. arbitrary short names
    used in tests).

    Args:
        source (str): Source identifier string in format '<algo>_seed_<nn>_rank_<k>'.

    Returns:
        tuple[str, int]: (algorithm_name, seed_int).
    """
    parts = source.split("_seed_")
    if len(parts) < 2:
        return source, 0
    algo = parts[0]
    seed_part = parts[1].split("_")[0]
    try:
        return algo, int(seed_part)
    except ValueError:
        return source, 0
