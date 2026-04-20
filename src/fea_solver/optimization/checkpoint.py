"""Checkpoint and result serializers for the optimization pipeline.

All result dataclasses serialize to plain JSON so they can be inspected by
hand (and resumed across restarts). NumPy arrays are stored as nested lists.

HistoryPoint:        Per-generation log entry for SeedResult.history.
SeedResult:          Final outcome of one global-search seed (DE or CMA-ES).
PolishResult:        Outcome of one SLSQP polish job.
EnsembleResult:      Aggregate of all seeds + all polish results + winner.
save_seed_result / load_seed_result:    JSON round-trip for SeedResult.
save_polish_result / load_polish_result: JSON round-trip for PolishResult.
save_ensemble_result / load_ensemble_result: JSON round-trip for EnsembleResult.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fea_solver.optimization.objective import EvalResult


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    """One entry in a seed's per-generation log.

    Fields:
        generation (int): Generation / iteration number.
        best_penalty (float): Best penalized fitness in the population so far.
        mean_penalty (float): Mean penalized fitness across the current population.
        n_feasible (int): Number of feasible individuals in the current population.
    """

    generation: int
    best_penalty: float
    mean_penalty: float
    n_feasible: int


@dataclass(frozen=True, slots=True)
class SeedResult:
    """Final outcome of one global-search seed.

    Fields:
        algorithm (str): "DE" or "CMA-ES".
        seed (int): RNG seed used.
        best_x (NDArray[np.float64]): Best design vector found, shape (n_vars,).
        best_eval (EvalResult): evaluate(best_x) outcome.
        best_penalty (float): penalized_objective(best_x) value.
        history (tuple[HistoryPoint, ...]): Per-generation log.
        wall_clock_s (float): Total wall-clock seconds for this seed.
        checkpoint_path (Path): Where the in-flight checkpoint was last written.
    """

    algorithm: str
    seed: int
    best_x: NDArray[np.float64]
    best_eval: EvalResult
    best_penalty: float
    history: tuple[HistoryPoint, ...]
    wall_clock_s: float
    checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class PolishResult:
    """Outcome of one SLSQP polish job.

    Fields:
        source (str): Identifier of the source seed (e.g. "DE_seed_07_rank_0").
        x_polished (NDArray[np.float64]): Final design vector, shape (n_vars,).
        eval_polished (EvalResult): evaluate(x_polished).
        success (bool): SLSQP success flag.
        n_iter (int): Number of SLSQP iterations.
        message (str): SLSQP exit message.
    """

    source: str
    x_polished: NDArray[np.float64]
    eval_polished: EvalResult
    success: bool
    n_iter: int
    message: str


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Aggregate of all seeds, all polish results, and the winning design.

    Fields:
        winner_x (NDArray[np.float64]): Best design vector overall, shape (n_vars,).
        winner_eval (EvalResult): evaluate(winner_x).
        winner_origin (tuple[str, int]): (algorithm, seed) of the candidate
            that produced the winner before / after polish.
        all_seeds (tuple[SeedResult, ...]): Every seed's final result.
        all_polish (tuple[PolishResult, ...]): Every polish job's outcome.
        wall_clock_s (float): Total wall-clock seconds for the ensemble.
        feasible (bool): True iff the winner survived hard selection rule 2.
    """

    winner_x: NDArray[np.float64]
    winner_eval: EvalResult
    winner_origin: tuple[str, int]
    all_seeds: tuple[SeedResult, ...]
    all_polish: tuple[PolishResult, ...]
    wall_clock_s: float
    feasible: bool


# ------------------------- helpers -------------------------


def _eval_to_dict(er: EvalResult) -> dict:
    """Convert an EvalResult to a plain dict suitable for JSON serialization.

    Args:
        er (EvalResult): The evaluation result to convert.

    Returns:
        dict: Plain dictionary representation of the EvalResult.
    """
    return asdict(er)


def _eval_from_dict(d: dict) -> EvalResult:
    """Reconstruct an EvalResult from a plain dict (as loaded from JSON).

    Args:
        d (dict): Dictionary with EvalResult field names as keys.

    Returns:
        EvalResult: Reconstructed dataclass instance.
    """
    return EvalResult(
        tip_disp=float(d["tip_disp"]),
        max_stress=float(d["max_stress"]),
        max_buckling_ratio=float(d["max_buckling_ratio"]),
        min_length=float(d["min_length"]),
        stress_violations=tuple(float(v) for v in d["stress_violations"]),
        buckling_violations=tuple(float(v) for v in d["buckling_violations"]),
        length_violations=tuple(float(v) for v in d["length_violations"]),
        feasible=bool(d["feasible"]),
        solve_ok=bool(d["solve_ok"]),
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write a dict as JSON to path atomically (write to .tmp then rename).

    Args:
        path (Path): Destination file path.
        payload (dict): Data to serialize as JSON.

    Returns:
        None

    Notes:
        Uses a sibling .tmp file and os.replace semantics (via Path.replace)
        to avoid leaving a partial file on crash.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ------------------------- SeedResult IO -------------------------


def save_seed_result(sr: SeedResult, path: Path) -> None:
    """Serialize a SeedResult to JSON via atomic write.

    Args:
        sr (SeedResult): Result to serialize.
        path (Path): Destination JSON file.

    Returns:
        None
    """
    payload = {
        "algorithm": sr.algorithm,
        "seed": sr.seed,
        "best_x": sr.best_x.tolist(),
        "best_eval": _eval_to_dict(sr.best_eval),
        "best_penalty": sr.best_penalty,
        "history": [asdict(h) for h in sr.history],
        "wall_clock_s": sr.wall_clock_s,
        "checkpoint_path": str(sr.checkpoint_path),
    }
    _atomic_write_json(path, payload)


def load_seed_result(path: Path) -> SeedResult:
    """Deserialize a SeedResult from JSON.

    Args:
        path (Path): Source JSON file.

    Returns:
        SeedResult

    Raises:
        ValueError: If the file is not valid JSON or schema is invalid.
    """
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt checkpoint at {path}: {exc}") from exc
    return SeedResult(
        algorithm=str(d["algorithm"]),
        seed=int(d["seed"]),
        best_x=np.asarray(d["best_x"], dtype=np.float64),
        best_eval=_eval_from_dict(d["best_eval"]),
        best_penalty=float(d["best_penalty"]),
        history=tuple(
            HistoryPoint(
                generation=int(h["generation"]),
                best_penalty=float(h["best_penalty"]),
                mean_penalty=float(h["mean_penalty"]),
                n_feasible=int(h["n_feasible"]),
            )
            for h in d["history"]
        ),
        wall_clock_s=float(d["wall_clock_s"]),
        checkpoint_path=Path(d["checkpoint_path"]),
    )


def save_polish_result(pr: PolishResult, path: Path) -> None:
    """Serialize a PolishResult to JSON via atomic write.

    Args:
        pr (PolishResult): Result to serialize.
        path (Path): Destination JSON file.

    Returns:
        None
    """
    payload = {
        "source": pr.source,
        "x_polished": pr.x_polished.tolist(),
        "eval_polished": _eval_to_dict(pr.eval_polished),
        "success": pr.success,
        "n_iter": pr.n_iter,
        "message": pr.message,
    }
    _atomic_write_json(path, payload)


def load_polish_result(path: Path) -> PolishResult:
    """Deserialize a PolishResult from JSON.

    Args:
        path (Path): Source JSON file.

    Returns:
        PolishResult

    Raises:
        ValueError: If the file is not valid JSON or schema is invalid.
    """
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt checkpoint at {path}: {exc}") from exc
    return PolishResult(
        source=str(d["source"]),
        x_polished=np.asarray(d["x_polished"], dtype=np.float64),
        eval_polished=_eval_from_dict(d["eval_polished"]),
        success=bool(d["success"]),
        n_iter=int(d["n_iter"]),
        message=str(d["message"]),
    )


def save_ensemble_result(er: EnsembleResult, path: Path) -> None:
    """Serialize an EnsembleResult to JSON via atomic write.

    Args:
        er (EnsembleResult): Result to serialize.
        path (Path): Destination JSON file.

    Returns:
        None
    """
    payload = {
        "winner_x": er.winner_x.tolist(),
        "winner_eval": _eval_to_dict(er.winner_eval),
        "winner_origin": list(er.winner_origin),
        "all_seeds": [
            {
                "algorithm": s.algorithm,
                "seed": s.seed,
                "best_x": s.best_x.tolist(),
                "best_eval": _eval_to_dict(s.best_eval),
                "best_penalty": s.best_penalty,
                "history": [asdict(h) for h in s.history],
                "wall_clock_s": s.wall_clock_s,
                "checkpoint_path": str(s.checkpoint_path),
            }
            for s in er.all_seeds
        ],
        "all_polish": [
            {
                "source": p.source,
                "x_polished": p.x_polished.tolist(),
                "eval_polished": _eval_to_dict(p.eval_polished),
                "success": p.success,
                "n_iter": p.n_iter,
                "message": p.message,
            }
            for p in er.all_polish
        ],
        "wall_clock_s": er.wall_clock_s,
        "feasible": er.feasible,
    }
    _atomic_write_json(path, payload)


def load_ensemble_result(path: Path) -> EnsembleResult:
    """Deserialize an EnsembleResult from JSON.

    Args:
        path (Path): Source JSON file.

    Returns:
        EnsembleResult

    Raises:
        ValueError: If the file is not valid JSON or schema is invalid.
    """
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt checkpoint at {path}: {exc}") from exc
    return EnsembleResult(
        winner_x=np.asarray(d["winner_x"], dtype=np.float64),
        winner_eval=_eval_from_dict(d["winner_eval"]),
        winner_origin=(str(d["winner_origin"][0]), int(d["winner_origin"][1])),
        all_seeds=tuple(
            SeedResult(
                algorithm=str(s["algorithm"]),
                seed=int(s["seed"]),
                best_x=np.asarray(s["best_x"], dtype=np.float64),
                best_eval=_eval_from_dict(s["best_eval"]),
                best_penalty=float(s["best_penalty"]),
                history=tuple(
                    HistoryPoint(
                        generation=int(h["generation"]),
                        best_penalty=float(h["best_penalty"]),
                        mean_penalty=float(h["mean_penalty"]),
                        n_feasible=int(h["n_feasible"]),
                    )
                    for h in s["history"]
                ),
                wall_clock_s=float(s["wall_clock_s"]),
                checkpoint_path=Path(s["checkpoint_path"]),
            )
            for s in d["all_seeds"]
        ),
        all_polish=tuple(
            PolishResult(
                source=str(p["source"]),
                x_polished=np.asarray(p["x_polished"], dtype=np.float64),
                eval_polished=_eval_from_dict(p["eval_polished"]),
                success=bool(p["success"]),
                n_iter=int(p["n_iter"]),
                message=str(p["message"]),
            )
            for p in d["all_polish"]
        ),
        wall_clock_s=float(d["wall_clock_s"]),
        feasible=bool(d["feasible"]),
    )
