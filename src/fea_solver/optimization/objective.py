"""Single-FE-solve objective evaluator for geometry optimization.

evaluate(x, problem) runs exactly one forward FE solve and packages every
metric the global penalty wrapper and the SLSQP constraint callables need
into a single EvalResult. Failure modes (coincident nodes, near-singular
stiffness) are caught and converted to a finite sentinel so optimizer
workers never crash.

EvalResult:           Frozen container for per-call FE metrics and violations.
SENTINEL_TIP_DISP:    Tip-disp value used when solve_ok is False; finite and
                      huge so the penalty wrapper produces 1e12-scale fitness.
evaluate:             One FE solve -> EvalResult, with degeneracy guards.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.buckling import compute_member_P_cr
from fea_solver.models import DOFType, ElementType
from fea_solver.optimization.problem import GeometryOptimizationProblem, apply_x_to_model
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.solver import run_solve_pipeline

logger = logging.getLogger(__name__)

SENTINEL_TIP_DISP: float = 1.0e12


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Per-call FE metrics and constraint violations.

    Fields:
        tip_disp (float): |v_y(load_node)| in the model's length unit.
            Set to SENTINEL_TIP_DISP when solve_ok is False.
        max_stress (float): max |sigma_e| across all elements (MPa for the
            problem_7 unit system; generally pressure units of the model).
        max_buckling_ratio (float): max |N_e|/P_cr_e across compression members,
            0.0 if no compressive members.
        min_length (float): min L_e across all elements.
        stress_violations (tuple[float, ...]): max(0, |sigma_e|/sigma_max - 1)
            per element, in element id order.
        buckling_violations (tuple[float, ...]): max(0, |N_e|/P_cr_e - 1)
            per element, 0.0 for tension members, in element id order.
        length_violations (tuple[float, ...]): max(0, 1 - L_e/L_min) per element.
        feasible (bool): True iff solve_ok and all violations are zero.
        solve_ok (bool): True iff the FE solve completed and no element had
            length below 1e-6 (model length unit).
    """

    tip_disp: float
    max_stress: float
    max_buckling_ratio: float
    min_length: float
    stress_violations: tuple[float, ...]
    buckling_violations: tuple[float, ...]
    length_violations: tuple[float, ...]
    feasible: bool
    solve_ok: bool


def _sentinel_result(n_elements: int) -> EvalResult:
    """Return an EvalResult representing a degenerate (unsolvable) configuration.

    Args:
        n_elements (int): Number of elements in the model.

    Returns:
        EvalResult: tip_disp=SENTINEL_TIP_DISP, all violations large, infeasible.
    """
    huge = (1.0,) * n_elements
    return EvalResult(
        tip_disp=SENTINEL_TIP_DISP,
        max_stress=SENTINEL_TIP_DISP,
        max_buckling_ratio=SENTINEL_TIP_DISP,
        min_length=0.0,
        stress_violations=huge,
        buckling_violations=huge,
        length_violations=huge,
        feasible=False,
        solve_ok=False,
    )


def evaluate(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> EvalResult:
    """Run one forward FE solve and return all metrics + violations.

    Steps:
      1. apply_x_to_model -> FEAModel.
      2. Guard: any element length < 1e-6 -> sentinel result.
      3. Try assembly + solve + postprocess. On any exception -> sentinel.
      4. Compute per-element stress, buckling ratio (compression only), and
         length violations relative to problem thresholds.
      5. Pack into EvalResult.

    Args:
        x (NDArray[np.float64]): Design vector, shape (problem.n_vars,).
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        EvalResult: All metrics for this candidate.

    Notes:
        Stress is computed from axial force as sigma = N / A on each element.
        For compression members (N < 0), buckling ratio = |N| / P_cr.
        For tension members, buckling ratio is 0.0 (no contribution).
    """
    n_elements = len(problem.baseline_model.mesh.elements)
    try:
        model = apply_x_to_model(x, problem)
    except (ValueError, ZeroDivisionError):
        logger.debug("evaluate: apply_x_to_model failed, returning sentinel")
        return _sentinel_result(n_elements)

    # Length guard
    lengths = [e.length for e in model.mesh.elements]
    if min(lengths) < 1.0e-6:
        logger.debug("evaluate: degenerate edge length, returning sentinel")
        return _sentinel_result(n_elements)

    try:
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        element_results = postprocess_all_elements(model, result, n_stations=2)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as exc:
        logger.debug("evaluate: solve failed (%s), returning sentinel", exc)
        return _sentinel_result(n_elements)

    # Tip displacement
    try:
        tip_disp = abs(float(
            result.displacements[dof_map.index(problem.load_node_id, DOFType.V)]
        ))
    except KeyError:
        return _sentinel_result(n_elements)

    # Per-element stress and buckling ratio (in element id order)
    er_by_id = {er.element_id: er for er in element_results}
    stress_violations: list[float] = []
    buckling_violations: list[float] = []
    length_violations: list[float] = []
    max_stress = 0.0
    max_buck = 0.0
    for elem in model.mesh.elements:
        er = er_by_id[elem.id]
        sigma = abs(er.axial_force) / elem.material.A
        max_stress = max(max_stress, sigma)
        s_v = max(0.0, sigma / problem.sigma_max - 1.0)
        stress_violations.append(s_v)

        if elem.element_type == ElementType.TRUSS and er.axial_force < 0.0 and elem.material.I > 0.0:
            P_cr = compute_member_P_cr(elem)
            ratio = abs(er.axial_force) / P_cr
            max_buck = max(max_buck, ratio)
            buckling_violations.append(max(0.0, ratio - 1.0))
        else:
            buckling_violations.append(0.0)

        length_violations.append(max(0.0, 1.0 - elem.length / problem.L_min))

    min_length = min(lengths)
    feasible = (
        all(v == 0.0 for v in stress_violations)
        and all(v == 0.0 for v in buckling_violations)
        and all(v == 0.0 for v in length_violations)
    )

    return EvalResult(
        tip_disp=tip_disp,
        max_stress=max_stress,
        max_buckling_ratio=max_buck,
        min_length=min_length,
        stress_violations=tuple(stress_violations),
        buckling_violations=tuple(buckling_violations),
        length_violations=tuple(length_violations),
        feasible=feasible,
        solve_ok=True,
    )
