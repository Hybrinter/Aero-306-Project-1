"""Euler buckling analysis for 2D pin-jointed TRUSS members.

Provides pure functions (no state) that compute the classical Euler critical
load P_cr = pi^2 * E * I / L^2 for each TRUSS element and flag compressive
members whose axial force magnitude meets or exceeds P_cr.

Pin-pin end conditions are implicit (effective length factor K = 1). Non-TRUSS
elements are skipped. See
docs/superpowers/specs/2026-04-17-truss-buckling-design.md for the formulation.

compute_member_P_cr:     P_cr for a single TRUSS element; raises on I <= 0.
compute_truss_buckling:  Iterate TRUSS elements in a model and build one
                         MemberBuckling per element by combining P_cr with the
                         axial force from the matching ElementResult.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from fea_solver.models import (
    Element,
    ElementResult,
    ElementType,
    FEAModel,
    MemberBuckling,
)

logger = logging.getLogger(__name__)


def compute_member_P_cr(element: Element) -> float:
    """Compute Euler critical load for one element.

    P_cr = pi^2 * E * I / L^2

    Args:
        element (Element): Any element with E, I, and length defined. The caller
            is responsible for filtering to TRUSS-only if desired.

    Returns:
        float: Critical load P_cr in the canonical force units of the model.
            Always positive.

    Raises:
        ValueError: If element.material.I <= 0. Guards placeholder I values in
            YAML inputs that would otherwise yield a zero or negative P_cr.

    Notes:
        Assumes pin-pin end conditions (K = 1, effective length = L).
    """
    I = element.material.I
    if I <= 0.0:
        raise ValueError(
            f"Element {element.id}: I must be > 0 for buckling (got {I})"
        )
    E = element.material.E
    L = element.length
    P_cr = math.pi**2 * E * I / (L * L)
    logger.debug("Element %d: P_cr = %.4e (E=%.3e, I=%.3e, L=%.3e)",
                 element.id, P_cr, E, I, L)
    return float(P_cr)


def compute_truss_buckling(
    model: FEAModel,
    element_results: Sequence[ElementResult],
) -> tuple[MemberBuckling, ...]:
    """Classify every TRUSS element's buckling status.

    For each element with element_type == TRUSS, compute P_cr via
    compute_member_P_cr, look up the matching ElementResult by element_id,
    and build a MemberBuckling. Non-TRUSS elements and elements without a
    matching ElementResult are skipped silently.

    Args:
        model (FEAModel): FEA problem containing the mesh.
        element_results (Sequence[ElementResult]): Post-processed results
            (typically from postprocess_all_elements). Only axial_force is read.

    Returns:
        tuple[MemberBuckling, ...]: One entry per TRUSS element in the same
            order as model.mesh.elements. Empty tuple if no TRUSS elements
            are present.

    Notes:
        Classification rules (|N| = abs(axial_force)):
          * axial_force >= 0  -> ratio = 0.0, is_buckled = False (tension/zero).
          * axial_force <  0  -> ratio = |N| / P_cr,
                                 is_buckled = (|N| >= P_cr).
        Tension members are retained in the tuple so the reporter can distinguish
        TENSION from SAFE compressive members by sign.
    """
    result_by_id: dict[int, ElementResult] = {er.element_id: er for er in element_results}
    out: list[MemberBuckling] = []
    for element in model.mesh.elements:
        if element.element_type != ElementType.TRUSS:
            continue
        er = result_by_id.get(element.id)
        if er is None:
            continue
        P_cr = compute_member_P_cr(element)
        N = er.axial_force
        if N < 0.0:
            ratio = abs(N) / P_cr
            is_buckled = abs(N) >= P_cr
        else:
            ratio = 0.0
            is_buckled = False
        out.append(MemberBuckling(
            element_id=element.id,
            P_cr=P_cr,
            axial_force=float(N),
            ratio=float(ratio),
            is_buckled=is_buckled,
        ))
        logger.debug(
            "Element %d: P_cr=%.4e, N=%.4e, ratio=%.3f, buckled=%s",
            element.id, P_cr, N, ratio, is_buckled,
        )
    return tuple(out)
