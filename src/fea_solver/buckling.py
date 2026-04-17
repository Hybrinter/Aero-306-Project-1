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
