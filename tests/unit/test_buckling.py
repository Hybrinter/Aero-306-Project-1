"""Tests for Euler buckling analysis on TRUSS members."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fea_solver.models import (
    Element,
    ElementResult,
    ElementType,
    FEAModel,
    LinearConstraint,
    MaterialProperties,
    MemberBuckling,
    Mesh,
    Node,
)


class TestMemberBucklingDataclass:
    """MemberBuckling must be a frozen slotted dataclass with required fields."""

    def test_member_buckling_construction(self) -> None:
        """Construct MemberBuckling and verify all fields are readable."""
        mb = MemberBuckling(
            element_id=7,
            P_cr=1000.0,
            axial_force=-1200.0,
            ratio=1.2,
            is_buckled=True,
        )
        assert mb.element_id == 7
        assert mb.P_cr == 1000.0
        assert mb.axial_force == -1200.0
        assert mb.ratio == 1.2
        assert mb.is_buckled is True

    def test_member_buckling_is_frozen(self) -> None:
        """Frozen dataclass must raise on attribute assignment."""
        mb = MemberBuckling(element_id=1, P_cr=1.0, axial_force=0.0, ratio=0.0, is_buckled=False)
        with pytest.raises(Exception):
            mb.element_id = 99  # type: ignore[misc]


def _make_truss_element(
    element_id: int = 1,
    length: float = 1.0,
    E: float = 200e9,
    A: float = 1e-4,
    I: float = 1e-8,
) -> Element:
    """Build a single 2-node TRUSS element along x-axis for tests."""
    mat = MaterialProperties(E=E, A=A, I=I)
    n1 = Node(1, (0.0, 0.0))
    n2 = Node(2, (length, 0.0))
    return Element(id=element_id, node_i=n1, node_j=n2,
                   element_type=ElementType.TRUSS, material=mat)


class TestComputeMemberPCr:
    """compute_member_P_cr returns pi^2 * E * I / L^2 and guards I <= 0."""

    def test_known_value(self) -> None:
        """E=200e9, I=1e-8, L=1.0 yields P_cr = pi^2 * 200e9 * 1e-8 / 1 ~= 19739.2."""
        from fea_solver.buckling import compute_member_P_cr
        elem = _make_truss_element(E=200e9, I=1e-8, length=1.0)
        expected = math.pi**2 * 200e9 * 1e-8 / 1.0**2
        assert compute_member_P_cr(elem) == pytest.approx(expected, rel=1e-12)

    def test_length_scaling(self) -> None:
        """Doubling L divides P_cr by 4."""
        from fea_solver.buckling import compute_member_P_cr
        e1 = _make_truss_element(length=1.0)
        e2 = _make_truss_element(length=2.0)
        assert compute_member_P_cr(e1) / compute_member_P_cr(e2) == pytest.approx(4.0)

    def test_raises_on_zero_I(self) -> None:
        """I == 0 raises ValueError mentioning the element id."""
        from fea_solver.buckling import compute_member_P_cr
        elem = _make_truss_element(element_id=42, I=0.0)
        with pytest.raises(ValueError, match="42"):
            compute_member_P_cr(elem)

    def test_raises_on_negative_I(self) -> None:
        """I < 0 raises ValueError."""
        from fea_solver.buckling import compute_member_P_cr
        elem = _make_truss_element(I=-1.0)
        with pytest.raises(ValueError):
            compute_member_P_cr(elem)
