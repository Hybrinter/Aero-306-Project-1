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
