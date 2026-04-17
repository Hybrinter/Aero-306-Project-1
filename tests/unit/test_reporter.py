"""Tests for reporter.py -- rich table output functions."""
from __future__ import annotations

import numpy as np
import pytest

from fea_solver.models import (
    DOFMap,
    DOFType,
    Element,
    ElementResult,
    ElementType,
    FEAModel,
    LinearConstraint,
    MaterialProperties,
    Mesh,
    Node,
    SolutionResult,
)


def _make_simple_model() -> FEAModel:
    """Build a minimal two-node bar FEAModel for reporter tests."""
    mat = MaterialProperties(E=200e9, A=1e-4, I=1e-8)
    n1 = Node(1, (0.0, 0.0))
    n2 = Node(2, (1.0, 0.0))
    e = Element(id=1, node_i=n1, node_j=n2,
                element_type=ElementType.BAR, material=mat)
    return FEAModel(
        mesh=Mesh(nodes=(n1, n2), elements=(e,)),
        boundary_conditions=(),
        nodal_loads=(),
        distributed_loads=(),
        label="test",
    )


class TestPrintBucklingSummary:
    """Tests for print_buckling_summary rich table output."""

    def _mb(self, element_id: int, P_cr: float, N: float) -> "MemberBuckling":
        from fea_solver.models import MemberBuckling
        if N < 0:
            ratio = abs(N) / P_cr
            is_buckled = abs(N) >= P_cr
        else:
            ratio = 0.0
            is_buckled = False
        return MemberBuckling(
            element_id=element_id, P_cr=P_cr, axial_force=N,
            ratio=ratio, is_buckled=is_buckled,
        )

    def _make_model(self):
        from fea_solver.models import (
            Element, ElementType, FEAModel, MaterialProperties,
            Mesh, Node,
        )
        mat = MaterialProperties(E=200e9, A=1e-4, I=1e-8)
        n1 = Node(1, (0.0, 0.0))
        n2 = Node(2, (1.0, 0.0))
        e = Element(id=1, node_i=n1, node_j=n2,
                    element_type=ElementType.TRUSS, material=mat)
        return FEAModel(
            mesh=Mesh(nodes=(n1, n2), elements=(e,)),
            boundary_conditions=(),
            nodal_loads=(),
            distributed_loads=(),
            label="t",
        )

    def test_empty_bucklings_prints_nothing(self, capsys) -> None:
        """Empty tuple produces no console output."""
        from fea_solver.reporter import print_buckling_summary
        print_buckling_summary((), self._make_model())
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_mixed_statuses_renders_all_three_tokens(self, capsys) -> None:
        """Output contains BUCKLED, SAFE, and TENSION tokens for a mixed input."""
        from fea_solver.reporter import print_buckling_summary
        bucklings = (
            self._mb(1, P_cr=100.0, N=-200.0),  # BUCKLED
            self._mb(2, P_cr=100.0, N=-50.0),   # SAFE
            self._mb(3, P_cr=100.0, N=+75.0),   # TENSION
        )
        print_buckling_summary(bucklings, self._make_model())
        out = capsys.readouterr().out
        assert "BUCKLED" in out
        assert "SAFE" in out
        assert "TENSION" in out
        # Ratio formatting: BUCKLED member has ratio 2.0, expect "2.00" token.
        assert "2.00" in out
