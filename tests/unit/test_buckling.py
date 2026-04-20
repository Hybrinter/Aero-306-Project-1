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


def _make_truss_model_two_elements() -> FEAModel:
    """Build a 3-node / 2-TRUSS-element model for compute_truss_buckling tests."""
    mat = MaterialProperties(E=200e9, A=1e-4, I=1e-8)
    n1 = Node(1, (0.0, 0.0))
    n2 = Node(2, (1.0, 0.0))
    n3 = Node(3, (2.0, 0.0))
    e1 = Element(id=1, node_i=n1, node_j=n2,
                 element_type=ElementType.TRUSS, material=mat)
    e2 = Element(id=2, node_i=n2, node_j=n3,
                 element_type=ElementType.TRUSS, material=mat)
    return FEAModel(
        mesh=Mesh(nodes=(n1, n2, n3), elements=(e1, e2)),
        boundary_conditions=(
            LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0)),
            LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0)),
        ),
        nodal_loads=(),
        distributed_loads=(),
        label="buckle_fixture",
    )


def _make_element_result(element_id: int, axial_force: float) -> ElementResult:
    """Produce an ElementResult with the given axial force; other fields zeroed."""
    n = 5
    return ElementResult(
        element_id=element_id,
        axial_force=axial_force,
        shear_forces=np.zeros(n),
        bending_moments=np.zeros(n),
        x_stations=np.linspace(0.0, 1.0, n),
        transverse_displacements=np.zeros(n),
        axial_displacements=np.zeros(n),
        rotations=np.zeros(n),
    )


class TestComputeTrussBuckling:
    """Per-element classification: BUCKLED / SAFE / TENSION."""

    def test_compression_buckled(self) -> None:
        """N=-2*P_cr marks is_buckled True and ratio=2.0."""
        from fea_solver.buckling import compute_member_P_cr, compute_truss_buckling
        model = _make_truss_model_two_elements()
        P_cr = compute_member_P_cr(model.mesh.elements[0])
        results = [
            _make_element_result(1, -2.0 * P_cr),
            _make_element_result(2, +P_cr * 0.5),
        ]
        bucklings = compute_truss_buckling(model, results)
        assert len(bucklings) == 2
        by_id = {mb.element_id: mb for mb in bucklings}
        assert by_id[1].is_buckled is True
        assert by_id[1].ratio == pytest.approx(2.0)

    def test_compression_safe(self) -> None:
        """N=-0.5*P_cr (compressive but below threshold) yields is_buckled=False, ratio=0.5."""
        from fea_solver.buckling import compute_member_P_cr, compute_truss_buckling
        model = _make_truss_model_two_elements()
        P_cr = compute_member_P_cr(model.mesh.elements[0])
        results = [
            _make_element_result(1, -0.5 * P_cr),
            _make_element_result(2, 0.0),
        ]
        bucklings = compute_truss_buckling(model, results)
        by_id = {mb.element_id: mb for mb in bucklings}
        assert by_id[1].is_buckled is False
        assert by_id[1].ratio == pytest.approx(0.5)
        assert by_id[2].is_buckled is False
        assert by_id[2].ratio == pytest.approx(0.0)

    def test_tension_marked_safe(self) -> None:
        """Positive N produces ratio=0.0 and is_buckled=False irrespective of magnitude."""
        from fea_solver.buckling import compute_truss_buckling
        model = _make_truss_model_two_elements()
        results = [
            _make_element_result(1, 1.0e20),
            _make_element_result(2, 1.0),
        ]
        bucklings = compute_truss_buckling(model, results)
        for mb in bucklings:
            assert mb.ratio == 0.0
            assert mb.is_buckled is False
            assert mb.axial_force > 0.0

    def test_exactly_critical_compressive_is_buckled(self) -> None:
        """N = -P_cr (equality) marks is_buckled True; ratio = 1.0 exactly."""
        from fea_solver.buckling import compute_member_P_cr, compute_truss_buckling
        model = _make_truss_model_two_elements()
        P_cr = compute_member_P_cr(model.mesh.elements[0])
        results = [
            _make_element_result(1, -P_cr),
            _make_element_result(2, 0.0),
        ]
        by_id = {mb.element_id: mb for mb in compute_truss_buckling(model, results)}
        assert by_id[1].is_buckled is True
        assert by_id[1].ratio == pytest.approx(1.0)

    def test_non_truss_skipped(self) -> None:
        """Mixed mesh: BAR elements produce no MemberBuckling entry."""
        from fea_solver.buckling import compute_truss_buckling
        mat = MaterialProperties(E=200e9, A=1e-4, I=1e-8)
        n1 = Node(1, (0.0, 0.0))
        n2 = Node(2, (1.0, 0.0))
        e_truss = Element(id=10, node_i=n1, node_j=n2,
                          element_type=ElementType.TRUSS, material=mat)
        e_bar = Element(id=11, node_i=n1, node_j=n2,
                        element_type=ElementType.BAR, material=mat)
        model = FEAModel(
            mesh=Mesh(nodes=(n1, n2), elements=(e_truss, e_bar)),
            boundary_conditions=(),
            nodal_loads=(),
            distributed_loads=(),
            label="mixed",
        )
        results = [_make_element_result(10, -1.0), _make_element_result(11, -1.0)]
        bucklings = compute_truss_buckling(model, results)
        assert len(bucklings) == 1
        assert bucklings[0].element_id == 10

    def test_missing_element_result_skipped(self) -> None:
        """A TRUSS element with no matching ElementResult is silently omitted."""
        from fea_solver.buckling import compute_truss_buckling
        model = _make_truss_model_two_elements()
        results = [_make_element_result(1, -100.0)]  # element 2 missing
        bucklings = compute_truss_buckling(model, results)
        assert len(bucklings) == 1
        assert bucklings[0].element_id == 1

    def test_returns_tuple(self) -> None:
        """Return type is tuple (immutable), not list."""
        from fea_solver.buckling import compute_truss_buckling
        model = _make_truss_model_two_elements()
        results = [_make_element_result(1, 0.0), _make_element_result(2, 0.0)]
        assert isinstance(compute_truss_buckling(model, results), tuple)

    def test_zero_I_element_skipped_with_warning(self, caplog) -> None:
        """A TRUSS element with material.I <= 0 is skipped and a warning is logged."""
        import logging

        from fea_solver.buckling import compute_truss_buckling
        mat_good = MaterialProperties(E=200e9, A=1e-4, I=1e-8)
        mat_bad = MaterialProperties(E=200e9, A=1e-4, I=0.0)
        n1 = Node(1, (0.0, 0.0))
        n2 = Node(2, (1.0, 0.0))
        n3 = Node(3, (2.0, 0.0))
        e_good = Element(id=1, node_i=n1, node_j=n2,
                         element_type=ElementType.TRUSS, material=mat_good)
        e_bad = Element(id=2, node_i=n2, node_j=n3,
                        element_type=ElementType.TRUSS, material=mat_bad)
        model = FEAModel(
            mesh=Mesh(nodes=(n1, n2, n3), elements=(e_good, e_bad)),
            boundary_conditions=(),
            nodal_loads=(),
            distributed_loads=(),
            label="partial_I",
        )
        results = [_make_element_result(1, -1.0), _make_element_result(2, -1.0)]

        with caplog.at_level(logging.WARNING, logger="fea_solver.buckling"):
            bucklings = compute_truss_buckling(model, results)

        assert len(bucklings) == 1
        assert bucklings[0].element_id == 1
        assert any("Element 2" in rec.message and "buckling skipped" in rec.message
                   for rec in caplog.records)
