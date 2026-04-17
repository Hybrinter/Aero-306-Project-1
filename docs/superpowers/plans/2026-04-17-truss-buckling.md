# Truss Euler Buckling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute Euler buckling load for every TRUSS member, flag compressive members whose axial force exceeds P_cr, overlay a half-sine mode shape on each buckled member in the deformed-truss plot, and emit a dedicated console summary table per solution.

**Architecture:** New frozen dataclass `MemberBuckling` in `models.py`; new module `buckling.py` with pure functions (no state); existing `plot_truss_deformed` gains one optional kwarg; new reporter function; 3-line wire-up in `main.py`. Non-truss code is untouched.

**Tech Stack:** Python 3.12, numpy, matplotlib, rich (console/table), pytest

---

## File Map

| File | Change |
|------|--------|
| `src/fea_solver/models.py` | Add `MemberBuckling` frozen dataclass |
| `src/fea_solver/buckling.py` | New module: `compute_member_P_cr`, `compute_truss_buckling` |
| `src/fea_solver/plotter.py` | Add optional `buckling` kwarg to `plot_truss_deformed`; import `MemberBuckling` |
| `src/fea_solver/reporter.py` | New function `print_buckling_summary`; import `MemberBuckling` |
| `main.py` | Compute bucklings per truss solution; pass to reporter + plotter |
| `tests/unit/test_buckling.py` | New: unit tests for pure buckling functions |
| `tests/unit/test_plotter.py` | Add one test for new kwarg |
| `tests/unit/test_reporter.py` | Add two tests for summary function |
| `tests/integration/test_truss_buckling.py` | New: full pipeline on `problem_7.yaml` |

---

### Task 1: Add `MemberBuckling` dataclass to `models.py`

**Files:**
- Modify: `src/fea_solver/models.py` (append new dataclass after `SolutionSeries`, near end of file)
- Test: `tests/unit/test_buckling.py` (new file)

- [ ] **Step 1: Create the failing test file**

Create `tests/unit/test_buckling.py` with content:

```python
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
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_buckling.py::TestMemberBucklingDataclass -v`

Expected: `ImportError: cannot import name 'MemberBuckling' from 'fea_solver.models'`

- [ ] **Step 3: Add the dataclass to `models.py`**

Append to the very end of `src/fea_solver/models.py` (after `SolutionSeries`):

```python


# ---------------------------------------------------------------------------
# Buckling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemberBuckling:
    """Euler buckling result for one TRUSS member.

    Fields:
        element_id (int): Identifier of the TRUSS element this result belongs to.
        P_cr (float): Euler critical load pi^2 * E * I / L^2 [force units].
            Always positive.
        axial_force (float): Signed axial force [force units] copied from
            ElementResult.axial_force. Positive = tension, negative = compression.
        ratio (float): abs(axial_force) / P_cr when axial_force < 0; 0.0 when
            axial_force >= 0. Dimensionless.
        is_buckled (bool): True iff axial_force < 0 and abs(axial_force) >= P_cr.

    Notes:
        Frozen and slotted; immutable after construction. Produced only for
        elements with element_type == TRUSS. Tension members are included in
        the returned tuple with ratio = 0.0 and is_buckled = False so that the
        downstream reporter can distinguish TENSION from SAFE compressive
        members by the sign of axial_force.
    """

    element_id: int
    P_cr: float
    axial_force: float
    ratio: float
    is_buckled: bool
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_buckling.py::TestMemberBucklingDataclass -v`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fea_solver/models.py tests/unit/test_buckling.py
git commit -m "feat: add MemberBuckling dataclass"
```

---

### Task 2: Add `compute_member_P_cr` to new `buckling.py`

**Files:**
- Create: `src/fea_solver/buckling.py`
- Modify: `tests/unit/test_buckling.py` (append new class)

- [ ] **Step 1: Append failing test class to `tests/unit/test_buckling.py`**

```python


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
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_buckling.py::TestComputeMemberPCr -v`

Expected: `ModuleNotFoundError: No module named 'fea_solver.buckling'`

- [ ] **Step 3: Create `src/fea_solver/buckling.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_buckling.py::TestComputeMemberPCr -v`

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fea_solver/buckling.py tests/unit/test_buckling.py
git commit -m "feat: add compute_member_P_cr to new buckling module"
```

---

### Task 3: Add `compute_truss_buckling` to `buckling.py`

**Files:**
- Modify: `src/fea_solver/buckling.py` (append function)
- Modify: `tests/unit/test_buckling.py` (append new class)

- [ ] **Step 1: Append failing tests to `tests/unit/test_buckling.py`**

```python


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
```

- [ ] **Step 2: Run to confirm failures**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_buckling.py::TestComputeTrussBuckling -v`

Expected: `ImportError: cannot import name 'compute_truss_buckling' from 'fea_solver.buckling'`

- [ ] **Step 3: Append `compute_truss_buckling` to `src/fea_solver/buckling.py`**

Add at the bottom of the file:

```python


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
```

- [ ] **Step 4: Run all buckling tests to confirm they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_buckling.py -v`

Expected: `13 passed` (2 from Task 1 + 4 from Task 2 + 7 from Task 3)

- [ ] **Step 5: Commit**

```bash
git add src/fea_solver/buckling.py tests/unit/test_buckling.py
git commit -m "feat: add compute_truss_buckling per-member classifier"
```

---

### Task 4: Add `print_buckling_summary` to `reporter.py`

**Files:**
- Modify: `src/fea_solver/reporter.py`
- Modify: `tests/unit/test_reporter.py`

- [ ] **Step 1: Inspect the existing reporter tests so our new ones match style**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_reporter.py -v --collect-only | head -30`

Note the existing helpers and the `capsys` pattern used there (if none, we use a fresh `Console` with `record=True` via monkeypatching).

- [ ] **Step 2: Append failing tests to `tests/unit/test_reporter.py`**

Add at the bottom of `tests/unit/test_reporter.py`:

```python


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
```

- [ ] **Step 3: Run the failing tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_reporter.py::TestPrintBucklingSummary -v`

Expected: `ImportError: cannot import name 'print_buckling_summary' from 'fea_solver.reporter'`

- [ ] **Step 4: Add `print_buckling_summary` to `reporter.py`**

Add to `src/fea_solver/reporter.py`. First, extend the imports near the top:

```python
from fea_solver.models import (
    DOFMap,
    DOFType,
    ElementResult,
    ElementType,
    FEAModel,
    MemberBuckling,
    SolutionResult,
)
```

Then append this function to the end of the file:

```python


def print_buckling_summary(
    bucklings: "tuple[MemberBuckling, ...]",
    model: FEAModel,
) -> None:
    """Print a rich table summarizing Euler buckling status per TRUSS member.

    Columns:
        Element      -- element id
        N            -- signed axial force [force unit]
        P_cr         -- Euler critical load [force unit]
        |N|/P_cr     -- ratio formatted as %.2f (0.00 for tension)
        Status       -- BUCKLED (bold red) / SAFE (green) / TENSION (dim)

    Args:
        bucklings (tuple[MemberBuckling, ...]): Output of compute_truss_buckling.
            Empty tuple produces no output at all.
        model (FEAModel): Used for the force-unit label in column headers.

    Returns:
        None

    Notes:
        Uses the module-level rich Console (_console). Tension members are
        included in the table so the user can see every member on one sheet.
    """
    if not bucklings:
        return

    force_unit = _lbl(model)["force"]

    table = Table(title="Buckling Summary", show_header=True, header_style="bold")
    table.add_column("Element", justify="right")
    table.add_column(f"N [{force_unit}]", justify="right")
    table.add_column(f"P_cr [{force_unit}]", justify="right")
    table.add_column("|N|/P_cr", justify="right")
    table.add_column("Status", justify="center")

    for mb in bucklings:
        if mb.is_buckled:
            status = "[bold red]BUCKLED[/bold red]"
        elif mb.axial_force < 0.0:
            status = "[green]SAFE[/green]"
        else:
            status = "[dim]TENSION[/dim]"
        table.add_row(
            str(mb.element_id),
            f"{mb.axial_force:.3e}",
            f"{mb.P_cr:.3e}",
            f"{mb.ratio:.2f}",
            status,
        )

    _console.print(table)
    logger.info("Buckling summary printed for %d members", len(bucklings))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_reporter.py::TestPrintBucklingSummary -v`

Expected: `2 passed`

If the test fails because rich markup like `[bold red]BUCKLED[/bold red]` is not rendered as `BUCKLED` in the captured output, tweak the test to search for the raw tag content rather than the rendered token. Rich with `Console()` (default) prints rendered output; the markup is stripped and only `BUCKLED` remains in plain text. Verify by running with `-s` if needed.

- [ ] **Step 6: Commit**

```bash
git add src/fea_solver/reporter.py tests/unit/test_reporter.py
git commit -m "feat: add print_buckling_summary reporter function"
```

---

### Task 5: Extend `plot_truss_deformed` with optional `buckling` kwarg

**Files:**
- Modify: `src/fea_solver/plotter.py`
- Modify: `tests/unit/test_plotter.py`

- [ ] **Step 1: Append failing test to `tests/unit/test_plotter.py`**

Add to the bottom of `tests/unit/test_plotter.py`:

```python


class TestTrussDeformedWithBuckling:
    """plot_truss_deformed gains an optional buckling overlay."""

    def test_default_kwarg_is_none_backward_compatible(self) -> None:
        """Calling without buckling kwarg yields the same figure as before."""
        from fea_solver.plotter import plot_truss_deformed
        sol = _make_truss_series(axial_force=-100.0)
        fig = plot_truss_deformed(sol)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_overlay_draws_extra_line_per_buckled_member(self) -> None:
        """One extra Line2D artist per buckled member appears on the axes."""
        from fea_solver.models import MemberBuckling
        from fea_solver.plotter import plot_truss_deformed
        sol = _make_truss_series(axial_force=-1000.0)

        # Count baseline lines (no buckling overlay).
        fig0 = plot_truss_deformed(sol)
        n0 = len(fig0.axes[0].get_lines())
        plt.close(fig0)

        # Provide a single buckled member for element id 1.
        mb = MemberBuckling(
            element_id=1, P_cr=500.0, axial_force=-1000.0,
            ratio=2.0, is_buckled=True,
        )
        fig1 = plot_truss_deformed(sol, buckling=(mb,))
        n1 = len(fig1.axes[0].get_lines())
        plt.close(fig1)

        assert n1 == n0 + 1, (
            f"Expected one extra Line2D for the buckled bow "
            f"(baseline {n0}, with overlay {n1})."
        )

    def test_non_buckled_entry_adds_no_overlay(self) -> None:
        """A MemberBuckling with is_buckled=False produces no extra lines."""
        from fea_solver.models import MemberBuckling
        from fea_solver.plotter import plot_truss_deformed
        sol = _make_truss_series(axial_force=-10.0)

        fig0 = plot_truss_deformed(sol)
        n0 = len(fig0.axes[0].get_lines())
        plt.close(fig0)

        mb = MemberBuckling(
            element_id=1, P_cr=500.0, axial_force=-10.0,
            ratio=0.02, is_buckled=False,
        )
        fig1 = plot_truss_deformed(sol, buckling=(mb,))
        n1 = len(fig1.axes[0].get_lines())
        plt.close(fig1)

        assert n1 == n0
```

- [ ] **Step 2: Run the tests to confirm failure**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestTrussDeformedWithBuckling -v`

Expected: `TypeError: plot_truss_deformed() got an unexpected keyword argument 'buckling'` on the second/third tests. The first (backward-compat) may already pass.

- [ ] **Step 3: Extend `plot_truss_deformed` in `src/fea_solver/plotter.py`**

First, extend the imports in `plotter.py`:

```python
from fea_solver.models import DOFType, ElementResult, FEAModel, MemberBuckling, SolutionSeries
```

Update `plot_truss_deformed`'s signature and body. Replace the existing function with:

```python
def plot_truss_deformed(
    sol: SolutionSeries,
    title: str = "Truss Deformed Shape",
    output_path: Path | None = None,
    buckling: tuple[MemberBuckling, ...] | None = None,
) -> plt.Figure:
    """Plot 2D truss geometry in its deformed state with coolwarm gradient by axial force.

    Node positions are shifted by (scale * U, scale * V) where scale is chosen
    automatically so the largest displacement equals 10% of the bounding-box
    diagonal. The scale factor is appended to the plot title.

    When buckling is provided, each entry whose is_buckled == True draws a
    half-sine lateral bow on top of the deformed member line. The bow has
    amplitude 0.1 * element.length (original undeformed length) along the unit
    perpendicular to the deformed-member axis; it is drawn as a black dashed
    line so it reads as a buckling-mode indicator rather than additional
    geometry.

    Args:
        sol (SolutionSeries): Solution bundle. sol.result.displacements provides
            nodal translations; sol.element_results provides axial forces for color.
        title (str): Base plot title. Scale factor is appended automatically.
            Default "Truss Deformed Shape".
        output_path (Path | None): If provided, save figure to this path as PNG.
        buckling (tuple[MemberBuckling, ...] | None): Optional per-element
            buckling results. None (default) preserves the prior behaviour
            exactly. When provided, members with is_buckled=True receive a
            half-sine overlay.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        Scale factor formula: scale = 0.1 * bbox_diagonal / max_abs_displacement.
        Falls back to scale = 1.0 when all displacements are zero.
        bbox_diagonal = hypot(max_x - min_x, max_y - min_y) over original node coords.
        If output_path is provided, saves figure at 150 dpi.
    """
    model = sol.model
    lbl = _unit_labels(model)
    result_by_id = {er.element_id: er for er in sol.element_results}
    nodes_by_id = {n.id: n for n in model.mesh.nodes}
    node_disps = _truss_node_displacements(sol)
    buckled_ids: set[int] = (
        {mb.element_id for mb in buckling if mb.is_buckled}
        if buckling is not None else set()
    )

    # Auto-compute displacement scale factor
    xs = np.array([n.x for n in model.mesh.nodes])
    ys = np.array([n.y for n in model.mesh.nodes])
    bbox_diag = float(np.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min())))
    if bbox_diag == 0.0:
        bbox_diag = 1.0
    all_disp_mags = [abs(d) for U, V in node_disps.values() for d in (U, V)]
    max_disp = max(all_disp_mags) if all_disp_mags else 0.0
    scale = 0.1 * bbox_diag / max_disp if max_disp > 0.0 else 1.0

    forces = [
        result_by_id[e.id].axial_force if e.id in result_by_id else 0.0
        for e in model.mesh.elements
    ]
    cmap, norm = _truss_colormap_norm(forces)

    fig, ax = plt.subplots(figsize=(10, 6))

    for element, N in zip(model.mesh.elements, forces):
        n_i = nodes_by_id[element.node_i.id]
        n_j = nodes_by_id[element.node_j.id]
        U_i, V_i = node_disps[n_i.id]
        U_j, V_j = node_disps[n_j.id]
        x_i_def = n_i.x + scale * U_i
        y_i_def = n_i.y + scale * V_i
        x_j_def = n_j.x + scale * U_j
        y_j_def = n_j.y + scale * V_j
        color = cmap(norm(N))
        ax.plot([x_i_def, x_j_def], [y_i_def, y_j_def], color=color, linewidth=2.5)
        mid_x = (x_i_def + x_j_def) / 2.0
        mid_y = (y_i_def + y_j_def) / 2.0
        ax.text(mid_x, mid_y, f"{N:.3g}", fontsize=7, ha="center", va="bottom",
                color="black")

        # Buckling mode overlay (half-sine bow along deformed-member perpendicular)
        if element.id in buckled_ids:
            dx = x_j_def - x_i_def
            dy = y_j_def - y_i_def
            chord = float(np.hypot(dx, dy))
            if chord > 0.0:
                cos_a = dx / chord
                sin_a = dy / chord
                amp = 0.1 * element.length  # undeformed length per spec
                xi = np.linspace(0.0, 1.0, 30)
                bow = amp * np.sin(np.pi * xi)
                bx = x_i_def + xi * dx + bow * (-sin_a)
                by = y_i_def + xi * dy + bow * ( cos_a)
                ax.plot(bx, by, color="black", linestyle="--",
                        linewidth=1.5, zorder=4)

    for node in model.mesh.nodes:
        U, V = node_disps[node.id]
        x_def = node.x + scale * U
        y_def = node.y + scale * V
        ax.plot(x_def, y_def, "ko", markersize=4, zorder=5)
        ax.text(x_def, y_def, f" {node.id}", fontsize=8, va="bottom")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"N [{lbl['force']}]")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"y [{lbl['length']}]")
    ax.set_title(f"{title} (scale {scale:.2g}x)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Truss deformed plot saved to %s", output_path)

    return fig
```

- [ ] **Step 4: Update the module docstring for `plotter.py`**

In the top-of-file docstring of `src/fea_solver/plotter.py`, update the `plot_truss_deformed` bullet:

Replace:

```
  - plot_truss_deformed:            2D deformed wireframe with auto-scale, colored by axial force
```

With:

```
  - plot_truss_deformed:            2D deformed wireframe with auto-scale, colored by axial force;
                                    optional buckling overlay draws a half-sine bow on failed members.
```

- [ ] **Step 5: Run the plotter tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py -v`

Expected: all tests pass, including the three new ones under `TestTrussDeformedWithBuckling`.

- [ ] **Step 6: Commit**

```bash
git add src/fea_solver/plotter.py tests/unit/test_plotter.py
git commit -m "feat: add buckling-mode overlay kwarg to plot_truss_deformed"
```

---

### Task 6: Wire buckling into `main.py` for all-truss problems

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add a failing integration-style smoke test**

Create `tests/integration/test_truss_buckling.py`:

```python
"""Full-pipeline test: solve problem_7.yaml and compute truss buckling.

This test asserts that compute_truss_buckling returns one entry per TRUSS
element and that the tuple contains the expected statuses.
"""
from __future__ import annotations

from pathlib import Path

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.buckling import compute_truss_buckling
from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.solver import run_solve_pipeline


def test_problem_7_buckling_count_matches_truss_elements() -> None:
    """problem_7 has 16 TRUSS elements; buckling tuple length must match."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "config" / "problem_7.yaml"
    (model,) = load_models_from_yaml(cfg)

    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    element_results = postprocess_all_elements(model, result, n_stations=10)

    bucklings = compute_truss_buckling(model, element_results)
    truss_count = sum(1 for e in model.mesh.elements if e.element_type.name == "TRUSS")
    assert len(bucklings) == truss_count == 16


def test_problem_7_has_at_least_one_compressive_member() -> None:
    """At least one member must carry compressive force for the feature to be meaningful."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "config" / "problem_7.yaml"
    (model,) = load_models_from_yaml(cfg)

    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    element_results = postprocess_all_elements(model, result, n_stations=10)

    bucklings = compute_truss_buckling(model, element_results)
    assert any(mb.axial_force < 0.0 for mb in bucklings)
```

- [ ] **Step 2: Run the new integration tests (must already pass since the
      library code is complete)**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_truss_buckling.py -v`

Expected: `2 passed`

- [ ] **Step 3: Wire buckling into `main.py`**

Open `main.py` and update the imports to include the new symbols. Replace:

```python
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.reporter import (
    generate_report,
    print_dof_table,
    print_element_forces,
    print_nodal_results,
    print_reaction_forces,
)
```

With:

```python
from fea_solver.buckling import compute_truss_buckling
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.reporter import (
    generate_report,
    print_buckling_summary,
    print_dof_table,
    print_element_forces,
    print_nodal_results,
    print_reaction_forces,
)
```

Then, inside the `if all_truss:` branch, replace the existing per-solution loop:

```python
        if all_truss:
            # 2D truss: three gradient plots per solution (overlay not meaningful)
            for sol in all_series:
                safe_sol = _sanitize_label(sol.label)

                forces_path = (save_dir / f"{safe_sol}_truss_forces.png") if save_dir else None
                figures.append(
                    plot_truss_forces(sol,
                                      title=f"Truss Forces: {sol.label}",
                                      output_path=forces_path)
                )

                deformed_path = (save_dir / f"{safe_sol}_truss_deformed.png") if save_dir else None
                figures.append(
                    plot_truss_deformed(sol,
                                        title=f"Truss Deformed: {sol.label}",
                                        output_path=deformed_path)
                )

                stress_path = (save_dir / f"{safe_sol}_truss_stress.png") if save_dir else None
                figures.append(
                    plot_truss_stress(sol,
                                      title=f"Truss Stress: {sol.label}",
                                      output_path=stress_path)
                )
```

With:

```python
        if all_truss:
            # 2D truss: three gradient plots per solution + buckling overlay/summary.
            for sol in all_series:
                safe_sol = _sanitize_label(sol.label)

                bucklings = compute_truss_buckling(sol.model, list(sol.element_results))
                print_buckling_summary(bucklings, sol.model)

                forces_path = (save_dir / f"{safe_sol}_truss_forces.png") if save_dir else None
                figures.append(
                    plot_truss_forces(sol,
                                      title=f"Truss Forces: {sol.label}",
                                      output_path=forces_path)
                )

                deformed_path = (save_dir / f"{safe_sol}_truss_deformed.png") if save_dir else None
                figures.append(
                    plot_truss_deformed(sol,
                                        title=f"Truss Deformed: {sol.label}",
                                        output_path=deformed_path,
                                        buckling=bucklings)
                )

                stress_path = (save_dir / f"{safe_sol}_truss_stress.png") if save_dir else None
                figures.append(
                    plot_truss_stress(sol,
                                      title=f"Truss Stress: {sol.label}",
                                      output_path=stress_path)
                )
```

- [ ] **Step 4: Smoke test via the CLI on `problem_7.yaml`**

Run: `.venv/Scripts/python.exe main.py config/problem_7.yaml --save-plots outputs/buckling_smoke --no-plot=False 2>&1 | tail -20`

Actually, the CLI uses `--no-plot` as a store_true flag; to exercise the plotting path run:

Run: `.venv/Scripts/python.exe main.py config/problem_7.yaml --save-plots outputs/buckling_smoke`

Expected:
- Exit code 0.
- Stdout contains a "Buckling Summary" table.
- File `outputs/buckling_smoke/problem_7_truss_deformed.png` exists.

- [ ] **Step 5: Run the entire test suite to verify nothing regressed**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`

Expected: all tests pass, including the new `tests/unit/test_buckling.py`,
the new integration file `tests/integration/test_truss_buckling.py`, and
the updated `tests/unit/test_plotter.py` and `tests/unit/test_reporter.py`.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/integration/test_truss_buckling.py
git commit -m "feat: wire buckling summary and overlay into main pipeline"
```

---

### Task 7: Final verification and cleanup

**Files:**
- No edits. Pure verification.

- [ ] **Step 1: Full test suite with verbose output**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`

Expected: all tests pass (all previous tests plus 13 new buckling unit tests,
2 new integration tests, 3 new plotter tests, 2 new reporter tests).

- [ ] **Step 2: Verify the deformed plot contains buckling overlay on `problem_7.yaml`**

Run: `.venv/Scripts/python.exe main.py config/problem_7.yaml --save-plots outputs/verify`

Open `outputs/verify/problem_7_truss_deformed.png` and confirm visually
that any member whose axial force exceeded P_cr is drawn with a dashed
black half-sine bow.

- [ ] **Step 3: Verify non-truss problems still behave unchanged**

Run: `.venv/Scripts/python.exe main.py config/example_case_02_cantilever_beam.yaml --save-plots outputs/beam_regression`

Expected:
- No "Buckling Summary" output (all_truss = False).
- Same shear/moment/disp/rotation plots as before in
  `outputs/beam_regression/`.

- [ ] **Step 4: Optional manual smoke of an all-truss problem with no buckled members**

Locate any all-truss YAML with small loads (e.g. `config/problem_4.yaml` if
it is all-truss — inspect first). Run and confirm the summary prints `SAFE`
or `TENSION` only, and the deformed plot carries no dashed-black overlay.

- [ ] **Step 5: No commit required (verification-only task)**

---

## Self-Review Checklist (done inline during plan authoring)

- Spec coverage -- each section of `2026-04-17-truss-buckling-design.md`
  is implemented: dataclass (Task 1), `compute_member_P_cr` and
  `compute_truss_buckling` (Tasks 2-3), reporter (Task 4), plotter overlay
  (Task 5), wiring (Task 6), regression and smoke (Task 7).
- No placeholders -- every code step contains the exact code to paste.
- Type consistency -- `MemberBuckling(element_id, P_cr, axial_force,
  ratio, is_buckled)` signature identical in every task; `compute_truss_buckling`
  signature returns `tuple[MemberBuckling, ...]` everywhere; plotter kwarg
  named `buckling` consistently; reporter function named
  `print_buckling_summary` consistently.
- TDD -- every behavioural change has a failing test written before the
  implementation step.
- Frequent commits -- one commit per task.
