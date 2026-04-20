# Constraint Architecture Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace keyword-based `BoundaryCondition` with general `LinearConstraint` and switch from the reduction method to the penalty method throughout the FEA solver.

**Architecture:** Each constraint is a scalar linear equation `a_U*u + a_V*v + a_THETA*theta = rhs` stored as a `tuple[float, float, float]` coefficient vector. Enforcement adds `k_penalty * outer(g, g)` to K and `k_penalty * rhs * g` to F before solving the full (not partitioned) system. Reactions are the per-constraint penalty residuals `k_penalty * (a^T * u - rhs)`.

**Tech Stack:** Python 3.12, numpy, pydantic v2, pytest, PyYAML

---

## File Map

| File | Action |
|------|--------|
| `src/fea_solver/models.py` | Remove `BoundaryConditionType`, `BoundaryCondition`; add `LinearConstraint`; update `FEAModel` |
| `src/fea_solver/constraints.py` | Full rewrite: penalty method replaces reduction method |
| `src/fea_solver/solver.py` | Solve full system; replace reaction computation |
| `src/fea_solver/io_yaml.py` | Replace `_BCSchema`/`_BC_TYPE_MAP`; add `penalty_alpha` parsing |
| `src/fea_solver/reporter.py` | Update reaction table to use constraints not DOF indices |
| `config/*.yaml` (all 15) | Translate BC blocks to `coefficients:` entries |
| `tests/conftest.py` | Replace `BoundaryCondition` fixtures with `LinearConstraint` |
| `tests/unit/test_constraints.py` | Full rewrite: penalty method tests |
| `tests/unit/test_io_yaml.py` | Update BC YAML strings; update `test_boundary_condition_type` |
| `tests/unit/test_assembler.py` | Update fixtures to use `LinearConstraint` |
| `tests/unit/test_solver.py` | Remove partitioned-solve tests; rewrite pipeline tests |
| `tests/unit/test_postprocessor.py` | Update fixtures to use `LinearConstraint` |
| `tests/unit/test_plotter.py` | Update fixture to use `LinearConstraint` |
| `tests/unit/test_properties.py` | Replace `get_constrained_dof_indices` property test |
| `tests/integration/test_beam_cases.py` | Rewrite `test_support_reactions_sum_to_applied_load` |
| `tests/integration/test_multi_solution.py` | Update inline YAML strings |
| `tests/integration/test_inclined_roller.py` | New test: inclined 45-degree roller |

### DOF validity per element type (critical for YAML migration)

| Element | Valid non-zero coefficient positions |
|---------|--------------------------------------|
| BAR     | `[a, 0, 0]` — U only                |
| BEAM    | `[0, b, c]` — V and THETA, no U     |
| FRAME   | `[a, b, c]` — all three              |
| TRUSS   | `[a, b, 0]` — U and V, no THETA     |

### Keyword translation table

| Old keyword  | New constraint entries (element-type-aware)     |
|--------------|--------------------------------------------------|
| `fixed_u`    | `[1.0, 0.0, 0.0]`                               |
| `fixed_v`    | `[0.0, 1.0, 0.0]`                               |
| `fixed_theta`| `[0.0, 0.0, 1.0]`                               |
| `pin` on BAR/TRUSS | `[1.0, 0.0, 0.0]` + `[0.0, 1.0, 0.0]`   |
| `pin` on BEAM | `[0.0, 1.0, 0.0]` only (no U DOF)             |
| `pin` on FRAME | `[1.0, 0.0, 0.0]` + `[0.0, 1.0, 0.0]`       |
| `roller` on BEAM/TRUSS/FRAME | `[0.0, 1.0, 0.0]`                |
| `fixed_all` on BAR | `[1.0, 0.0, 0.0]`                          |
| `fixed_all` on BEAM | `[0.0, 1.0, 0.0]` + `[0.0, 0.0, 1.0]`    |
| `fixed_all` on FRAME | `[1.0, 0.0, 0.0]` + `[0.0, 1.0, 0.0]` + `[0.0, 0.0, 1.0]` |

---

## Task 1: Update models.py — add LinearConstraint, update FEAModel

**Files:**
- Modify: `src/fea_solver/models.py`

- [x] **Step 1: Remove `BoundaryConditionType` enum and `BoundaryCondition` dataclass, add `LinearConstraint`**

Replace the section between `class BoundaryConditionType(Enum):` and `class LoadType(Enum):` (lines 79–101) with:

```python
class LoadType(Enum):
```

And replace the `BoundaryCondition` dataclass (lines 244–258) with:

```python
@dataclass(frozen=True, slots=True)
class LinearConstraint:
    """One scalar linear constraint equation applied at a node.

    Encodes the constraint: a_U*u + a_V*v + a_THETA*theta = rhs

    Fields:
        node_id (int): Node at which the constraint is applied.
        coefficients (tuple[float, float, float]): Constraint direction vector
            in [U, V, THETA] DOF order (global coordinates). Must be a unit
            vector (normalized in _schema_to_model before construction).
            Non-zero components for DOFs absent at the node raise ValueError
            during constraint application.
        rhs (float): Prescribed value. Default 0.0 (homogeneous constraint).

    Notes:
        Applied via the penalty method: adds k_penalty * outer(g, g) to K
        and k_penalty * rhs * g to F, where g is the global DOF coefficient
        vector built from coefficients and the node's DOF indices.
    """

    node_id: int
    coefficients: tuple[float, float, float]
    rhs: float = 0.0
```

- [ ] **Step 2: Update `FEAModel` field types and add `penalty_alpha`**

In `FEAModel` (around line 323), update the `boundary_conditions` field and the docstring, and add `penalty_alpha`:

```python
@dataclass(frozen=True, slots=True)
class FEAModel:
    """Complete FEA problem definition containing geometry, loads, and constraints.

    Fields:
        mesh (Mesh): Spatial discretization with nodes and elements.
        boundary_conditions (tuple[LinearConstraint, ...]): Penalty-enforced
            kinematic constraints. Each entry is one scalar linear equation
            in the node's [U, V, THETA] DOF space.
        nodal_loads (tuple[NodalLoad, ...]): Concentrated forces and moments.
        distributed_loads (tuple[DistributedLoad, ...]): Distributed loads over elements.
        label (str): Optional descriptive label. Default "unnamed".
        unit_system (UnitSystem): Canonical unit system all numeric values are stored in.
            Default UnitSystem.SI. Determines reporter column-header labels.
        penalty_alpha (float): Scale factor for penalty stiffness. The penalty
            parameter used during constraint enforcement is computed as
            penalty_alpha * max(abs(diag(K))). Default 1e8.

    Notes:
        Frozen and slotted. Immutable after construction; use dataclasses.replace()
        to create modified copies. All field values must already be expressed in
        the canonical units for unit_system before construction.
    """

    mesh: Mesh
    boundary_conditions: tuple[LinearConstraint, ...]
    nodal_loads: tuple[NodalLoad, ...]
    distributed_loads: tuple[DistributedLoad, ...]
    label: str = "unnamed"
    unit_system: UnitSystem = UnitSystem.SI
    penalty_alpha: float = 1e8
```

- [ ] **Step 3: Update the module docstring header to remove references to `BoundaryCondition`/`BoundaryConditionType` and add `LinearConstraint`**

Change `BoundaryCondition, NodalLoad, DistributedLoad: applied constraints and loads.` to:
`LinearConstraint, NodalLoad, DistributedLoad: applied constraints and loads.`

Remove `BoundaryCondition` and `BoundaryConditionType` from the `Key types:` list in the module docstring.
Add `LinearConstraint: a_U*u + a_V*v + a_THETA*theta = rhs scalar constraint with penalty enforcement.`

- [ ] **Step 4: Verify models.py compiles cleanly**

```bash
.venv/Scripts/python.exe -c "from fea_solver.models import LinearConstraint, FEAModel; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/fea_solver/models.py
git commit -m "feat: replace BoundaryCondition with LinearConstraint in models"
```

---

## Task 2: Rewrite constraints.py — penalty method

**Files:**
- Modify: `src/fea_solver/constraints.py`
- Modify: `tests/unit/test_constraints.py`

- [ ] **Step 1: Write the new failing unit tests**

Replace `tests/unit/test_constraints.py` entirely with:

```python
"""Tests for penalty constraint enforcement."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.constraints import apply_penalty_constraints, compute_constraint_residuals
from fea_solver.models import (
    DOFMap,
    DOFType,
    Element,
    ElementType,
    FEAModel,
    LinearConstraint,
    LoadType,
    MaterialProperties,
    Mesh,
    Node,
    NodalLoad,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_bar_dof_map() -> DOFMap:
    """2-node bar DOF map: node 1 -> DOF 0, node 2 -> DOF 1 (U only)."""
    dm = DOFMap()
    dm.mapping[(1, DOFType.U)] = 0
    dm.mapping[(2, DOFType.U)] = 1
    dm.total_dofs = 2
    return dm


def make_truss_dof_map() -> DOFMap:
    """2-node truss DOF map: node 1 -> [0,1], node 2 -> [2,3] (U,V)."""
    dm = DOFMap()
    dm.mapping[(1, DOFType.U)] = 0
    dm.mapping[(1, DOFType.V)] = 1
    dm.mapping[(2, DOFType.U)] = 2
    dm.mapping[(2, DOFType.V)] = 3
    dm.total_dofs = 4
    return dm


def make_frame_dof_map() -> DOFMap:
    """2-node frame DOF map: node 1 -> [0,1,2], node 2 -> [3,4,5] (U,V,THETA)."""
    dm = DOFMap()
    dm.mapping[(1, DOFType.U)] = 0
    dm.mapping[(1, DOFType.V)] = 1
    dm.mapping[(1, DOFType.THETA)] = 2
    dm.mapping[(2, DOFType.U)] = 3
    dm.mapping[(2, DOFType.V)] = 4
    dm.mapping[(2, DOFType.THETA)] = 5
    dm.total_dofs = 6
    return dm


# ---------------------------------------------------------------------------
# Tests for apply_penalty_constraints
# ---------------------------------------------------------------------------


class TestApplyPenaltyConstraints:
    """Tests for apply_penalty_constraints."""

    def test_axis_aligned_u_constraint_adds_to_diagonal(self) -> None:
        """U-direction constraint adds k_penalty to K[0,0]."""
        K = np.zeros((2, 2))
        F = np.zeros(2)
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        K_mod, F_mod = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        assert K_mod[0, 0] == pytest.approx(1e6)
        assert K_mod[1, 1] == pytest.approx(0.0)

    def test_axis_aligned_v_constraint_adds_to_v_diagonal(self) -> None:
        """V-direction constraint adds k_penalty to K[V,V] diagonal."""
        K = np.zeros((4, 4))
        F = np.zeros(4)
        dof_map = make_truss_dof_map()
        constraint = LinearConstraint(node_id=2, coefficients=(0.0, 1.0, 0.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        v_idx = dof_map.index(2, DOFType.V)
        assert K_mod[v_idx, v_idx] == pytest.approx(1e6)

    def test_inclined_constraint_adds_off_diagonal_coupling(self) -> None:
        """Inclined constraint (45-deg) adds coupling between U and V DOFs."""
        K = np.zeros((4, 4))
        F = np.zeros(4)
        dof_map = make_truss_dof_map()
        n = 1.0 / math.sqrt(2.0)
        constraint = LinearConstraint(node_id=2, coefficients=(n, n, 0.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        u_idx = dof_map.index(2, DOFType.U)
        v_idx = dof_map.index(2, DOFType.V)
        assert K_mod[u_idx, v_idx] == pytest.approx(0.5e6)
        assert K_mod[v_idx, u_idx] == pytest.approx(0.5e6)

    def test_rotation_constraint_adds_to_theta_diagonal(self) -> None:
        """THETA constraint adds k_penalty to K[THETA,THETA] diagonal."""
        K = np.zeros((6, 6))
        F = np.zeros(6)
        dof_map = make_frame_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(0.0, 0.0, 1.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        theta_idx = dof_map.index(1, DOFType.THETA)
        assert K_mod[theta_idx, theta_idx] == pytest.approx(1e6)

    def test_nonzero_rhs_modifies_force_vector(self) -> None:
        """Non-zero rhs adds k_penalty * rhs * g to F."""
        K = np.zeros((2, 2))
        F = np.zeros(2)
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.005)
        _, F_mod = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        assert F_mod[0] == pytest.approx(1e6 * 0.005)

    def test_does_not_mutate_input_arrays(self) -> None:
        """Input K and F are not modified in place."""
        K = np.eye(2)
        F = np.ones(2)
        K_copy = K.copy()
        F_copy = F.copy()
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        np.testing.assert_array_equal(K, K_copy)
        np.testing.assert_array_equal(F, F_copy)

    def test_nonzero_coefficient_for_missing_dof_raises(self) -> None:
        """Non-zero coefficient for a DOF absent at the node raises ValueError."""
        K = np.zeros((2, 2))
        F = np.zeros(2)
        dof_map = make_bar_dof_map()
        # Bar has no V DOF; [0.0, 1.0, 0.0] at node 1 should raise
        constraint = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0), rhs=0.0)
        with pytest.raises(ValueError, match="V"):
            apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)

    def test_returned_K_is_symmetric(self) -> None:
        """Modified K is symmetric."""
        K = np.zeros((4, 4))
        F = np.zeros(4)
        dof_map = make_truss_dof_map()
        n = 1.0 / math.sqrt(2.0)
        constraint = LinearConstraint(node_id=2, coefficients=(n, n, 0.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (constraint,), dof_map, k_penalty=1e6)
        np.testing.assert_allclose(K_mod, K_mod.T, atol=1e-12)

    def test_multiple_constraints_accumulate(self) -> None:
        """Multiple constraints accumulate on K correctly."""
        K = np.zeros((4, 4))
        F = np.zeros(4)
        dof_map = make_truss_dof_map()
        c1 = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        c2 = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0), rhs=0.0)
        K_mod, _ = apply_penalty_constraints(K, F, (c1, c2), dof_map, k_penalty=1e6)
        assert K_mod[0, 0] == pytest.approx(1e6)   # U at node 1
        assert K_mod[1, 1] == pytest.approx(1e6)   # V at node 1


# ---------------------------------------------------------------------------
# Tests for compute_constraint_residuals
# ---------------------------------------------------------------------------


class TestComputeConstraintResiduals:
    """Tests for compute_constraint_residuals."""

    def test_zero_residual_at_exact_constraint(self) -> None:
        """Zero displacement at constrained DOF gives zero residual."""
        u = np.zeros(2)
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        R = compute_constraint_residuals(u, (constraint,), dof_map, k_penalty=1e6)
        assert R.shape == (1,)
        assert R[0] == pytest.approx(0.0)

    def test_residual_scales_with_displacement(self) -> None:
        """Residual is k_penalty * a^T * u when rhs=0."""
        u = np.array([0.001, 0.0])
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        R = compute_constraint_residuals(u, (constraint,), dof_map, k_penalty=1e6)
        assert R[0] == pytest.approx(1e6 * 0.001)

    def test_residual_accounts_for_rhs(self) -> None:
        """Residual is k_penalty * (a^T*u - rhs)."""
        u = np.array([0.005, 0.0])
        dof_map = make_bar_dof_map()
        constraint = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.005)
        R = compute_constraint_residuals(u, (constraint,), dof_map, k_penalty=1e6)
        assert R[0] == pytest.approx(0.0, abs=1e-9)

    def test_output_shape_equals_number_of_constraints(self) -> None:
        """Output length equals number of constraints."""
        u = np.zeros(4)
        dof_map = make_truss_dof_map()
        constraints = (
            LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0),
            LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0), rhs=0.0),
            LinearConstraint(node_id=2, coefficients=(0.0, 1.0, 0.0), rhs=0.0),
        )
        R = compute_constraint_residuals(u, constraints, dof_map, k_penalty=1e6)
        assert R.shape == (3,)

    def test_near_zero_residuals_after_full_solve(self) -> None:
        """Constraint residuals are near zero (< 1/k_p) after penalty solve."""
        # Single bar: E=A=L=1, constrain U at node 1, apply P=1 at node 2
        mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
        n1, n2 = Node(1, (0.0, 0.0)), Node(2, (1.0, 0.0))
        elem = Element(id=1, node_i=n1, node_j=n2,
                       element_type=ElementType.BAR, material=mat)
        c = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
        load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_X, magnitude=1.0)
        model = FEAModel(
            mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
            boundary_conditions=(c,),
            nodal_loads=(load,),
            distributed_loads=(),
            label="test",
            penalty_alpha=1e8,
        )
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        k_penalty = model.penalty_alpha * float(np.max(np.abs(np.diag(K))))
        K_mod, F_mod = apply_penalty_constraints(K, F, model.boundary_conditions, dof_map, k_penalty)
        u = np.linalg.solve(K_mod, F_mod)
        R = compute_constraint_residuals(u, model.boundary_conditions, dof_map, k_penalty)
        # Constraint violation should be extremely small
        assert abs(u[dof_map.index(1, DOFType.U)]) < 1.0 / model.penalty_alpha
        assert R.shape == (1,)
```

- [ ] **Step 2: Run the new tests to verify they all fail (functions not yet defined)**

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_constraints.py -v 2>&1 | head -30
```

Expected: `ImportError` or all `FAILED` — `apply_penalty_constraints` and `compute_constraint_residuals` not yet defined.

- [ ] **Step 3: Rewrite `constraints.py` with the penalty method**

Replace `src/fea_solver/constraints.py` entirely with:

```python
"""Kinematic constraint application via the penalty method.

The penalty method:
  1. For each LinearConstraint, build a global coefficient vector g from
     the constraint's (node_id, coefficients) and the DOF map.
  2. Add k_penalty * outer(g, g) to K and k_penalty * rhs * g to F.
  3. Solve the full (unpartitioned) modified system K_mod * u = F_mod.
  4. Recover per-constraint reaction forces as k_penalty * (a^T * u - rhs).

The penalty parameter is computed as penalty_alpha * max(abs(diag(K_natural)))
so that it scales with the problem stiffness without needing manual tuning.

apply_penalty_constraints: add penalty terms to K and F for all constraints.
compute_constraint_residuals: compute per-constraint reaction magnitudes post-solve.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fea_solver.models import DOFMap, DOFType, LinearConstraint

logger = logging.getLogger(__name__)

# Canonical mapping from coefficient position to DOFType
_COEFF_INDEX_TO_DOF: tuple[DOFType, DOFType, DOFType] = (
    DOFType.U,
    DOFType.V,
    DOFType.THETA,
)


def apply_penalty_constraints(
    K: NDArray[np.float64],
    F: NDArray[np.float64],
    constraints: tuple[LinearConstraint, ...],
    dof_map: DOFMap,
    k_penalty: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply kinematic constraints via the penalty method.

    For each constraint with coefficient vector a = [a_U, a_V, a_THETA]:
      1. Build global coefficient vector g (length n_dofs) by placing a[i]
         at the global DOF index for (node_id, DOFType_i) for each non-zero a[i].
      2. K_mod += k_penalty * outer(g, g)
      3. F_mod += k_penalty * rhs * g

    Args:
        K (NDArray[np.float64]): Global stiffness matrix, shape (n, n).
        F (NDArray[np.float64]): Global force vector, shape (n,).
        constraints (tuple[LinearConstraint, ...]): All constraints to apply.
        dof_map (DOFMap): DOF index mapping for (node_id, DOFType) lookups.
        k_penalty (float): Penalty stiffness parameter (problem-scale-dependent).

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: New (K_mod, F_mod)
            arrays. The input K and F are not mutated.

    Raises:
        ValueError: If a constraint has a non-zero coefficient for a DOF that
            does not exist at the node (e.g., U component on a BEAM-only node).

    Notes:
        Returns copies; input arrays are never mutated.
        The penalty parameter k_penalty is typically computed as
        model.penalty_alpha * max(abs(diag(K_natural))) by the caller.
    """
    K_mod = K.copy()
    F_mod = F.copy()
    n = K_mod.shape[0]

    for constraint in constraints:
        g = np.zeros(n)
        for coeff_idx, dof_type in enumerate(_COEFF_INDEX_TO_DOF):
            a_i = constraint.coefficients[coeff_idx]
            if a_i == 0.0:
                continue
            if not dof_map.has_dof(constraint.node_id, dof_type):
                raise ValueError(
                    f"Constraint at node {constraint.node_id} has non-zero "
                    f"coefficient for {dof_type.value} ({a_i}), but that DOF "
                    f"does not exist at this node."
                )
            g[dof_map.index(constraint.node_id, dof_type)] = a_i

        K_mod += k_penalty * np.outer(g, g)
        F_mod += k_penalty * constraint.rhs * g

    logger.debug(
        "Penalty constraints applied: %d constraints, k_penalty=%.3e",
        len(constraints), k_penalty,
    )
    return K_mod, F_mod


def compute_constraint_residuals(
    u: NDArray[np.float64],
    constraints: tuple[LinearConstraint, ...],
    dof_map: DOFMap,
    k_penalty: float,
) -> NDArray[np.float64]:
    """Compute per-constraint reaction force magnitudes after solving.

    For each constraint i with coefficient vector a_i and prescribed value rhs_i:
        reactions[i] = k_penalty * (a_i^T * u_node_i - rhs_i)

    This is the constraint force: the force the support applies to the structure.

    Args:
        u (NDArray[np.float64]): Full displacement vector, shape (n_dofs,).
        constraints (tuple[LinearConstraint, ...]): All constraints.
        dof_map (DOFMap): DOF index mapping for (node_id, DOFType) lookups.
        k_penalty (float): Penalty stiffness parameter used during assembly.

    Returns:
        NDArray[np.float64]: Reaction magnitudes, shape (n_constraints,).
            reactions[i] corresponds to constraints[i].

    Notes:
        For axis-aligned constraints with unit coefficients, the reaction equals
        the physical support force (e.g., k_p * v_1 is the vertical reaction).
        Residuals are near zero when the constraint is well-enforced.
    """
    reactions = np.empty(len(constraints))
    for i, constraint in enumerate(constraints):
        a_dot_u = 0.0
        for coeff_idx, dof_type in enumerate(_COEFF_INDEX_TO_DOF):
            a_i = constraint.coefficients[coeff_idx]
            if a_i != 0.0 and dof_map.has_dof(constraint.node_id, dof_type):
                a_dot_u += a_i * u[dof_map.index(constraint.node_id, dof_type)]
        reactions[i] = k_penalty * (a_dot_u - constraint.rhs)
    logger.debug("Constraint residuals: %s", reactions)
    return reactions
```

- [ ] **Step 4: Run new constraint tests**

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_constraints.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fea_solver/constraints.py tests/unit/test_constraints.py
git commit -m "feat: rewrite constraints.py with penalty method; rewrite test_constraints.py"
```

---

## Task 3: Update solver.py — solve full system, update reaction recovery

**Files:**
- Modify: `src/fea_solver/solver.py`
- Modify: `tests/unit/test_solver.py`

- [ ] **Step 1: Rewrite `solver.py`**

Replace `src/fea_solver/solver.py` entirely with:

```python
"""Displacement solver using the penalty-modified full stiffness system.

Solves K_mod * u = F_mod where K_mod and F_mod include penalty constraint
terms. Reactions are computed as penalty residuals per constraint.

compute_penalty_parameter: compute k_penalty from penalty_alpha and K diagonal.
solve_system:              np.linalg.solve with condition number check.
run_solve_pipeline:        orchestrate full solve (penalty apply -> solve -> reactions).
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fea_solver.constraints import apply_penalty_constraints, compute_constraint_residuals
from fea_solver.models import DOFMap, FEAModel, SolutionResult

logger = logging.getLogger(__name__)


def compute_penalty_parameter(K: NDArray[np.float64], penalty_alpha: float) -> float:
    """Compute the penalty stiffness parameter from the natural stiffness matrix.

    k_penalty = penalty_alpha * max(abs(diag(K)))

    Args:
        K (NDArray[np.float64]): Natural (pre-penalty) global stiffness matrix.
        penalty_alpha (float): Scale factor from FEAModel.penalty_alpha.

    Returns:
        float: Penalty stiffness parameter.

    Notes:
        Scaling by max(diag(K)) makes the penalty parameter dimensionally
        consistent across problem scales, so penalty_alpha=1e8 works for both
        steel aerospace structures and soft-material unit-stiffness problems.
    """
    return penalty_alpha * float(np.max(np.abs(np.diag(K))))


def solve_system(
    K_mod: NDArray[np.float64],
    F_mod: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the penalty-modified system K_mod * u = F_mod.

    Args:
        K_mod (NDArray[np.float64]): Penalty-modified stiffness matrix, shape (n, n).
        F_mod (NDArray[np.float64]): Penalty-modified force vector, shape (n,).

    Returns:
        NDArray[np.float64]: Full displacement vector u, shape (n,).

    Raises:
        np.linalg.LinAlgError: If K_mod is singular.

    Notes:
        Condition number is logged. Penalty-modified matrices have condition
        numbers proportional to penalty_alpha, so warnings above 1e14 are
        expected and do not indicate a problem for well-posed models.
    """
    cond = float(np.linalg.cond(K_mod))
    logger.debug("K_mod condition number: %.3e", cond)
    if cond > 1e14:
        logger.warning(
            "K_mod is nearly singular (cond=%.3e). For penalty-method models "
            "this is expected when penalty_alpha is large. Check that all DOFs "
            "have at least one constraint or stiffness contribution.", cond
        )

    try:
        u = np.linalg.solve(K_mod, F_mod)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Stiffness matrix is singular -- check boundary conditions. "
            f"Original error: {exc}"
        ) from exc

    logger.debug("Max displacement: %.6e", float(np.max(np.abs(u))))
    return u


def run_solve_pipeline(
    model: FEAModel,
    dof_map: DOFMap,
    K: NDArray[np.float64],
    F: NDArray[np.float64],
) -> SolutionResult:
    """Orchestrate the full penalty-method solve pipeline.

    Steps:
      1. compute_penalty_parameter -> k_penalty
      2. apply_penalty_constraints -> K_mod, F_mod
      3. solve_system -> u (full displacement vector)
      4. compute_constraint_residuals -> reactions (per constraint)
      5. Return SolutionResult

    Args:
        model (FEAModel): FEA model supplying constraints and penalty_alpha.
        dof_map (DOFMap): DOF index mapping.
        K (NDArray[np.float64]): Natural global stiffness matrix.
        F (NDArray[np.float64]): Global force vector.

    Returns:
        SolutionResult: Displacements, per-constraint reactions, dof_map, model.

    Notes:
        reactions[i] in the returned SolutionResult corresponds to
        model.boundary_conditions[i]. The penalty parameter is stored
        implicitly in model.penalty_alpha and reconstructed from K's diagonal.
    """
    k_penalty = compute_penalty_parameter(K, model.penalty_alpha)
    K_mod, F_mod = apply_penalty_constraints(
        K, F, model.boundary_conditions, dof_map, k_penalty
    )
    u = solve_system(K_mod, F_mod)
    reactions = compute_constraint_residuals(
        u, model.boundary_conditions, dof_map, k_penalty
    )

    logger.info(
        "Solve complete: max|u|=%.4e, max|R|=%.4e",
        float(np.max(np.abs(u))),
        float(np.max(np.abs(reactions))) if len(reactions) > 0 else 0.0,
    )

    return SolutionResult(
        displacements=u,
        reactions=reactions,
        dof_map=dof_map,
        model=model,
    )
```

- [ ] **Step 2: Rewrite `tests/unit/test_solver.py`**

Replace `tests/unit/test_solver.py` entirely with:

```python
"""Tests for the penalty-method solver pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.models import (
    DOFType,
    Element,
    ElementType,
    FEAModel,
    LinearConstraint,
    LoadType,
    MaterialProperties,
    Mesh,
    Node,
    NodalLoad,
    SolutionResult,
)
from fea_solver.solver import compute_penalty_parameter, run_solve_pipeline, solve_system


def _make_cantilever_bar_model() -> FEAModel:
    """Single bar element: E=A=L=1, fixed U at node 1, P=1 at node 2."""
    mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
    n1, n2 = Node(1, (0.0, 0.0)), Node(2, (1.0, 0.0))
    elem = Element(id=1, node_i=n1, node_j=n2,
                   element_type=ElementType.BAR, material=mat)
    c = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
    load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_X, magnitude=1.0)
    return FEAModel(
        mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
        boundary_conditions=(c,),
        nodal_loads=(load,),
        distributed_loads=(),
        label="cantilever_bar",
        penalty_alpha=1e10,
    )


class TestComputePenaltyParameter:
    """Tests for compute_penalty_parameter."""

    def test_scales_with_max_diagonal(self) -> None:
        """k_penalty = penalty_alpha * max(abs(diag(K)))."""
        K = np.diag([2.0, 5.0, 3.0])
        k = compute_penalty_parameter(K, penalty_alpha=1e8)
        assert k == pytest.approx(5.0e8)

    def test_uses_absolute_value_of_diagonal(self) -> None:
        """Uses abs of diagonal entries."""
        K = np.diag([-4.0, 1.0])
        k = compute_penalty_parameter(K, penalty_alpha=1e8)
        assert k == pytest.approx(4.0e8)


class TestSolveSystem:
    """Tests for solve_system."""

    def test_trivial_system(self) -> None:
        """Solves identity system correctly."""
        K = np.eye(3)
        F = np.array([1.0, 2.0, 3.0])
        u = solve_system(K, F)
        np.testing.assert_allclose(u, F)

    def test_singular_raises(self) -> None:
        """Singular matrix raises LinAlgError."""
        K = np.zeros((2, 2))
        F = np.array([1.0, 1.0])
        with pytest.raises(np.linalg.LinAlgError):
            solve_system(K, F)


class TestRunSolvePipeline:
    """Tests for run_solve_pipeline."""

    def test_cantilever_bar_tip_displacement(self) -> None:
        """Tip displacement equals P*L/(E*A) = 1.0."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        u_tip = result.displacements[dof_map.index(2, DOFType.U)]
        assert u_tip == pytest.approx(1.0, rel=1e-6)

    def test_returns_solution_result_type(self) -> None:
        """run_solve_pipeline returns SolutionResult."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        assert isinstance(result, SolutionResult)

    def test_reactions_shape_matches_constraints(self) -> None:
        """reactions array length equals number of constraints."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        assert result.reactions.shape == (len(model.boundary_conditions),)

    def test_reaction_equals_applied_load(self) -> None:
        """For a single-constraint bar, reaction magnitude equals applied load."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        # The pin reaction at node 1 must balance the applied load P=1
        assert abs(result.reactions[0]) == pytest.approx(1.0, rel=1e-4)

    def test_fixed_node_displacement_near_zero(self) -> None:
        """Constrained node displacement is near zero (within penalty tolerance)."""
        model = _make_cantilever_bar_model()
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        u_fixed = result.displacements[dof_map.index(1, DOFType.U)]
        assert abs(u_fixed) < 1.0 / model.penalty_alpha
```

- [ ] **Step 3: Run solver tests**

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_solver.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/fea_solver/solver.py tests/unit/test_solver.py
git commit -m "feat: rewrite solver.py for penalty method; rewrite test_solver.py"
```

---

## Task 4: Update io_yaml.py — new schema and penalty_alpha

**Files:**
- Modify: `src/fea_solver/io_yaml.py`

- [ ] **Step 1: Replace `_BCSchema` with `_LinearConstraintSchema` and remove `_BC_TYPE_MAP`**

In `io_yaml.py`:

a. Update imports — remove `BoundaryCondition`, `BoundaryConditionType`, add `LinearConstraint`:

```python
from fea_solver.models import (
    DistributedLoad,
    Element,
    ElementType,
    FEAModel,
    LinearConstraint,
    LoadType,
    MaterialProperties,
    Mesh,
    Node,
    NodalLoad,
)
```

b. Remove the `_BC_TYPE_MAP` dict entirely (lines 94–101).

c. Replace `_BCSchema` class with `_LinearConstraintSchema`:

```python
class _LinearConstraintSchema(BaseModel):
    """Pydantic schema for a single linear constraint entry.

    Args:
        node_id (int): ID of the constrained node.
        coefficients (list[float]): Constraint direction in [U, V, THETA] DOF order.
            Must have exactly 3 elements. The magnitude must be > 0. The vector
            is normalized to unit length during conversion in _schema_to_model.
        rhs (float): Prescribed displacement/rotation value. Default 0.0.

    Notes:
        Non-zero coefficients for DOFs that do not exist at the node raise
        ValueError during constraint application, not here.
    """

    node_id: int
    coefficients: list[float]
    rhs: float = 0.0
```

d. In `_FEAModelSchema`, change:
- `boundary_conditions: list[_BCSchema]` → `boundary_conditions: list[_LinearConstraintSchema]`
- Add: `penalty_alpha: float = 1e8`

e. In `_SolutionEntrySchema`, change:
- `boundary_conditions: list[_BCSchema]` → `boundary_conditions: list[_LinearConstraintSchema]`
- (Do NOT add `penalty_alpha` here — it is a top-level field, not per-solution.)

f. In `_MultiSolutionFileSchema`, add:
- `penalty_alpha: float = 1e8`

g. In `_parse_multi_solution`, add `penalty_alpha=file_schema.penalty_alpha` to the `_FEAModelSchema(...)` constructor call:

```python
        merged_schema = _FEAModelSchema(
            label=composite_label,
            unit_system=file_schema.unit_system,
            units=file_schema.units,
            mesh=sol.mesh,
            materials=sol.materials,
            boundary_conditions=sol.boundary_conditions,
            loads=sol.loads,
            penalty_alpha=file_schema.penalty_alpha,   # ADD THIS LINE
        )
```

h. In `_schema_to_model`, replace the entire `--- Boundary Conditions ---` block:

```python
    # --- Boundary Conditions ---
    penalty_alpha: float = schema.penalty_alpha

    constraints: list[LinearConstraint] = []
    for bc in schema.boundary_conditions:
        if len(bc.coefficients) != 3:
            raise ValueError(
                f"Constraint at node {bc.node_id}: coefficients must have exactly "
                f"3 elements [a_U, a_V, a_THETA], got {len(bc.coefficients)}."
            )
        if bc.node_id not in node_ids:
            raise ValueError(f"Constraint references unknown node_id={bc.node_id}")
        raw = bc.coefficients
        mag = math.sqrt(sum(c * c for c in raw))
        if mag == 0.0:
            raise ValueError(
                f"Constraint at node {bc.node_id}: coefficient vector is zero."
            )
        normalized = tuple(c / mag for c in raw)
        constraints.append(
            LinearConstraint(
                node_id=bc.node_id,
                coefficients=(normalized[0], normalized[1], normalized[2]),
                rhs=bc.rhs,
            )
        )
```

i. Update the `FEAModel(...)` constructor call at the bottom of `_schema_to_model` to:

```python
    return FEAModel(
        mesh=mesh,
        boundary_conditions=tuple(constraints),
        nodal_loads=tuple(nodal_loads),
        distributed_loads=tuple(dist_loads),
        label=label,
        unit_system=unit_system,
        penalty_alpha=penalty_alpha,
    )
```

j. Update the module docstring line `boundary_conditions: [{node_id: int, type: str}, ...]` to:

```
  boundary_conditions:
    - node_id: int
      coefficients: [float, float, float]   # [a_U, a_V, a_THETA], normalized to unit length
      rhs: float                             # optional, default 0.0
  penalty_alpha: float                       # optional, default 1.0e8
```

- [ ] **Step 2: Verify io_yaml parses old-style single constraints**

Create a temporary test file inline (no need to save):

```bash
.venv/Scripts/python.exe -c "
import tempfile, pathlib
from fea_solver.io_yaml import load_model_from_yaml
yaml_text = '''
label: tmp
mesh:
  nodes:
    - {id: 1, x: 0.0, y: 0.0}
    - {id: 2, x: 1.0, y: 0.0}
  elements:
    - {id: 1, node_i: 1, node_j: 2, type: bar, material: s}
materials:
  s: {E: 1.0, A: 1.0}
boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
loads:
  nodal:
    - {node_id: 2, type: point_force_x, magnitude: 1.0}
'''
with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
    f.write(yaml_text)
    p = pathlib.Path(f.name)
m = load_model_from_yaml(p)
c = m.boundary_conditions[0]
print('node_id:', c.node_id, 'coefficients:', c.coefficients)
p.unlink()
"
```

Expected output: `node_id: 1 coefficients: (1.0, 0.0, 0.0)`

- [ ] **Step 3: Commit**

```bash
git add src/fea_solver/io_yaml.py
git commit -m "feat: update io_yaml.py schema for LinearConstraint and penalty_alpha"
```

---

## Task 5: Update reporter.py — new reaction table

**Files:**
- Modify: `src/fea_solver/reporter.py`

- [ ] **Step 1: Replace the reaction table function body**

Locate the function that builds the reaction table (around lines 148–193). Replace the body from `from fea_solver.constraints import get_constrained_dof_indices` onward with:

```python
    model = result.model

    lbl = _lbl(model)
    table = Table(title=f"Reaction Forces -- {model.label}", show_header=True,
                  header_style="bold red")
    table.add_column("Node ID", justify="right")
    table.add_column("Direction [U, V, Th]", justify="center")
    table.add_column("Prescribed", justify="right")
    table.add_column(f"Reaction [{lbl['force']} or {lbl['moment']}]", justify="right")

    for i, constraint in enumerate(model.boundary_conditions):
        c = constraint.coefficients
        coeff_str = f"[{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]"
        table.add_row(
            str(constraint.node_id),
            coeff_str,
            f"{constraint.rhs:.4e}",
            f"{result.reactions[i]:.6e}",
        )

    _console.print(table)
```

- [ ] **Step 2: Update the function docstring** to reflect new columns:

```
Notes:
    Output is printed to console via rich.console.Console.
    Each row corresponds to one LinearConstraint in model.boundary_conditions.
    Reaction[i] = k_penalty * (a_i^T * u_node_i - rhs_i).
    Force/moment units match the model's canonical unit system.
```

- [ ] **Step 3: Commit**

```bash
git add src/fea_solver/reporter.py
git commit -m "feat: update reporter reaction table for LinearConstraint"
```

---

## Task 6: Migrate all 15 config YAML files

**Files:**
- Modify: all `config/*.yaml`

Translate every `boundary_conditions:` block according to the keyword table in the File Map above. Apply the DOF validity rules — no `[1,0,0]` for BEAM nodes, no `[0,0,1]` for TRUSS nodes.

- [ ] **Step 1: Update the 8 example case configs**

**`config/example_case_01_bar_axial.yaml`** (BAR, `fixed_u` at node 1):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
```

**`config/example_case_02_cantilever_beam.yaml`** (BEAM, `fixed_all` at node 1 — no U DOF):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
```

**`config/example_case_03_simply_supported.yaml`** (BEAM, `pin` at node 1, `roller` at node 5 — no U DOF):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 5, coefficients: [0.0, 1.0, 0.0]}
```

**`config/example_case_04_fixed_fixed.yaml`** (BEAM, `fixed_all` at nodes 1 and 3):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
  - {node_id: 3, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 3, coefficients: [0.0, 0.0, 1.0]}
```

**`config/example_case_05_combined_bar_beam.yaml`** (FRAME, `fixed_all` at node 1 — all 3 DOFs):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
```

**`config/example_case_06_distributed_load.yaml`** (BEAM, `pin` at node 1, `roller` at node 5):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 5, coefficients: [0.0, 1.0, 0.0]}
```

**`config/example_case_07_multi_material.yaml`** (BAR, `fixed_u` at node 1):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
```

**`config/example_case_08_point_moment.yaml`** (BEAM, `pin` at node 1, `roller` at node 3):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 3, coefficients: [0.0, 1.0, 0.0]}
```

- [ ] **Step 2: Update the 7 problem configs**

**`config/problem_1.yaml`** (BEAM, multi-solution: `pin` at nodes 1 and 3, `fixed_all` at node 2 — no U DOF on BEAM):

For each solution block, replace `boundary_conditions:` with:
```yaml
    boundary_conditions:
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 2, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 2, coefficients: [0.0, 0.0, 1.0]}
      - {node_id: 3, coefficients: [0.0, 1.0, 0.0]}
```

**`config/problem_2.yaml`** (BEAM, multi-solution: `fixed_all` at node 1):

For each solution block:
```yaml
    boundary_conditions:
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
```

**`config/problem_3.yaml`** (BEAM, multi-solution: `fixed_all` at node 1, `pin` at node 2/node 3):

Coarse solution:
```yaml
    boundary_conditions:
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
      - {node_id: 2, coefficients: [0.0, 1.0, 0.0]}
```

Fine solution (`pin` at node 3 instead of node 2):
```yaml
    boundary_conditions:
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
      - {node_id: 3, coefficients: [0.0, 1.0, 0.0]}
```

**`config/problem_4.yaml`** (BEAM, multi-solution: `fixed_all` at nodes 1 and 5 coarse / 1 and 9 fine; `pin` at nodes 2 and 4 coarse / 3 and 7 fine):

Coarse:
```yaml
    boundary_conditions:
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
      - {node_id: 2, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 4, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 5, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 5, coefficients: [0.0, 0.0, 1.0]}
```

Fine:
```yaml
    boundary_conditions:
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
      - {node_id: 3, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 7, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 9, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 9, coefficients: [0.0, 0.0, 1.0]}
```

**`config/problem_5.yaml`** (TRUSS, `pin` at nodes 1 and 3 — U and V, no THETA):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 3, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 3, coefficients: [0.0, 1.0, 0.0]}
```

**`config/problem_6.yaml`** (TRUSS, `roller` at node 1 = V only, `pin` at node 5 = U+V):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 5, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 5, coefficients: [0.0, 1.0, 0.0]}
```

**`config/problem_7.yaml`** (TRUSS, `pin` at node 1 = U+V, `fixed_U` at node 4 = U only):
```yaml
boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 4, coefficients: [1.0, 0.0, 0.0]}
```

- [ ] **Step 3: Smoke-test all 15 configs parse without error**

```bash
.venv/Scripts/python.exe -c "
from pathlib import Path
from fea_solver.io_yaml import load_models_from_yaml
configs = list(Path('config').glob('*.yaml'))
for p in sorted(configs):
    try:
        models = load_models_from_yaml(p)
        print(f'OK  {p.name} ({len(models)} model(s))')
    except Exception as e:
        print(f'ERR {p.name}: {e}')
"
```

Expected: all lines start with `OK`.

- [ ] **Step 4: Commit**

```bash
git add config/
git commit -m "feat: migrate all 15 config YAMLs to LinearConstraint coefficients schema"
```

---

## Task 7: Update all unit test fixtures that reference BoundaryCondition

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/unit/test_assembler.py`
- Modify: `tests/unit/test_postprocessor.py`
- Modify: `tests/unit/test_plotter.py`
- Modify: `tests/unit/test_properties.py`

In every file, make these mechanical substitutions:

1. Remove `BoundaryCondition, BoundaryConditionType,` from all imports
2. Add `LinearConstraint,` to the `fea_solver.models` import
3. Replace all `BoundaryCondition(node_id=N, bc_type=BoundaryConditionType.FIXED_U)` with `LinearConstraint(node_id=N, coefficients=(1.0, 0.0, 0.0))`
4. Replace all `BoundaryCondition(node_id=N, bc_type=BoundaryConditionType.FIXED_ALL)` with the appropriate multi-entry tuples for the element type in that fixture
5. Replace all `BoundaryCondition(node_id=N, bc_type=BoundaryConditionType.FIXED_V)` with `LinearConstraint(node_id=N, coefficients=(0.0, 1.0, 0.0))`
6. Replace all `BoundaryCondition(node_id=N, bc_type=BoundaryConditionType.PIN)` with the appropriate entries

- [ ] **Step 1: Update `tests/conftest.py`**

```python
# Old imports:
from fea_solver.models import (
    BoundaryCondition, BoundaryConditionType, ...
)

# New imports (remove BoundaryCondition/BoundaryConditionType, add LinearConstraint):
from fea_solver.models import (
    DistributedLoad, Element, ElementType, FEAModel, LinearConstraint,
    LoadType, MaterialProperties, Mesh, Node, NodalLoad,
)
```

`two_node_bar_model` fixture — BAR element, `FIXED_U` at node 1:
```python
    c = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0))
    return FEAModel(mesh=mesh, boundary_conditions=(c,),
                    nodal_loads=(load,), distributed_loads=(), label="two_node_bar")
```

`cantilever_beam_model` fixture — BEAM element, `FIXED_ALL` at node 1 (no U DOF):
```python
    c_v = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0))
    c_t = LinearConstraint(node_id=1, coefficients=(0.0, 0.0, 1.0))
    return FEAModel(mesh=mesh, boundary_conditions=(c_v, c_t),
                    nodal_loads=(load,), distributed_loads=(), label="cantilever_beam")
```

- [ ] **Step 2: Update `tests/unit/test_assembler.py`**

Find all `BoundaryCondition(...)` calls. The assembler tests create bar and beam models.

Bar fixture (FIXED_U at node 1):
```python
    c = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0))
    return FEAModel(mesh=mesh, boundary_conditions=(c,), nodal_loads=(load,), ...)
```

Beam fixture (FIXED_ALL at node 1, BEAM element — no U):
```python
    c_v = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0))
    c_t = LinearConstraint(node_id=1, coefficients=(0.0, 0.0, 1.0))
    return FEAModel(mesh=mesh, boundary_conditions=(c_v, c_t), nodal_loads=(load,), ...)
```

Inline `BoundaryCondition(1, BoundaryConditionType.FIXED_U)` (for BAR):
→ `LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0))`

Inline `BoundaryCondition(1, BoundaryConditionType.FIXED_ALL)` (for BEAM):
→ `(LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0)), LinearConstraint(node_id=1, coefficients=(0.0, 0.0, 1.0)))`

Models with `boundary_conditions=()` — leave as-is.

- [ ] **Step 3: Update `tests/unit/test_postprocessor.py`**

BEAM fixtures — `FIXED_ALL` → `(c_v, c_t)` as above.
BAR fixtures — `FIXED_U` → single U constraint.
TRUSS fixture (`PIN` at node 1, `FIXED_V` at node 2):
```python
    bc_pin = (
        LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0)),
        LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0)),
    )
    bc_roller = LinearConstraint(node_id=2, coefficients=(0.0, 1.0, 0.0))
    boundary_conditions=(bc_pin[0], bc_pin[1], bc_roller),
```

- [ ] **Step 4: Update `tests/unit/test_plotter.py`**

BEAM fixture (`FIXED_ALL` at node 1):
```python
    c_v = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0))
    c_t = LinearConstraint(node_id=1, coefficients=(0.0, 0.0, 1.0))
    boundary_conditions=(c_v, c_t),
```

- [ ] **Step 5: Update `tests/unit/test_properties.py`**

Replace the property test that uses `get_constrained_dof_indices` / `get_free_dof_indices` with a penalty enforcement property:

```python
from hypothesis import given, settings
import hypothesis.strategies as st
from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.constraints import apply_penalty_constraints, compute_constraint_residuals
from fea_solver.models import (
    Element, ElementType, FEAModel, LinearConstraint,
    MaterialProperties, Mesh, Node,
)
from fea_solver.solver import compute_penalty_parameter


@given(
    node_i_id=st.integers(min_value=1, max_value=10),
    L=st.floats(min_value=0.01, max_value=100.0),
)
@settings(max_examples=50)
def test_penalty_constraint_enforced_after_solve(node_i_id: int, L: float) -> None:
    """After penalty solve, constrained DOF displacement is < 1/penalty_alpha."""
    node_j_id = node_i_id + 1
    mat = MaterialProperties(E=1.0, A=1.0, I=0.0)
    ni = Node(id=node_i_id, pos=(0.0, 0.0))
    nj = Node(id=node_j_id, pos=(L, 0.0))
    elem = Element(id=1, node_i=ni, node_j=nj,
                   element_type=ElementType.BAR, material=mat)
    c = LinearConstraint(node_id=node_i_id, coefficients=(1.0, 0.0, 0.0), rhs=0.0)
    model = FEAModel(
        mesh=Mesh(nodes=(ni, nj), elements=(elem,)),
        boundary_conditions=(c,),
        nodal_loads=(),
        distributed_loads=(),
        penalty_alpha=1e8,
    )
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    k_penalty = compute_penalty_parameter(K, model.penalty_alpha)
    K_mod, F_mod = apply_penalty_constraints(K, F, model.boundary_conditions, dof_map, k_penalty)
    import numpy as np
    u = np.linalg.solve(K_mod, F_mod)
    R = compute_constraint_residuals(u, model.boundary_conditions, dof_map, k_penalty)
    assert R.shape == (1,)
    # Constraint residual gives zero (no load applied, so all DOFs are zero)
    assert abs(u[dof_map.index(node_i_id, DOFType.U)]) < 1.0 / model.penalty_alpha
```

Remove the old import of `get_constrained_dof_indices`, `get_free_dof_indices`, `BoundaryCondition`, `BoundaryConditionType`.

- [ ] **Step 6: Run all unit tests (excluding test_io_yaml which is next)**

```bash
.venv/Scripts/python.exe -m pytest tests/unit/ -v --ignore=tests/unit/test_io_yaml.py
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/unit/test_assembler.py tests/unit/test_postprocessor.py tests/unit/test_plotter.py tests/unit/test_properties.py
git commit -m "test: update all unit test fixtures to use LinearConstraint"
```

---

## Task 8: Update test_io_yaml.py

**Files:**
- Modify: `tests/unit/test_io_yaml.py`

- [ ] **Step 1: Update imports**

Remove `BoundaryConditionType` from the import. Add `LinearConstraint` if not already present.

- [ ] **Step 2: Update `test_boundary_condition_type`**

Replace the existing test (which checks `bc_type == BoundaryConditionType.FIXED_U`) with:

```python
def test_boundary_condition_parsed(self, bar_yaml_path: Path) -> None:
    """Boundary condition is parsed into a LinearConstraint with correct fields."""
    model = load_model_from_yaml(bar_yaml_path)
    assert len(model.boundary_conditions) == 1
    c = model.boundary_conditions[0]
    assert c.node_id == 1
    # [1.0, 0.0, 0.0] after normalization (already unit length)
    assert c.coefficients == pytest.approx((1.0, 0.0, 0.0))
    assert c.rhs == pytest.approx(0.0)
```

- [ ] **Step 3: Update all inline YAML strings in `test_io_yaml.py`**

Find every `boundary_conditions:` block inside YAML strings (the `_MULTI_SOL_YAML` constant and others) and replace `type: fixed_all` / `type: pin` / etc. with the appropriate `coefficients:` entries, following the same keyword translation table as Task 6.

For BEAM elements (`type: fixed_all` → two entries):
```yaml
    boundary_conditions:
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
```

For BAR elements (`type: fixed_u`):
```yaml
    boundary_conditions:
      - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
```

- [ ] **Step 4: Run test_io_yaml.py**

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_io_yaml.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_io_yaml.py
git commit -m "test: update test_io_yaml.py for LinearConstraint schema"
```

---

## Task 9: Update integration tests

**Files:**
- Modify: `tests/integration/test_beam_cases.py`
- Modify: `tests/integration/test_multi_solution.py`

- [ ] **Step 1: Update `test_beam_cases.py` — remove `get_constrained_dof_indices` usage**

The `test_support_reactions_sum_to_applied_load` test in `TestCase03SimplySupported` currently calls `get_constrained_dof_indices` to look up which reactions correspond to V at nodes 1 and 5. With the new system, reactions are indexed by constraint order. After migrating `example_case_03_simply_supported.yaml`, the constraints are:
- Index 0: node 1, [0,1,0] — V reaction
- Index 1: node 5, [0,1,0] — V reaction

Replace the test:

```python
def test_support_reactions_sum_to_applied_load(self) -> None:
    """Support reactions balance applied load."""
    model, dof_map, result, _ = full_solve(CONFIG / "example_case_03_simply_supported.yaml")
    # Both support V-reactions must sum to +1 (upward, balancing P=-1 downward).
    # After migration: constraint 0 = V at node 1, constraint 1 = V at node 5.
    total_v_reaction = sum(
        result.reactions[i]
        for i, c in enumerate(model.boundary_conditions)
        if c.coefficients[1] == pytest.approx(1.0)   # V-direction constraint
    )
    assert abs(total_v_reaction) == pytest.approx(1.0, rel=0.01)
```

Remove the import of `get_constrained_dof_indices` from `test_beam_cases.py`.

- [ ] **Step 2: Update `test_multi_solution.py` — update inline YAML strings**

Find all inline YAML strings in `test_multi_solution.py` that contain `boundary_conditions:` blocks and update them to use `coefficients:` entries instead of `type:` keywords, following the same rules as Tasks 6 and 8.

- [ ] **Step 3: Run integration tests (bar + beam + combined)**

```bash
.venv/Scripts/python.exe -m pytest tests/integration/test_bar_cases.py tests/integration/test_beam_cases.py tests/integration/test_combined_cases.py tests/integration/test_multi_solution.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_beam_cases.py tests/integration/test_multi_solution.py
git commit -m "test: update integration tests for LinearConstraint and penalty reactions"
```

---

## Task 10: Add inclined roller integration test

**Files:**
- Create: `tests/integration/test_inclined_roller.py`
- Create: `config/test_inclined_roller_45deg.yaml`

- [ ] **Step 1: Create the YAML config**

Save as `config/test_inclined_roller_45deg.yaml`:

```yaml
# Inclined roller test: triangle truss, 45-degree roller at node 2
# Node 1 (0,0): pin  Node 2 (2,0): 45-deg roller  Node 3 (1,1): applied load
# EA = 1 for all members (E=1, A=1)
label: "inclined_roller_45deg"
unit_system: SI

mesh:
  nodes:
    - {id: 1, x: 0.0, y: 0.0}
    - {id: 2, x: 2.0, y: 0.0}
    - {id: 3, x: 1.0, y: 1.0}
  elements:
    - {id: 1, node_i: 1, node_j: 3, type: truss, material: unit}
    - {id: 2, node_i: 3, node_j: 2, type: truss, material: unit}
    - {id: 3, node_i: 1, node_j: 2, type: truss, material: unit}

materials:
  unit:
    E: 1.0
    A: 1.0
    I: 0.0

boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}   # pin: fix U at node 1
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}   # pin: fix V at node 1
  # 45-degree inclined roller at node 2: constrain direction (1/sqrt(2), 1/sqrt(2))
  # Normalized: [0.7071067812, 0.7071067812, 0.0]
  - {node_id: 2, coefficients: [0.7071067812, 0.7071067812, 0.0]}

loads:
  nodal:
    - {node_id: 3, type: point_force_y, magnitude: -1.0}
  distributed: []
```

- [ ] **Step 2: Create `tests/integration/test_inclined_roller.py`**

```python
"""Integration test for 45-degree inclined roller using the penalty method.

This test validates the primary new capability of LinearConstraint: constraints
in arbitrary directions that cannot be expressed with the old keyword system.

Triangle truss: nodes at (0,0), (2,0), (1,1). EA=1 for all members.
  - Node 1: pin (fix U and V)
  - Node 2: 45-degree inclined roller (constrain direction [1/sqrt(2), 1/sqrt(2)])
  - Node 3: applied downward load F_y = -1.0 N
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.io_yaml import load_model_from_yaml
from fea_solver.models import DOFType
from fea_solver.solver import run_solve_pipeline

_CONFIG = Path(__file__).parent.parent.parent / "config"


@pytest.fixture(scope="module")
def inclined_roller_solution():
    """Solve the inclined roller test case once for the module."""
    yaml_path = _CONFIG / "example_case_09_inclined_roller.yaml"
    if not yaml_path.exists():
        pytest.skip("example_case_09_inclined_roller.yaml not found")
    model = load_model_from_yaml(yaml_path)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    return model, dof_map, result


class TestInclinedRoller45Deg:
    """Tests for the 45-degree inclined roller triangle truss."""

    def test_solves_without_error(self, inclined_roller_solution) -> None:
        """Model solves without raising any exception."""
        model, dof_map, result = inclined_roller_solution
        assert result.displacements is not None

    def test_inclined_constraint_satisfied(self, inclined_roller_solution) -> None:
        """Inclined roller constraint is satisfied to within penalty tolerance.

        The constraint enforces U_2/sqrt(2) + V_2/sqrt(2) = 0,
        i.e. U_2 = -V_2. Checks that the constraint residual is near zero.
        """
        model, dof_map, result = inclined_roller_solution
        u2 = result.displacements[dof_map.index(2, DOFType.U)]
        v2 = result.displacements[dof_map.index(2, DOFType.V)]
        n = 1.0 / math.sqrt(2.0)
        constraint_violation = abs(n * u2 + n * v2)
        # Violation should be on the order of 1/penalty_alpha
        assert constraint_violation < 1e-5

    def test_loaded_node_deflects_downward(self, inclined_roller_solution) -> None:
        """Node 3 (loaded node) deflects downward under downward applied load."""
        model, dof_map, result = inclined_roller_solution
        v3 = result.displacements[dof_map.index(3, DOFType.V)]
        assert v3 < 0.0

    def test_reactions_shape(self, inclined_roller_solution) -> None:
        """reactions array has one entry per constraint (3 total)."""
        model, dof_map, result = inclined_roller_solution
        assert result.reactions.shape == (3,)

    def test_vertical_equilibrium(self, inclined_roller_solution) -> None:
        """Sum of vertical reaction components balances applied load.

        Constraint 0 (node 1, [1,0,0]): pure horizontal reaction.
        Constraint 1 (node 1, [0,1,0]): pure vertical reaction R1_V.
        Constraint 2 (node 2, [n,n,0]): mixed reaction; vertical component = R2 * n.

        Equilibrium: R1_V + R2 * n = 1.0 (upward = -F_y applied)
        """
        model, dof_map, result = inclined_roller_solution
        n = 1.0 / math.sqrt(2.0)
        r1_v = result.reactions[1]   # pure V at node 1
        r2 = result.reactions[2]     # inclined roller magnitude
        total_vertical = r1_v + r2 * n
        assert abs(total_vertical) == pytest.approx(1.0, rel=0.01)

    def test_displacements_non_zero(self, inclined_roller_solution) -> None:
        """At least one DOF has non-zero displacement under applied load."""
        _, _, result = inclined_roller_solution
        assert not np.allclose(result.displacements, 0.0)
```

- [ ] **Step 3: Run the new test**

```bash
.venv/Scripts/python.exe -m pytest tests/integration/test_inclined_roller.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add config/example_case_09_inclined_roller.yaml tests/integration/test_inclined_roller.py
git commit -m "test: add inclined 45-degree roller integration test"
```

---

## Task 11: Run truss integration tests and full suite

**Files:** (no new changes — verification only)

- [ ] **Step 1: Run truss integration tests**

```bash
.venv/Scripts/python.exe -m pytest tests/integration/test_truss_cases.py -v
```

Expected: all PASS. (These tests only check counts, non-zero displacements, and shear/moment — none reference `get_constrained_dof_indices` or old reaction indexing.)

- [ ] **Step 2: Run full test suite**

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all PASS. If failures remain, they are caused by missed `BoundaryCondition` references — search with:

```bash
.venv/Scripts/python.exe -m pytest tests/ -v 2>&1 | grep FAILED
```

Then grep for remaining old references:

```bash
grep -r "BoundaryCondition\|get_constrained_dof_indices\|get_free_dof_indices\|apply_constraints_reduction" tests/ src/
```

Fix any remaining occurrences following the same substitution rules as Tasks 7–9.

- [ ] **Step 3: Final commit**

```bash
git add -u
git commit -m "feat: complete constraint architecture redesign — LinearConstraint + penalty method"
```
