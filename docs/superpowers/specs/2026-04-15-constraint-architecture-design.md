# Constraint Architecture Redesign

**Date:** 2026-04-15
**Status:** Approved

## Problem Statement

The current constraint system uses a fixed set of named keyword presets
(`pin`, `roller`, `fixed_u`, etc.) mapped to global DOF types via a lookup
table. This has two concrete problems:

1. **Semantic ambiguity across element types.** `PIN` silently constrains
   only the DOFs that exist at a node. A `PIN` on a BEAM node constrains only
   V (not U), because BEAM has no U DOF. This behaviour is non-obvious and
   can cause incorrect models without any error.

2. **No support for oblique constraints.** There is no way to express a
   constraint in an arbitrary direction (e.g. a roller on a 30-degree inclined
   surface). The keyword set is finite and hardcoded to global axes.

## Goals

- Replace keyword-based constraints with a general linear constraint equation
  per entry.
- Support constraints in any direction without special-casing.
- Switch constraint enforcement from the reduction (elimination) method to the
  penalty method, which applies uniformly to all constraint types.
- Clean break: existing YAML config files are rewritten; no backward
  compatibility shims.

## Architecture

### 1. Data Model

`BoundaryCondition` and `BoundaryConditionType` are removed entirely from
`models.py`. They are replaced by:

```python
@dataclass(frozen=True, slots=True)
class LinearConstraint:
    """One scalar linear constraint equation applied at a node.

    Encodes: a_U * u + a_V * v + a_THETA * theta = rhs

    Fields:
        node_id (int): Node at which the constraint is applied.
        coefficients (tuple[float, float, float]): Constraint direction vector
            in [U, V, THETA] DOF order (global coordinates). Must be non-zero.
            Components for DOFs that do not exist at the node must be 0.0;
            a non-zero component for a missing DOF raises ValueError at
            constraint application time.
        rhs (float): Prescribed displacement/rotation value. Default 0.0
            (homogeneous constraint).
    """
    node_id: int
    coefficients: tuple[float, float, float]
    rhs: float = 0.0
```

`FEAModel` changes:
- `boundary_conditions: tuple[BoundaryCondition, ...]` becomes
  `boundary_conditions: tuple[LinearConstraint, ...]`
- A new optional field `penalty_alpha: float = 1e8` is added. The penalty
  stiffness is computed as `k_penalty = penalty_alpha * max(abs(diag(K)))`.

`SolutionResult` changes:
- `reactions: NDArray[np.float64]` changes meaning from "reaction force
  indexed by constrained global DOF" to "constraint force magnitude indexed
  by constraint", with `reactions[i] = k_penalty * (a_i^T * u - rhs_i)`.
  Shape is `(n_constraints,)`.

### 2. Constraint Enforcement (constraints.py)

The reduction-method functions are removed:
- `get_constrained_dof_indices`
- `get_free_dof_indices`
- `apply_constraints_reduction`

Two new functions replace them:

**`apply_penalty_constraints(K, F, constraints, dof_map, k_penalty) -> (K, F)`**

Accepts copies of K and F; returns new modified arrays (does not mutate the
inputs, consistent with the codebase's immutability style).

For each `LinearConstraint`:
1. For each of the three DOF types `[U, V, THETA]` in order, if the
   corresponding coefficient is non-zero, look up its global index via
   `dof_map.index(node_id, dof_type)`. If the DOF does not exist at the node,
   raise `ValueError` (no silent projection).
2. Build a global coefficient vector `g` (length `n_dofs`) with nonzero
   entries placed at the resolved DOF indices.
3. `K_out = K + k_penalty * outer(g, g)`
4. `F_out = F + k_penalty * rhs * g`

Returns the new `(K_out, F_out)` pair.

**`compute_constraint_residuals(u, constraints, dof_map, k_penalty) -> NDArray`**

For each constraint, returns `k_penalty * (a^T * u_node - rhs)` as a scalar.
Result shape: `(n_constraints,)`.

### 3. Solver (solver.py)

The solver no longer partitions `K`/`F`. It solves the full `n x n` system
`K_mod * u = F_mod` where `K_mod` and `F_mod` are the penalty-modified
matrices. The existing condition number check is retained; it will produce
large condition numbers (proportional to `penalty_alpha`) as expected.

Reaction recovery changes from `K_cf * u_f - F_c` to
`compute_constraint_residuals(u, constraints, dof_map, k_penalty)`.

### 4. YAML Schema (io_yaml.py)

`_BCSchema` is replaced by `_LinearConstraintSchema`:

```python
class _LinearConstraintSchema(BaseModel):
    node_id: int
    coefficients: list[float]   # must have exactly 3 elements, non-zero magnitude
    rhs: float = 0.0
```

`_schema_to_model` normalizes each coefficient vector to unit length before
constructing the `LinearConstraint`. This ensures that reported residual
magnitudes from `compute_constraint_residuals` are in consistent physical
units (force or moment) regardless of how the user scales their input vector.
Users may supply any non-zero vector; `[1, 0, 0]` and `[2, 0, 0]` produce
identical constraints after normalization.

`_BC_TYPE_MAP` is removed. `_FEAModelSchema` and `_SolutionEntrySchema` gain
`penalty_alpha: float = 1e8`.

Example YAML entries (common support conditions):

```yaml
# Pin at node 1 (fix U and V)
boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}

# Roller at node 3 (fix V only)
  - {node_id: 3, coefficients: [0.0, 1.0, 0.0]}

# Inclined roller at 45 degrees at node 5
  - {node_id: 5, coefficients: [0.707, 0.707, 0.0]}

# Fixed support at node 6 (fix U, V, and THETA)
  - {node_id: 6, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 6, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 6, coefficients: [0.0, 0.0, 1.0]}
```

Coefficient mapping for all previous keywords:

| Old keyword  | New entries (one row = one constraint)    |
|--------------|-------------------------------------------|
| `fixed_u`    | `[1, 0, 0]`                               |
| `fixed_v`    | `[0, 1, 0]`                               |
| `fixed_theta`| `[0, 0, 1]`                               |
| `fixed_all`  | `[1, 0, 0]`, `[0, 1, 0]`, `[0, 0, 1]`    |
| `pin`        | `[1, 0, 0]`, `[0, 1, 0]`                 |
| `roller`     | `[0, 1, 0]`                               |

All 15 existing config files in `config/` are rewritten to use the new schema.

### 5. Testing Strategy

**Unit tests (`tests/unit/test_constraints.py`)** — rewritten:
- `apply_penalty_constraints`: verify K/F modifications for axis-aligned,
  inclined, and rotational constraints; verify `ValueError` for non-zero
  coefficient on a missing DOF.
- `compute_constraint_residuals`: verify near-zero residuals post-solve;
  verify magnitude matches `k_penalty * (a^T * u - rhs)`.
- Penalty scaling: verify `k_penalty = penalty_alpha * max(abs(diag(K)))`.

**Integration tests** — existing bar/beam/frame/truss test cases updated to
new YAML schema. Results must match old golden values to within `1e-6`
relative tolerance (the penalty approximation error is ~`1/penalty_alpha`).

**New inclined roller test** — 2-node truss with a 45-degree roller at one
end, verified against an analytical solution. This test cannot be expressed
with the old keyword system.

## Files Changed

| File | Change |
|------|--------|
| `src/fea_solver/models.py` | Remove `BoundaryCondition`, `BoundaryConditionType`; add `LinearConstraint`; update `FEAModel` |
| `src/fea_solver/constraints.py` | Full rewrite: penalty method replaces reduction method |
| `src/fea_solver/solver.py` | Solve full system; update reaction recovery |
| `src/fea_solver/io_yaml.py` | Replace `_BCSchema`/`_BC_TYPE_MAP`; add `penalty_alpha` |
| `config/*.yaml` (all 15) | Rewrite boundary_conditions blocks |
| `tests/unit/test_constraints.py` | Full rewrite |
| `tests/integration/test_*.py` | Update to new YAML schema; add inclined roller test |
