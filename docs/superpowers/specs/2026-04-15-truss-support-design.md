# Truss Element Support — Design Spec

**Date:** 2026-04-15  
**Author:** aiden_kamp  
**Status:** Approved

---

## Context

The FEA solver currently handles 1D structural elements (BAR, BEAM, FRAME) where all nodes
lie along the global x-axis. Problems 5, 6, and 7 require 2D pin-jointed truss structures
where elements may be inclined at arbitrary angles.

Adding truss support requires:
1. 2D node coordinates
2. A coordinate-transformation-based stiffness matrix (local -> global)
3. New element type with 2 DOFs/node (U, V — no rotation)
4. Updated post-processing, reporting, and plotting

Problems 5-7 are all pure trusses (pin-connected joints, members carry axial force only).
Problem 7 specifies I = 1/12 mm^4 for completeness but this is not used in the stiffness.

---

## Constraints

- Must remain fully backward-compatible with problems 1-4 (BAR, BEAM, FRAME)
- Existing YAML configs will be updated to include `y: 0.0` on every node (required field)
- No new files for the core solver modules — extend existing files only
- TDD: failing tests written before each implementation step
- Strong typing throughout; all new dataclasses use `@dataclass(frozen=True, slots=True)`

---

## Architecture

### 1. Data Models (`src/fea_solver/models.py`)

**`Node`**: Replace `x: float` with `pos: tuple[float, float]`. Add `@property x` and `@property y`
for backward compatibility of all access expressions. Constructor call sites (~28 total) updated
to `Node(id, (x, y))`.

**`ElementType`**: Add `TRUSS = auto()` — 2 DOFs/node: [U, V] (global x/y displacement). No
rotation DOF (pin joints by definition).

**`Element.length`**: Update from `abs(node_j.x - node_i.x)` to `sqrt((Dx)^2 + (Dy)^2)`.
Backward-compatible since existing elements have y=0.

No changes to `MaterialProperties`, `BoundaryCondition`, `NodalLoad`, `DistributedLoad`,
`DOFMap`, `SolutionResult`, `ElementResult`, `SolutionSeries`.

---

### 2. Element Stiffness (`src/fea_solver/elements.py`)

New function `truss_stiffness_matrix(element: Element) -> NDArray[np.float64]`:

```
c = (xj - xi) / L
s = (yj - yi) / L

         [ c^2   cs   -c^2  -cs ]
k_g = EA/L * [ cs    s^2  -cs  -s^2 ]
         [-c^2  -cs    c^2   cs ]
         [-cs   -s^2   cs    s^2 ]

DOF ordering: [U_i, V_i, U_j, V_j]
```

`element_stiffness_matrix()` dispatch router: add `TRUSS -> truss_stiffness_matrix`.

`element_load_vector()`: raise `NotImplementedError` for `TRUSS` (no distributed loads on
truss members — distributed loading is a beam/frame concept).

---

### 3. Assembly (`src/fea_solver/assembler.py`)

Add to `_ELEMENT_DOFS`:

```python
ElementType.TRUSS: (DOFType.U, DOFType.V)
```

The existing `_dofs_for_node()` union logic and `build_dof_map()` already handle this
correctly — a node connected to both a TRUSS and another element type will get the
union of DOF types. No other changes needed.

---

### 4. Constraints (`src/fea_solver/constraints.py`)

No changes required. Existing BC types cover all truss support conditions:

- `PIN` -> constrains U and V (standard truss pin support)
- `ROLLER` -> constrains V only (horizontal roller)
- `FIXED_U` -> constrains U only (vertical roller)

---

### 5. YAML Parser (`src/fea_solver/io_yaml.py`)

- `_NodeSchema`: add `y: float` (required, no default — all node entries must include y)
- `_schema_to_model()`: construct `Node(id=n.id, pos=(conv.convert(n.x, "length"), conv.convert(n.y, "length")))`
- `_ELEMENT_TYPE_MAP`: add `"truss": ElementType.TRUSS`
- Distributed load path: unchanged — TRUSS elements will raise `NotImplementedError` in
  `element_load_vector()` if distributed loads are applied, giving clear feedback

---

### 6. Post-Processing (`src/fea_solver/postprocessor.py`)

New function `compute_truss_axial_force(element_id, model, dof_map, u) -> float`:

```
c = (xj - xi) / L,  s = (yj - yi) / L
N = (EA / L) * (c * (U_j - U_i) + s * (V_j - V_i))
```

`postprocess_all_elements()` gains a TRUSS branch:
- `axial_force = N` (constant along element)
- `x_stations` = linspace from `xi` to `xj` (global x coordinate along element)
- `axial_displacements` = linear interpolation of the projected axial deformation at each station
- `shear_forces`, `bending_moments`, `transverse_displacements`, `rotations` = zeros

---

### 7. Plotter (`src/fea_solver/plotter.py`)

New function `plot_truss_structure(series: list[SolutionSeries], ...)`:
- Draws undeformed wireframe (grey dashed) and deformed wireframe (solid)
- Member forces annotated: tension = blue, compression = red
- Works with the existing `SolutionSeries` container

Main entry-point guard: if all elements are TRUSS type, skip existing 1D plot functions
(`plot_sfd`, `plot_bmd`, `plot_deformed_shape`) and call `plot_truss_structure` instead.

---

### 8. Reporter (`src/fea_solver/reporter.py`)

Node displacement table: add `y` column when any node has `y != 0.0`.

Element results table: for TRUSS elements, show axial force only — suppress shear/moment rows.

---

## Files Modified

| File | Change |
|------|--------|
| `src/fea_solver/models.py` | Node.pos, ElementType.TRUSS |
| `src/fea_solver/elements.py` | truss_stiffness_matrix, dispatcher |
| `src/fea_solver/assembler.py` | _ELEMENT_DOFS entry |
| `src/fea_solver/io_yaml.py` | _NodeSchema.y, Node constructor, element type map |
| `src/fea_solver/postprocessor.py` | compute_truss_axial_force, TRUSS branch |
| `src/fea_solver/plotter.py` | plot_truss_structure, 1D plot guard |
| `src/fea_solver/reporter.py` | y column, TRUSS result row |
| `tests/conftest.py` | Node constructor updates |
| `tests/unit/test_elements.py` | Node constructors + new truss tests |
| `tests/unit/test_assembler.py` | Node constructors |
| `tests/unit/test_constraints.py` | Node constructors |
| `tests/unit/test_solver.py` | Node constructors |
| `tests/unit/test_postprocessor.py` | Node constructors + new truss tests |
| `tests/unit/test_properties.py` | Node constructors |
| `tests/unit/test_plotter.py` | Node constructors |
| `tests/unit/test_io_yaml.py` | New truss YAML parsing tests |
| `tests/integration/test_truss_cases.py` | NEW — problems 5, 6, 7 integration tests |
| `config/problem_1.yaml` | Add y: 0.0 to all nodes |
| `config/problem_2.yaml` | Add y: 0.0 to all nodes |
| `config/problem_3.yaml` | Add y: 0.0 to all nodes |
| `config/problem_4.yaml` | Add y: 0.0 to all nodes |
| `config/problem_5.yaml` | NEW — diamond truss (4 nodes, 5 elements, EA=2000) |
| `config/problem_6.yaml` | NEW — bridge truss (8 nodes, EA=2020, F=50 at 3 nodes) |
| `config/problem_7.yaml` | NEW — 9-node truss (E=3003 MPa, A=1 mm^2, F=50 N) |

---

## TDD Order

Each group: write failing test -> implement -> confirm pass.

1. `Node.pos` + `x`/`y` properties (models.py) + migrate all constructors
2. `Element.length` 2D formula
3. `ElementType.TRUSS` + `_ELEMENT_DOFS` wiring
4. `truss_stiffness_matrix` — unit tests for 0 deg, 45 deg, 90 deg
5. YAML parser — `y` field required, `"truss"` element type
6. `compute_truss_axial_force` — unit tests
7. `postprocess_all_elements` TRUSS branch
8. Integration tests for problems 5, 6, 7 (YAML configs written first)
9. `plot_truss_structure` + plotter guard
10. Reporter `y` column + TRUSS row

---

## Verification

```bash
# All existing tests pass (regression check after each step)
.venv/Scripts/python.exe -m pytest tests/ -v

# Truss-specific integration tests
.venv/Scripts/python.exe -m pytest tests/integration/test_truss_cases.py -v

# Smoke-run each problem config
uv run python main.py config/problem_5.yaml --no-plot
uv run python main.py config/problem_6.yaml --no-plot
uv run python main.py config/problem_7.yaml --no-plot
```

Expected verification targets (from problem statements):

**Problem 5** (diamond truss, EA=2000, P=100 at node 1):
- All nodal displacements match textbook solution (JW notes Ch.4 p.14)
- 5 member forces computed, tensile/compressive signs correct

**Problem 6** (bridge truss, EA=2020, F=50 at nodes 6, 7, 8 downward):
- Reactions at nodes 1 and 5 (pin and roller supports)
- 13 member forces

**Problem 7** (9-node truss, E=3003 MPa, A=1 mm^2, F=50 N):
- Nodal displacements in mm
- Member forces in N
