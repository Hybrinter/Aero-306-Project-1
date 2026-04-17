# Truss Plotting Enhancement — Design Spec

**Date:** 2026-04-17  
**Branch:** optimization-challenge  
**Status:** Approved

---

## Problem Statement

The existing `plot_truss_axial_forces` function draws truss geometry with binary tension/compression coloring (blue/red by sign). It provides no magnitude gradient, no deformed shape, and no stress visualization. This spec replaces it with three distinct, gradient-colored plots.

---

## Goals

1. Replace the binary tension/compression plot with a continuous `coolwarm` gradient showing force magnitude.
2. Add a deformed-shape plot with auto-computed displacement scale factor.
3. Add an axial stress (`sigma = N/A`) plot with the same gradient approach.

---

## Section 1: Data Model

### Change: `SolutionSeries` gains `result: SolutionResult`

**File:** `src/fea_solver/models.py`

Add one required field to `SolutionSeries`:

```python
result: SolutionResult
```

`SolutionResult` is already defined earlier in the same file — no import changes needed. The field is required (no default) to preserve the project's strong-typing convention. Every real solve produces a `SolutionResult`, so a default would be meaningless.

**Construction sites updated:**
- `main.py` — one site, pass `result` as the fourth keyword argument
- `tests/unit/test_plotter.py` — `_make_series` helper updated to pass a minimal stub `SolutionResult`

Truss integration tests (`test_truss_cases.py`) do not construct `SolutionSeries` and are unaffected.

---

## Section 2: `plotter.py` Changes

### Remove

`plot_truss_axial_forces` — deleted entirely.

### Add: Private helpers

**`_truss_colormap_norm(values: list[float]) -> tuple[Colormap, TwoSlopeNorm]`**

Returns the `coolwarm` colormap and a `TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)`.
Falls back to `vmin=-1, vmax=1` when all values are zero to avoid a degenerate norm.

**`_truss_node_displacements(sol: SolutionSeries) -> dict[int, tuple[float, float]]`**

Iterates `sol.model.mesh.nodes`, extracts `(U, V)` per node from `sol.result.displacements`
indexed via `sol.result.dof_map`. Returns `{node_id: (U, V)}`.

### Add: Three public plot functions

All follow the existing signature pattern:
```python
def plot_truss_<name>(
    sol: SolutionSeries,
    title: str = "...",
    output_path: Path | None = None,
) -> plt.Figure:
```

---

#### `plot_truss_forces`

**What it shows:** Undeformed geometry, members colored by axial force `N`.

- Geometry: original node coordinates.
- Color: `coolwarm` colormap, normalized to `[-max_abs_N, +max_abs_N]`.
- Colorbar labeled: `N [force_unit]` (e.g. `N [N]` for SI, `N [lb]` for empirical).
- Midpoint text annotation: force value per member.
- Node markers (black circles) + node ID labels.

---

#### `plot_truss_deformed`

**What it shows:** Deformed geometry (displaced node positions), members colored by axial force `N`.

- Scale factor: `scale = 0.1 * bbox_diagonal / max_abs_displacement`
  - `bbox_diagonal = sqrt((max_x - min_x)^2 + (max_y - min_y)^2)` over original node coords.
  - Falls back to `scale = 1.0` if all displacements are zero.
- Node positions: `(x + scale * U, y + scale * V)`.
- Color: same `coolwarm` + `TwoSlopeNorm` as `plot_truss_forces`.
- Colorbar labeled: `N [force_unit]`.
- Midpoint text annotation: force value per member, at deformed midpoints.
- Node markers at deformed positions + node ID labels.
- Scale factor in title: `f"{title} (scale {scale:.2g}x)"`.

---

#### `plot_truss_stress`

**What it shows:** Undeformed geometry, members colored by axial stress `sigma = N/A`.

- Geometry: original node coordinates.
- Stress per element: `sigma = er.axial_force / element.material.A`.
- Color: `coolwarm` normalized to `[-max_abs_sigma, +max_abs_sigma]`.
- Stress unit label composed inline: `f"{lbl['force']}/{lbl['length']}^2"`
  - SI → `"N/m^2"`, Empirical → `"lb/in^2"`. No changes to `units.py`.
- Colorbar labeled: `sigma [stress_unit]`.
- Midpoint text annotation: stress value per member.
- Node markers (black circles) + node ID labels.

---

### Module docstring update

The `plotter.py` module docstring is updated to list:
- `plot_truss_forces`
- `plot_truss_deformed`
- `plot_truss_stress`

And remove the old `plot_truss_axial_forces` entry.

---

## Section 3: `main.py` Changes

### `SolutionSeries` construction

```python
all_series.append(SolutionSeries(
    label=model.label.split("/")[-1],
    element_results=tuple(element_results),
    model=model,
    result=result,          # new
))
```

### Truss plot block

Replace the single `plot_truss_axial_forces` call with three calls per solution:

| Output filename                    | Function              |
|------------------------------------|-----------------------|
| `{label}_truss_forces.png`         | `plot_truss_forces`   |
| `{label}_truss_deformed.png`       | `plot_truss_deformed` |
| `{label}_truss_stress.png`         | `plot_truss_stress`   |

### Import block

Remove `plot_truss_axial_forces`. Add `plot_truss_forces`, `plot_truss_deformed`, `plot_truss_stress`.

---

## Section 4: Test Changes

### `tests/unit/test_plotter.py`

**`_make_series` helper** — updated to build a minimal stub `SolutionResult`:
- Zero displacements array sized to the model's DOF count.
- Zero reactions array.
- A real `DOFMap` built from the model.
- The model itself.

Passed as `result=` to `SolutionSeries`.

**New `TestTrussPlots` class** — uses a minimal two-node TRUSS model (not the existing BEAM model) so DOF types and displacement indexing are correct.

Tests:
- `test_truss_forces_returns_figure`
- `test_truss_forces_saves_to_file`
- `test_truss_deformed_returns_figure`
- `test_truss_deformed_saves_to_file`
- `test_truss_stress_returns_figure`
- `test_truss_stress_saves_to_file`
- `test_truss_deformed_scale_annotation` — asserts scale factor string appears in plot title

---

## Edge Cases

| Situation | Handling |
|-----------|----------|
| All member forces zero | `_truss_colormap_norm` uses fallback `vmin=-1, vmax=1` |
| All displacements zero | `plot_truss_deformed` uses `scale = 1.0` |
| Single-member truss | Functions work; colorbar still shown |
| Mixed sign forces | `TwoSlopeNorm` centers at zero naturally |

---

## Files Changed

| File | Change |
|------|--------|
| `src/fea_solver/models.py` | Add `result: SolutionResult` to `SolutionSeries` |
| `src/fea_solver/plotter.py` | Remove old function; add 2 helpers + 3 public functions; update docstring |
| `main.py` | Pass `result` to `SolutionSeries`; call three new plot functions; update imports |
| `tests/unit/test_plotter.py` | Update `_make_series`; add `TestTrussPlots`; update imports |
