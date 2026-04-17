# Truss Euler Buckling Analysis -- Design

**Date**: 2026-04-17
**Branch**: optimization-challenge
**Scope**: Add Euler buckling analysis for 2D pin-jointed truss members and
visualize failed members as sinusoidal mode-shape overlays on the existing
deformed-truss plot.

## Goals

1. Compute Euler critical load `P_cr = pi**2 * E * I / L**2` for every TRUSS
   element in a solved model.
2. Flag members where compressive axial force magnitude exceeds `P_cr` as
   buckled (`|N| >= P_cr` and `N < 0`).
3. Overlay a half-sine lateral bow on each buckled member in the existing
   truss deformed-shape plot.
4. Emit a dedicated console "Buckling Summary" table per solution listing
   `N`, `P_cr`, `|N|/P_cr`, and status (`SAFE` / `BUCKLED` / `TENSION`) per
   member.

## Non-Goals

- No Johnson parabolic transition, no yield-stress interaction, no
  slenderness check.
- No per-element effective length factor `K` (implicit `K = 1`, pin-pin).
- No eigenvalue buckling analysis (geometric stiffness is not assembled).
- No buckling support for BAR / BEAM / FRAME elements; truss-only feature.
- No new CLI flag (amplitude is fixed at `0.1 * L`).

## Formulation

For each TRUSS element `e` with material `(E, A, I)` and length `L`:

    P_cr(e) = pi**2 * E * I / L**2

`P_cr` is always positive; the element's axial force `N` carries sign
(positive tension, negative compression per existing postprocessor).

Definitions:

- `ratio = 0.0` when `N >= 0` (tension or zero).
- `ratio = abs(N) / P_cr` when `N < 0`.
- `is_buckled = (N < 0) and (abs(N) >= P_cr)`.

## Architecture

Pipeline extension:

    YAML -> FEAModel -> DOFMap -> K,F -> solve -> postprocess (ElementResult)
                                                 |
                                                 v
                                compute_truss_buckling(model, element_results)
                                                 |
                                                 v
                              tuple[MemberBuckling, ...]  (truss elements only)
                                                 |
                             +-------------------+-------------------+
                             v                                       v
                 print_buckling_summary(...)             plot_truss_deformed(buckling=...)

No new driver wiring outside the existing `all_truss` branch in `main.py`.

## Components

### 1. `models.py` -- new dataclass

    @dataclass(frozen=True, slots=True)
    class MemberBuckling:
        element_id: int
        P_cr: float         # always positive, [force]
        axial_force: float  # signed, from ElementResult.axial_force
        ratio: float        # |N|/P_cr for compression, 0.0 for tension
        is_buckled: bool    # True iff N < 0 and |N| >= P_cr

Immutable container. No new fields on `ElementResult` or `SolutionResult`
(keeps those structs free of feature-specific data).

### 2. `buckling.py` -- new module

Public functions:

- `compute_member_P_cr(element: Element) -> float`
  Returns `pi**2 * E * I / L**2`. Raises `ValueError` when
  `element.material.I <= 0` (catches placeholder I values in truss YAMLs).

- `compute_truss_buckling(model: FEAModel, element_results: Sequence[ElementResult]) -> tuple[MemberBuckling, ...]`
  Iterates `model.mesh.elements`, skips non-TRUSS elements, looks up the
  matching `ElementResult.axial_force` by `element_id`, builds one
  `MemberBuckling` per truss element. Returns a tuple (immutable).

Module docstring summarises formulation and both functions.

### 3. `plotter.py` -- extend `plot_truss_deformed`

Add one optional parameter:

    def plot_truss_deformed(
        sol: SolutionSeries,
        title: str = "Truss Deformed Shape",
        output_path: Path | None = None,
        buckling: tuple[MemberBuckling, ...] | None = None,
    ) -> plt.Figure

When `buckling` is `None` the function behaves identically to today
(baseline unchanged). When provided, for each element whose matching
`MemberBuckling.is_buckled` is `True`:

1. Sample `xi = linspace(0, 1, 30)` along the deformed member axis.
2. Compute the deformed-member axis unit vector `(cos_a, sin_a)` from
   deformed endpoint coordinates and its unit perpendicular
   `(-sin_a, cos_a)`.
3. Amplitude `A = 0.1 * L_def` where `L_def` is the deformed chord length.
4. Plot curve
       `x(xi) = x_i_def + xi * (x_j_def - x_i_def) + A * sin(pi*xi) * (-sin_a)`
       `y(xi) = y_i_def + xi * (y_j_def - y_i_def) + A * sin(pi*xi) * (cos_a)`
   as a black dashed line, `linewidth=1.5`, `zorder=4` (above member line
   at default zorder=2, below node markers at `zorder=5`).

No changes to `plot_truss_forces` or `plot_truss_stress`.

### 4. `reporter.py` -- new function

    def print_buckling_summary(
        bucklings: Sequence[MemberBuckling],
        model: FEAModel,
    ) -> None

Uses `rich.table.Table` titled "Buckling Summary". Columns:

| Element | N [force_unit] | P_cr [force_unit] | \|N\|/P_cr | Status |

Status rendering:

- `BUCKLED` -- bold red -- when `is_buckled` is True.
- `SAFE` -- green -- when `N < 0` and `ratio < 1.0`.
- `TENSION` -- dim white -- when `N >= 0`.

Force units pulled from `UNIT_LABELS[model.unit_system]["force"]`.

No-op (empty bucklings tuple) prints nothing.

### 5. `main.py` -- wire-up

Inside the existing `if all_truss:` branch, per solution:

1. `bucklings = compute_truss_buckling(sol.model, sol.element_results)`
2. `print_buckling_summary(bucklings, sol.model)` (console output).
3. Pass `buckling=bucklings` to `plot_truss_deformed(...)`.

`plot_truss_forces` and `plot_truss_stress` calls remain unchanged.

## Data Flow (per solution, all-truss branch)

1. Existing pipeline produces `SolutionSeries` with `element_results`.
2. `compute_truss_buckling` reads `element.material.I`, `element.length`,
   `element.material.E`, and `ElementResult.axial_force` to build the
   tuple.
3. `print_buckling_summary` emits one console table.
4. `plot_truss_deformed(buckling=...)` draws deformed members plus bows
   on buckled ones.

## Error Handling

- `I <= 0` on any TRUSS element -> `compute_member_P_cr` raises
  `ValueError(f"Element {id}: I must be > 0 for buckling")`. Surfaces bad
  YAML input early.
- Non-TRUSS element encountered inside the truss loop: skipped (no entry
  in the returned tuple).
- `ElementResult` missing for an element id: skipped (defensive; should
  not occur since `postprocess_all_elements` produces one result per
  element).
- Empty mesh or non-truss mesh: `compute_truss_buckling` returns empty
  tuple; downstream consumers tolerate empty input as no-op.

## Testing

Placed in existing test tree with style matching current unit tests.

### Unit tests (`tests/unit/test_buckling.py`)

1. `test_compute_member_P_cr_known_values` -- `E=200e9`, `A=1e-4`,
   `I=1e-8`, `L=1.0` yields `P_cr ~= 19.739 N` within 1e-6 relative.
2. `test_compute_member_P_cr_raises_on_zero_I`.
3. `test_compute_truss_buckling_compression_buckled` -- craft
   `ElementResult.axial_force = -1000.0` against a member whose `P_cr`
   is `100.0`; assert `is_buckled`, `ratio == 10.0`.
4. `test_compute_truss_buckling_compression_safe` -- `N = -50`, `P_cr =
   100`; assert `is_buckled is False`, `ratio == 0.5`.
5. `test_compute_truss_buckling_tension_marked_safe` -- `N = +500`;
   assert `ratio == 0.0`, `is_buckled is False`.
6. `test_compute_truss_buckling_skips_non_truss` -- mixed mesh with one
   BAR element returns tuple of length equal to the TRUSS element count.

### Unit tests (`tests/unit/test_plotter.py` extension)

7. `test_plot_truss_deformed_accepts_buckling_kwarg` -- call with
   `buckling=None` and with a non-empty buckling tuple; assert
   `Figure` returned and correct number of `Line2D` artists on the
   axes (2 * N_members baseline + 1 per buckled member for the bow).

### Unit tests (`tests/unit/test_reporter.py` extension)

8. `test_print_buckling_summary_empty_prints_nothing` -- captured
   output is empty string.
9. `test_print_buckling_summary_renders_statuses` -- capture stdout
   and assert all three tokens (`BUCKLED`, `SAFE`, `TENSION`) appear
   given a mixed input tuple.

### Integration test (`tests/integration/test_truss_buckling.py`)

10. Load `config/problem_7.yaml`, run full solve + postprocess +
    `compute_truss_buckling`, assert tuple length equals truss element
    count (16) and that at least one compressive member exists.

### Smoke

11. `uv run python main.py config/problem_7.yaml --save-plots outputs/`
    completes without exception and produces the deformed-plot PNG.

## Files Changed / Added

- `src/fea_solver/models.py` -- add `MemberBuckling` dataclass.
- `src/fea_solver/buckling.py` -- new file (~60 lines).
- `src/fea_solver/plotter.py` -- one optional kwarg, ~20 lines of overlay
  logic in `plot_truss_deformed`.
- `src/fea_solver/reporter.py` -- new `print_buckling_summary` (~40 lines).
- `main.py` -- 3-line wiring inside `all_truss` branch.
- `tests/unit/test_buckling.py` -- new file.
- `tests/unit/test_plotter.py` -- one new test.
- `tests/unit/test_reporter.py` -- two new tests.
- `tests/integration/test_truss_buckling.py` -- new file.

## Out of Scope / Future Work

- Per-element effective length factor `K`.
- Johnson short-column transition or yield interaction.
- Full eigenvalue buckling (geometric stiffness matrix).
- Buckling mode-shape overlay for BAR / BEAM / FRAME elements.
