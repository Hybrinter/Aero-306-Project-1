# Truss Plotting Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the binary tension/compression truss plot with three gradient-colored plots: undeformed+forces, deformed shape, and undeformed+stress.

**Architecture:** Add `result: SolutionResult` to `SolutionSeries` so plot functions have displacement access; replace `plot_truss_axial_forces` with three new public functions that share two private helpers; update all four call/construction sites in `main.py` and tests.

**Tech Stack:** Python 3.12, matplotlib (TwoSlopeNorm, ScalarMappable, coolwarm colormap), numpy, pytest

---

## File Map

| File | Change |
|------|--------|
| `src/fea_solver/models.py` | Add `result: SolutionResult` field to `SolutionSeries` |
| `src/fea_solver/plotter.py` | Remove `plot_truss_axial_forces`; add 2 private helpers + 3 public functions; update imports and module docstring |
| `main.py` | Pass `result=result` to `SolutionSeries`; replace single plot call with three; update imports |
| `tests/unit/test_plotter.py` | Update `_make_series`; add `_make_truss_series`; add `TestTrussPlots`; update imports |
| `tests/integration/test_multi_solution.py` | Pass `result=result` at two `SolutionSeries` construction sites |

---

### Task 1: Add `result` field to `SolutionSeries` and fix all construction sites

**Files:**
- Modify: `src/fea_solver/models.py` (SolutionSeries dataclass, ~lines 467–494)
- Modify: `tests/unit/test_plotter.py` (`_make_series` helper and imports)
- Modify: `tests/integration/test_multi_solution.py` (two construction sites)

- [ ] **Step 1: Write a failing test**

Add to the bottom of `tests/unit/test_plotter.py`:

```python
class TestSolutionSeriesResult:
    """Tests that SolutionSeries carries a SolutionResult."""

    def test_solution_series_has_result_field(self) -> None:
        """SolutionSeries constructed with result field is accessible via .result."""
        from fea_solver.assembler import build_dof_map
        from fea_solver.models import DOFMap, SolutionResult
        model = _make_model()
        dof_map = build_dof_map(model)
        result_stub = SolutionResult(
            displacements=np.zeros(dof_map.total_dofs),
            reactions=np.zeros(0),
            dof_map=dof_map,
            model=model,
        )
        er = make_element_result()
        sol = SolutionSeries(
            label="test",
            element_results=(er,),
            model=model,
            result=result_stub,
        )
        assert sol.result is result_stub
```

- [ ] **Step 2: Run to confirm it fails**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestSolutionSeriesResult -v
```

Expected: `FAILED` — `TypeError: SolutionSeries.__init__() got an unexpected keyword argument 'result'`

- [ ] **Step 3: Add `result` field to `SolutionSeries` in `models.py`**

In `src/fea_solver/models.py`, find the `SolutionSeries` dataclass (near line 467) and add the new field after `model`:

```python
@dataclass(frozen=True, slots=True)
class SolutionSeries:
    """Bundle of post-processed results for one named mesh refinement.

    Groups element-level results with their parent model so that plotting
    functions can render multiple solutions on shared axes without receiving
    parallel lists of disparate types.

    Fields:
        label (str): Short human-readable name for this solution (e.g. "coarse", "fine").
            Used for legend entries and per-solution max/min annotations.
        element_results (tuple[ElementResult, ...]): Immutable sequence of post-processed
            element results for all elements in this solution's mesh.
        model (FEAModel): The FEA model that produced these results; used to retrieve
            unit labels for axis annotations.
        result (SolutionResult): Full solution result carrying the displacement vector
            and DOF map. Required by truss deformed-shape plots to recover nodal (U, V).

    Notes:
        Frozen and slotted, matching the style of all other result containers.
        element_results is a tuple (not list) to satisfy the frozen invariant.
        At call sites, wrap a list with tuple(): tuple(element_results_list).
        The model is carried here (not recovered from ElementResult) because
        ElementResult does not hold a model reference.
    """

    label: str
    element_results: tuple[ElementResult, ...]
    model: FEAModel
    result: SolutionResult
```

- [ ] **Step 4: Update `_make_series` in `tests/unit/test_plotter.py`**

Replace the existing `_make_series` function and update the imports block at the top of the file:

Updated imports (replace the existing `from fea_solver.models import ...` line):

```python
from fea_solver.assembler import build_dof_map
from fea_solver.models import (
    DOFMap, DOFType, Element, ElementResult, ElementType,
    FEAModel, LinearConstraint, MaterialProperties, Mesh, Node,
    SolutionResult, SolutionSeries, UnitSystem,
)
```

Updated `_make_series`:

```python
def _make_series(
    label: str = "test",
    x_start: float = 0.0,
    x_end: float = 1.0,
    unit_system: UnitSystem = UnitSystem.SI,
) -> SolutionSeries:
    """Build a SolutionSeries with a single ElementResult for plotter tests."""
    model = _make_model(unit_system)
    dof_map = build_dof_map(model)
    result_stub = SolutionResult(
        displacements=np.zeros(dof_map.total_dofs),
        reactions=np.zeros(0),
        dof_map=dof_map,
        model=model,
    )
    er = make_element_result(x_start=x_start, x_end=x_end)
    return SolutionSeries(label=label, element_results=(er,), model=model, result=result_stub)
```

- [ ] **Step 5: Update `tests/integration/test_multi_solution.py`**

At lines 93–97, add `result=result`:

```python
all_series.append(SolutionSeries(
    label=model.label.split("/")[-1],
    element_results=tuple(element_results),
    model=model,
    result=result,
))
```

Apply the same change at lines 120–126 (the second construction site, same pattern).

- [ ] **Step 6: Run tests to confirm passing**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py tests/integration/test_multi_solution.py -v
```

Expected: all existing tests pass plus the new `TestSolutionSeriesResult` test.

- [ ] **Step 7: Commit**

```
git add src/fea_solver/models.py tests/unit/test_plotter.py tests/integration/test_multi_solution.py
git commit -m "feat: add result field to SolutionSeries for truss displacement access"
```

---

### Task 2: Add private helpers to `plotter.py`

**Files:**
- Modify: `src/fea_solver/plotter.py` (imports + two new private functions)
- Modify: `tests/unit/test_plotter.py` (new test class)

- [ ] **Step 1: Write failing tests for both helpers**

Add the two helper functions to `tests/unit/test_plotter.py` (after `_make_series`, before the existing `TestPlotShearForceDiagram` class). All imports needed (`build_dof_map`, `DOFType`, `SolutionResult`, etc.) are already present from Task 1 Step 4.

```python
def _make_truss_model() -> FEAModel:
    """Build a minimal two-node TRUSS FEAModel for truss plotter tests."""
    mat = MaterialProperties(E=200e9, A=0.01, I=0.0)
    n1 = Node(1, (0.0, 0.0))
    n2 = Node(2, (3.0, 4.0))
    elem = Element(id=1, node_i=n1, node_j=n2, element_type=ElementType.TRUSS, material=mat)
    c1 = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0))
    c2 = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0))
    return FEAModel(
        mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
        boundary_conditions=(c1, c2),
        nodal_loads=(),
        distributed_loads=(),
        label="test_truss",
    )


def _make_truss_series(axial_force: float = 50.0) -> SolutionSeries:
    """Build a SolutionSeries with a single TRUSS ElementResult."""
    model = _make_truss_model()
    dof_map = build_dof_map(model)
    # TRUSS 2-node: DOFs are (node1,U)=0, (node1,V)=1, (node2,U)=2, (node2,V)=3
    u = np.zeros(dof_map.total_dofs)
    u[dof_map.index(2, DOFType.U)] = 0.001
    u[dof_map.index(2, DOFType.V)] = -0.002
    result = SolutionResult(
        displacements=u,
        reactions=np.zeros(0),
        dof_map=dof_map,
        model=model,
    )
    er = ElementResult(
        element_id=1,
        axial_force=axial_force,
        shear_forces=np.zeros(5),
        bending_moments=np.zeros(5),
        x_stations=np.linspace(0.0, 3.0, 5),
        transverse_displacements=np.zeros(5),
        axial_displacements=np.zeros(5),
        rotations=np.zeros(5),
    )
    return SolutionSeries(label="test", element_results=(er,), model=model, result=result)


class TestTrussHelpers:
    """Tests for private truss plotting helpers."""

    def test_colormap_norm_symmetric_range(self) -> None:
        """Norm vmin and vmax are symmetric about zero."""
        from fea_solver.plotter import _truss_colormap_norm
        cmap, norm = _truss_colormap_norm([10.0, -5.0, 3.0])
        assert norm.vmax == pytest.approx(10.0)
        assert norm.vmin == pytest.approx(-10.0)
        assert norm.vcenter == pytest.approx(0.0)

    def test_colormap_norm_all_zero_fallback(self) -> None:
        """All-zero values use fallback norm vmin=-1, vmax=1."""
        from fea_solver.plotter import _truss_colormap_norm
        cmap, norm = _truss_colormap_norm([0.0, 0.0])
        assert norm.vmax == pytest.approx(1.0)
        assert norm.vmin == pytest.approx(-1.0)

    def test_colormap_returns_coolwarm(self) -> None:
        """Returned colormap is coolwarm."""
        from fea_solver.plotter import _truss_colormap_norm
        cmap, _ = _truss_colormap_norm([1.0])
        assert "coolwarm" in cmap.name

    def test_node_displacements_keys(self) -> None:
        """Node displacement dict has an entry for every node."""
        from fea_solver.plotter import _truss_node_displacements
        sol = _make_truss_series()
        disps = _truss_node_displacements(sol)
        assert set(disps.keys()) == {1, 2}

    def test_node_displacements_values(self) -> None:
        """Extracted displacements match what was put into the displacement vector."""
        from fea_solver.plotter import _truss_node_displacements
        sol = _make_truss_series()
        disps = _truss_node_displacements(sol)
        assert disps[1] == pytest.approx((0.0, 0.0))
        assert disps[2][0] == pytest.approx(0.001)
        assert disps[2][1] == pytest.approx(-0.002)
```

- [ ] **Step 2: Run to confirm tests fail**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestTrussHelpers -v
```

Expected: `FAILED` — `ImportError: cannot import name '_truss_colormap_norm' from 'fea_solver.plotter'`

- [ ] **Step 3: Update `plotter.py` imports**

Add these imports to the existing import block in `src/fea_solver/plotter.py`:

```python
from matplotlib.colors import Colormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
```

Update the `from fea_solver.models import ...` line to add `DOFType`:

```python
from fea_solver.models import DOFType, ElementResult, FEAModel, SolutionSeries
```

- [ ] **Step 4: Add the two private helpers to `plotter.py`**

Insert these two functions immediately after the `_plot_extremes` function (around line 155), before `plot_shear_force_diagram`:

```python
def _truss_colormap_norm(values: list[float]) -> tuple[Colormap, TwoSlopeNorm]:
    """Return a coolwarm Colormap and a TwoSlopeNorm centered at zero.

    The norm is symmetric: vmin = -max_abs, vcenter = 0, vmax = max_abs.
    Falls back to vmin=-1, vmax=1 when all values are zero to avoid a
    degenerate norm.

    Args:
        values (list[float]): Scalar values to be represented (e.g. axial forces
            or stresses). May contain positive, negative, or zero entries.

    Returns:
        tuple[Colormap, TwoSlopeNorm]: The coolwarm colormap and the norm.

    Notes:
        Tension (positive) maps to red end; compression (negative) maps to blue
        end of coolwarm, consistent with the convention that red = high, blue = low.
        The symmetric range ensures zero always maps to the neutral white midpoint.
    """
    max_abs = max((abs(v) for v in values), default=0.0)
    if max_abs == 0.0:
        max_abs = 1.0
    cmap: Colormap = plt.cm.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    return cmap, norm


def _truss_node_displacements(sol: SolutionSeries) -> dict[int, tuple[float, float]]:
    """Extract per-node (U, V) global displacements from a SolutionSeries.

    Args:
        sol (SolutionSeries): Solution bundle whose result.displacements and
            result.dof_map are used to recover nodal translations.

    Returns:
        dict[int, tuple[float, float]]: Maps node_id to (U, V) displacement tuple
            in the model's canonical length units. All nodes in sol.model.mesh.nodes
            are present as keys.

    Notes:
        TRUSS elements always carry both U and V DOFs at every node, so this
        function is safe for all truss meshes without a guard. Using it on
        non-truss models (BAR, BEAM) that lack U or V DOFs will raise KeyError.
    """
    u_vec = sol.result.displacements
    dof_map = sol.result.dof_map
    disps: dict[int, tuple[float, float]] = {}
    for node in sol.model.mesh.nodes:
        U = float(u_vec[dof_map.index(node.id, DOFType.U)])
        V = float(u_vec[dof_map.index(node.id, DOFType.V)])
        disps[node.id] = (U, V)
    return disps
```

- [ ] **Step 5: Run tests to confirm helpers pass**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestTrussHelpers -v
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 6: Commit**

```
git add src/fea_solver/plotter.py tests/unit/test_plotter.py
git commit -m "feat: add _truss_colormap_norm and _truss_node_displacements helpers"
```

---

### Task 3: Implement `plot_truss_forces` and remove old function

**Files:**
- Modify: `src/fea_solver/plotter.py` (remove old function, add new, update docstring)
- Modify: `tests/unit/test_plotter.py` (new test class)

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_plotter.py`:

```python
class TestPlotTrussForces:
    """Tests for plot_truss_forces."""

    def test_returns_figure(self) -> None:
        """plot_truss_forces returns a matplotlib Figure."""
        from fea_solver.plotter import plot_truss_forces
        fig = plot_truss_forces(_make_truss_series())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """plot_truss_forces saves a PNG when output_path is given."""
        from fea_solver.plotter import plot_truss_forces
        out = tmp_path / "forces.png"
        fig = plot_truss_forces(_make_truss_series(), output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_has_colorbar(self) -> None:
        """plot_truss_forces figure contains a colorbar axes."""
        from fea_solver.plotter import plot_truss_forces
        fig = plot_truss_forces(_make_truss_series())
        # colorbar creates an extra Axes in the figure
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_title_set(self) -> None:
        """Custom title is applied to the axes."""
        from fea_solver.plotter import plot_truss_forces
        fig = plot_truss_forces(_make_truss_series(), title="My Forces")
        assert fig.axes[0].get_title() == "My Forces"
        plt.close(fig)

    def test_compression_series_no_crash(self) -> None:
        """plot_truss_forces handles all-compression (negative) forces."""
        from fea_solver.plotter import plot_truss_forces
        fig = plot_truss_forces(_make_truss_series(axial_force=-80.0))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
```

- [ ] **Step 2: Run to confirm tests fail**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestPlotTrussForces -v
```

Expected: `FAILED` — `ImportError: cannot import name 'plot_truss_forces'`

- [ ] **Step 3: Remove `plot_truss_axial_forces` from `plotter.py`**

Delete the entire `plot_truss_axial_forces` function (approximately lines 465–535 in the original file). Also remove the `Line2D` import if it is no longer used elsewhere (it was only used in the legend of the old function).

Verify `Line2D` is not referenced anywhere else:
```
.venv/Scripts/python.exe -c "import ast, pathlib; src=pathlib.Path('src/fea_solver/plotter.py').read_text(); print('Line2D refs:', src.count('Line2D'))"
```
If count is 0, remove `from matplotlib.lines import Line2D` from imports.

- [ ] **Step 4: Add `plot_truss_forces` to `plotter.py`**

Add after the `plot_rotation` function:

```python
def plot_truss_forces(
    sol: SolutionSeries,
    title: str = "Truss Member Forces",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot 2D truss geometry (undeformed) with coolwarm gradient coloring by axial force.

    Members are colored on a continuous diverging scale: blue for compression
    (N < 0), white for zero force, red for tension (N > 0). A colorbar shows
    the force scale. Each member is annotated at its midpoint with the numeric
    force value.

    Args:
        sol (SolutionSeries): Solution bundle containing element results and model.
        title (str): Plot title. Default "Truss Member Forces".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        Node coordinates are the original (undeformed) positions.
        If output_path is provided, saves figure at 150 dpi.
    """
    model = sol.model
    lbl = _unit_labels(model)
    result_by_id = {er.element_id: er for er in sol.element_results}
    nodes_by_id = {n.id: n for n in model.mesh.nodes}

    forces = [
        result_by_id[e.id].axial_force if e.id in result_by_id else 0.0
        for e in model.mesh.elements
    ]
    cmap, norm = _truss_colormap_norm(forces)

    fig, ax = plt.subplots(figsize=(10, 6))

    for element, N in zip(model.mesh.elements, forces):
        n_i = nodes_by_id[element.node_i.id]
        n_j = nodes_by_id[element.node_j.id]
        color = cmap(norm(N))
        ax.plot([n_i.x, n_j.x], [n_i.y, n_j.y], color=color, linewidth=2.5)
        mid_x = (n_i.x + n_j.x) / 2.0
        mid_y = (n_i.y + n_j.y) / 2.0
        ax.text(mid_x, mid_y, f"{N:.3g}", fontsize=7, ha="center", va="bottom",
                color="black")

    for node in model.mesh.nodes:
        ax.plot(node.x, node.y, "ko", markersize=4, zorder=5)
        ax.text(node.x, node.y, f" {node.id}", fontsize=8, va="bottom")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"N [{lbl['force']}]")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"y [{lbl['length']}]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Truss forces plot saved to %s", output_path)

    return fig
```

- [ ] **Step 5: Update the module docstring in `plotter.py`**

Replace the `plot_truss_axial_forces` entry in the module-level docstring with the three new function entries. The updated function list section should read:

```
  - plot_truss_forces:          2D undeformed wireframe colored by axial force gradient
  - plot_truss_deformed:        2D deformed wireframe with auto-scale, colored by axial force
  - plot_truss_stress:          2D undeformed wireframe colored by axial stress (N/A) gradient
  - show_all_plots:             plt.show() wrapper
```

Also add `_truss_colormap_norm` and `_truss_node_displacements` to the private helper list:

```
_truss_colormap_norm:   Returns coolwarm Colormap + TwoSlopeNorm centered at zero.
_truss_node_displacements: Extracts per-node (U, V) from SolutionSeries.result.
```

- [ ] **Step 6: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestPlotTrussForces -v
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 7: Commit**

```
git add src/fea_solver/plotter.py tests/unit/test_plotter.py
git commit -m "feat: add plot_truss_forces with coolwarm gradient, remove binary plot"
```

---

### Task 4: Implement `plot_truss_deformed`

**Files:**
- Modify: `src/fea_solver/plotter.py`
- Modify: `tests/unit/test_plotter.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_plotter.py`:

```python
class TestPlotTrussDeformed:
    """Tests for plot_truss_deformed."""

    def test_returns_figure(self) -> None:
        """plot_truss_deformed returns a matplotlib Figure."""
        from fea_solver.plotter import plot_truss_deformed
        fig = plot_truss_deformed(_make_truss_series())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """plot_truss_deformed saves PNG when output_path given."""
        from fea_solver.plotter import plot_truss_deformed
        out = tmp_path / "deformed.png"
        fig = plot_truss_deformed(_make_truss_series(), output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_scale_factor_in_title(self) -> None:
        """Scale factor string appears in the plot title."""
        from fea_solver.plotter import plot_truss_deformed
        fig = plot_truss_deformed(_make_truss_series(), title="Deformed")
        title_text = fig.axes[0].get_title()
        assert "scale" in title_text
        assert "x" in title_text
        plt.close(fig)

    def test_has_colorbar(self) -> None:
        """plot_truss_deformed figure contains a colorbar axes."""
        from fea_solver.plotter import plot_truss_deformed
        fig = plot_truss_deformed(_make_truss_series())
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_zero_displacement_no_crash(self) -> None:
        """plot_truss_deformed handles zero displacement (scale fallback to 1.0)."""
        from fea_solver.plotter import plot_truss_deformed
        model = _make_truss_model()
        dof_map = build_dof_map(model)
        result_zero = SolutionResult(
            displacements=np.zeros(dof_map.total_dofs),
            reactions=np.zeros(0),
            dof_map=dof_map,
            model=model,
        )
        er = ElementResult(
            element_id=1,
            axial_force=0.0,
            shear_forces=np.zeros(5),
            bending_moments=np.zeros(5),
            x_stations=np.linspace(0.0, 3.0, 5),
            transverse_displacements=np.zeros(5),
            axial_displacements=np.zeros(5),
            rotations=np.zeros(5),
        )
        sol = SolutionSeries(label="zero", element_results=(er,), model=model, result=result_zero)
        fig = plot_truss_deformed(sol)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
```

- [ ] **Step 2: Run to confirm tests fail**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestPlotTrussDeformed -v
```

Expected: `FAILED` — `ImportError: cannot import name 'plot_truss_deformed'`

- [ ] **Step 3: Add `plot_truss_deformed` to `plotter.py`**

Add immediately after `plot_truss_forces`:

```python
def plot_truss_deformed(
    sol: SolutionSeries,
    title: str = "Truss Deformed Shape",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot 2D truss geometry in its deformed state with coolwarm gradient by axial force.

    Node positions are shifted by (scale * U, scale * V) where scale is chosen
    automatically so the largest displacement equals 10 % of the bounding-box
    diagonal. The scale factor is appended to the plot title.

    Args:
        sol (SolutionSeries): Solution bundle. sol.result.displacements provides
            nodal translations; sol.element_results provides axial forces for color.
        title (str): Base plot title. Scale factor is appended automatically.
            Default "Truss Deformed Shape".
        output_path (Path | None): If provided, save figure to this path as PNG.

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

- [ ] **Step 4: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestPlotTrussDeformed -v
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 5: Commit**

```
git add src/fea_solver/plotter.py tests/unit/test_plotter.py
git commit -m "feat: add plot_truss_deformed with auto-scale displacement"
```

---

### Task 5: Implement `plot_truss_stress`

**Files:**
- Modify: `src/fea_solver/plotter.py`
- Modify: `tests/unit/test_plotter.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_plotter.py`:

```python
class TestPlotTrussStress:
    """Tests for plot_truss_stress."""

    def test_returns_figure(self) -> None:
        """plot_truss_stress returns a matplotlib Figure."""
        from fea_solver.plotter import plot_truss_stress
        fig = plot_truss_stress(_make_truss_series())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """plot_truss_stress saves PNG when output_path given."""
        from fea_solver.plotter import plot_truss_stress
        out = tmp_path / "stress.png"
        fig = plot_truss_stress(_make_truss_series(), output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_has_colorbar(self) -> None:
        """plot_truss_stress figure contains a colorbar axes."""
        from fea_solver.plotter import plot_truss_stress
        fig = plot_truss_stress(_make_truss_series())
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_colorbar_label_contains_stress_unit(self) -> None:
        """Colorbar label contains force and length units for SI model."""
        from fea_solver.plotter import plot_truss_stress
        fig = plot_truss_stress(_make_truss_series())
        # Colorbar Axes ylabel is the label
        colorbar_ax = fig.axes[1]
        ylabel = colorbar_ax.get_ylabel()
        assert "N" in ylabel   # force unit
        assert "m" in ylabel   # length unit
        plt.close(fig)

    def test_title_set(self) -> None:
        """Custom title is applied."""
        from fea_solver.plotter import plot_truss_stress
        fig = plot_truss_stress(_make_truss_series(), title="My Stress")
        assert fig.axes[0].get_title() == "My Stress"
        plt.close(fig)
```

- [ ] **Step 2: Run to confirm tests fail**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestPlotTrussStress -v
```

Expected: `FAILED` — `ImportError: cannot import name 'plot_truss_stress'`

- [ ] **Step 3: Add `plot_truss_stress` to `plotter.py`**

Add immediately after `plot_truss_deformed`:

```python
def plot_truss_stress(
    sol: SolutionSeries,
    title: str = "Truss Member Stresses",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot 2D truss geometry (undeformed) with coolwarm gradient coloring by axial stress.

    Axial stress per member: sigma = N / A, where N is the axial force from
    ElementResult and A is the element's cross-sectional area. Members are
    colored on a continuous diverging scale (blue = compression, red = tension)
    with a colorbar showing the stress magnitude. Each member is annotated at
    its midpoint with the numeric stress value.

    Args:
        sol (SolutionSeries): Solution bundle containing element results and model.
        title (str): Plot title. Default "Truss Member Stresses".
        output_path (Path | None): If provided, save figure to this path as PNG.

    Returns:
        plt.Figure: The matplotlib Figure.

    Notes:
        Stress unit label is composed as f"{force_unit}/{length_unit}^2"
        (e.g. "N/m^2" for SI, "lb/in^2" for empirical). No changes to units.py needed.
        Node coordinates are the original (undeformed) positions.
        If output_path is provided, saves figure at 150 dpi.
    """
    model = sol.model
    lbl = _unit_labels(model)
    result_by_id = {er.element_id: er for er in sol.element_results}
    nodes_by_id = {n.id: n for n in model.mesh.nodes}

    stresses = [
        (result_by_id[e.id].axial_force if e.id in result_by_id else 0.0) / e.material.A
        for e in model.mesh.elements
    ]
    cmap, norm = _truss_colormap_norm(stresses)
    stress_unit = f"{lbl['force']}/{lbl['length']}^2"

    fig, ax = plt.subplots(figsize=(10, 6))

    for element, sigma in zip(model.mesh.elements, stresses):
        n_i = nodes_by_id[element.node_i.id]
        n_j = nodes_by_id[element.node_j.id]
        color = cmap(norm(sigma))
        ax.plot([n_i.x, n_j.x], [n_i.y, n_j.y], color=color, linewidth=2.5)
        mid_x = (n_i.x + n_j.x) / 2.0
        mid_y = (n_i.y + n_j.y) / 2.0
        ax.text(mid_x, mid_y, f"{sigma:.3g}", fontsize=7, ha="center", va="bottom",
                color="black")

    for node in model.mesh.nodes:
        ax.plot(node.x, node.y, "ko", markersize=4, zorder=5)
        ax.text(node.x, node.y, f" {node.id}", fontsize=8, va="bottom")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"sigma [{stress_unit}]")

    ax.set_xlabel(f"x [{lbl['length']}]")
    ax.set_ylabel(f"y [{lbl['length']}]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Truss stress plot saved to %s", output_path)

    return fig
```

- [ ] **Step 4: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/unit/test_plotter.py::TestPlotTrussStress -v
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 5: Commit**

```
git add src/fea_solver/plotter.py tests/unit/test_plotter.py
git commit -m "feat: add plot_truss_stress with N/A gradient and stress unit label"
```

---

### Task 6: Update `main.py`

**Files:**
- Modify: `main.py` (imports, SolutionSeries construction, truss plot block)

- [ ] **Step 1: Update the import block in `main.py`**

Replace the `plot_truss_axial_forces` import with the three new names. The updated plotter import block should be:

```python
from fea_solver.plotter import (
    plot_axial_displacement,
    plot_bending_moment_diagram,
    plot_rotation,
    plot_shear_force_diagram,
    plot_transverse_displacement,
    plot_truss_deformed,
    plot_truss_forces,
    plot_truss_stress,
    show_all_plots,
)
```

- [ ] **Step 2: Pass `result` to `SolutionSeries` construction**

Find the `all_series.append(SolutionSeries(...))` call (around line 231) and add `result=result`:

```python
all_series.append(SolutionSeries(
    label=model.label.split("/")[-1],
    element_results=tuple(element_results),
    model=model,
    result=result,
))
```

- [ ] **Step 3: Replace the truss plot block**

Find the `if all_truss:` block (around lines 253–260) and replace its body:

```python
        if all_truss:
            # 2D truss: three plots per solution (gradient force, deformed, stress)
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

- [ ] **Step 4: Run the full test suite**

```
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all tests `PASSED`.

- [ ] **Step 5: Commit**

```
git add main.py
git commit -m "feat: wire up three truss plot functions in main pipeline"
```

---

### Task 7: Final verification and cleanup

**Files:** Read-only verification pass

- [ ] **Step 1: Run full test suite one final time**

```
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all tests `PASSED` with zero failures or errors.

- [ ] **Step 2: Smoke-test the CLI against a truss config**

```
.venv/Scripts/python.exe main.py config/problem_5.yaml --save-plots outputs/
```

Expected: exits with code 0; `outputs/` contains `problem_5_truss_forces.png`, `problem_5_truss_deformed.png`, `problem_5_truss_stress.png`.

- [ ] **Step 3: Confirm no reference to old function remains**

```
.venv/Scripts/python.exe -c "import pathlib; files = list(pathlib.Path('.').rglob('*.py')); hits = [(f, i+1, l) for f in files for i, l in enumerate(f.read_text().splitlines()) if 'plot_truss_axial_forces' in l]; print(hits)"
```

Expected: empty list `[]`.

- [ ] **Step 4: Final commit if any cleanup was needed**

```
git add -u
git commit -m "chore: final cleanup after truss plotting enhancement"
```
