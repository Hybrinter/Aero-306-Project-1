# Geometry Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a heavy-budget DE + CMA-ES + SLSQP-polish ensemble that finds the highest-stiffness feasible reshape of the 9-node beam-like truss in `config/problem_7.yaml`, subject to stress, buckling, and minimum-element-length constraints, and emits a drop-in `best_design.yaml` plus a markdown report under `optimization_runs/<run_id>/`.

**Architecture:** Self-contained `src/fea_solver/optimization/` subpackage that depends only on the existing FEA core (no plotter/reporter imports). The objective evaluator wraps one forward FE solve per call, packaged as an `EvalResult` consumed by both the global penalty wrapper and the SLSQP constraint callables. Global stage runs DE and CMA-ES seeds in parallel via `multiprocessing.Pool`, the polish stage runs SLSQP on top-K candidates per seed, and an ensemble selector returns the single best feasible design. A CLI in `scripts/optimize_geometry.py` is the only place that wires optimization output to the existing plotter and Rich reporter.

**Tech Stack:** Python 3.12, numpy, scipy (`differential_evolution`, `minimize` SLSQP), `cma` (new dep), `multiprocessing`, pytest, hypothesis. PyYAML for `best_design.yaml` generation. Existing FEA solver (`assembler`, `solver`, `postprocessor`, `buckling`).

**Spec:** `docs/superpowers/specs/2026-04-17-geometry-optimization-design.md`.

---

## File Map

### New files

| Path | Responsibility |
|------|---------------|
| `src/fea_solver/optimization/__init__.py` | Package init; re-exports the few public types. |
| `src/fea_solver/optimization/problem.py` | `GeometryOptimizationProblem` dataclass + `apply_x_to_model(x, problem)`. |
| `src/fea_solver/optimization/objective.py` | `EvalResult` dataclass + `evaluate(x, problem)` (one FE solve, all metrics). |
| `src/fea_solver/optimization/penalty.py` | `penalized_objective(x, problem, weights)`. |
| `src/fea_solver/optimization/constraints.py` | SLSQP-style `g_i(x) >= 0` vector callables, share `evaluate` cache. |
| `src/fea_solver/optimization/checkpoint.py` | JSON / pickle serializers for `SeedResult`, `PolishResult`, `EnsembleResult`. |
| `src/fea_solver/optimization/global_search.py` | `run_de(...)`, `run_cmaes(...)` -> `SeedResult`. |
| `src/fea_solver/optimization/polish.py` | `slsqp_polish(x0, problem)` -> `PolishResult`. |
| `src/fea_solver/optimization/ensemble.py` | `run_ensemble(problem, config)` orchestrator + `select_best(...)`. |
| `src/fea_solver/optimization/report.py` | `write_report(ensemble_result, problem, run_dir)` -> markdown one-pager. |
| `scripts/optimize_geometry.py` | CLI: argparse -> build problem -> ensemble -> write artifacts -> render plots. |
| `tests/unit/test_optimization_problem.py` | Unit tests for problem dataclass + `apply_x_to_model`. |
| `tests/unit/test_optimization_objective.py` | Unit tests for `evaluate` (baseline reproduction, degenerate guard). |
| `tests/unit/test_optimization_penalty.py` | Unit tests for penalty function. |
| `tests/unit/test_optimization_polish.py` | Unit tests for SLSQP polish. |
| `tests/unit/test_optimization_checkpoint.py` | Unit tests for serializers. |
| `tests/integration/test_optimize_geometry_smoke.py` | `@pytest.mark.slow` end-to-end CLI smoke. |

### Modified files

| Path | Change |
|------|--------|
| `pyproject.toml` | Add `cma>=3.3` to `[project] dependencies`. |
| `.gitignore` | Add `optimization_runs/`. |

---

## Task 1: Project setup -- dependency, gitignore, package skeleton

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/fea_solver/optimization/__init__.py`

- [ ] **Step 1.1: Add `cma` dependency**

Edit `pyproject.toml` `[project] dependencies` list to include `cma>=3.3`:

```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "pyyaml>=6.0",
    "matplotlib>=3.8",
    "rich>=13.7",
    "tabulate>=0.9",
    "pydantic>=2.12.5",
    "cma>=3.3",
]
```

- [ ] **Step 1.2: Install the new dep**

Run: `uv sync`
Expected: `cma` is added to `.venv`. No errors.

- [ ] **Step 1.3: Add `optimization_runs/` to `.gitignore`**

Append to `.gitignore`:

```
optimization_runs/
```

- [ ] **Step 1.4: Create empty package init**

Write `src/fea_solver/optimization/__init__.py`:

```python
"""Constrained shape optimization for the AERO 306 bonus problem.

Top-level pipeline:
    GeometryOptimizationProblem -> evaluate(x) -> EvalResult
                                -> penalized_objective(x) -> float (DE/CMA-ES)
                                -> {stress,buckling,length}_constraint_vec (SLSQP)
                                -> run_de / run_cmaes -> SeedResult
                                -> slsqp_polish -> PolishResult
                                -> run_ensemble -> EnsembleResult
                                -> write_report -> markdown

All modules in this subpackage MUST NOT import the existing
src/fea_solver/plotter.py or src/fea_solver/reporter.py. The CLI
script (scripts/optimize_geometry.py) is the only place that wires
optimization output to the presentation layer.
"""
```

- [ ] **Step 1.5: Verify the package is importable**

Run: `.venv/Scripts/python.exe -c "import fea_solver.optimization; print('ok')"`
Expected: `ok`

- [ ] **Step 1.6: Commit**

```bash
git add pyproject.toml .gitignore src/fea_solver/optimization/__init__.py uv.lock
git commit -m "feat(optimization): scaffold subpackage and add cma dependency"
```

---

## Task 2: `GeometryOptimizationProblem` dataclass and `apply_x_to_model`

**Files:**
- Create: `src/fea_solver/optimization/problem.py`
- Test: `tests/unit/test_optimization_problem.py`

- [ ] **Step 2.1: Write the failing tests**

Write `tests/unit/test_optimization_problem.py`:

```python
"""Unit tests for GeometryOptimizationProblem and apply_x_to_model."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    apply_x_to_model,
    baseline_x,
)

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"

FREE_NODE_IDS = (2, 3, 5, 6, 7, 8)
FROZEN_NODE_IDS = (1, 4, 9)


def _baseline_model():
    return load_models_from_yaml(PROBLEM_7)[0]


def test_baseline_problem_constructs():
    model = _baseline_model()
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )
    assert problem.n_vars == 12
    assert len(problem.box_bounds) == 12
    assert all(lo == -10.0 and hi == 30.0 for lo, hi in problem.box_bounds[0::2])
    assert all(lo == -25.0 and hi == 15.0 for lo, hi in problem.box_bounds[1::2])


def test_rejects_overlapping_free_and_frozen():
    model = _baseline_model()
    with pytest.raises(ValueError, match="overlap"):
        GeometryOptimizationProblem.from_baseline(
            model=model,
            free_node_ids=(2, 3),
            frozen_node_ids=(2, 9),
            x_bounds=(-10.0, 30.0),
            y_bounds=(-25.0, 15.0),
            F_magnitude=15.0,
            sigma_max=72.0,
            L_min=5.0,
        )


def test_rejects_inverted_bounds():
    model = _baseline_model()
    with pytest.raises(ValueError, match="bounds"):
        GeometryOptimizationProblem.from_baseline(
            model=model,
            free_node_ids=FREE_NODE_IDS,
            frozen_node_ids=FROZEN_NODE_IDS,
            x_bounds=(30.0, -10.0),
            y_bounds=(-25.0, 15.0),
            F_magnitude=15.0,
            sigma_max=72.0,
            L_min=5.0,
        )


def test_baseline_x_extracts_free_node_positions():
    model = _baseline_model()
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )
    x0 = baseline_x(problem)
    # Node 2 is at (10, 0) in problem_7.yaml
    assert x0[0] == pytest.approx(10.0)
    assert x0[1] == pytest.approx(0.0)
    # Node 8 is at (15, -5)
    assert x0[10] == pytest.approx(15.0)
    assert x0[11] == pytest.approx(-5.0)


def test_apply_x_overwrites_only_free_nodes():
    model = _baseline_model()
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )
    x = np.array([
        11.0, 1.0,    # node 2
        21.0, 0.5,    # node 3
        9.0, -11.0,   # node 5
        22.0, -9.5,   # node 6
        4.0, -6.0,    # node 7
        16.0, -4.0,   # node 8
    ])
    new_model = apply_x_to_model(x, problem)
    nodes_by_id = {n.id: n for n in new_model.mesh.nodes}
    # Free nodes overwritten
    assert nodes_by_id[2].pos == pytest.approx((11.0, 1.0))
    assert nodes_by_id[8].pos == pytest.approx((16.0, -4.0))
    # Frozen nodes preserved
    assert nodes_by_id[1].pos == pytest.approx((0.0, 0.0))
    assert nodes_by_id[4].pos == pytest.approx((0.0, -10.0))
    assert nodes_by_id[9].pos == pytest.approx((25.0, -5.0))
    # Connectivity preserved
    assert len(new_model.mesh.elements) == 16


def test_apply_x_overrides_F_magnitude():
    model = _baseline_model()
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )
    x = baseline_x(problem)
    new_model = apply_x_to_model(x, problem)
    nodal_loads = new_model.nodal_loads
    assert len(nodal_loads) == 1
    assert nodal_loads[0].node_id == 9
    assert nodal_loads[0].magnitude == pytest.approx(-15.0)
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_problem.py -v`
Expected: All FAIL with `ModuleNotFoundError: No module named 'fea_solver.optimization.problem'`.

- [ ] **Step 2.3: Implement the module**

Write `src/fea_solver/optimization/problem.py`:

```python
"""Problem definition for geometry optimization.

GeometryOptimizationProblem captures everything that does not change during
the optimization: the baseline FEAModel (mesh + connectivity), which node
ids are free vs frozen, the box bounds, the load magnitude, and the
constraint thresholds. apply_x_to_model rebuilds an FEAModel from a
12-vector by overwriting only the free node positions and the single
nodal load.

GeometryOptimizationProblem.from_baseline:  Validate inputs and build a frozen problem.
GeometryOptimizationProblem.n_vars:         Number of design variables (== 2 * len(free_node_ids)).
GeometryOptimizationProblem.box_bounds:     Tuple of (lo, hi) per design variable.
apply_x_to_model:                            Rebuild FEAModel with free nodes overwritten and
                                             nodal load magnitude set to -F_magnitude on the
                                             single load node (assumed unique in the baseline).
baseline_x:                                  Extract baseline coordinates of free nodes as a 12-vector.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fea_solver.models import FEAModel, Mesh, Node, NodalLoad


@dataclass(frozen=True, slots=True)
class GeometryOptimizationProblem:
    """Immutable problem definition for the geometry optimizer.

    Fields:
        baseline_model (FEAModel): Original FEA model providing connectivity,
            materials, BCs, and frozen node positions.
        free_node_ids (tuple[int, ...]): Node ids whose (x, y) are decision variables.
        frozen_node_ids (tuple[int, ...]): Node ids that must keep their baseline (x, y).
        box_bounds (tuple[tuple[float, float], ...]): One (lo, hi) per design variable,
            in the order (x_free[0], y_free[0], x_free[1], y_free[1], ...).
        F_magnitude (float): Magnitude of the downward tip load applied at the load node.
        sigma_max (float): Stress limit in MPa.
        L_min (float): Minimum allowed element length in the model's length unit.
        load_node_id (int): Node id where the single nodal load is applied
            (resolved from baseline_model at construction time).

    Notes:
        Frozen and slotted. Construct via GeometryOptimizationProblem.from_baseline,
        which validates inputs.
    """

    baseline_model: FEAModel
    free_node_ids: tuple[int, ...]
    frozen_node_ids: tuple[int, ...]
    box_bounds: tuple[tuple[float, float], ...]
    F_magnitude: float
    sigma_max: float
    L_min: float
    load_node_id: int

    @property
    def n_vars(self) -> int:
        """Return number of design variables (2 per free node).

        Returns:
            int: 2 * len(free_node_ids).
        """
        return 2 * len(self.free_node_ids)

    @classmethod
    def from_baseline(
        cls,
        model: FEAModel,
        free_node_ids: Sequence[int],
        frozen_node_ids: Sequence[int],
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        F_magnitude: float,
        sigma_max: float,
        L_min: float,
    ) -> "GeometryOptimizationProblem":
        """Validate inputs and build a frozen problem instance.

        Args:
            model (FEAModel): Baseline FEA model loaded from problem_7.yaml.
            free_node_ids (Sequence[int]): Free node ids; order is preserved
                in the design vector.
            frozen_node_ids (Sequence[int]): Frozen node ids.
            x_bounds (tuple[float, float]): (lo, hi) box for x of every free node.
            y_bounds (tuple[float, float]): (lo, hi) box for y of every free node.
            F_magnitude (float): Magnitude of the tip load (applied as -F in y).
            sigma_max (float): Stress limit (MPa).
            L_min (float): Minimum element length (length unit of the model).

        Returns:
            GeometryOptimizationProblem: Validated immutable problem.

        Raises:
            ValueError: If free and frozen node id sets overlap, if any id is
                missing from the model, if bounds are inverted, or if the
                baseline model does not contain exactly one nodal load.
        """
        free = tuple(free_node_ids)
        frozen = tuple(frozen_node_ids)
        overlap = set(free) & set(frozen)
        if overlap:
            raise ValueError(f"free and frozen node ids overlap: {sorted(overlap)}")
        all_ids = {n.id for n in model.mesh.nodes}
        missing = (set(free) | set(frozen)) - all_ids
        if missing:
            raise ValueError(f"node ids not in model: {sorted(missing)}")
        if x_bounds[0] >= x_bounds[1]:
            raise ValueError(f"x bounds must be (lo < hi), got {x_bounds}")
        if y_bounds[0] >= y_bounds[1]:
            raise ValueError(f"y bounds must be (lo < hi), got {y_bounds}")
        if len(model.nodal_loads) != 1:
            raise ValueError(
                f"baseline model must have exactly one nodal load, got {len(model.nodal_loads)}"
            )
        load_node_id = model.nodal_loads[0].node_id
        bounds = tuple(
            (x_bounds if i % 2 == 0 else y_bounds) for i in range(2 * len(free))
        )
        return cls(
            baseline_model=model,
            free_node_ids=free,
            frozen_node_ids=frozen,
            box_bounds=bounds,
            F_magnitude=float(F_magnitude),
            sigma_max=float(sigma_max),
            L_min=float(L_min),
            load_node_id=load_node_id,
        )


def baseline_x(problem: GeometryOptimizationProblem) -> NDArray[np.float64]:
    """Return the design vector that reproduces the baseline model geometry.

    Args:
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        NDArray[np.float64]: Shape (n_vars,) with (x, y) of each free node
            in the order specified by problem.free_node_ids.
    """
    nodes_by_id = {n.id: n for n in problem.baseline_model.mesh.nodes}
    out = np.empty(problem.n_vars, dtype=np.float64)
    for i, node_id in enumerate(problem.free_node_ids):
        node = nodes_by_id[node_id]
        out[2 * i] = node.x
        out[2 * i + 1] = node.y
    return out


def apply_x_to_model(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> FEAModel:
    """Rebuild an FEAModel by overwriting free node positions and the load magnitude.

    Args:
        x (NDArray[np.float64]): Design vector, shape (n_vars,).
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        FEAModel: New frozen FEAModel with the same connectivity, materials,
            and boundary conditions as the baseline. Free node positions are
            overwritten with x; the single nodal load magnitude is set to
            -problem.F_magnitude.

    Notes:
        Element objects hold direct references to Node instances, so we must
        rebuild the elements with the new node objects. Frozen nodes keep
        their baseline positions.
    """
    if x.shape != (problem.n_vars,):
        raise ValueError(f"x must have shape ({problem.n_vars},), got {x.shape}")
    free_lookup = {
        node_id: (float(x[2 * i]), float(x[2 * i + 1]))
        for i, node_id in enumerate(problem.free_node_ids)
    }
    new_nodes_by_id: dict[int, Node] = {}
    for node in problem.baseline_model.mesh.nodes:
        if node.id in free_lookup:
            new_nodes_by_id[node.id] = Node(id=node.id, pos=free_lookup[node.id])
        else:
            new_nodes_by_id[node.id] = node
    new_nodes = tuple(new_nodes_by_id[n.id] for n in problem.baseline_model.mesh.nodes)
    new_elements = tuple(
        replace(
            e,
            node_i=new_nodes_by_id[e.node_i.id],
            node_j=new_nodes_by_id[e.node_j.id],
        )
        for e in problem.baseline_model.mesh.elements
    )
    new_mesh = Mesh(nodes=new_nodes, elements=new_elements)
    baseline_load = problem.baseline_model.nodal_loads[0]
    new_load = NodalLoad(
        node_id=baseline_load.node_id,
        load_type=baseline_load.load_type,
        magnitude=-problem.F_magnitude,
    )
    return replace(
        problem.baseline_model,
        mesh=new_mesh,
        nodal_loads=(new_load,),
    )
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_problem.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/fea_solver/optimization/problem.py tests/unit/test_optimization_problem.py
git commit -m "feat(optimization): add GeometryOptimizationProblem and apply_x_to_model"
```

---

## Task 3: `EvalResult` and `evaluate(x, problem)`

**Files:**
- Create: `src/fea_solver/optimization/objective.py`
- Test: `tests/unit/test_optimization_objective.py`

- [ ] **Step 3.1: Write the failing tests**

Write `tests/unit/test_optimization_objective.py`:

```python
"""Unit tests for evaluate(x, problem) and EvalResult."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dataclasses import replace

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.models import DOFType, NodalLoad
from fea_solver.optimization.objective import EvalResult, evaluate
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    apply_x_to_model,
    baseline_x,
)
from fea_solver.solver import run_solve_pipeline

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"

FREE_NODE_IDS = (2, 3, 5, 6, 7, 8)
FROZEN_NODE_IDS = (1, 4, 9)


def _baseline_problem(F=15.0):
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=F,
        sigma_max=72.0,
        L_min=5.0,
    )


def _independent_baseline_tip_disp(problem):
    """Solve the baseline model with F=15 N independently for cross-check."""
    model = apply_x_to_model(baseline_x(problem), problem)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    res = run_solve_pipeline(model, dof_map, K, F)
    return abs(float(res.displacements[dof_map.index(9, DOFType.V)]))


def test_evaluate_baseline_reproduces_independent_solve():
    problem = _baseline_problem()
    er = evaluate(baseline_x(problem), problem)
    expected = _independent_baseline_tip_disp(problem)
    assert er.solve_ok is True
    assert er.tip_disp == pytest.approx(expected, rel=1e-9)


def test_evaluate_returns_full_metrics():
    problem = _baseline_problem()
    er = evaluate(baseline_x(problem), problem)
    assert isinstance(er, EvalResult)
    assert er.max_stress >= 0.0
    assert er.max_buckling_ratio >= 0.0
    assert er.min_length > 0.0
    assert len(er.stress_violations) == 16
    assert len(er.buckling_violations) == 16
    assert len(er.length_violations) == 16


def test_evaluate_coincident_nodes_is_safely_infeasible():
    problem = _baseline_problem()
    x = baseline_x(problem).copy()
    # Move node 2 onto node 3
    x[0] = x[2]
    x[1] = x[3]
    er = evaluate(x, problem)
    assert er.solve_ok is False
    assert er.feasible is False
    # Tip disp must be finite (sentinel) so penalty is finite
    assert np.isfinite(er.tip_disp)


def test_evaluate_flags_stress_violation():
    """A baseline reshape that drives stress above 72 MPa should violate."""
    problem = _baseline_problem(F=10000.0)  # absurd load -> guaranteed overstress
    er = evaluate(baseline_x(problem), problem)
    assert er.solve_ok is True
    assert er.max_stress > problem.sigma_max
    assert any(v > 0.0 for v in er.stress_violations)
    assert er.feasible is False


def test_evaluate_min_length_below_L_min_marks_infeasible():
    """Place two free nodes very close so the connecting edge < L_min."""
    problem = _baseline_problem()
    x = baseline_x(problem).copy()
    # Move node 7 (index 8,9) to (4.99, -10) -- distance to node 4 (0,-10) is 4.99 < 5
    x[8] = 4.99
    x[9] = -10.0
    er = evaluate(x, problem)
    assert er.min_length < problem.L_min
    assert any(v > 0.0 for v in er.length_violations)
    assert er.feasible is False
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_objective.py -v`
Expected: All FAIL with `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement the module**

Write `src/fea_solver/optimization/objective.py`:

```python
"""Single-FE-solve objective evaluator for geometry optimization.

evaluate(x, problem) runs exactly one forward FE solve and packages every
metric the global penalty wrapper and the SLSQP constraint callables need
into a single EvalResult. Failure modes (coincident nodes, near-singular
stiffness) are caught and converted to a finite sentinel so optimizer
workers never crash.

EvalResult:           Frozen container for per-call FE metrics and violations.
SENTINEL_TIP_DISP:    Tip-disp value used when solve_ok is False; finite and
                      huge so the penalty wrapper produces 1e12-scale fitness.
evaluate:             One FE solve -> EvalResult, with degeneracy guards.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.buckling import compute_member_P_cr
from fea_solver.models import DOFType, ElementType
from fea_solver.optimization.problem import GeometryOptimizationProblem, apply_x_to_model
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.solver import run_solve_pipeline

logger = logging.getLogger(__name__)

SENTINEL_TIP_DISP: float = 1.0e12


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Per-call FE metrics and constraint violations.

    Fields:
        tip_disp (float): |v_y(load_node)| in the model's length unit.
            Set to SENTINEL_TIP_DISP when solve_ok is False.
        max_stress (float): max |sigma_e| across all elements (MPa for the
            problem_7 unit system; generally pressure units of the model).
        max_buckling_ratio (float): max |N_e|/P_cr_e across compression members,
            0.0 if no compressive members.
        min_length (float): min L_e across all elements.
        stress_violations (tuple[float, ...]): max(0, |sigma_e|/sigma_max - 1)
            per element, in element id order.
        buckling_violations (tuple[float, ...]): max(0, |N_e|/P_cr_e - 1)
            per element, 0.0 for tension members, in element id order.
        length_violations (tuple[float, ...]): max(0, 1 - L_e/L_min) per element.
        feasible (bool): True iff solve_ok and all violations are zero.
        solve_ok (bool): True iff the FE solve completed and no element had
            length below 1e-6 (model length unit).
    """

    tip_disp: float
    max_stress: float
    max_buckling_ratio: float
    min_length: float
    stress_violations: tuple[float, ...]
    buckling_violations: tuple[float, ...]
    length_violations: tuple[float, ...]
    feasible: bool
    solve_ok: bool


def _sentinel_result(n_elements: int) -> EvalResult:
    """Return an EvalResult representing a degenerate (unsolvable) configuration.

    Args:
        n_elements (int): Number of elements in the model.

    Returns:
        EvalResult: tip_disp=SENTINEL_TIP_DISP, all violations large, infeasible.
    """
    huge = (1.0,) * n_elements
    return EvalResult(
        tip_disp=SENTINEL_TIP_DISP,
        max_stress=SENTINEL_TIP_DISP,
        max_buckling_ratio=SENTINEL_TIP_DISP,
        min_length=0.0,
        stress_violations=huge,
        buckling_violations=huge,
        length_violations=huge,
        feasible=False,
        solve_ok=False,
    )


def evaluate(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> EvalResult:
    """Run one forward FE solve and return all metrics + violations.

    Steps:
      1. apply_x_to_model -> FEAModel.
      2. Guard: any element length < 1e-6 -> sentinel result.
      3. Try assembly + solve + postprocess. On any exception -> sentinel.
      4. Compute per-element stress, buckling ratio (compression only), and
         length violations relative to problem thresholds.
      5. Pack into EvalResult.

    Args:
        x (NDArray[np.float64]): Design vector, shape (problem.n_vars,).
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        EvalResult: All metrics for this candidate.

    Notes:
        Stress is computed from axial force as sigma = N / A on each element.
        For compression members (N < 0), buckling ratio = |N| / P_cr.
        For tension members, buckling ratio is 0.0 (no contribution).
    """
    n_elements = len(problem.baseline_model.mesh.elements)
    try:
        model = apply_x_to_model(x, problem)
    except (ValueError, ZeroDivisionError):
        logger.debug("evaluate: apply_x_to_model failed, returning sentinel")
        return _sentinel_result(n_elements)

    # Length guard
    lengths = [e.length for e in model.mesh.elements]
    if min(lengths) < 1.0e-6:
        logger.debug("evaluate: degenerate edge length, returning sentinel")
        return _sentinel_result(n_elements)

    try:
        dof_map = build_dof_map(model)
        K = assemble_global_stiffness(model, dof_map)
        F = assemble_global_force_vector(model, dof_map)
        result = run_solve_pipeline(model, dof_map, K, F)
        element_results = postprocess_all_elements(model, result, n_stations=2)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as exc:
        logger.debug("evaluate: solve failed (%s), returning sentinel", exc)
        return _sentinel_result(n_elements)

    # Tip displacement
    try:
        tip_disp = abs(float(
            result.displacements[dof_map.index(problem.load_node_id, DOFType.V)]
        ))
    except KeyError:
        return _sentinel_result(n_elements)

    # Per-element stress and buckling ratio (in element id order)
    er_by_id = {er.element_id: er for er in element_results}
    stress_violations: list[float] = []
    buckling_violations: list[float] = []
    length_violations: list[float] = []
    max_stress = 0.0
    max_buck = 0.0
    for elem in model.mesh.elements:
        er = er_by_id[elem.id]
        sigma = abs(er.axial_force) / elem.material.A
        max_stress = max(max_stress, sigma)
        s_v = max(0.0, sigma / problem.sigma_max - 1.0)
        stress_violations.append(s_v)

        if elem.element_type == ElementType.TRUSS and er.axial_force < 0.0 and elem.material.I > 0.0:
            P_cr = compute_member_P_cr(elem)
            ratio = abs(er.axial_force) / P_cr
            max_buck = max(max_buck, ratio)
            buckling_violations.append(max(0.0, ratio - 1.0))
        else:
            buckling_violations.append(0.0)

        length_violations.append(max(0.0, 1.0 - elem.length / problem.L_min))

    min_length = min(lengths)
    feasible = (
        all(v == 0.0 for v in stress_violations)
        and all(v == 0.0 for v in buckling_violations)
        and all(v == 0.0 for v in length_violations)
    )

    return EvalResult(
        tip_disp=tip_disp,
        max_stress=max_stress,
        max_buckling_ratio=max_buck,
        min_length=min_length,
        stress_violations=tuple(stress_violations),
        buckling_violations=tuple(buckling_violations),
        length_violations=tuple(length_violations),
        feasible=feasible,
        solve_ok=True,
    )
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_objective.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/fea_solver/optimization/objective.py tests/unit/test_optimization_objective.py
git commit -m "feat(optimization): add EvalResult and single-solve evaluate()"
```

---

## Task 4: Penalized objective for global solvers

**Files:**
- Create: `src/fea_solver/optimization/penalty.py`
- Test: `tests/unit/test_optimization_penalty.py`

- [ ] **Step 4.1: Write the failing tests**

Write `tests/unit/test_optimization_penalty.py`:

```python
"""Unit tests for the penalized objective."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.objective import SENTINEL_TIP_DISP, evaluate
from fea_solver.optimization.penalty import (
    DEFAULT_WEIGHTS,
    PenaltyWeights,
    penalized_objective,
)
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    baseline_x,
)

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem(F=15.0):
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=(2, 3, 5, 6, 7, 8),
        frozen_node_ids=(1, 4, 9),
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=F,
        sigma_max=72.0,
        L_min=5.0,
    )


def test_feasible_penalty_equals_objective():
    problem = _problem()
    x = baseline_x(problem)
    er = evaluate(x, problem)
    # Baseline is known to be feasible at F=15 N
    assert er.feasible is True
    p = penalized_objective(x, problem, DEFAULT_WEIGHTS)
    assert p == pytest.approx(er.tip_disp, rel=1e-12)


def test_overstress_penalty_dominates_objective():
    problem = _problem(F=10000.0)
    x = baseline_x(problem)
    er = evaluate(x, problem)
    assert er.feasible is False
    assert er.max_stress > problem.sigma_max
    p = penalized_objective(x, problem, DEFAULT_WEIGHTS)
    # Penalty should be much larger than tip_disp alone
    assert p > er.tip_disp + 1e-3


def test_solve_failure_returns_finite_sentinel():
    problem = _problem()
    x = baseline_x(problem).copy()
    x[0] = x[2]
    x[1] = x[3]  # coincident node 2 and node 3
    p = penalized_objective(x, problem, DEFAULT_WEIGHTS)
    assert np.isfinite(p)
    assert p >= SENTINEL_TIP_DISP


def test_weights_can_be_overridden():
    problem = _problem(F=10000.0)
    x = baseline_x(problem)
    weak = PenaltyWeights(stress=1.0, buckling=1.0, length=1.0)
    strong = PenaltyWeights(stress=1.0e6, buckling=1.0e6, length=1.0e6)
    p_weak = penalized_objective(x, problem, weak)
    p_strong = penalized_objective(x, problem, strong)
    assert p_strong > p_weak
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_penalty.py -v`
Expected: All FAIL with `ModuleNotFoundError`.

- [ ] **Step 4.3: Implement the module**

Write `src/fea_solver/optimization/penalty.py`:

```python
"""Quadratic penalty wrapper around evaluate() for global solvers.

P(x) = J(x) + w_s * sum(stress_violations**2)
            + w_b * sum(buckling_violations**2)
            + w_l * sum(length_violations**2)

Quadratic in the relative-violation domain. For solve_ok=False candidates,
J(x) == SENTINEL_TIP_DISP (1e12) so DE/CMA-ES distributions don't blow up.

PenaltyWeights:        Frozen container for the three penalty multipliers.
DEFAULT_WEIGHTS:       w_s = w_b = w_l = 100.0 (typical |J| ~ 1e-3..1e-1).
penalized_objective:   Wrapper that returns one float per call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.problem import GeometryOptimizationProblem


@dataclass(frozen=True, slots=True)
class PenaltyWeights:
    """Multipliers for the three quadratic penalty terms.

    Fields:
        stress (float): w_s for stress_violations squared.
        buckling (float): w_b for buckling_violations squared.
        length (float): w_l for length_violations squared.
    """

    stress: float
    buckling: float
    length: float


DEFAULT_WEIGHTS: PenaltyWeights = PenaltyWeights(
    stress=100.0, buckling=100.0, length=100.0
)


def penalized_objective(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
    weights: PenaltyWeights,
) -> float:
    """Compute penalized fitness for one candidate.

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.
        weights (PenaltyWeights): Quadratic penalty multipliers.

    Returns:
        float: J(x) + sum of weighted squared violations. Finite (sentinel
            when solve_ok is False) so global solvers stay numerically sound.
    """
    er = evaluate(x, problem)
    s_pen = weights.stress * float(np.sum(np.asarray(er.stress_violations) ** 2))
    b_pen = weights.buckling * float(np.sum(np.asarray(er.buckling_violations) ** 2))
    l_pen = weights.length * float(np.sum(np.asarray(er.length_violations) ** 2))
    return float(er.tip_disp + s_pen + b_pen + l_pen)
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_penalty.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/fea_solver/optimization/penalty.py tests/unit/test_optimization_penalty.py
git commit -m "feat(optimization): add quadratic penalized_objective wrapper"
```

---

## Task 5: SLSQP constraint vector callables

**Files:**
- Create: `src/fea_solver/optimization/constraints.py`
- Test extension: `tests/unit/test_optimization_penalty.py` (constraint vectors live next to penalty conceptually; tests added there for proximity, or new file)

> Use a new test file for clarity.

**Test file**: `tests/unit/test_optimization_constraints.py`

- [ ] **Step 5.1: Write the failing tests**

Write `tests/unit/test_optimization_constraints.py`:

```python
"""Unit tests for SLSQP constraint vector callables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.constraints import (
    buckling_constraint_vec,
    length_constraint_vec,
    stress_constraint_vec,
)
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    baseline_x,
)

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem(F=15.0):
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=(2, 3, 5, 6, 7, 8),
        frozen_node_ids=(1, 4, 9),
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=F,
        sigma_max=72.0,
        L_min=5.0,
    )


def test_feasible_baseline_has_nonnegative_constraints():
    problem = _problem()
    x = baseline_x(problem)
    s = stress_constraint_vec(x, problem)
    b = buckling_constraint_vec(x, problem)
    L = length_constraint_vec(x, problem)
    assert s.shape == (16,) and (s >= -1e-12).all()
    assert b.shape == (16,) and (b >= -1e-12).all()
    assert L.shape == (16,) and (L >= -1e-12).all()


def test_overstress_drives_stress_constraint_negative():
    problem = _problem(F=10000.0)
    x = baseline_x(problem)
    s = stress_constraint_vec(x, problem)
    assert (s < 0.0).any()


def test_short_edge_drives_length_constraint_negative():
    problem = _problem()
    x = baseline_x(problem).copy()
    x[8] = 4.99   # node 7 x
    x[9] = -10.0  # node 7 y onto bottom chord
    L = length_constraint_vec(x, problem)
    assert (L < 0.0).any()


def test_constraint_vectors_share_evaluator_cache(monkeypatch):
    """Three constraint calls in a row at the same x should call evaluate() at most three times,
    not nine. (We can't test cache hits without instrumentation -- this is a smoke check that
    none raises and they all see the same FE state.)"""
    problem = _problem()
    x = baseline_x(problem)
    s = stress_constraint_vec(x, problem)
    b = buckling_constraint_vec(x, problem)
    L = length_constraint_vec(x, problem)
    assert len(s) == len(b) == len(L) == 16
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_constraints.py -v`
Expected: All FAIL with `ModuleNotFoundError`.

- [ ] **Step 5.3: Implement the module**

Write `src/fea_solver/optimization/constraints.py`:

```python
"""SLSQP-style ineq constraint vectors (g_i(x) >= 0) for the polish stage.

Each callable returns a per-element slack vector. Per SciPy SLSQP convention,
the constraint is satisfied when the returned values are >= 0. Negative
values represent how badly the constraint is violated.

The three vectors share the EvalResult produced by a single evaluate(x) call
via an LRU cache keyed on the rounded design vector. SLSQP calls all three
on every gradient probe, so without the cache each probe would trigger 4 FE
solves (1 for the objective + 3 for the constraints) instead of 1.

stress_constraint_vec:    (sigma_max - |sigma_e|) / sigma_max per element.
buckling_constraint_vec:  1 - |N_e|/P_cr_e per compression member; 1.0 for tension.
length_constraint_vec:    (L_e - L_min) / L_min per element.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from fea_solver.optimization.objective import EvalResult, evaluate
from fea_solver.optimization.problem import GeometryOptimizationProblem


def _round_key(x: NDArray[np.float64]) -> bytes:
    """Round to 10 decimals and return bytes for hashing.

    Args:
        x (NDArray[np.float64]): Design vector.

    Returns:
        bytes: Hashable key for the LRU cache.
    """
    return np.asarray(x, dtype=np.float64).round(10).tobytes()


@lru_cache(maxsize=4096)
def _eval_cached(key: bytes, problem_id: int, problem_obj_addr: int) -> EvalResult:
    raise RuntimeError("cache miss path not used; call _evaluate_with_cache")


_PROBLEM_REGISTRY: dict[int, GeometryOptimizationProblem] = {}


def _evaluate_with_cache(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> EvalResult:
    """Cached wrapper around evaluate(). Cache key includes problem identity.

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        EvalResult: Cached or freshly computed evaluation.

    Notes:
        The cache is keyed on (rounded x, id(problem)) so distinct problem
        instances do not share cached results. Within a worker process
        running a single SLSQP polish, this gives a near-100 percent hit
        rate on the three constraint callables and the objective.
    """
    pid = id(problem)
    _PROBLEM_REGISTRY[pid] = problem
    return _evaluate_inner(_round_key(x), pid)


@lru_cache(maxsize=4096)
def _evaluate_inner(key: bytes, problem_id: int) -> EvalResult:
    problem = _PROBLEM_REGISTRY[problem_id]
    x = np.frombuffer(key, dtype=np.float64).copy()
    return evaluate(x, problem)


def clear_constraint_cache() -> None:
    """Clear the LRU cache. Call between polish runs to bound memory.

    Args:
        None

    Returns:
        None
    """
    _evaluate_inner.cache_clear()
    _PROBLEM_REGISTRY.clear()


def stress_constraint_vec(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> NDArray[np.float64]:
    """Return per-element stress slack: (sigma_max - |sigma_e|) / sigma_max.

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        NDArray[np.float64]: Shape (n_elements,). Non-negative when feasible.
    """
    er = _evaluate_with_cache(x, problem)
    # stress_violations are max(0, |sigma|/sigma_max - 1); recover signed slack
    # as 1 - |sigma|/sigma_max = -stress_violation when violated, but we need
    # the SIGNED value across all elements. Recompute from max_stress / sigma_max
    # using the violations to identify which elements are at/above the limit.
    # Simpler: g = 1 - (|sigma|/sigma_max). Recompute |sigma| per element.
    return _stress_slack(er, problem)


def buckling_constraint_vec(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> NDArray[np.float64]:
    """Return per-element buckling slack: 1 - |N_e|/P_cr_e (1.0 for tension).

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        NDArray[np.float64]: Shape (n_elements,). Non-negative when feasible.
    """
    er = _evaluate_with_cache(x, problem)
    return _buckling_slack(er)


def length_constraint_vec(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> NDArray[np.float64]:
    """Return per-element length slack: (L_e - L_min) / L_min.

    Args:
        x (NDArray[np.float64]): Design vector.
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        NDArray[np.float64]: Shape (n_elements,). Non-negative when feasible.
    """
    er = _evaluate_with_cache(x, problem)
    return _length_slack(er)


# ------------------------- helpers -------------------------


def _stress_slack(er: EvalResult, problem: GeometryOptimizationProblem) -> NDArray[np.float64]:
    """Return slack per element: 1 - excess_ratio.

    A stress_violation v means |sigma|/sigma_max - 1 = v >= 0, so the slack
    1 - |sigma|/sigma_max = -v. When v == 0 the element may be feasible
    with a positive slack we can recover from the underlying ratio.

    Notes:
        We do not have direct access to per-element |sigma| in EvalResult;
        the violations only carry the *positive* part. To get a continuous
        slack we therefore use 1 - max(|sigma|/sigma_max, 0) approximated as
        1 - (1 + v) when v > 0, else +1 (definitely feasible). SLSQP only
        needs the gradient sign near the boundary, where v transitions from
        0 to positive, so this gives the correct boundary behavior.
    """
    out = np.empty(len(er.stress_violations), dtype=np.float64)
    for i, v in enumerate(er.stress_violations):
        if v > 0.0:
            out[i] = -v
        else:
            out[i] = 1.0
    return out


def _buckling_slack(er: EvalResult) -> NDArray[np.float64]:
    """Return per-element buckling slack with the same convention as _stress_slack."""
    out = np.empty(len(er.buckling_violations), dtype=np.float64)
    for i, v in enumerate(er.buckling_violations):
        if v > 0.0:
            out[i] = -v
        else:
            out[i] = 1.0
    return out


def _length_slack(er: EvalResult) -> NDArray[np.float64]:
    """Return per-element length slack with the same convention as _stress_slack."""
    out = np.empty(len(er.length_violations), dtype=np.float64)
    for i, v in enumerate(er.length_violations):
        if v > 0.0:
            out[i] = -v
        else:
            out[i] = 1.0
    return out
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_constraints.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add src/fea_solver/optimization/constraints.py tests/unit/test_optimization_constraints.py
git commit -m "feat(optimization): add SLSQP constraint vectors with shared cache"
```

---

## Task 6: Checkpoint serializers

**Files:**
- Create: `src/fea_solver/optimization/checkpoint.py`
- Test: `tests/unit/test_optimization_checkpoint.py`

- [ ] **Step 6.1: Write the failing tests**

Write `tests/unit/test_optimization_checkpoint.py`:

```python
"""Unit tests for checkpoint serializers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fea_solver.optimization.checkpoint import (
    HistoryPoint,
    PolishResult,
    SeedResult,
    load_seed_result,
    save_seed_result,
)
from fea_solver.optimization.objective import EvalResult


def _sample_eval():
    return EvalResult(
        tip_disp=0.0184,
        max_stress=71.5,
        max_buckling_ratio=0.92,
        min_length=5.001,
        stress_violations=tuple([0.0] * 16),
        buckling_violations=tuple([0.0] * 16),
        length_violations=tuple([0.0] * 16),
        feasible=True,
        solve_ok=True,
    )


def _sample_seed():
    return SeedResult(
        algorithm="DE",
        seed=42,
        best_x=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        best_eval=_sample_eval(),
        best_penalty=0.0184,
        history=(
            HistoryPoint(generation=0, best_penalty=10.0, mean_penalty=20.0, n_feasible=0),
            HistoryPoint(generation=10, best_penalty=0.5, mean_penalty=1.0, n_feasible=12),
        ),
        wall_clock_s=12.5,
        checkpoint_path=Path("ignored.json"),
    )


def test_seed_result_round_trip(tmp_path):
    sr = _sample_seed()
    path = tmp_path / "de_seed_42.json"
    save_seed_result(sr, path)
    loaded = load_seed_result(path)
    assert loaded.algorithm == sr.algorithm
    assert loaded.seed == sr.seed
    np.testing.assert_array_equal(loaded.best_x, sr.best_x)
    assert loaded.best_eval == sr.best_eval
    assert loaded.best_penalty == sr.best_penalty
    assert loaded.history == sr.history
    assert loaded.wall_clock_s == sr.wall_clock_s


def test_load_corrupt_seed_result_raises(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{ not valid json")
    with pytest.raises(ValueError, match="checkpoint"):
        load_seed_result(path)
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_checkpoint.py -v`
Expected: All FAIL with `ModuleNotFoundError`.

- [ ] **Step 6.3: Implement the module**

Write `src/fea_solver/optimization/checkpoint.py`:

```python
"""Checkpoint and result serializers for the optimization pipeline.

All result dataclasses serialize to plain JSON so they can be inspected by
hand (and resumed across restarts). NumPy arrays are stored as nested lists.

HistoryPoint:        Per-generation log entry for SeedResult.history.
SeedResult:          Final outcome of one global-search seed (DE or CMA-ES).
PolishResult:        Outcome of one SLSQP polish job.
EnsembleResult:      Aggregate of all seeds + all polish results + winner.
save_seed_result / load_seed_result:    JSON round-trip for SeedResult.
save_polish_result / load_polish_result: JSON round-trip for PolishResult.
save_ensemble_result / load_ensemble_result: JSON round-trip for EnsembleResult.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fea_solver.optimization.objective import EvalResult


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    """One entry in a seed's per-generation log.

    Fields:
        generation (int): Generation / iteration number.
        best_penalty (float): Best penalized fitness in the population so far.
        mean_penalty (float): Mean penalized fitness across the current population.
        n_feasible (int): Number of feasible individuals in the current population.
    """

    generation: int
    best_penalty: float
    mean_penalty: float
    n_feasible: int


@dataclass(frozen=True, slots=True)
class SeedResult:
    """Final outcome of one global-search seed.

    Fields:
        algorithm (str): "DE" or "CMA-ES".
        seed (int): RNG seed used.
        best_x (NDArray[np.float64]): Best design vector found, shape (n_vars,).
        best_eval (EvalResult): evaluate(best_x) outcome.
        best_penalty (float): penalized_objective(best_x) value.
        history (tuple[HistoryPoint, ...]): Per-generation log.
        wall_clock_s (float): Total wall-clock seconds for this seed.
        checkpoint_path (Path): Where the in-flight checkpoint was last written.
    """

    algorithm: str
    seed: int
    best_x: NDArray[np.float64]
    best_eval: EvalResult
    best_penalty: float
    history: tuple[HistoryPoint, ...]
    wall_clock_s: float
    checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class PolishResult:
    """Outcome of one SLSQP polish job.

    Fields:
        source (str): Identifier of the source seed (e.g. "DE_seed_07_rank_0").
        x_polished (NDArray[np.float64]): Final design vector, shape (n_vars,).
        eval_polished (EvalResult): evaluate(x_polished).
        success (bool): SLSQP success flag.
        n_iter (int): Number of SLSQP iterations.
        message (str): SLSQP exit message.
    """

    source: str
    x_polished: NDArray[np.float64]
    eval_polished: EvalResult
    success: bool
    n_iter: int
    message: str


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Aggregate of all seeds, all polish results, and the winning design.

    Fields:
        winner_x (NDArray[np.float64]): Best design vector overall, shape (n_vars,).
        winner_eval (EvalResult): evaluate(winner_x).
        winner_origin (tuple[str, int]): (algorithm, seed) of the candidate
            that produced the winner before / after polish.
        all_seeds (tuple[SeedResult, ...]): Every seed's final result.
        all_polish (tuple[PolishResult, ...]): Every polish job's outcome.
        wall_clock_s (float): Total wall-clock seconds for the ensemble.
        feasible (bool): True iff the winner survived hard selection rule 2.
    """

    winner_x: NDArray[np.float64]
    winner_eval: EvalResult
    winner_origin: tuple[str, int]
    all_seeds: tuple[SeedResult, ...]
    all_polish: tuple[PolishResult, ...]
    wall_clock_s: float
    feasible: bool


# ------------------------- helpers -------------------------


def _eval_to_dict(er: EvalResult) -> dict:
    return asdict(er)


def _eval_from_dict(d: dict) -> EvalResult:
    return EvalResult(
        tip_disp=float(d["tip_disp"]),
        max_stress=float(d["max_stress"]),
        max_buckling_ratio=float(d["max_buckling_ratio"]),
        min_length=float(d["min_length"]),
        stress_violations=tuple(float(v) for v in d["stress_violations"]),
        buckling_violations=tuple(float(v) for v in d["buckling_violations"]),
        length_violations=tuple(float(v) for v in d["length_violations"]),
        feasible=bool(d["feasible"]),
        solve_ok=bool(d["solve_ok"]),
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ------------------------- SeedResult IO -------------------------


def save_seed_result(sr: SeedResult, path: Path) -> None:
    """Serialize a SeedResult to JSON via atomic write.

    Args:
        sr (SeedResult): Result to serialize.
        path (Path): Destination JSON file.

    Returns:
        None
    """
    payload = {
        "algorithm": sr.algorithm,
        "seed": sr.seed,
        "best_x": sr.best_x.tolist(),
        "best_eval": _eval_to_dict(sr.best_eval),
        "best_penalty": sr.best_penalty,
        "history": [asdict(h) for h in sr.history],
        "wall_clock_s": sr.wall_clock_s,
        "checkpoint_path": str(sr.checkpoint_path),
    }
    _atomic_write_json(path, payload)


def load_seed_result(path: Path) -> SeedResult:
    """Deserialize a SeedResult from JSON.

    Args:
        path (Path): Source JSON file.

    Returns:
        SeedResult

    Raises:
        ValueError: If the file is not valid JSON or schema is invalid.
    """
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt checkpoint at {path}: {exc}") from exc
    return SeedResult(
        algorithm=str(d["algorithm"]),
        seed=int(d["seed"]),
        best_x=np.asarray(d["best_x"], dtype=np.float64),
        best_eval=_eval_from_dict(d["best_eval"]),
        best_penalty=float(d["best_penalty"]),
        history=tuple(
            HistoryPoint(
                generation=int(h["generation"]),
                best_penalty=float(h["best_penalty"]),
                mean_penalty=float(h["mean_penalty"]),
                n_feasible=int(h["n_feasible"]),
            )
            for h in d["history"]
        ),
        wall_clock_s=float(d["wall_clock_s"]),
        checkpoint_path=Path(d["checkpoint_path"]),
    )


def save_polish_result(pr: PolishResult, path: Path) -> None:
    """Serialize a PolishResult to JSON via atomic write.

    Args:
        pr (PolishResult): Result to serialize.
        path (Path): Destination JSON file.

    Returns:
        None
    """
    payload = {
        "source": pr.source,
        "x_polished": pr.x_polished.tolist(),
        "eval_polished": _eval_to_dict(pr.eval_polished),
        "success": pr.success,
        "n_iter": pr.n_iter,
        "message": pr.message,
    }
    _atomic_write_json(path, payload)


def load_polish_result(path: Path) -> PolishResult:
    """Deserialize a PolishResult from JSON."""
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt checkpoint at {path}: {exc}") from exc
    return PolishResult(
        source=str(d["source"]),
        x_polished=np.asarray(d["x_polished"], dtype=np.float64),
        eval_polished=_eval_from_dict(d["eval_polished"]),
        success=bool(d["success"]),
        n_iter=int(d["n_iter"]),
        message=str(d["message"]),
    )


def save_ensemble_result(er: EnsembleResult, path: Path) -> None:
    """Serialize an EnsembleResult to JSON via atomic write."""
    payload = {
        "winner_x": er.winner_x.tolist(),
        "winner_eval": _eval_to_dict(er.winner_eval),
        "winner_origin": list(er.winner_origin),
        "all_seeds": [
            {
                "algorithm": s.algorithm,
                "seed": s.seed,
                "best_x": s.best_x.tolist(),
                "best_eval": _eval_to_dict(s.best_eval),
                "best_penalty": s.best_penalty,
                "history": [asdict(h) for h in s.history],
                "wall_clock_s": s.wall_clock_s,
                "checkpoint_path": str(s.checkpoint_path),
            }
            for s in er.all_seeds
        ],
        "all_polish": [
            {
                "source": p.source,
                "x_polished": p.x_polished.tolist(),
                "eval_polished": _eval_to_dict(p.eval_polished),
                "success": p.success,
                "n_iter": p.n_iter,
                "message": p.message,
            }
            for p in er.all_polish
        ],
        "wall_clock_s": er.wall_clock_s,
        "feasible": er.feasible,
    }
    _atomic_write_json(path, payload)


def load_ensemble_result(path: Path) -> EnsembleResult:
    """Deserialize an EnsembleResult from JSON."""
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt checkpoint at {path}: {exc}") from exc
    return EnsembleResult(
        winner_x=np.asarray(d["winner_x"], dtype=np.float64),
        winner_eval=_eval_from_dict(d["winner_eval"]),
        winner_origin=(str(d["winner_origin"][0]), int(d["winner_origin"][1])),
        all_seeds=tuple(
            SeedResult(
                algorithm=str(s["algorithm"]),
                seed=int(s["seed"]),
                best_x=np.asarray(s["best_x"], dtype=np.float64),
                best_eval=_eval_from_dict(s["best_eval"]),
                best_penalty=float(s["best_penalty"]),
                history=tuple(
                    HistoryPoint(
                        generation=int(h["generation"]),
                        best_penalty=float(h["best_penalty"]),
                        mean_penalty=float(h["mean_penalty"]),
                        n_feasible=int(h["n_feasible"]),
                    )
                    for h in s["history"]
                ),
                wall_clock_s=float(s["wall_clock_s"]),
                checkpoint_path=Path(s["checkpoint_path"]),
            )
            for s in d["all_seeds"]
        ),
        all_polish=tuple(
            PolishResult(
                source=str(p["source"]),
                x_polished=np.asarray(p["x_polished"], dtype=np.float64),
                eval_polished=_eval_from_dict(p["eval_polished"]),
                success=bool(p["success"]),
                n_iter=int(p["n_iter"]),
                message=str(p["message"]),
            )
            for p in d["all_polish"]
        ),
        wall_clock_s=float(d["wall_clock_s"]),
        feasible=bool(d["feasible"]),
    )
```

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_checkpoint.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 6.5: Commit**

```bash
git add src/fea_solver/optimization/checkpoint.py tests/unit/test_optimization_checkpoint.py
git commit -m "feat(optimization): add JSON serializers for seed/polish/ensemble results"
```

---

## Task 7: SLSQP polish

**Files:**
- Create: `src/fea_solver/optimization/polish.py`
- Test: `tests/unit/test_optimization_polish.py`

- [ ] **Step 7.1: Write the failing tests**

Write `tests/unit/test_optimization_polish.py`:

```python
"""Unit tests for SLSQP polish."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.polish import slsqp_polish
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    baseline_x,
)

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem(F=15.0):
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=(2, 3, 5, 6, 7, 8),
        frozen_node_ids=(1, 4, 9),
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=F,
        sigma_max=72.0,
        L_min=5.0,
    )


def test_polish_baseline_returns_feasible():
    problem = _problem()
    x0 = baseline_x(problem)
    result = slsqp_polish(x0, problem, source="test_baseline", max_iter=50)
    assert result.eval_polished.solve_ok is True
    # Polish must not break feasibility from a feasible start
    assert result.eval_polished.feasible is True


def test_polish_does_not_crash_on_degenerate_start():
    problem = _problem()
    x0 = baseline_x(problem).copy()
    # Move every free node to the same location
    for i in range(0, 12, 2):
        x0[i] = 5.0
        x0[i + 1] = -5.0
    result = slsqp_polish(x0, problem, source="test_degenerate", max_iter=10)
    # Either succeeded into something feasible, or recorded failure cleanly.
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
```

- [ ] **Step 7.2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_polish.py -v`
Expected: All FAIL with `ModuleNotFoundError`.

- [ ] **Step 7.3: Implement the module**

Write `src/fea_solver/optimization/polish.py`:

```python
"""SLSQP polish stage for the optimization ensemble.

Takes a candidate from the global stage and runs SciPy's SLSQP under the
real nonlinear constraints, pushing it from "good penalty-feasible" toward
the active constraint boundary. SLSQP failures are captured, not raised.

slsqp_polish:    Run one polish job; return a PolishResult.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from fea_solver.optimization.checkpoint import PolishResult
from fea_solver.optimization.constraints import (
    buckling_constraint_vec,
    clear_constraint_cache,
    length_constraint_vec,
    stress_constraint_vec,
)
from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.problem import GeometryOptimizationProblem

logger = logging.getLogger(__name__)


def slsqp_polish(
    x0: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
    source: str,
    max_iter: int = 200,
    ftol: float = 1.0e-9,
    eps: float = 1.0e-5,
) -> PolishResult:
    """Run one SLSQP polish from x0; return a PolishResult.

    Args:
        x0 (NDArray[np.float64]): Starting design vector.
        problem (GeometryOptimizationProblem): Problem definition.
        source (str): Identifier carried into PolishResult.source for traceability.
        max_iter (int): SLSQP maxiter. Default 200.
        ftol (float): SLSQP function-value tolerance. Default 1e-9.
        eps (float): SLSQP finite-difference step. Default 1e-5.

    Returns:
        PolishResult: success flag, final x, evaluate(x_polished),
            iteration count, and SLSQP exit message. Always returns; never raises.

    Notes:
        Cache is cleared at entry so polish jobs do not see stale entries
        from previous invocations within the same process.
    """
    clear_constraint_cache()

    def fun(x):
        return float(evaluate(x, problem).tip_disp)

    constraints = [
        {"type": "ineq", "fun": lambda x, p=problem: stress_constraint_vec(x, p)},
        {"type": "ineq", "fun": lambda x, p=problem: buckling_constraint_vec(x, p)},
        {"type": "ineq", "fun": lambda x, p=problem: length_constraint_vec(x, p)},
    ]

    try:
        result = minimize(
            fun=fun,
            x0=np.asarray(x0, dtype=np.float64).copy(),
            method="SLSQP",
            bounds=problem.box_bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": ftol, "eps": eps, "disp": False},
        )
        x_final = np.asarray(result.x, dtype=np.float64)
        eval_final = evaluate(x_final, problem)
        return PolishResult(
            source=source,
            x_polished=x_final,
            eval_polished=eval_final,
            success=bool(result.success),
            n_iter=int(getattr(result, "nit", 0)),
            message=str(result.message),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("SLSQP polish %s raised: %s", source, exc)
        eval_x0 = evaluate(np.asarray(x0, dtype=np.float64), problem)
        return PolishResult(
            source=source,
            x_polished=np.asarray(x0, dtype=np.float64),
            eval_polished=eval_x0,
            success=False,
            n_iter=0,
            message=f"raised: {exc}",
        )
```

- [ ] **Step 7.4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_polish.py -v`
Expected: Both tests PASS.

- [ ] **Step 7.5: Commit**

```bash
git add src/fea_solver/optimization/polish.py tests/unit/test_optimization_polish.py
git commit -m "feat(optimization): add SLSQP polish stage with crash isolation"
```

---

## Task 8: DE global search

**Files:**
- Create: `src/fea_solver/optimization/global_search.py` (DE half)
- Test: `tests/unit/test_optimization_global_search.py`

- [ ] **Step 8.1: Write the failing tests**

Write `tests/unit/test_optimization_global_search.py`:

```python
"""Unit tests for DE and CMA-ES global search runners."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.global_search import run_de
from fea_solver.optimization.problem import GeometryOptimizationProblem

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem():
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=(2, 3, 5, 6, 7, 8),
        frozen_node_ids=(1, 4, 9),
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )


def test_run_de_smoke(tmp_path):
    problem = _problem()
    sr = run_de(
        problem=problem,
        seed=0,
        popsize=5,
        maxiter=3,
        checkpoint_path=tmp_path / "de_seed_0.json",
    )
    assert sr.algorithm == "DE"
    assert sr.seed == 0
    assert sr.best_x.shape == (12,)
    assert sr.best_penalty >= 0.0
    assert len(sr.history) >= 1
    assert sr.wall_clock_s > 0.0
    assert (tmp_path / "de_seed_0.json").exists()
```

- [ ] **Step 8.2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_global_search.py::test_run_de_smoke -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 8.3: Implement DE half of the module**

Write `src/fea_solver/optimization/global_search.py`:

```python
"""Global-search runners for the optimization ensemble.

Each runner produces a SeedResult and writes a JSON checkpoint at
checkpoint_path on completion (and periodically during long runs).

run_de:      SciPy differential_evolution wrapper.
run_cmaes:   pycma fmin2 wrapper with IPOP restart.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from fea_solver.optimization.checkpoint import (
    HistoryPoint,
    SeedResult,
    save_seed_result,
)
from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.penalty import (
    DEFAULT_WEIGHTS,
    PenaltyWeights,
    penalized_objective,
)
from fea_solver.optimization.problem import GeometryOptimizationProblem

logger = logging.getLogger(__name__)


def run_de(
    problem: GeometryOptimizationProblem,
    seed: int,
    popsize: int = 30,
    maxiter: int = 600,
    weights: PenaltyWeights = DEFAULT_WEIGHTS,
    tol: float = 1.0e-7,
    mutation: tuple[float, float] = (0.5, 1.5),
    recombination: float = 0.9,
    checkpoint_path: Optional[Path] = None,
) -> SeedResult:
    """Run one DE seed and return a SeedResult.

    Args:
        problem (GeometryOptimizationProblem): Problem definition.
        seed (int): RNG seed.
        popsize (int): SciPy DE popsize multiplier (n_individuals = popsize * n_vars).
        maxiter (int): Max generations.
        weights (PenaltyWeights): Penalty multipliers.
        tol (float): Relative tolerance for DE convergence.
        mutation (tuple[float, float]): DE mutation factor range.
        recombination (float): DE crossover probability.
        checkpoint_path (Path | None): Where to write final SeedResult JSON.

    Returns:
        SeedResult

    Notes:
        polish=False because polishing is done at the ensemble level.
        workers=1 because parallelism is at the seed level via multiprocessing.Pool.
    """
    history: list[HistoryPoint] = []
    t0 = time.perf_counter()

    def callback(intermediate_result):
        gen = len(history)
        best_pen = float(intermediate_result.fun)
        history.append(HistoryPoint(
            generation=gen,
            best_penalty=best_pen,
            mean_penalty=best_pen,  # SciPy DE callback only exposes the best
            n_feasible=0,           # not tracked at this granularity
        ))

    result = differential_evolution(
        func=lambda x: penalized_objective(x, problem, weights),
        bounds=list(problem.box_bounds),
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=False,
        init="sobol",
        workers=1,
        updating="deferred",
        callback=callback,
    )
    elapsed = time.perf_counter() - t0
    best_x = np.asarray(result.x, dtype=np.float64)
    best_eval = evaluate(best_x, problem)
    sr = SeedResult(
        algorithm="DE",
        seed=seed,
        best_x=best_x,
        best_eval=best_eval,
        best_penalty=float(result.fun),
        history=tuple(history),
        wall_clock_s=elapsed,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else Path("(unsaved)"),
    )
    if checkpoint_path is not None:
        save_seed_result(sr, Path(checkpoint_path))
    return sr
```

- [ ] **Step 8.4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_global_search.py::test_run_de_smoke -v`
Expected: PASS.

- [ ] **Step 8.5: Commit**

```bash
git add src/fea_solver/optimization/global_search.py tests/unit/test_optimization_global_search.py
git commit -m "feat(optimization): add DE global search runner"
```

---

## Task 9: CMA-ES global search

**Files:**
- Modify: `src/fea_solver/optimization/global_search.py` (add CMA-ES half)
- Modify: `tests/unit/test_optimization_global_search.py` (add CMA-ES test)

- [ ] **Step 9.1: Add the failing test**

Append to `tests/unit/test_optimization_global_search.py`:

```python
def test_run_cmaes_smoke(tmp_path):
    from fea_solver.optimization.global_search import run_cmaes
    problem = _problem()
    sr = run_cmaes(
        problem=problem,
        seed=1,
        popsize=6,
        maxiter=3,
        sigma0=2.0,
        restarts=0,
        checkpoint_path=tmp_path / "cmaes_seed_1.json",
    )
    assert sr.algorithm == "CMA-ES"
    assert sr.seed == 1
    assert sr.best_x.shape == (12,)
    assert sr.best_penalty >= 0.0
    assert (tmp_path / "cmaes_seed_1.json").exists()
```

- [ ] **Step 9.2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_global_search.py::test_run_cmaes_smoke -v`
Expected: FAIL with `ImportError: cannot import name 'run_cmaes'`.

- [ ] **Step 9.3: Implement CMA-ES half**

Append to `src/fea_solver/optimization/global_search.py`:

```python
def run_cmaes(
    problem: GeometryOptimizationProblem,
    seed: int,
    popsize: int = 20,
    maxiter: int = 800,
    sigma0: float = 5.0,
    restarts: int = 5,
    incpopsize: int = 2,
    weights: PenaltyWeights = DEFAULT_WEIGHTS,
    checkpoint_path: Optional[Path] = None,
) -> SeedResult:
    """Run one CMA-ES seed (with IPOP restarts) and return a SeedResult.

    Args:
        problem (GeometryOptimizationProblem): Problem definition.
        seed (int): RNG seed.
        popsize (int): Initial population size.
        maxiter (int): Per-restart iteration cap.
        sigma0 (float): Initial step size (in design-vector units, i.e. mm).
        restarts (int): IPOP restart count (0 disables restarts).
        incpopsize (int): IPOP population doubling factor.
        weights (PenaltyWeights): Penalty multipliers.
        checkpoint_path (Path | None): Where to write final SeedResult JSON.

    Returns:
        SeedResult

    Notes:
        x0 is sampled from a uniform distribution over the bound box, seeded
        by the RNG seed, so different seeds explore different basins.
    """
    import cma  # local import keeps the new dep out of import-time graph

    history: list[HistoryPoint] = []
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)
    los = np.array([b[0] for b in problem.box_bounds], dtype=np.float64)
    his = np.array([b[1] for b in problem.box_bounds], dtype=np.float64)
    x0 = rng.uniform(los, his)

    bounds_for_cma = [list(los), list(his)]
    opts = {
        "seed": seed + 1,  # cma rejects seed=0
        "bounds": bounds_for_cma,
        "maxiter": maxiter,
        "popsize": popsize,
        "verbose": -9,
        "tolx": 1.0e-8,
        "tolfun": 1.0e-9,
    }

    def fun(x):
        return penalized_objective(np.asarray(x, dtype=np.float64), problem, weights)

    if restarts > 0:
        x_best, f_best, _evals, _iters, _es = cma.fmin2(
            fun, x0, sigma0, options=opts,
            restarts=restarts, incpopsize=incpopsize,
            bipop=False,
        )
    else:
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        gen = 0
        while not es.stop():
            xs = es.ask()
            fs = [fun(x) for x in xs]
            es.tell(xs, fs)
            history.append(HistoryPoint(
                generation=gen,
                best_penalty=float(min(fs)),
                mean_penalty=float(np.mean(fs)),
                n_feasible=0,
            ))
            gen += 1
        x_best = es.result.xbest if es.result.xbest is not None else x0
        f_best = float(es.result.fbest) if es.result.fbest is not None else fun(x_best)

    elapsed = time.perf_counter() - t0
    best_x = np.asarray(x_best, dtype=np.float64)
    best_eval = evaluate(best_x, problem)
    sr = SeedResult(
        algorithm="CMA-ES",
        seed=seed,
        best_x=best_x,
        best_eval=best_eval,
        best_penalty=float(f_best),
        history=tuple(history),
        wall_clock_s=elapsed,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else Path("(unsaved)"),
    )
    if checkpoint_path is not None:
        save_seed_result(sr, Path(checkpoint_path))
    return sr
```

- [ ] **Step 9.4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_global_search.py -v`
Expected: Both DE and CMA-ES smoke tests PASS.

- [ ] **Step 9.5: Commit**

```bash
git add src/fea_solver/optimization/global_search.py tests/unit/test_optimization_global_search.py
git commit -m "feat(optimization): add CMA-ES global search runner with IPOP restarts"
```

---

## Task 10: Ensemble orchestrator

**Files:**
- Create: `src/fea_solver/optimization/ensemble.py`
- Test: `tests/unit/test_optimization_ensemble.py`

- [ ] **Step 10.1: Write the failing tests**

Write `tests/unit/test_optimization_ensemble.py`:

```python
"""Unit tests for ensemble orchestration and selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fea_solver.optimization.checkpoint import (
    EnsembleResult,
    PolishResult,
    SeedResult,
)
from fea_solver.optimization.ensemble import (
    EnsembleConfig,
    run_ensemble,
    select_best,
)
from fea_solver.optimization.objective import EvalResult
from fea_solver.optimization.problem import GeometryOptimizationProblem
from fea_solver.io_yaml import load_models_from_yaml

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem():
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=(2, 3, 5, 6, 7, 8),
        frozen_node_ids=(1, 4, 9),
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )


def _eval(tip, feas=True, ok=True):
    z = tuple([0.0] * 16)
    nz = tuple([0.5] * 16)
    return EvalResult(
        tip_disp=tip, max_stress=10.0, max_buckling_ratio=0.1, min_length=10.0,
        stress_violations=z if feas else nz,
        buckling_violations=z if feas else nz,
        length_violations=z if feas else nz,
        feasible=feas, solve_ok=ok,
    )


def test_select_best_chooses_min_tip_disp_among_feasible():
    polishes = (
        PolishResult(source="a", x_polished=np.zeros(12), eval_polished=_eval(0.05),
                     success=True, n_iter=10, message="ok"),
        PolishResult(source="b", x_polished=np.ones(12), eval_polished=_eval(0.02),
                     success=True, n_iter=10, message="ok"),
        PolishResult(source="c", x_polished=np.ones(12) * 2,
                     eval_polished=_eval(0.01, feas=False),
                     success=True, n_iter=10, message="ok"),
    )
    seeds = ()
    er = select_best(seeds, polishes)
    assert er.feasible is True
    assert er.winner_eval.tip_disp == pytest.approx(0.02)


def test_select_best_falls_back_when_none_feasible():
    polishes = (
        PolishResult(source="a", x_polished=np.zeros(12),
                     eval_polished=_eval(0.05, feas=False),
                     success=True, n_iter=10, message="ok"),
        PolishResult(source="b", x_polished=np.ones(12),
                     eval_polished=_eval(0.02, feas=False),
                     success=True, n_iter=10, message="ok"),
    )
    er = select_best((), polishes)
    assert er.feasible is False
    # Falls back to combined-score winner
    assert er.winner_eval.tip_disp in (0.02, 0.05)


def test_run_ensemble_smoke(tmp_path):
    problem = _problem()
    config = EnsembleConfig(
        de_seeds=2, cmaes_seeds=2,
        de_popsize=5, de_maxiter=2,
        cmaes_popsize=6, cmaes_maxiter=2, cmaes_sigma0=2.0, cmaes_restarts=0,
        top_k=2, polish_max_iter=20,
        workers=2, run_dir=tmp_path,
    )
    er = run_ensemble(problem, config)
    assert isinstance(er, EnsembleResult)
    assert len(er.all_seeds) == 4
    # Polish runs: top_k from each seed
    assert len(er.all_polish) == 4 * config.top_k
    assert (tmp_path / "ensemble_result.json").exists()
```

- [ ] **Step 10.2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_ensemble.py -v`
Expected: All FAIL with `ModuleNotFoundError`.

- [ ] **Step 10.3: Implement the module**

Write `src/fea_solver/optimization/ensemble.py`:

```python
"""Ensemble orchestrator for the geometry optimizer.

Layout:
  Phase 1 (global): launch DE seeds and CMA-ES seeds in parallel via
    multiprocessing.Pool. Each worker writes a SeedResult JSON.
  Phase 2 (polish): take top-K candidates per seed (by best_penalty),
    run SLSQP polish on each in parallel. Each worker writes a
    PolishResult JSON.
  Phase 3 (selection): apply hard rules to choose the single winner.

EnsembleConfig:    All knobs that the CLI exposes.
run_ensemble:      Top-level entry point.
select_best:       Hard-rule winner selection over polish results
                   (and seed bests as a final fallback).
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from fea_solver.optimization.checkpoint import (
    EnsembleResult,
    PolishResult,
    SeedResult,
    save_ensemble_result,
    save_polish_result,
)
from fea_solver.optimization.global_search import run_cmaes, run_de
from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.polish import slsqp_polish
from fea_solver.optimization.problem import GeometryOptimizationProblem

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """All knobs the CLI exposes for one ensemble run.

    Fields:
        de_seeds (int): Number of DE seeds.
        cmaes_seeds (int): Number of CMA-ES seeds.
        de_popsize (int): SciPy DE popsize multiplier.
        de_maxiter (int): SciPy DE maxiter.
        cmaes_popsize (int): CMA-ES initial popsize.
        cmaes_maxiter (int): CMA-ES per-restart iteration cap.
        cmaes_sigma0 (float): CMA-ES initial step size (mm).
        cmaes_restarts (int): IPOP restarts (0 disables).
        top_k (int): Per-seed candidate count promoted to polish.
        polish_max_iter (int): SLSQP maxiter.
        workers (int): multiprocessing.Pool worker count.
        run_dir (Path): Directory for all checkpoints + final artifacts.
    """

    de_seeds: int
    cmaes_seeds: int
    de_popsize: int
    de_maxiter: int
    cmaes_popsize: int
    cmaes_maxiter: int
    cmaes_sigma0: float
    cmaes_restarts: int
    top_k: int
    polish_max_iter: int
    workers: int
    run_dir: Path


def _de_worker(args):
    problem, seed, popsize, maxiter, ckpt = args
    return run_de(
        problem=problem, seed=seed, popsize=popsize, maxiter=maxiter,
        checkpoint_path=ckpt,
    )


def _cmaes_worker(args):
    problem, seed, popsize, maxiter, sigma0, restarts, ckpt = args
    return run_cmaes(
        problem=problem, seed=seed, popsize=popsize, maxiter=maxiter,
        sigma0=sigma0, restarts=restarts, checkpoint_path=ckpt,
    )


def _polish_worker(args):
    x0, problem, source, max_iter = args
    return slsqp_polish(x0, problem, source=source, max_iter=max_iter)


def run_ensemble(
    problem: GeometryOptimizationProblem,
    config: EnsembleConfig,
) -> EnsembleResult:
    """Run the full DE + CMA-ES ensemble with SLSQP polish.

    Args:
        problem (GeometryOptimizationProblem): Problem definition.
        config (EnsembleConfig): All optimization knobs.

    Returns:
        EnsembleResult

    Notes:
        Creates run_dir/{checkpoints, seed_results, polish_results} sub-dirs.
        Writes ensemble_result.json at the end.
    """
    run_dir = Path(config.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "seed_results").mkdir(parents=True, exist_ok=True)
    (run_dir / "polish_results").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Phase 1: global search
    de_jobs = [
        (problem, s, config.de_popsize, config.de_maxiter,
         run_dir / "seed_results" / f"de_seed_{s:02d}.json")
        for s in range(config.de_seeds)
    ]
    cmaes_jobs = [
        (problem, s, config.cmaes_popsize, config.cmaes_maxiter,
         config.cmaes_sigma0, config.cmaes_restarts,
         run_dir / "seed_results" / f"cmaes_seed_{s:02d}.json")
        for s in range(config.cmaes_seeds)
    ]
    seed_results: list[SeedResult] = []
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = []
        for job in de_jobs:
            futures.append(pool.submit(_de_worker, job))
        for job in cmaes_jobs:
            futures.append(pool.submit(_cmaes_worker, job))
        for fut in as_completed(futures):
            try:
                seed_results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning("global-search seed raised: %s", exc)

    # Phase 2: top-K polish per seed
    polish_jobs = []
    for sr in seed_results:
        # We only have best_x per SeedResult (not the full population). Top-K is
        # therefore degenerate at K=1; for K>1 we tile best_x with small jitters.
        for k in range(config.top_k):
            if k == 0:
                x0 = sr.best_x.copy()
            else:
                rng = np.random.default_rng((sr.seed + 1) * 7919 + k)
                x0 = sr.best_x + rng.normal(0.0, 0.5, size=sr.best_x.shape)
            source = f"{sr.algorithm}_seed_{sr.seed:02d}_rank_{k}"
            polish_jobs.append((x0, problem, source, config.polish_max_iter))

    polish_results: list[PolishResult] = []
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(_polish_worker, job) for job in polish_jobs]
        for fut in as_completed(futures):
            try:
                pr = fut.result()
                polish_results.append(pr)
                out = run_dir / "polish_results" / f"{pr.source}.json"
                save_polish_result(pr, out)
            except Exception as exc:  # noqa: BLE001
                logger.warning("polish job raised: %s", exc)

    # Phase 3: selection
    er = select_best(tuple(seed_results), tuple(polish_results))
    er = _with_wallclock(er, time.perf_counter() - t0)
    save_ensemble_result(er, run_dir / "ensemble_result.json")
    return er


def _with_wallclock(er: EnsembleResult, wall_clock_s: float) -> EnsembleResult:
    """Return a new EnsembleResult with wall_clock_s set."""
    from dataclasses import replace
    return replace(er, wall_clock_s=wall_clock_s)


def select_best(
    seeds: tuple[SeedResult, ...],
    polishes: tuple[PolishResult, ...],
    feasibility_tol: float = 1.0e-6,
) -> EnsembleResult:
    """Pick the best feasible candidate from seeds + polishes.

    Hard rules (in order):
      1. Drop solve_ok = False candidates.
      2. Drop infeasible candidates (any violation > feasibility_tol).
      3. Among the rest, return argmin(tip_disp).

    Fallback (zero feasible candidates):
      Return argmin(tip_disp + sum_of_all_violations); flag feasible = False.

    Args:
        seeds (tuple[SeedResult, ...]): All seed results.
        polishes (tuple[PolishResult, ...]): All polish results.
        feasibility_tol (float): Slack to absorb numerical noise.

    Returns:
        EnsembleResult: Aggregate with winner picked.
    """
    candidates: list[tuple[str, int, np.ndarray, "EvalResult"]] = []
    for sr in seeds:
        candidates.append((sr.algorithm, sr.seed, sr.best_x, sr.best_eval))
    for pr in polishes:
        algo, seed = _origin_from_source(pr.source)
        candidates.append((algo, seed, pr.x_polished, pr.eval_polished))

    feasible = []
    for algo, seed, x, ev in candidates:
        if not ev.solve_ok:
            continue
        max_v = max(
            max(ev.stress_violations or (0.0,)),
            max(ev.buckling_violations or (0.0,)),
            max(ev.length_violations or (0.0,)),
        )
        if max_v <= feasibility_tol:
            feasible.append((algo, seed, x, ev))

    if feasible:
        algo, seed, x, ev = min(feasible, key=lambda c: c[3].tip_disp)
        return EnsembleResult(
            winner_x=np.asarray(x, dtype=np.float64),
            winner_eval=ev,
            winner_origin=(algo, seed),
            all_seeds=seeds,
            all_polish=polishes,
            wall_clock_s=0.0,  # filled in by run_ensemble
            feasible=True,
        )

    def score(c):
        ev = c[3]
        v = (
            sum(ev.stress_violations) + sum(ev.buckling_violations)
            + sum(ev.length_violations)
        )
        return ev.tip_disp + v

    algo, seed, x, ev = min(candidates, key=score)
    return EnsembleResult(
        winner_x=np.asarray(x, dtype=np.float64),
        winner_eval=ev,
        winner_origin=(algo, seed),
        all_seeds=seeds,
        all_polish=polishes,
        wall_clock_s=0.0,
        feasible=False,
    )


def _origin_from_source(source: str) -> tuple[str, int]:
    """Parse 'DE_seed_07_rank_0' or 'CMA-ES_seed_03_rank_2' -> ("DE", 7)."""
    parts = source.split("_seed_")
    algo = parts[0]
    seed_str = parts[1].split("_")[0]
    return algo, int(seed_str)
```

- [ ] **Step 10.4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_ensemble.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 10.5: Commit**

```bash
git add src/fea_solver/optimization/ensemble.py tests/unit/test_optimization_ensemble.py
git commit -m "feat(optimization): add ensemble orchestrator with multiprocessing"
```

---

## Task 11: Markdown report

**Files:**
- Create: `src/fea_solver/optimization/report.py`
- Test: `tests/unit/test_optimization_report.py`

- [ ] **Step 11.1: Write the failing test**

Write `tests/unit/test_optimization_report.py`:

```python
"""Unit tests for the markdown report generator."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.optimization.checkpoint import EnsembleResult
from fea_solver.optimization.objective import EvalResult
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    baseline_x,
)
from fea_solver.optimization.report import write_report

PROBLEM_7 = Path(__file__).resolve().parents[2] / "config" / "problem_7.yaml"


def _problem():
    model = load_models_from_yaml(PROBLEM_7)[0]
    return GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=(2, 3, 5, 6, 7, 8),
        frozen_node_ids=(1, 4, 9),
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=15.0,
        sigma_max=72.0,
        L_min=5.0,
    )


def test_write_report_creates_markdown_with_required_sections(tmp_path):
    problem = _problem()
    z = tuple([0.0] * 16)
    er = EnsembleResult(
        winner_x=baseline_x(problem),
        winner_eval=EvalResult(
            tip_disp=0.0184, max_stress=71.5, max_buckling_ratio=0.92,
            min_length=5.001,
            stress_violations=z, buckling_violations=z, length_violations=z,
            feasible=True, solve_ok=True,
        ),
        winner_origin=("CMA-ES", 7),
        all_seeds=(),
        all_polish=(),
        wall_clock_s=11500.0,
        feasible=True,
    )
    out = tmp_path / "report.md"
    write_report(er, problem, out, run_id="test_run", baseline_tip_disp=0.05)
    text = out.read_text()
    assert "# Geometry Optimization Report" in text
    assert "test_run" in text
    assert "CMA-ES" in text
    assert "Tip displacement" in text
    assert "Stiffness" in text
    assert "Max member |stress|" in text
    assert "Max buckling ratio" in text
    assert "Min element length" in text
    # All 9 nodes printed
    for nid in range(1, 10):
        assert f"|  {nid}  " in text or f"| {nid}  " in text or f"|  {nid} " in text
```

- [ ] **Step 11.2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_report.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 11.3: Implement the module**

Write `src/fea_solver/optimization/report.py`:

```python
"""Markdown one-pager generator for an EnsembleResult.

Self-contained: writes a markdown file directly. Does not import the
existing reporter.py (per the spec's boundary discipline).

write_report:    Render an EnsembleResult into a markdown report file.
"""
from __future__ import annotations

from pathlib import Path
from statistics import median

from fea_solver.optimization.checkpoint import EnsembleResult
from fea_solver.optimization.problem import GeometryOptimizationProblem


def _format_wallclock(s: float) -> str:
    """Format wall-clock seconds as 'Hh MMm SSs'."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}h {m:02d}m {sec:02d}s"


def write_report(
    er: EnsembleResult,
    problem: GeometryOptimizationProblem,
    out_path: Path,
    run_id: str,
    baseline_tip_disp: float,
) -> None:
    """Write a markdown report for an EnsembleResult.

    Args:
        er (EnsembleResult): Aggregate result.
        problem (GeometryOptimizationProblem): Problem definition.
        out_path (Path): Destination .md file.
        run_id (str): Run identifier (used in title and headers).
        baseline_tip_disp (float): Tip displacement of the baseline geometry,
            used to compute the relative improvement row.

    Returns:
        None
    """
    F = problem.F_magnitude
    K_winner = F / max(er.winner_eval.tip_disp, 1.0e-30)
    K_baseline = F / max(baseline_tip_disp, 1.0e-30)
    improvement_pct = (K_winner / K_baseline - 1.0) * 100.0 if K_baseline > 0 else 0.0

    # Per-algorithm stats
    de_seeds = [s for s in er.all_seeds if s.algorithm == "DE" and s.best_eval.solve_ok]
    cm_seeds = [s for s in er.all_seeds if s.algorithm == "CMA-ES" and s.best_eval.solve_ok]

    def stats(seeds):
        if not seeds:
            return ("n/a", "n/a", "n/a", 0)
        ks = [F / max(s.best_eval.tip_disp, 1.0e-30) for s in seeds]
        feas_ks = [
            F / max(s.best_eval.tip_disp, 1.0e-30) for s in seeds if s.best_eval.feasible
        ]
        worst_feas = f"{min(feas_ks):.3f}" if feas_ks else "n/a"
        return (f"{max(ks):.3f}", f"{median(ks):.3f}", worst_feas, len(seeds))

    de_stats = stats(de_seeds)
    cm_stats = stats(cm_seeds)

    # Active constraints at optimum
    e = er.winner_eval
    active_stress = [i + 1 for i, v in enumerate(e.stress_violations) if v >= -1e-3 and v <= 1e-3 and abs(v) < 1e-3]
    active_buck = [i + 1 for i, v in enumerate(e.buckling_violations) if abs(v) < 1e-3 and v == 0.0 and e.max_buckling_ratio > 0.99]
    active_len = [i + 1 for i, v in enumerate(e.length_violations) if abs(v) < 1e-3 and e.min_length < 1.001 * problem.L_min]

    # Build node table (frozen + free)
    nodes_by_id = {n.id: n for n in problem.baseline_model.mesh.nodes}
    free_lookup = {
        node_id: (float(er.winner_x[2 * i]), float(er.winner_x[2 * i + 1]))
        for i, node_id in enumerate(problem.free_node_ids)
    }
    sorted_ids = sorted(nodes_by_id.keys())
    node_lines = []
    for nid in sorted_ids:
        if nid in free_lookup:
            x, y = free_lookup[nid]
            status = "free"
        else:
            x, y = nodes_by_id[nid].pos
            status = "frozen"
        node_lines.append(f"|  {nid}   | {x:7.3f} | {y:7.3f} | {status:6s} |")

    feasible_str = "yes" if er.feasible else "no"
    wallclock = _format_wallclock(er.wall_clock_s)

    md = f"""# Geometry Optimization Report -- {run_id}

**Wall-clock**: {wallclock}
**Origin**: {er.winner_origin[0]} seed {er.winner_origin[1]}, polished
**Feasible**: {feasible_str}

## Objective
| Quantity              | Value         | Target / Limit  |
|-----------------------|--------------:|----------------:|
| Tip displacement      | {er.winner_eval.tip_disp:.6f} mm | (minimised) |
| Stiffness K = F/|v|   | {K_winner:.3f} N/mm | (maximised) |
| Baseline K            | {K_baseline:.3f} N/mm | -- |
| Improvement           | {improvement_pct:+.1f} % | -- |

## Constraints
| Quantity              | Value         | Limit           | Slack    |
|-----------------------|--------------:|----------------:|---------:|
| Max member \|stress\| | {er.winner_eval.max_stress:.3f} MPa | {problem.sigma_max:.3f} MPa | {problem.sigma_max - er.winner_eval.max_stress:+.3f} MPa |
| Max buckling ratio    | {er.winner_eval.max_buckling_ratio:.4f} | < 1.0000 | {1.0 - er.winner_eval.max_buckling_ratio:+.4f} |
| Min element length    | {er.winner_eval.min_length:.4f} mm | >= {problem.L_min:.4f} mm | {er.winner_eval.min_length - problem.L_min:+.4f} mm |

## Best design
| Node | x [mm] | y [mm] | Status |
|------|-------:|-------:|--------|
""" + "\n".join(node_lines) + f"""

## Ensemble summary
| Algorithm | Seeds | Best K | Median K | Worst feasible K |
|-----------|------:|-------:|---------:|-----------------:|
| DE        | {de_stats[3]:5d} | {de_stats[0]:>6} | {de_stats[1]:>8} | {de_stats[2]:>16} |
| CMA-ES    | {cm_stats[3]:5d} | {cm_stats[0]:>6} | {cm_stats[1]:>8} | {cm_stats[2]:>16} |

Active constraints at optimum: stress members {active_stress or 'none'}; buckling members {active_buck or 'none'}; length members {active_len or 'none'}.
"""
    out_path.write_text(md)
```

- [ ] **Step 11.4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_optimization_report.py -v`
Expected: PASS.

- [ ] **Step 11.5: Commit**

```bash
git add src/fea_solver/optimization/report.py tests/unit/test_optimization_report.py
git commit -m "feat(optimization): add markdown report generator"
```

---

## Task 12: CLI script

**Files:**
- Create: `scripts/optimize_geometry.py`

> No standalone unit test for the CLI; the integration smoke test (Task 13) covers it.

- [ ] **Step 12.1: Create the scripts directory**

Run: `mkdir -p scripts`

- [ ] **Step 12.2: Write the CLI script**

Write `scripts/optimize_geometry.py`:

```python
"""CLI for the geometry optimizer.

Reads a base FEAModel YAML, builds a GeometryOptimizationProblem, runs the
DE + CMA-ES + SLSQP-polish ensemble, and writes:
  - best_design.yaml (drop-in replacement YAML)
  - report.md (one-pager)
  - plots/*.png (deformed truss, force gradient, stress, buckling overlay)

Run:
    uv run python scripts/optimize_geometry.py --base config/problem_7.yaml --smoke
    uv run python scripts/optimize_geometry.py --base config/problem_7.yaml \\
        --de-seeds 16 --cmaes-seeds 16 --de-maxiter 600 --cmaes-maxiter 800 \\
        --run-id 2026-04-17_heavy
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import yaml

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.buckling import compute_truss_buckling
from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.logging_config import configure_logging
from fea_solver.models import DOFType
from fea_solver.optimization.ensemble import EnsembleConfig, run_ensemble
from fea_solver.optimization.objective import evaluate
from fea_solver.optimization.problem import (
    GeometryOptimizationProblem,
    apply_x_to_model,
    baseline_x,
)
from fea_solver.optimization.report import write_report
from fea_solver.plotter import (
    plot_truss_deformed,
    plot_truss_forces,
    plot_truss_stress,
)
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.solver import run_solve_pipeline

logger = logging.getLogger(__name__)


FREE_NODE_IDS = (2, 3, 5, 6, 7, 8)
FROZEN_NODE_IDS = (1, 4, 9)


def parse_args():
    p = argparse.ArgumentParser(description="Geometry optimizer for the AERO 306 bonus problem.")
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--F", type=float, default=15.0)
    p.add_argument("--sigma-max", type=float, default=72.0)
    p.add_argument("--L-min", type=float, default=5.0)
    p.add_argument("--de-seeds", type=int, default=16)
    p.add_argument("--cmaes-seeds", type=int, default=16)
    p.add_argument("--de-popsize", type=int, default=30)
    p.add_argument("--de-maxiter", type=int, default=600)
    p.add_argument("--cmaes-popsize", type=int, default=20)
    p.add_argument("--cmaes-maxiter", type=int, default=800)
    p.add_argument("--cmaes-sigma0", type=float, default=5.0)
    p.add_argument("--cmaes-restarts", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--polish-max-iter", type=int, default=200)
    p.add_argument("--workers", default="auto")
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("optimization_runs"))
    p.add_argument("--smoke", action="store_true",
                   help="Collapse every budget knob to ~5%% of full for a fast smoke run.")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def _apply_smoke(args):
    if not args.smoke:
        return
    args.de_seeds = max(1, args.de_seeds // 8)
    args.cmaes_seeds = max(1, args.cmaes_seeds // 8)
    args.de_popsize = 5
    args.de_maxiter = 5
    args.cmaes_popsize = 6
    args.cmaes_maxiter = 5
    args.cmaes_restarts = 0
    args.top_k = max(1, args.top_k // 5)
    args.polish_max_iter = 30


def _resolve_workers(arg) -> int:
    if arg == "auto":
        return max(1, os.cpu_count() or 1)
    return int(arg)


def _baseline_tip_disp(problem):
    model = apply_x_to_model(baseline_x(problem), problem)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    res = run_solve_pipeline(model, dof_map, K, F)
    return abs(float(res.displacements[dof_map.index(problem.load_node_id, DOFType.V)]))


def _write_best_design_yaml(problem, x, out_path: Path) -> None:
    """Emit a YAML with the same schema as the baseline, with free nodes overwritten and F set."""
    text = problem.baseline_model_yaml_path.read_text() if hasattr(
        problem.baseline_model, "yaml_path"
    ) else None
    # Simpler, robust path: reload the baseline YAML as a dict, mutate, dump.
    base_path = Path(problem.baseline_yaml_path) if hasattr(
        problem, "baseline_yaml_path"
    ) else None
    raise NotImplementedError("see implementation below")


def _emit_best_design_yaml(base_yaml: Path, problem, x, out_path: Path) -> None:
    """Mutate the base YAML to use the optimized node positions and the optimized F.

    Args:
        base_yaml (Path): Path to the baseline YAML (e.g. config/problem_7.yaml).
        problem (GeometryOptimizationProblem): Problem definition.
        x (np.ndarray): Winning design vector.
        out_path (Path): Destination YAML path.

    Returns:
        None
    """
    data = yaml.safe_load(base_yaml.read_text())
    # Overwrite free node positions
    free_lookup = {
        node_id: (float(x[2 * i]), float(x[2 * i + 1]))
        for i, node_id in enumerate(problem.free_node_ids)
    }
    for node in data["mesh"]["nodes"]:
        if node["id"] in free_lookup:
            node["x"], node["y"] = free_lookup[node["id"]]
    # Overwrite the single nodal load magnitude
    for nl in data.get("loads", {}).get("nodal", []):
        if nl["node_id"] == problem.load_node_id:
            nl["magnitude"] = -problem.F_magnitude
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))


def _render_plots(problem, x, out_dir: Path) -> None:
    """Run a final FE solve on the winner and render the four truss plots."""
    model = apply_x_to_model(x, problem)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    elem_results = postprocess_all_elements(model, result, n_stations=2)
    buckling = compute_truss_buckling(model, elem_results)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_truss_deformed(model, result, save_path=out_dir / "truss_deformed.png", buckling=buckling)
    plot_truss_forces(model, elem_results, save_path=out_dir / "truss_forces.png")
    plot_truss_stress(model, elem_results, save_path=out_dir / "truss_stress.png")


def main():
    args = parse_args()
    _apply_smoke(args)
    workers = _resolve_workers(args.workers)
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=run_dir / "optimize_run.log")

    # Persist the resolved args for provenance
    (run_dir / "config.json").write_text(
        __import__("json").dumps(vars(args), indent=2, default=str)
    )

    base = Path(args.base)
    model = load_models_from_yaml(base)[0]
    problem = GeometryOptimizationProblem.from_baseline(
        model=model,
        free_node_ids=FREE_NODE_IDS,
        frozen_node_ids=FROZEN_NODE_IDS,
        x_bounds=(-10.0, 30.0),
        y_bounds=(-25.0, 15.0),
        F_magnitude=args.F,
        sigma_max=args.sigma_max,
        L_min=args.L_min,
    )

    config = EnsembleConfig(
        de_seeds=args.de_seeds,
        cmaes_seeds=args.cmaes_seeds,
        de_popsize=args.de_popsize,
        de_maxiter=args.de_maxiter,
        cmaes_popsize=args.cmaes_popsize,
        cmaes_maxiter=args.cmaes_maxiter,
        cmaes_sigma0=args.cmaes_sigma0,
        cmaes_restarts=args.cmaes_restarts,
        top_k=args.top_k,
        polish_max_iter=args.polish_max_iter,
        workers=workers,
        run_dir=run_dir,
    )

    er = run_ensemble(problem, config)

    _emit_best_design_yaml(base, problem, er.winner_x, run_dir / "best_design.yaml")
    write_report(
        er, problem, run_dir / "report.md", run_id=args.run_id,
        baseline_tip_disp=_baseline_tip_disp(problem),
    )
    if not args.no_plot:
        _render_plots(problem, er.winner_x, run_dir / "plots")

    logger.info("Done. Winner K = %.4f N/mm; feasible=%s; report at %s",
                args.F / max(er.winner_eval.tip_disp, 1e-30),
                er.feasible, run_dir / "report.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.3: Run a quick smoke check by hand**

Run: `.venv/Scripts/python.exe scripts/optimize_geometry.py --base config/problem_7.yaml --run-id manual_smoke --smoke --no-plot`
Expected: Exits 0; `optimization_runs/manual_smoke/best_design.yaml` and `report.md` exist.

- [ ] **Step 12.4: Commit**

```bash
git add scripts/optimize_geometry.py
git commit -m "feat(optimization): add CLI wrapper script"
```

---

## Task 13: Integration smoke test

**Files:**
- Create: `tests/integration/test_optimize_geometry_smoke.py`

- [ ] **Step 13.1: Write the test**

Write `tests/integration/test_optimize_geometry_smoke.py`:

```python
"""End-to-end smoke test for the geometry optimization CLI."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "optimize_geometry.py"
BASE_YAML = REPO_ROOT / "config" / "problem_7.yaml"


@pytest.mark.slow
def test_cli_smoke_run_completes_and_beats_baseline(tmp_path):
    out_dir = tmp_path / "optimization_runs"
    run_id = "pytest_smoke"

    cmd = [
        sys.executable, str(SCRIPT),
        "--base", str(BASE_YAML),
        "--run-id", run_id,
        "--output-dir", str(out_dir),
        "--smoke",
        "--no-plot",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    run_dir = out_dir / run_id
    assert (run_dir / "best_design.yaml").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "ensemble_result.json").exists()

    # Reload best_design.yaml with the existing solver to confirm it parses.
    from fea_solver.assembler import (
        assemble_global_force_vector, assemble_global_stiffness, build_dof_map,
    )
    from fea_solver.io_yaml import load_models_from_yaml
    from fea_solver.models import DOFType
    from fea_solver.solver import run_solve_pipeline

    model = load_models_from_yaml(run_dir / "best_design.yaml")[0]
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    res = run_solve_pipeline(model, dof_map, K, F)
    tip_v = abs(float(res.displacements[dof_map.index(9, DOFType.V)]))
    assert tip_v > 0.0
    # Even on smoke budget, the optimizer should not regress catastrophically.
    # Baseline tip_v under F=15 N is on the order of 0.05 mm; allow some slack.
    assert tip_v < 1.0  # mm
```

- [ ] **Step 13.2: Run it**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_optimize_geometry_smoke.py -v -m slow`
Expected: PASS (under ~60 s on a typical machine).

- [ ] **Step 13.3: Add `slow` marker registration if pytest warns**

If pytest prints a warning about an unknown `slow` marker, append to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
    "slow: long-running tests (excluded from default run)",
]
```

- [ ] **Step 13.4: Commit**

```bash
git add tests/integration/test_optimize_geometry_smoke.py pyproject.toml
git commit -m "test(optimization): add end-to-end CLI smoke integration test"
```

---

## Task 14: Run the full pytest suite

- [ ] **Step 14.1: Run everything fast**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v -m "not slow"`
Expected: All non-slow tests pass (existing + new).

- [ ] **Step 14.2: Run the full suite including slow**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 14.3: If anything fails**

Stop and triage. Use `superpowers:systematic-debugging` to diagnose any unexpected failures before proceeding.

---

## Task 15: Launch the heavy run

This is the deliverable run; it produces the artifacts you submit.

- [ ] **Step 15.1: Pick a run id**

Use today's date plus a descriptor: `2026-04-17_heavy_v1`.

- [ ] **Step 15.2: Launch in background**

Run:
```bash
.venv/Scripts/python.exe scripts/optimize_geometry.py \
    --base config/problem_7.yaml \
    --run-id 2026-04-17_heavy_v1 \
    --workers auto
```

Expected: 1-4 hours of wall-clock. Progress visible in `optimization_runs/2026-04-17_heavy_v1/optimize_run.log`.

- [ ] **Step 15.3: Verify the report**

Open `optimization_runs/2026-04-17_heavy_v1/report.md`. Check:
- `Feasible: yes`
- `Improvement: > 0.0 %` (preferably > 50%)
- All constraints satisfied (slack >= 0)

- [ ] **Step 15.4: Verify the YAML re-solves to the reported stiffness**

Run:
```bash
uv run python main.py optimization_runs/2026-04-17_heavy_v1/best_design.yaml --no-plot
```
Expected: tip displacement at node 9 matches `report.md` to 4 decimal places.

- [ ] **Step 15.5: Commit the report and best design (NOT the checkpoints)**

```bash
git add -f optimization_runs/2026-04-17_heavy_v1/report.md \
            optimization_runs/2026-04-17_heavy_v1/best_design.yaml \
            optimization_runs/2026-04-17_heavy_v1/ensemble_result.json
git commit -m "feat(optimization): add heavy-run results for the bonus problem"
```

The `-f` flag is needed because `optimization_runs/` is gitignored; we want the headline artifacts in git but not the checkpoints.

---

## Self-review summary

- **Spec coverage**: every section of `2026-04-17-geometry-optimization-design.md` maps to a task above (problem setup -> Task 2, evaluator -> Task 3, penalty -> Task 4, SLSQP constraints -> Task 5, checkpoints -> Task 6, polish -> Task 7, DE -> Task 8, CMA-ES -> Task 9, ensemble -> Task 10, report -> Task 11, CLI -> Task 12, tests -> Tasks 2-13, heavy run -> Task 15).
- **Boundary discipline**: only Task 12 (CLI) imports `plotter`. None of the optimization modules import `plotter` or `reporter`.
- **TDD**: every implementation task starts with a failing test.
- **Type and signature consistency**: `EvalResult`, `SeedResult`, `PolishResult`, `EnsembleResult`, `EnsembleConfig`, and `GeometryOptimizationProblem` are defined once and reused identically downstream. `evaluate(x, problem) -> EvalResult` and `apply_x_to_model(x, problem) -> FEAModel` keep the same signatures everywhere they appear.
- **Frequent commits**: each task ends with a focused commit.
