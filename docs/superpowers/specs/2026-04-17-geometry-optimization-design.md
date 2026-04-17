# Geometry Optimization for Beam-Like Truss -- Design

**Date**: 2026-04-17
**Branch**: optimization-challenge
**Scope**: Implement a constrained shape optimizer for the bonus problem
described in `problems/optimization_bonus.png`. Maximize the structural
stiffness of the 9-node beam-like truss in `config/problem_7.yaml` by
varying the reference positions of nodes 2, 3, 5, 6, 7, 8 subject to
stress, buckling, and minimum-element-length constraints.

## Problem Statement

The bonus optimizes the structure from problem 3 of the assignment
(`config/problem_7.yaml`). Inputs:

- **Structure**: 9 nodes, 16 truss elements, fixed connectivity.
- **Frozen state**:
  - Node 1 = (0, 0), pinned (FIXED_U + FIXED_V).
  - Node 4 = (0, -10), x-roller (FIXED_U).
  - Node 9 = (25, -5), free DOFs.
  - Material: `E = 3003 MPa`, `A = 1 mm**2`, `I = 1/12 mm**4` for every member.
- **Load**: `F_y = -15 N` at node 9 (modified per the bonus statement).
- **Free design variables**: full `(x, y)` of nodes 2, 3, 5, 6, 7, 8 -- 12
  continuous variables in mm.
- **Constraints**:
  - Stress in any member: `|sigma_e| <= 72 MPa`.
  - No member buckled: `|N_e| < P_cr,e = pi**2 * E * I / L_e**2` for every
    compression member.
  - Element length: `L_e >= 5 mm` for every element.
- **Objective**: maximize structural stiffness `K = F / |v_y(node 9)|`.
  Equivalent to minimizing `J(x) = |v_y(node 9)|` under the fixed 15 N load.

## Goals

1. Find the highest-stiffness feasible design within a heavy compute budget
   (target: hours of wall-clock, parallel across CPU cores, resumable).
2. Produce a drop-in replacement YAML (`best_design.yaml`) that the existing
   FEA solver can load and re-verify without modification.
3. Produce a one-page markdown report summarizing stiffness, all 12 free node
   positions, max stress, max buckling ratio, min element length, and
   ensemble convergence statistics.
4. Keep the optimization core fully decoupled from the existing solver,
   plotter, and reporter. Optimization code never imports presentation
   modules; the CLI script is the only seam.

## Non-Goals

- No connectivity changes; the 16-edge topology of `config/problem_7.yaml`
  is locked.
- No material or section changes; `(E, A, I)` are frozen.
- No multi-load-case optimization; only the single 15 N tip load is
  considered.
- No symbolic gradients of the FE response; finite-difference jacobians are
  sufficient at 12-D.
- No GUI; CLI only.

## Decision Vector and Bounds

```
x = [x2, y2, x3, y3, x5, y5, x6, y6, x7, y7, x8, y8]   (12 floats, mm)
```

Box bounds (per coordinate, applied uniformly across all six free nodes):

- `x_i in [-10, 30] mm`
- `y_i in [-25, 15] mm`

Bounds are encoded once in `GeometryOptimizationProblem`. They can be
tightened after early runs reveal optimizer behavior near the box edges.

## Failure Modes the Formulation Must Survive

The objective evaluator must never crash an optimizer worker. The following
failure modes are caught and converted to a finite sentinel result:

- Two nodes coincident -> element of length 0 -> assembly division by zero.
  Mitigation: `evaluate` short-circuits with `solve_ok = False` whenever any
  `L_e < 1e-6 mm`.
- Near-collinear or otherwise singular configuration -> `LinAlgError` from
  the linear solver. Mitigation: `try/except` around `run_solve_pipeline`,
  return `solve_ok = False`.
- Tiny `L_e` -> astronomical `P_cr` (harmless for buckling) but inflated
  stress. The stress constraint catches this; nothing additional is needed.

A `solve_ok = False` evaluator result feeds a `+1e12` sentinel to the
penalty function -- finite (so DE/CMA-ES distributions don't blow up) and
huge (so no genuine feasible candidate ever loses to it).

## Architecture

### Module layout

```
src/fea_solver/
  optimization/
    __init__.py
    problem.py         GeometryOptimizationProblem dataclass: bounds, frozen
                       nodes, constraint thresholds, F magnitude.
    objective.py       build_model_from_x(x, problem) -> FEAModel
                       evaluate(x, problem) -> EvalResult
    penalty.py         penalized_objective(x, problem, weights) -> float
    constraints.py     SLSQP-style g_i(x) >= 0 callables for the polish stage
    global_search.py   run_de(problem, seed, ...) and
                       run_cmaes(problem, seed, ...) -> SeedResult
    polish.py          slsqp_polish(x0, problem) -> PolishResult
    ensemble.py        Orchestrate parallel global + polish; selection rule
    report.py          Markdown one-pager generator
    checkpoint.py      JSON / pickle (for cma) serializers and run-dir IO
scripts/
  optimize_geometry.py CLI wrapper
tests/unit/
  test_optimization_problem.py
  test_optimization_objective.py
  test_optimization_penalty.py
  test_optimization_polish.py
  test_optimization_checkpoint.py
tests/integration/
  test_optimize_geometry_smoke.py    @pytest.mark.slow
```

### Boundary discipline

- Modules under `src/fea_solver/optimization/` import only from the FEA
  solver core (`assembler`, `solver`, `postprocessor`, `buckling`, `models`,
  `io_yaml`).
- Modules under `src/fea_solver/optimization/` MUST NOT import `plotter`,
  `reporter`, or any presentation-layer module.
- The CLI script (`scripts/optimize_geometry.py`) is the only place that
  wires optimization output to the existing plotter and reporter.

### New runtime dependency

`cma` (PyPI; MIT-licensed; pure Python) added to `pyproject.toml`. DE and
SLSQP come from SciPy, which is already a project dependency.

## Optimization Formulation

### Single evaluation

`objective.py::evaluate(x, problem) -> EvalResult` performs **exactly one**
forward FE solve per call:

```python
@dataclass(frozen=True, slots=True)
class EvalResult:
    tip_disp: float                        # |v_y(node 9)| in mm
    max_stress: float                      # max |sigma_e| in MPa
    max_buckling_ratio: float              # max |N_e|/P_cr_e (compression only)
    min_length: float                      # min L_e in mm
    stress_violations: tuple[float, ...]   # max(0, |sigma|/sigma_max - 1) per element
    buckling_violations: tuple[float, ...] # max(0, |N|/P_cr - 1) per compression member
    length_violations: tuple[float, ...]   # max(0, 1 - L/L_min) per element
    feasible: bool                         # all violations zero AND solve_ok
    solve_ok: bool                         # FE solve did not raise / coincident-node guard
```

The penalty wrapper and the SLSQP constraint callables both consume the
same `EvalResult`; an LRU cache (key = `bytes(np.asarray(x).round(10))`)
prevents duplicate solves during polish.

### Penalty function (global stage)

```
P(x) = J(x) + w_s * sum(stress_violations**2)
            + w_b * sum(buckling_violations**2)
            + w_l * sum(length_violations**2)
```

with default weights `w_s = w_b = w_l = 100.0`. Quadratic penalties are
smooth at the boundary and dominate `J` (typically `1e-3` to `1e-1` mm)
when violated. Weights are exposed as CLI flags so they can be raised if
the optimizer spends its budget in infeasible regions.

`solve_ok = False` -> `P(x) = 1e12`.

### SLSQP constraint callables (polish stage)

`constraints.py` exposes three vector-valued ineq callables:

```
stress_constraint_vec(x)   -> array shape (16,)  -- (sigma_max - |sigma_e|) / sigma_max
buckling_constraint_vec(x) -> array shape (16,)  -- 1 - |N_e|/P_cr_e (zero for tension)
length_constraint_vec(x)   -> array shape (16,)  -- (L_e - L_min) / L_min
```

All three pull from the same cached `EvalResult` keyed on `x`. SLSQP's
`'ineq'` convention is `g(x) >= 0`.

## Global Search

Both arms run in isolated worker processes via `multiprocessing.Pool`,
parallel at the seed level. A crash in one seed does not kill the others;
the worker writes a `SeedResult` with `solve_ok = False` and a traceback
to its checkpoint and returns.

### Shared seed-result shape

```python
@dataclass(frozen=True, slots=True)
class SeedResult:
    algorithm: str                       # "DE" or "CMA-ES"
    seed: int
    best_x: NDArray[np.float64]          # shape (12,)
    best_eval: EvalResult                # evaluate(best_x)
    best_penalty: float                  # P(best_x)
    history: tuple[HistoryPoint, ...]    # (gen, best_pen, mean_pen, n_feas) per gen
    wall_clock_s: float
    checkpoint_path: Path
```

### DE arm (`run_de`)

`scipy.optimize.differential_evolution`:

- `strategy='best1bin'`, `popsize=30` (-> 360 individuals at 12-D),
  `maxiter=600`, `mutation=(0.5, 1.5)`, `recombination=0.9`,
  `init='sobol'`, `tol=1e-7`, `polish=False`, `workers=1`, `seed=seed`.
- Callback writes a checkpoint JSON every 25 generations.
- `polish=False` because polishing is done across the ensemble, not within
  each DE run.

### CMA-ES arm (`run_cmaes`)

`cma.fmin2(...)`:

- `x0` from a Sobol' sample of the bound box (per seed; not always the
  baseline) so seeds explore distinct basins.
- `sigma0 = 5.0 mm` (a few percent of the box width).
- `popsize=20`, `maxiter=800`, `seed=seed`, `bounds=[lower, upper]`,
  `verbose=-9`.
- IPOP restart: `restarts=5, incpopsize=2` (popsize doubles on stagnation).
- Checkpoint via `es.pickle_dumps()` every 50 iterations.

### Default seed plan ("heavy" budget)

- 16 DE seeds * (~360 pop * 600 gen)  ~= 3.5M evaluations.
- 16 CMA-ES seeds * (20 pop * 800 iter * (1 + restarts)) ~= 1.5M
  evaluations.
- Single FE solve ~ 2-5 ms on this 9-node mesh -> 3-4 hours wall-clock at
  full parallelism on an 8-core machine.

CLI knobs allow scaling: `--de-seeds`, `--cmaes-seeds`, `--de-maxiter`,
`--cmaes-maxiter`, plus `--smoke` (every knob ~5% of full) and
`--resume <run_id>` (read all seed checkpoints, continue from saved state).

## Polish Stage

After every global seed completes, the ensemble takes the **top-K = 5**
candidates from each seed (by `best_penalty`) and runs each through SLSQP
under the real nonlinear constraints. With 32 seeds * top-5, up to 160
polish jobs run in parallel through the same `multiprocessing.Pool`.

```python
result = scipy.optimize.minimize(
    fun=lambda x: evaluate(x, problem).tip_disp,
    jac=None,                                  # finite difference
    x0=candidate.best_x,
    method='SLSQP',
    bounds=problem.box_bounds,
    constraints=[
        {'type': 'ineq', 'fun': stress_constraint_vec},
        {'type': 'ineq', 'fun': buckling_constraint_vec},
        {'type': 'ineq', 'fun': length_constraint_vec},
    ],
    options={'maxiter': 200, 'ftol': 1e-9, 'eps': 1e-5},
)
```

`PolishResult` records `success`, final `x`, final `EvalResult`, iteration
count, and exit message. SLSQP failures do not abort the ensemble; the
candidate is just compared against everyone else's outcome.

## Ensemble Selection

Hard rules in order:

1. Drop any candidate where `EvalResult.solve_ok = False`.
2. Drop any candidate where `EvalResult.feasible = False` after polish
   (per-element slack tolerance: `1e-6` to absorb numerical noise).
3. Among the remainder, return `argmin(tip_disp)`.

If step 2 leaves zero candidates: log a loud warning, fall back to
`argmin(tip_disp + 1.0 * total_violation)`, and flag the design as
**infeasible** in the report. Should be unreachable on the heavy budget;
matters only for `--smoke`.

```python
@dataclass(frozen=True, slots=True)
class EnsembleResult:
    winner_x: NDArray[np.float64]
    winner_eval: EvalResult
    winner_origin: tuple[str, int]      # (algorithm, seed)
    all_seeds: tuple[SeedResult, ...]
    all_polish: tuple[PolishResult, ...]
    wall_clock_s: float
    feasible: bool
```

## CLI and Output Artifacts

### Invocation

```bash
uv run python scripts/optimize_geometry.py \
    --base config/problem_7.yaml \
    --F 15.0 \
    --sigma-max 72.0 \
    --L-min 5.0 \
    --de-seeds 16 --cmaes-seeds 16 \
    --de-maxiter 600 --cmaes-maxiter 800 \
    --top-k 5 \
    --workers auto \
    --run-id 2026-04-17_heavy \
    --output-dir optimization_runs \
    [--resume] [--smoke] [--no-plot]
```

### Run directory: `optimization_runs/<run_id>/`

```
optimize_run.log              full Python logger output, INFO+
config.json                   exact CLI args + resolved defaults
checkpoints/
  de_seed_00.json             incremental DE state (every 25 gens)
  cmaes_seed_00.pkl           pickled CMA-ES ES (every 50 iters)
  ...
seed_results/
  de_seed_00.json             final SeedResult per seed
  ...
polish_results/
  polish_de_seed_00_rank_0.json
  ...
ensemble_result.json          full EnsembleResult
best_design.yaml              drop-in replacement for problem_7.yaml
report.md                     human-readable one-page summary
plots/
  convergence.png             best-penalty vs generation (all seeds, both algos)
  truss_deformed.png          winner's deformed shape
  truss_forces.png            winner's member force gradient
  truss_stress.png            winner's stress map
  buckling_overlay.png        winner with any buckled members highlighted
```

`optimization_runs/` is added to `.gitignore`.

`best_design.yaml` is identical in schema to `config/problem_7.yaml`: same
connectivity, materials, BCs, with only the six free node coordinates and
the 15 N load magnitude updated. Verifying the optimizer is then just
`uv run python main.py optimization_runs/<run_id>/best_design.yaml`.

### `report.md` template

```markdown
# Geometry Optimization Report -- <run_id>

**Wall-clock**: 3h 12m
**Origin**: CMA-ES seed 7, polished
**Feasible**: yes

## Objective
| Quantity              | Value         | Target / Limit  |
|-----------------------|--------------:|----------------:|
| Tip displacement      | 0.0184 mm     | (minimised)     |
| Stiffness K = F/|v|   | 815.2 N/mm    | (maximised)     |
| Baseline K            | 312.4 N/mm    | --              |
| Improvement           | +161 %        | --              |

## Constraints
| Quantity              | Value         | Limit           | Slack    |
|-----------------------|--------------:|----------------:|---------:|
| Max member |stress|   | 71.94 MPa     | 72.00 MPa       | 0.06 MPa |
| Max buckling ratio    | 0.998         | < 1.000         | 0.002    |
| Min element length    | 5.001 mm      | >= 5.000 mm     | 0.001 mm |

## Best design
| Node | x [mm] | y [mm] | Status |
|------|-------:|-------:|--------|
|  1   |  0.000 |  0.000 | frozen |
|  2   |  ...   |  ...   | free   |
|  3   |  ...   |  ...   | free   |
|  4   |  0.000 | -10.000| frozen |
|  5   |  ...   |  ...   | free   |
|  6   |  ...   |  ...   | free   |
|  7   |  ...   |  ...   | free   |
|  8   |  ...   |  ...   | free   |
|  9   | 25.000 | -5.000 | frozen |

## Ensemble summary
| Algorithm | Seeds | Best K | Median K | Worst feasible K |
|-----------|------:|-------:|---------:|-----------------:|
| DE        |  16   |  ...   |   ...    |       ...        |
| CMA-ES    |  16   |  ...   |   ...    |       ...        |

Active constraints at optimum: stress (members 7, 13), buckling (member 5).
```

The convergence plot is one matplotlib figure with up to 32 series (one
per seed), DE blue, CMA-ES orange, log-y on penalty, winning seed
highlighted in bold.

## Testing Strategy

### Unit tests (fast, no `slow` mark)

| File | What it locks down |
|------|--------------------|
| `test_optimization_problem.py` | `GeometryOptimizationProblem` rejects malformed bounds, accepts the canonical baseline; `apply_x_to_model` produces a valid `FEAModel` (frozen nodes preserved, free nodes overwritten, 16 elements intact). |
| `test_optimization_objective.py` | `evaluate(x_baseline)` reproduces the known baseline tip displacement of `config/problem_7.yaml` with `F = 15 N` to 1e-6 mm; `evaluate` on a degenerate `x` (two nodes coincident) returns `solve_ok = False` without raising. |
| `test_optimization_penalty.py` | A feasible `x` gives `P(x) == J(x)` to machine precision; a single-element overstress gives `P(x) > J(x) + w_s * (excess**2)` within tolerance; weight scaling correct relative to `J`. |
| `test_optimization_polish.py` | A near-feasible point with one slightly-violated stress constraint converges under SLSQP to a feasible point in <= 50 iterations; an all-coincident start returns `success = False` without crashing. |
| `test_optimization_checkpoint.py` | Round-trip serialise/deserialise of `SeedResult` and `EnsembleResult` is bit-exact; a corrupt checkpoint file raises a clear error. |

### Integration test (`@pytest.mark.slow`, target <= 60 s)

`tests/integration/test_optimize_geometry_smoke.py` runs
`scripts/optimize_geometry.py --smoke --run-id pytest_smoke --no-plot`
end-to-end. Asserts:

1. The script exits 0.
2. `optimization_runs/pytest_smoke/best_design.yaml` exists and parses
   with the existing YAML loader.
3. The reported stiffness `K_best > K_baseline`.
4. `EnsembleResult.feasible` is True.
5. All `plots/*.png` are skipped under `--no-plot` and the run still
   succeeds.

A pytest fixture cleans up the run directory after the test.

### Property test (one, with `hypothesis`)

`evaluate(x)` is invariant under reordering of element ids in the
underlying YAML (sanity that the optimizer never relies on element
index ordering).

### What we deliberately do not test

- The exact best stiffness from a heavy run; that is an empirical
  deliverable, not a regression target. The integration test only checks
  "beats baseline."
- DE / CMA-ES internals; those are well-tested upstream libraries.
- Plot pixel content; the existing plotter tests already cover it.

### TDD discipline

Per `superpowers:test-driven-development`: write each unit test before
its production code, run it, watch it fail for the right reason, then
implement. The integration smoke test is added last.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Heavy-run ends infeasible (no candidate satisfies all constraints) | Penalty weights are CLI-tunable; `--smoke` validates the pipeline end-to-end first; ensemble selection has an infeasible fallback that still produces a flagged report. |
| FE solve raises on a degenerate `x` and kills a worker | `evaluate` short-circuits when any `L_e < 1e-6 mm`; remaining `LinAlgError` is caught and converted to `solve_ok = False`. |
| `cma` package incompatibility with current numpy/scipy versions | Pin a compatible version range in `pyproject.toml`; smoke test runs in CI. |
| Multiprocessing interaction with matplotlib's Agg backend | Workers do no plotting; only the parent process renders after the ensemble completes. |
| Checkpoint corruption from a partial write | Write to `*.tmp`, then atomic rename. Detected on resume by JSON / pickle parse failure with a clear error. |
| Best design overfits to the F = 15 N load case | Out of scope per problem statement; documented as a non-goal. The `--F` flag would let a future user re-optimize for a different load if the instructor changes it. |

## Open Questions

None at design time. Anything that surfaces during implementation will be
captured in the implementation plan.
