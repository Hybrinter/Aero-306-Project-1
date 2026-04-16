# AERO 306 FEA Solver

Finite element solver for bar/beam/frame (1D) and 2D pin-jointed truss structures. Reads YAML case files, assembles and solves the global stiffness system, post-processes internal forces, and generates plots and reports.

## Architecture

Pipeline: `YAML -> FEAModel -> DOFMap -> K/F assembly -> solve -> post-process -> report/plot`

```
src/fea_solver/
  models.py         -- all dataclasses and enums (Node, Element, FEAModel, DOFMap, SolutionResult, ...)
  io_yaml.py        -- YAML parser -> FEAModel
  assembler.py      -- global K and F assembly; DOF map (node sorted by id, DOF order: U->V->THETA)
  elements.py       -- local stiffness matrices: BAR (2x2), BEAM (4x4), FRAME (6x6), TRUSS (4x4 global); load vectors
  constraints.py    -- reduction method: partition K/F, apply BCs, recover full displacement
  solver.py         -- np.linalg.solve, reaction forces, condition number check
  postprocessor.py  -- Hermite shape functions for M(x), V(x) recovery; direction-cosine axial force for TRUSS
  reporter.py       -- rich-formatted tables; writes report file
  plotter.py        -- SFD, BMD, deformed shape plots (matplotlib); truss member force plot
  logging_config.py -- file + console logger setup

config/             -- 16 YAML test case definitions
tests/unit/         -- 7 unit test modules (one per source module)
tests/integration/  -- bar, beam, combined element, and truss integration tests
```

## Key Types

- `ElementType`: BAR | BEAM | FRAME | TRUSS
- `DOFType`: U | V | THETA
- `BoundaryConditionType`: FIXED_U | FIXED_V | FIXED_THETA | FIXED_ALL | PIN | ROLLER
- `LoadType`: POINT_FORCE_X | POINT_FORCE_Y | POINT_MOMENT | DISTRIBUTED_Y | DISTRIBUTED_LINEAR
- `FEAModel`: complete problem (mesh, BCs, loads)
- `DOFMap`: (node_id, DOFType) -> global index; single source of DOF ordering truth
- `SolutionResult`: displacements, reactions, dof_map, model
- `ElementResult`: axial_force, shear_forces, bending_moments, x_stations

All data structs use `@dataclass(frozen=True)` (or `slots=True`). Use `dataclasses.replace()` for copies.

## Entry Point

```bash
uv run python main.py <config.yaml> [--no-plot] [--save-plots <dir>] [--n-stations <int>]
```

## Conventions

- snake_case, ASCII only
- Strong typing everywhere, mypy strict
- No duck typing, no dynamic dispatch
- Docstrings required on all functions/classes (summary, inputs with types, outputs with types, notes)
- Tests in `tests/unit/` and `tests/integration/`; run with `.venv/Scripts/python.exe -m pytest tests/ -v`
