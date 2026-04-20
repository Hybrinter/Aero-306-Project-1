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
