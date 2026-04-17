"""Full-pipeline test: solve problem_7.yaml and compute truss buckling.

This test asserts that compute_truss_buckling returns one entry per TRUSS
element and that the tuple contains the expected statuses.
"""
from __future__ import annotations

from pathlib import Path

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.buckling import compute_truss_buckling
from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.solver import run_solve_pipeline


def test_problem_7_buckling_count_matches_truss_elements() -> None:
    """problem_7 has 16 TRUSS elements; buckling tuple length must match."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "config" / "problem_7.yaml"
    (model,) = load_models_from_yaml(cfg)

    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    element_results = postprocess_all_elements(model, result, n_stations=10)

    bucklings = compute_truss_buckling(model, element_results)
    truss_count = sum(1 for e in model.mesh.elements if e.element_type.name == "TRUSS")
    assert len(bucklings) == truss_count == 16


def test_problem_7_has_at_least_one_compressive_member() -> None:
    """At least one member must carry compressive force for the feature to be meaningful."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "config" / "problem_7.yaml"
    (model,) = load_models_from_yaml(cfg)

    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    element_results = postprocess_all_elements(model, result, n_stations=10)

    bucklings = compute_truss_buckling(model, element_results)
    assert any(mb.axial_force < 0.0 for mb in bucklings)
