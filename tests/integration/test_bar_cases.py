"""Integration tests for bar element cases against analytical solutions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from fea_solver.io_yaml import load_model_from_yaml
from fea_solver.assembler import build_dof_map, assemble_global_stiffness, assemble_global_force_vector
from fea_solver.solver import run_solve_pipeline
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.models import DOFType

CONFIG = Path("config")


def full_solve(yaml_path: Path):
    """Run the complete solve pipeline for a YAML config."""
    model = load_model_from_yaml(yaml_path)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    element_results = postprocess_all_elements(model, result)
    return model, dof_map, result, element_results


class TestCase01BarAxial:
    """case_01_bar_axial.yaml: 3-node bar, E=200GPa, A=0.01m², P=10kN at tip.

    Analytical: u_tip = PL/EA = 10000/(200e9*0.01) = 5e-6 m
    """
    def test_tip_displacement(self):
        model, dof_map, result, _ = full_solve(CONFIG / "case_01_bar_axial.yaml")
        # Tip is at node 3 (x=1.0)
        tip_node = max(model.mesh.nodes, key=lambda n: n.x)
        u_tip = result.displacements[dof_map.index(tip_node.id, DOFType.U)]
        analytical = 10000.0 / (200.0e9 * 0.01)  # 5e-6 m
        assert u_tip == pytest.approx(analytical, rel=0.01)

    def test_reaction_equals_applied_load(self):
        model, dof_map, result, _ = full_solve(CONFIG / "case_01_bar_axial.yaml")
        # Reaction at fixed node must equal -10000 N
        assert abs(result.reactions[0]) == pytest.approx(10000.0, rel=0.01)

    def test_axial_forces_equal_applied_load(self):
        _, _, result, element_results = full_solve(CONFIG / "case_01_bar_axial.yaml")
        for er in element_results:
            assert abs(er.axial_force) == pytest.approx(10000.0, rel=0.01)


class TestCase07MultiMaterial:
    """case_07_multi_material.yaml: 3-segment bar, different materials."""

    def test_tip_displacement_analytical(self):
        # u_tip = P * sum(L_i / (E_i * A_i))
        # L=1m each, E=[200e9, 70e9, 110e9], A=[0.01, 0.02, 0.015], P=50000
        P = 50000.0
        segments = [
            (1.0, 200.0e9, 0.01),
            (1.0, 70.0e9, 0.02),
            (1.0, 110.0e9, 0.015),
        ]
        u_analytical = P * sum(L / (E * A) for L, E, A in segments)
        model, dof_map, result, _ = full_solve(CONFIG / "case_07_multi_material.yaml")
        tip_node = max(model.mesh.nodes, key=lambda n: n.x)
        u_tip = result.displacements[dof_map.index(tip_node.id, DOFType.U)]
        assert u_tip == pytest.approx(u_analytical, rel=0.01)
