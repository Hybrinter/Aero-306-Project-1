"""Integration tests for combined/frame element cases."""
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


def full_solve(yaml_path: Path) -> tuple:
    model = load_model_from_yaml(yaml_path)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    element_results = postprocess_all_elements(model, result)
    return model, dof_map, result, element_results


class TestCase05CombinedBarBeam:
    """example_case_05_combined_bar_beam.yaml: frame element, axial + transverse loads."""
    def test_model_solves_without_error(self) -> None:
        """Frame element with combined loads solves without error."""
        model, dof_map, result, element_results = full_solve(CONFIG / "example_case_05_combined_bar_beam.yaml")
        assert result.displacements is not None
        assert len(element_results) == 1

    def test_axial_displacement_non_zero(self) -> None:
        """Axial displacement is non-zero."""
        model, dof_map, result, _ = full_solve(CONFIG / "example_case_05_combined_bar_beam.yaml")
        tip_node = max(model.mesh.nodes, key=lambda n: n.x)
        u_tip = result.displacements[dof_map.index(tip_node.id, DOFType.U)]
        assert abs(u_tip) > 0.0

    def test_transverse_displacement_non_zero(self) -> None:
        """Transverse displacement is non-zero."""
        model, dof_map, result, _ = full_solve(CONFIG / "example_case_05_combined_bar_beam.yaml")
        tip_node = max(model.mesh.nodes, key=lambda n: n.x)
        v_tip = result.displacements[dof_map.index(tip_node.id, DOFType.V)]
        assert abs(v_tip) > 0.0

    def test_internal_forces_non_zero(self) -> None:
        """Internal forces are non-zero."""
        _, _, _, element_results = full_solve(CONFIG / "example_case_05_combined_bar_beam.yaml")
        er = element_results[0]
        assert abs(er.axial_force) > 0.0

    def test_axial_displacement_matches_bar_formula(self) -> None:
        """Axial displacement matches bar formula."""
        # u_tip = P_x * L / (E * A)
        P_x = 10000.0
        L = 1.0
        E = 200.0e9
        A = 0.01
        analytical_u = P_x * L / (E * A)
        model, dof_map, result, _ = full_solve(CONFIG / "example_case_05_combined_bar_beam.yaml")
        tip_node = max(model.mesh.nodes, key=lambda n: n.x)
        u_tip = result.displacements[dof_map.index(tip_node.id, DOFType.U)]
        assert u_tip == pytest.approx(analytical_u, rel=0.01)

    def test_transverse_displacement_matches_beam_formula(self) -> None:
        """Transverse displacement matches beam formula."""
        # v_tip = P_y * L^3 / (3 * E * I)
        P_y = -5000.0
        L = 1.0
        E = 200.0e9
        I = 1.0e-4
        analytical_v = P_y * L**3 / (3.0 * E * I)
        model, dof_map, result, _ = full_solve(CONFIG / "example_case_05_combined_bar_beam.yaml")
        tip_node = max(model.mesh.nodes, key=lambda n: n.x)
        v_tip = result.displacements[dof_map.index(tip_node.id, DOFType.V)]
        assert v_tip == pytest.approx(analytical_v, rel=0.01)
