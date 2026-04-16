"""Integration tests for 2D truss problems (5, 6, 7) from YAML config files."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.io_yaml import load_model_from_yaml
from fea_solver.models import DOFType, ElementType
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.solver import run_solve_pipeline

_CONFIG = Path(__file__).parent.parent.parent / "config"


def _solve_from_yaml(yaml_path: Path) -> tuple:
    """Load YAML, assemble, and solve. Returns (model, dof_map, result)."""
    model = load_model_from_yaml(yaml_path)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    return model, dof_map, result


class TestProblem5:
    """Integration tests for problem 5: 4-node, 5-element planar truss."""

    @pytest.fixture(scope="class")
    def solution(self):
        """Solve problem 5 once for the whole class."""
        yaml_path = _CONFIG / "problem_5.yaml"
        if not yaml_path.exists():
            pytest.skip("problem_5.yaml not found")
        return _solve_from_yaml(yaml_path)

    def test_all_elements_are_truss(self, solution) -> None:
        """All elements in problem 5 are TRUSS type."""
        model, _, _ = solution
        for elem in model.mesh.elements:
            assert elem.element_type == ElementType.TRUSS

    def test_four_nodes(self, solution) -> None:
        """Problem 5 has exactly 4 nodes."""
        model, _, _ = solution
        assert len(model.mesh.nodes) == 4

    def test_five_elements(self, solution) -> None:
        """Problem 5 has exactly 5 elements."""
        model, _, _ = solution
        assert len(model.mesh.elements) == 5

    def test_displacement_vector_non_zero(self, solution) -> None:
        """At least one displacement DOF is non-zero under applied load."""
        _, _, result = solution
        assert not np.allclose(result.displacements, 0.0)

    def test_postprocess_returns_five_element_results(self, solution) -> None:
        """postprocess_all_elements returns one ElementResult per element."""
        model, _, result = solution
        ers = postprocess_all_elements(model, result, n_stations=5)
        assert len(ers) == 5

    def test_all_element_results_have_zero_shear(self, solution) -> None:
        """All TRUSS ElementResults have zero shear forces."""
        model, _, result = solution
        for er in postprocess_all_elements(model, result, n_stations=5):
            np.testing.assert_allclose(er.shear_forces, np.zeros(5), atol=1e-12)

    def test_all_element_results_have_zero_moment(self, solution) -> None:
        """All TRUSS ElementResults have zero bending moments."""
        model, _, result = solution
        for er in postprocess_all_elements(model, result, n_stations=5):
            np.testing.assert_allclose(er.bending_moments, np.zeros(5), atol=1e-12)

    def test_reactions_are_non_zero(self, solution) -> None:
        """Support reactions are non-zero under applied loads."""
        _, _, result = solution
        assert not np.allclose(result.reactions, 0.0)


class TestProblem6:
    """Integration tests for problem 6: 8-node, 13-element Pratt bridge truss."""

    @pytest.fixture(scope="class")
    def solution(self):
        """Solve problem 6 once for the whole class."""
        yaml_path = _CONFIG / "problem_6.yaml"
        if not yaml_path.exists():
            pytest.skip("problem_6.yaml not found")
        return _solve_from_yaml(yaml_path)

    def test_all_elements_are_truss(self, solution) -> None:
        """All elements in problem 6 are TRUSS type."""
        model, _, _ = solution
        for elem in model.mesh.elements:
            assert elem.element_type == ElementType.TRUSS

    def test_eight_nodes(self, solution) -> None:
        """Problem 6 has exactly 8 nodes."""
        model, _, _ = solution
        assert len(model.mesh.nodes) == 8

    def test_thirteen_elements(self, solution) -> None:
        """Problem 6 has exactly 13 elements."""
        model, _, _ = solution
        assert len(model.mesh.elements) == 13

    def test_displacement_vector_non_zero(self, solution) -> None:
        """At least one DOF displaces under applied load."""
        _, _, result = solution
        assert not np.allclose(result.displacements, 0.0)

    def test_lower_chord_nodes_deflect_downward(self, solution) -> None:
        """Lower chord nodes (6, 7, 8) have negative V displacement under downward load."""
        _, dof_map, result = solution
        for node_id in (6, 7, 8):
            v = result.displacements[dof_map.index(node_id, DOFType.V)]
            assert v < 0.0, f"Node {node_id} should deflect downward"

    def test_symmetry_nodes_6_and_8(self, solution) -> None:
        """By symmetry, nodes 6 and 8 have equal magnitude vertical displacement."""
        _, dof_map, result = solution
        v6 = result.displacements[dof_map.index(6, DOFType.V)]
        v8 = result.displacements[dof_map.index(8, DOFType.V)]
        assert abs(v6) == pytest.approx(abs(v8), rel=1e-6)

    def test_postprocess_returns_thirteen_results(self, solution) -> None:
        """postprocess_all_elements returns one result per element."""
        model, _, result = solution
        ers = postprocess_all_elements(model, result, n_stations=5)
        assert len(ers) == 13


class TestProblem7:
    """Integration tests for problem 7: 9-node, 15-element beam-like truss."""

    @pytest.fixture(scope="class")
    def solution(self):
        """Solve problem 7 once for the whole class."""
        yaml_path = _CONFIG / "problem_7.yaml"
        if not yaml_path.exists():
            pytest.skip("problem_7.yaml not found")
        return _solve_from_yaml(yaml_path)

    def test_all_elements_are_truss(self, solution) -> None:
        """All elements in problem 7 are TRUSS type."""
        model, _, _ = solution
        for elem in model.mesh.elements:
            assert elem.element_type == ElementType.TRUSS

    def test_nine_nodes(self, solution) -> None:
        """Problem 7 has exactly 9 nodes."""
        model, _, _ = solution
        assert len(model.mesh.nodes) == 9

    def test_fifteen_elements(self, solution) -> None:
        """Problem 7 has exactly 15 elements."""
        model, _, _ = solution
        assert len(model.mesh.elements) == 15

    def test_displacement_vector_non_zero(self, solution) -> None:
        """At least one DOF displaces under applied load."""
        _, _, result = solution
        assert not np.allclose(result.displacements, 0.0)

    def test_postprocess_returns_fifteen_results(self, solution) -> None:
        """postprocess_all_elements returns one result per element."""
        model, _, result = solution
        ers = postprocess_all_elements(model, result, n_stations=5)
        assert len(ers) == 15

    def test_all_element_results_have_zero_shear(self, solution) -> None:
        """All TRUSS ElementResults have zero shear forces."""
        model, _, result = solution
        for er in postprocess_all_elements(model, result, n_stations=5):
            np.testing.assert_allclose(er.shear_forces, np.zeros(5), atol=1e-12)
