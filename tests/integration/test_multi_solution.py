"""Integration tests for multi-solution YAML pipeline."""
from __future__ import annotations
from pathlib import Path
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.io_yaml import load_models_from_yaml
from fea_solver.models import SolutionSeries
from fea_solver.plotter import plot_shear_force_diagram
from fea_solver.postprocessor import postprocess_all_elements
from fea_solver.solver import run_solve_pipeline


_MULTI_SOL_YAML = """\
label: test_multi
unit_system: SI
solutions:
  - label: coarse
    mesh:
      nodes:
        - {id: 1, x: 0.0, y: 0.0}
        - {id: 2, x: 1.0, y: 0.0}
      elements:
        - {id: 1, node_i: 1, node_j: 2, type: beam, material: steel}
    materials:
      steel: {E: 200.0e9, A: 0.01, I: 1.0e-4}
    boundary_conditions:
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
    loads:
      nodal:
        - {node_id: 2, type: point_force_y, magnitude: -1000.0}
  - label: fine
    mesh:
      nodes:
        - {id: 1, x: 0.0, y: 0.0}
        - {id: 2, x: 0.5, y: 0.0}
        - {id: 3, x: 1.0, y: 0.0}
      elements:
        - {id: 1, node_i: 1, node_j: 2, type: beam, material: steel}
        - {id: 2, node_i: 2, node_j: 3, type: beam, material: steel}
    materials:
      steel: {E: 200.0e9, A: 0.01, I: 1.0e-4}
    boundary_conditions:
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
    loads:
      nodal:
        - {node_id: 3, type: point_force_y, magnitude: -1000.0}
"""

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _full_solve(model):
    """Run full solve pipeline for a model."""
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    return run_solve_pipeline(model, dof_map, K, F)


class TestMultiSolutionPipeline:
    """Integration tests for multi-solution YAML parsing and end-to-end solving."""

    def test_parse_and_solve_two_solutions(self, tmp_path: Path) -> None:
        """Two solutions parsed from YAML and solved produce non-empty ElementResults."""
        yaml_file = tmp_path / "multi.yaml"
        yaml_file.write_text(_MULTI_SOL_YAML)
        models = load_models_from_yaml(yaml_file)
        assert len(models) == 2
        for model in models:
            result = _full_solve(model)
            element_results = postprocess_all_elements(model, result, n_stations=10)
            assert len(element_results) > 0

    def test_overlay_plot_no_exception(self, tmp_path: Path) -> None:
        """Overlaid SFD from two solutions raises no exception and returns Figure."""
        yaml_file = tmp_path / "multi.yaml"
        yaml_file.write_text(_MULTI_SOL_YAML)
        models = load_models_from_yaml(yaml_file)
        all_series = []
        for model in models:
            result = _full_solve(model)
            element_results = postprocess_all_elements(model, result, n_stations=10)
            all_series.append(SolutionSeries(
                label=model.label.split("/")[-1],
                element_results=tuple(element_results),
                model=model,
                result=result,
            ))
        fig = plot_shear_force_diagram(all_series)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_backward_compat_via_load_models(self) -> None:
        """Old single-mesh YAML loaded via load_models_from_yaml returns list of one."""
        single_yaml = _CONFIG_DIR / "example_case_02_cantilever_beam.yaml"
        if not single_yaml.exists():
            pytest.skip("example_case_02_cantilever_beam.yaml not found")
        models = load_models_from_yaml(single_yaml)
        assert len(models) == 1
        assert len(models[0].mesh.elements) > 0

    def test_plot_saved_to_disk(self, tmp_path: Path) -> None:
        """Overlay plot saved to a file path produces a PNG file."""
        yaml_file = tmp_path / "multi.yaml"
        yaml_file.write_text(_MULTI_SOL_YAML)
        models = load_models_from_yaml(yaml_file)
        all_series = []
        for model in models:
            result = _full_solve(model)
            element_results = postprocess_all_elements(model, result, n_stations=10)
            all_series.append(SolutionSeries(
                label=model.label.split("/")[-1],
                element_results=tuple(element_results),
                model=model,
                result=result,
            ))
        out = tmp_path / "shear.png"
        fig = plot_shear_force_diagram(all_series, output_path=out)
        plt.close(fig)
        assert out.exists()
