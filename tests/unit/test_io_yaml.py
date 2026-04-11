"""Tests for YAML input parsing -- TDD first."""
from __future__ import annotations
from pathlib import Path
import pytest
from fea_solver.io_yaml import load_model_from_yaml
from fea_solver.models import ElementType, BoundaryConditionType, LoadType, DOFType

# Use a temporary YAML file for tests
MINIMAL_BAR_YAML = """\
label: "test_bar"
mesh:
  nodes:
    - id: 1
      x: 0.0
    - id: 2
      x: 1.0
  elements:
    - id: 1
      node_i: 1
      node_j: 2
      type: bar
      material: steel
materials:
  steel:
    E: 200.0e9
    A: 0.01
    I: 0.0
boundary_conditions:
  - node_id: 1
    type: fixed_u
loads:
  nodal:
    - node_id: 2
      type: point_force_x
      magnitude: 10000.0
  distributed: []
"""

MINIMAL_BEAM_YAML = """\
label: "test_beam"
mesh:
  nodes:
    - id: 1
      x: 0.0
    - id: 2
      x: 1.0
  elements:
    - id: 1
      node_i: 1
      node_j: 2
      type: beam
      material: concrete
materials:
  concrete:
    E: 30.0e9
    A: 0.05
    I: 1.0e-4
boundary_conditions:
  - node_id: 1
    type: fixed_all
loads:
  nodal:
    - node_id: 2
      type: point_force_y
      magnitude: -5000.0
  distributed: []
"""


@pytest.fixture
def bar_yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "bar.yaml"
    p.write_text(MINIMAL_BAR_YAML)
    return p


@pytest.fixture
def beam_yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "beam.yaml"
    p.write_text(MINIMAL_BEAM_YAML)
    return p


class TestLoadModelFromYaml:
    """Tests for load_model_from_yaml YAML parsing."""
    def test_returns_fea_model(self, bar_yaml_path: Path) -> None:
        """load_model_from_yaml returns FEAModel instance."""
        from fea_solver.models import FEAModel
        model = load_model_from_yaml(bar_yaml_path)
        assert isinstance(model, FEAModel)

    def test_label_parsed(self, bar_yaml_path: Path) -> None:
        """Model label is parsed from YAML."""
        model = load_model_from_yaml(bar_yaml_path)
        assert model.label == "test_bar"

    def test_nodes_count(self, bar_yaml_path: Path) -> None:
        """Correct number of nodes parsed."""
        model = load_model_from_yaml(bar_yaml_path)
        assert len(model.mesh.nodes) == 2

    def test_node_ids_and_coords(self, bar_yaml_path: Path) -> None:
        """Node IDs and coordinates are parsed correctly."""
        model = load_model_from_yaml(bar_yaml_path)
        nodes_by_id = {n.id: n for n in model.mesh.nodes}
        assert nodes_by_id[1].x == pytest.approx(0.0)
        assert nodes_by_id[2].x == pytest.approx(1.0)

    def test_element_count(self, bar_yaml_path: Path) -> None:
        """Correct number of elements parsed."""
        model = load_model_from_yaml(bar_yaml_path)
        assert len(model.mesh.elements) == 1

    def test_element_type_bar(self, bar_yaml_path: Path) -> None:
        """Bar element type is parsed correctly."""
        model = load_model_from_yaml(bar_yaml_path)
        assert model.mesh.elements[0].element_type == ElementType.BAR

    def test_element_type_beam(self, beam_yaml_path: Path) -> None:
        """Beam element type is parsed correctly."""
        model = load_model_from_yaml(beam_yaml_path)
        assert model.mesh.elements[0].element_type == ElementType.BEAM

    def test_material_properties(self, bar_yaml_path: Path) -> None:
        """Material properties are parsed correctly."""
        model = load_model_from_yaml(bar_yaml_path)
        mat = model.mesh.elements[0].material
        assert mat.E == pytest.approx(200.0e9)
        assert mat.A == pytest.approx(0.01)

    def test_boundary_condition_type(self, bar_yaml_path: Path) -> None:
        """Boundary condition type is parsed correctly."""
        model = load_model_from_yaml(bar_yaml_path)
        assert len(model.boundary_conditions) == 1
        assert model.boundary_conditions[0].bc_type == BoundaryConditionType.FIXED_U
        assert model.boundary_conditions[0].node_id == 1

    def test_nodal_load_parsed(self, bar_yaml_path: Path) -> None:
        """Nodal load is parsed correctly."""
        model = load_model_from_yaml(bar_yaml_path)
        assert len(model.nodal_loads) == 1
        load = model.nodal_loads[0]
        assert load.node_id == 2
        assert load.load_type == LoadType.POINT_FORCE_X
        assert load.magnitude == pytest.approx(10000.0)

    def test_file_not_found_raises(self) -> None:
        """Missing YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model_from_yaml(Path("/nonexistent/path.yaml"))

    def test_element_node_references_resolved(self, bar_yaml_path: Path) -> None:
        """Element node references are resolved to Node objects."""
        # Element's node_i and node_j must be Node objects, not ints
        model = load_model_from_yaml(bar_yaml_path)
        elem = model.mesh.elements[0]
        from fea_solver.models import Node
        assert isinstance(elem.node_i, Node)
        assert isinstance(elem.node_j, Node)
        assert elem.node_i.id == 1
        assert elem.node_j.id == 2
