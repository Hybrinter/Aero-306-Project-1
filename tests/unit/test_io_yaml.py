"""Tests for YAML input parsing -- TDD first."""
from __future__ import annotations
from pathlib import Path
import pytest
import pydantic
from fea_solver.io_yaml import load_model_from_yaml, load_models_from_yaml
from fea_solver.models import ElementType, LoadType, DOFType, LinearConstraint
from fea_solver.units import UnitSystem

# Use a temporary YAML file for tests
MINIMAL_BAR_YAML = """\
label: "test_bar"
mesh:
  nodes:
    - id: 1
      x: 0.0
      y: 0.0
    - id: 2
      x: 1.0
      y: 0.0
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
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
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
      y: 0.0
    - id: 2
      x: 1.0
      y: 0.0
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
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
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

    def test_boundary_condition_parsed(self, bar_yaml_path: Path) -> None:
        """Boundary condition is parsed into a LinearConstraint with correct fields."""
        model = load_model_from_yaml(bar_yaml_path)
        assert len(model.boundary_conditions) == 1
        c = model.boundary_conditions[0]
        assert c.node_id == 1
        # [1.0, 0.0, 0.0] after normalization (already unit length)
        assert c.coefficients == pytest.approx((1.0, 0.0, 0.0))
        assert c.rhs == pytest.approx(0.0)

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


# ---------------------------------------------------------------------------
# YAML fixtures for multi-solution tests
# ---------------------------------------------------------------------------

_MULTI_SOL_YAML = """\
label: "test_multi"
unit_system: SI
solutions:
  - label: "coarse"
    mesh:
      nodes:
        - {id: 1, x: 0.0, y: 0.0}
        - {id: 2, x: 1.0, y: 0.0}
      elements:
        - {id: 1, node_i: 1, node_j: 2, type: beam, material: steel}
    materials:
      steel: {E: 200.0e9, A: 0.01, I: 1.0e-4}
    boundary_conditions:
      - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
    loads:
      nodal:
        - {node_id: 2, type: point_force_y, magnitude: -1000.0}
  - label: "fine"
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
      - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
    loads:
      nodal:
        - {node_id: 3, type: point_force_y, magnitude: -1000.0}
"""

_MULTI_SOL_DESCRIPTION_YAML = """\
label: "test_with_desc"
unit_system: SI
solutions:
  - label: "coarse"
    description: "This field should be silently ignored by Pydantic v2"
    mesh:
      nodes:
        - {id: 1, x: 0.0, y: 0.0}
        - {id: 2, x: 1.0, y: 0.0}
      elements:
        - {id: 1, node_i: 1, node_j: 2, type: beam, material: steel}
    materials:
      steel: {E: 200.0e9, A: 0.01, I: 1.0e-4}
    boundary_conditions:
      - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
    loads: {}
"""

_MULTI_SOL_EMPIRICAL_YAML = """\
label: "test_emp"
unit_system: empirical
solutions:
  - label: "coarse"
    mesh:
      nodes:
        - {id: 1, x: 0.0, y: 0.0}
        - {id: 2, x: 12.0, y: 0.0}
      elements:
        - {id: 1, node_i: 1, node_j: 2, type: beam, material: alum}
    materials:
      alum: {E: 10000000.0, A: 0.01, I: 0.001}
    boundary_conditions:
      - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
    loads:
      nodal:
        - {node_id: 2, type: point_force_y, magnitude: -5.0}
  - label: "fine"
    mesh:
      nodes:
        - {id: 1, x: 0.0, y: 0.0}
        - {id: 2, x: 6.0, y: 0.0}
        - {id: 3, x: 12.0, y: 0.0}
      elements:
        - {id: 1, node_i: 1, node_j: 2, type: beam, material: alum}
        - {id: 2, node_i: 2, node_j: 3, type: beam, material: alum}
    materials:
      alum: {E: 10000000.0, A: 0.01, I: 0.001}
    boundary_conditions:
      - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
      - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
    loads:
      nodal:
        - {node_id: 3, type: point_force_y, magnitude: -5.0}
"""

_NO_MESH_NO_SOLUTIONS_YAML = """\
label: "bad"
unit_system: SI
materials:
  steel: {E: 200.0e9, A: 0.01, I: 1.0e-4}
"""

_EMPTY_SOLUTIONS_YAML = """\
label: "empty"
unit_system: SI
solutions: []
"""


@pytest.fixture
def multi_yaml_path(tmp_path: Path) -> Path:
    """Write multi-solution YAML to a temp file."""
    p = tmp_path / "multi.yaml"
    p.write_text(_MULTI_SOL_YAML)
    return p


class TestLoadModelsFromYaml:
    """Tests for the new load_models_from_yaml multi-solution parser."""

    def test_single_solution_backward_compat(self, bar_yaml_path: Path) -> None:
        """Old mesh-keyed YAML returns a list of exactly one model."""
        models = load_models_from_yaml(bar_yaml_path)
        assert len(models) == 1

    def test_single_solution_backward_compat_model_valid(self, bar_yaml_path: Path) -> None:
        """Single-solution backward compat model has correct label."""
        models = load_models_from_yaml(bar_yaml_path)
        assert models[0].label == "test_bar"

    def test_multi_solution_returns_two_models(self, multi_yaml_path: Path) -> None:
        """Multi-solution YAML with two entries returns list of two."""
        models = load_models_from_yaml(multi_yaml_path)
        assert len(models) == 2

    def test_composite_labels(self, multi_yaml_path: Path) -> None:
        """Labels are composite: top_label/solution_label."""
        models = load_models_from_yaml(multi_yaml_path)
        assert models[0].label == "test_multi/coarse"
        assert models[1].label == "test_multi/fine"

    def test_coarse_has_fewer_nodes(self, multi_yaml_path: Path) -> None:
        """Coarse solution has 2 nodes; fine has 3."""
        models = load_models_from_yaml(multi_yaml_path)
        assert len(models[0].mesh.nodes) == 2
        assert len(models[1].mesh.nodes) == 3

    def test_shared_unit_system_si(self, multi_yaml_path: Path) -> None:
        """Both models share the top-level unit_system (SI)."""
        models = load_models_from_yaml(multi_yaml_path)
        assert all(m.unit_system == UnitSystem.SI for m in models)

    def test_description_field_tolerated(self, tmp_path: Path) -> None:
        """Extra description field in solution entry does not raise."""
        p = tmp_path / "desc.yaml"
        p.write_text(_MULTI_SOL_DESCRIPTION_YAML)
        models = load_models_from_yaml(p)
        assert len(models) == 1

    def test_no_mesh_no_solutions_raises_value_error(self, tmp_path: Path) -> None:
        """YAML without mesh or solutions key raises ValueError."""
        p = tmp_path / "bad.yaml"
        p.write_text(_NO_MESH_NO_SOLUTIONS_YAML)
        with pytest.raises(ValueError):
            load_models_from_yaml(p)

    def test_empty_solutions_list_raises(self, tmp_path: Path) -> None:
        """solutions: [] raises pydantic.ValidationError (min_length=1)."""
        p = tmp_path / "empty.yaml"
        p.write_text(_EMPTY_SOLUTIONS_YAML)
        with pytest.raises(pydantic.ValidationError):
            load_models_from_yaml(p)

    def test_file_not_found_raises(self) -> None:
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_models_from_yaml(Path("/nonexistent/multi.yaml"))

    def test_empirical_unit_system_propagates(self, tmp_path: Path) -> None:
        """Empirical unit_system propagates to all solutions."""
        p = tmp_path / "emp.yaml"
        p.write_text(_MULTI_SOL_EMPIRICAL_YAML)
        models = load_models_from_yaml(p)
        assert len(models) == 2
        assert all(m.unit_system == UnitSystem.EMPIRICAL for m in models)


# ---------------------------------------------------------------------------
# Distributed load function tests
# ---------------------------------------------------------------------------

_DIST_FUNC_ALL_YAML = """\
label: "test_dist_all"
mesh:
  nodes:
    - {id: 1, x: 0.0, y: 0.0}
    - {id: 2, x: 1.0, y: 0.0}
    - {id: 3, x: 2.0, y: 0.0}
  elements:
    - {id: 1, node_i: 1, node_j: 2, type: beam, material: steel}
    - {id: 2, node_i: 2, node_j: 3, type: beam, material: steel}
materials:
  steel: {E: 200.0e9, A: 0.01, I: 1.0e-4}
boundary_conditions:
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 3, coefficients: [0.0, 1.0, 0.0]}
loads:
  distributed:
    - element_ids: all
      expression: "w0"
      parameters:
        w0: -500.0
"""

_DIST_FUNC_EXPLICIT_YAML = """\
label: "test_dist_explicit"
mesh:
  nodes:
    - {id: 1, x: 0.0, y: 0.0}
    - {id: 2, x: 1.0, y: 0.0}
    - {id: 3, x: 2.0, y: 0.0}
  elements:
    - {id: 1, node_i: 1, node_j: 2, type: beam, material: steel}
    - {id: 2, node_i: 2, node_j: 3, type: beam, material: steel}
materials:
  steel: {E: 200.0e9, A: 0.01, I: 1.0e-4}
boundary_conditions:
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 3, coefficients: [0.0, 1.0, 0.0]}
loads:
  distributed:
    - element_ids: [1]
      expression: "w0"
      parameters:
        w0: -500.0
"""

_DIST_FUNC_LINEAR_YAML = """\
label: "test_dist_linear"
mesh:
  nodes:
    - {id: 1, x: 0.0, y: 0.0}
    - {id: 2, x: 10.0, y: 0.0}
  elements:
    - {id: 1, node_i: 1, node_j: 2, type: beam, material: steel}
materials:
  steel: {E: 200.0e9, A: 0.01, I: 1.0e-4}
boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
loads:
  distributed:
    - element_ids: all
      expression: "a * x + b"
      parameters:
        a: 2.0
        b: 5.0
"""

_DIST_FUNC_BAD_ELEMENT_YAML = """\
label: "test_dist_bad"
mesh:
  nodes:
    - {id: 1, x: 0.0, y: 0.0}
    - {id: 2, x: 1.0, y: 0.0}
  elements:
    - {id: 1, node_i: 1, node_j: 2, type: beam, material: steel}
materials:
  steel: {E: 200.0e9, A: 0.01, I: 1.0e-4}
boundary_conditions:
  - {node_id: 1, coefficients: [1.0, 0.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
  - {node_id: 1, coefficients: [0.0, 0.0, 1.0]}
loads:
  distributed:
    - element_ids: [99]
      expression: "w0"
      parameters:
        w0: -1.0
"""


class TestDistributedLoadFunction:
    """Tests for function-based distributed load parsing."""

    def test_element_ids_all_applies_to_every_element(self, tmp_path: Path) -> None:
        """element_ids: all creates one DistributedLoad per element in the mesh."""
        p = tmp_path / "dist_all.yaml"
        p.write_text(_DIST_FUNC_ALL_YAML)
        model = load_model_from_yaml(p)
        assert len(model.distributed_loads) == 2
        assert {dl.element_id for dl in model.distributed_loads} == {1, 2}

    def test_element_ids_explicit_list_restricts_targets(self, tmp_path: Path) -> None:
        """Explicit element_ids list applies load only to listed elements."""
        p = tmp_path / "dist_explicit.yaml"
        p.write_text(_DIST_FUNC_EXPLICIT_YAML)
        model = load_model_from_yaml(p)
        assert len(model.distributed_loads) == 1
        assert model.distributed_loads[0].element_id == 1

    def test_constant_expression_correct_w_values(self, tmp_path: Path) -> None:
        """Constant expression produces equal w_i and w_j equal to the constant."""
        p = tmp_path / "dist_all.yaml"
        p.write_text(_DIST_FUNC_ALL_YAML)
        model = load_model_from_yaml(p)
        for dl in model.distributed_loads:
            assert dl.w_i == pytest.approx(-500.0)
            assert dl.w_j == pytest.approx(-500.0)

    def test_linear_expression_evaluated_at_node_x(self, tmp_path: Path) -> None:
        """Linear expression a*x+b is evaluated at node x positions (w_i at x=0, w_j at x=10)."""
        p = tmp_path / "dist_linear.yaml"
        p.write_text(_DIST_FUNC_LINEAR_YAML)
        model = load_model_from_yaml(p)
        dl = model.distributed_loads[0]
        # x_i=0: 2*0 + 5 = 5.0; x_j=10: 2*10 + 5 = 25.0
        assert dl.w_i == pytest.approx(5.0)
        assert dl.w_j == pytest.approx(25.0)

    def test_result_is_distributed_linear_type(self, tmp_path: Path) -> None:
        """Function-based loads always resolve to DISTRIBUTED_LINEAR internally."""
        p = tmp_path / "dist_all.yaml"
        p.write_text(_DIST_FUNC_ALL_YAML)
        model = load_model_from_yaml(p)
        for dl in model.distributed_loads:
            assert dl.load_type == LoadType.DISTRIBUTED_LINEAR

    def test_invalid_element_id_raises_value_error(self, tmp_path: Path) -> None:
        """Explicit element_ids list referencing a nonexistent ID raises ValueError."""
        p = tmp_path / "dist_bad.yaml"
        p.write_text(_DIST_FUNC_BAD_ELEMENT_YAML)
        with pytest.raises(ValueError, match="unknown element_id"):
            load_model_from_yaml(p)


_MINIMAL_TRUSS_YAML = """\
label: "test_truss"
mesh:
  nodes:
    - {id: 1, x: 0.0, y: 0.0}
    - {id: 2, x: 3.0, y: 4.0}
  elements:
    - {id: 1, node_i: 1, node_j: 2, type: truss, material: steel}
materials:
  steel:
    E: 200.0e9
    A: 0.01
    I: 0.0
boundary_conditions:
  - {node_id: 1, coefficients: [0.0, 1.0, 0.0]}
loads:
  nodal:
    - {node_id: 2, type: point_force_x, magnitude: 1000.0}
"""


class TestTrussYamlParsing:
    """Tests for TRUSS element type in YAML parsing."""

    def test_truss_element_type_parsed(self, tmp_path: Path) -> None:
        """YAML with type: truss produces ElementType.TRUSS."""
        p = tmp_path / "truss.yaml"
        p.write_text(_MINIMAL_TRUSS_YAML)
        model = load_model_from_yaml(p)
        assert model.mesh.elements[0].element_type == ElementType.TRUSS

    def test_truss_node_pos_parsed(self, tmp_path: Path) -> None:
        """Node x and y coordinates are parsed into pos tuple correctly."""
        p = tmp_path / "truss.yaml"
        p.write_text(_MINIMAL_TRUSS_YAML)
        model = load_model_from_yaml(p)
        n2 = next(n for n in model.mesh.nodes if n.id == 2)
        assert n2.x == pytest.approx(3.0)
        assert n2.y == pytest.approx(4.0)

    def test_truss_element_length_3_4_5(self, tmp_path: Path) -> None:
        """Element length for a 3-4-5 truss is 5.0."""
        p = tmp_path / "truss.yaml"
        p.write_text(_MINIMAL_TRUSS_YAML)
        model = load_model_from_yaml(p)
        import math
        assert math.isclose(model.mesh.elements[0].length, 5.0, rel_tol=1e-12)
