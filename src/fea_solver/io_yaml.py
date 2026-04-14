"""YAML input file parser -- converts config files into FEAModel instances.

Uses Pydantic schema models for structured parsing and field-level validation,
then converts validated schemas into frozen FEA dataclasses.

Expected YAML schema:
  label: str
  unit_system: "SI" | "empirical"   (optional, default "SI")
  units:                             (optional; overrides per-field input units)
    length: str        e.g. "ft", "m", "in"
    force: str         e.g. "lb", "N", "kN"
    modulus: str       e.g. "psi", "Pa", "MPa"
    area: str          e.g. "in^2", "m^2"
    second_moment: str e.g. "in^4", "m^4"
    distributed: str   e.g. "lb/ft", "N/m", "lb/in"
    moment: str        e.g. "in-lb", "N-m", "ft-lb"
  mesh:
    nodes: [{id: int, x: float}, ...]
    elements: [{id: int, node_i: int, node_j: int, type: str, material: str}, ...]
  materials:
    <name>: {E: float, A: float, I: float}
  boundary_conditions: [{node_id: int, type: str}, ...]
  loads:
    nodal: [{node_id: int, type: str, magnitude: float}, ...]
    distributed:
      - element_ids: all | [int, ...]   # "all" targets every element; or explicit list
        expression: str                 # Python expression; x = node position (input units)
        parameters: {name: float, ...}  # named constants injected into expression scope

_UnitsSchema:          Pydantic schema for the optional units block.
_NodeSchema:           Pydantic schema for a single node entry.
_MaterialSchema:       Pydantic schema for material properties; enforces E > 0, A > 0, I >= 0.
_ElementSchema:        Pydantic schema for a single element entry (refs resolved later).
_BCSchema:             Pydantic schema for a single boundary condition entry.
_NodalLoadSchema:      Pydantic schema for a single nodal load entry.
_DistributedLoadFunctionSchema: Pydantic schema for a function-based distributed load entry.
_LoadsSchema:          Pydantic schema for the loads block; defaults both sublists to empty.
_MeshSchema:           Pydantic schema for the mesh block.
_FEAModelSchema:       Top-level Pydantic schema for the entire YAML file.
_SolutionEntrySchema:  Pydantic schema for one entry in the 'solutions' list (multi-solution).
_MultiSolutionFileSchema: Top-level schema for multi-solution YAML files.
_schema_to_model:      Converts a validated _FEAModelSchema into a FEAModel dataclass,
                       resolving referential integrity (node refs, duplicate IDs, enum lookups)
                       and applying unit conversions via UnitConverter.
_parse_multi_solution: Converts a _MultiSolutionFileSchema into a list of FEAModels by
                       merging the shared unit_system/units with each solution's mesh/loads.
load_model_from_yaml:  Public API -- reads single-solution YAML, returns one FEAModel.
load_models_from_yaml: Public API -- detects single vs multi-solution YAML, returns list
                       of FEAModels. Use this for all new callers.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from fea_solver.models import (
    BoundaryCondition,
    BoundaryConditionType,
    DistributedLoad,
    Element,
    ElementType,
    FEAModel,
    LoadType,
    MaterialProperties,
    Mesh,
    Node,
    NodalLoad,
)
from fea_solver.units import (
    CANONICAL_UNITS,
    UnitConverter,
    UnitSystem,
    validate_unit,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# String -> Enum mappings  (used during schema -> dataclass conversion)
# ---------------------------------------------------------------------------

_ELEMENT_TYPE_MAP: dict[str, ElementType] = {
    "bar":   ElementType.BAR,
    "beam":  ElementType.BEAM,
    "frame": ElementType.FRAME,
}

_BC_TYPE_MAP: dict[str, BoundaryConditionType] = {
    "fixed_u":     BoundaryConditionType.FIXED_U,
    "fixed_v":     BoundaryConditionType.FIXED_V,
    "fixed_theta": BoundaryConditionType.FIXED_THETA,
    "fixed_all":   BoundaryConditionType.FIXED_ALL,
    "pin":         BoundaryConditionType.PIN,
    "roller":      BoundaryConditionType.ROLLER,
}

_LOAD_TYPE_MAP: dict[str, LoadType] = {
    "point_force_x":      LoadType.POINT_FORCE_X,
    "point_force_y":      LoadType.POINT_FORCE_Y,
    "point_moment":       LoadType.POINT_MOMENT,
    "distributed_y":      LoadType.DISTRIBUTED_Y,
    "distributed_linear": LoadType.DISTRIBUTED_LINEAR,
}


# ---------------------------------------------------------------------------
# Expression evaluator for function-based distributed loads
# ---------------------------------------------------------------------------


def _evaluate_expression(expression: str, x: float, parameters: dict[str, float]) -> float:
    """Evaluate a load distribution expression at a single x coordinate.

    Args:
        expression (str): Python expression string. May reference 'x', math
            builtins (sqrt, pi, sin, cos, tan, exp, log, abs), and any key
            present in 'parameters'.
        x (float): Position along the beam in input (YAML) units.
        parameters (dict[str, float]): Named constants injected into the
            expression namespace. Values must be in the same unit system as x.

    Returns:
        float: Load intensity in input distributed-load units (before unit
            conversion). Sign convention follows the YAML coordinate system.

    Notes:
        Builtins are restricted to the math namespace to prevent arbitrary code
        execution. The expression is evaluated with __builtins__ set to an empty
        dict so that global Python builtins (open, exec, import, etc.) are not
        accessible. Only the functions listed below are in scope:
        sqrt, pi, sin, cos, tan, exp, log, abs, x, and all parameter keys.
    """
    ns: dict[str, object] = {
        "x":    x,
        "pi":   math.pi,
        "sqrt": math.sqrt,
        "sin":  math.sin,
        "cos":  math.cos,
        "tan":  math.tan,
        "exp":  math.exp,
        "log":  math.log,
        "abs":  abs,
        **parameters,
    }
    return float(eval(expression, {"__builtins__": {}}, ns))  # noqa: S307


# ---------------------------------------------------------------------------
# Pydantic schema models  (private -- YAML structure only, no FEA logic)
# ---------------------------------------------------------------------------


class _UnitsSchema(BaseModel):
    """Pydantic schema for the optional YAML units block.

    Each field specifies the input unit for the corresponding quantity type.
    Omitted fields default to the canonical unit for the chosen unit_system.

    Args:
        length (str | None): Unit for node x positions and element lengths.
            E.g. "m", "ft", "in". Default None (use system canonical).
        force (str | None): Unit for point force load magnitudes.
            E.g. "N", "lb", "kN". Default None.
        modulus (str | None): Unit for Young's modulus E.
            E.g. "Pa", "psi", "MPa". Default None.
        area (str | None): Unit for cross-sectional area A.
            E.g. "m^2", "in^2". Default None.
        second_moment (str | None): Unit for second moment of area I.
            E.g. "m^4", "in^4". Default None.
        distributed (str | None): Unit for distributed load intensities w_i, w_j.
            E.g. "N/m", "lb/ft", "lb/in". Default None.
        moment (str | None): Unit for point moment load magnitudes.
            E.g. "N-m", "in-lb", "ft-lb". Default None.

    Notes:
        All fields are optional. Non-None values override the system canonical default.
        Unknown unit strings are validated in _schema_to_model via validate_unit.
    """

    length:        str | None = None
    force:         str | None = None
    modulus:       str | None = None
    area:          str | None = None
    second_moment: str | None = None
    distributed:   str | None = None
    moment:        str | None = None


class _NodeSchema(BaseModel):
    """Pydantic schema for a single node entry in the YAML mesh block.

    Args:
        id (int): Unique node identifier. Must be a positive integer.
        x (float): Position along the beam axis in metres.

    Notes:
        Pydantic coerces numeric strings to int/float automatically.
        Uniqueness is enforced later in _schema_to_model.
    """

    id: int
    x: float


class _MaterialSchema(BaseModel):
    """Pydantic schema for a named material entry in the YAML materials block.

    Args:
        E (float): Young's modulus in Pascals. Must be > 0.
        A (float): Cross-sectional area in m^2. Must be > 0.
        I (float): Second moment of area in m^4. Defaults to 0.0; must be >= 0.

    Notes:
        Field constraints (gt/ge) raise pydantic.ValidationError on bad input,
        eliminating the need for manual E/A checks in conversion code.
    """

    E: float = Field(gt=0, description="Young's modulus [Pa]")
    A: float = Field(gt=0, description="Cross-sectional area [m^2]")
    I: float = Field(default=0.0, ge=0.0, description="Second moment of area [m^4]")


class _ElementSchema(BaseModel):
    """Pydantic schema for a single element entry in the YAML mesh block.

    Args:
        id (int): Unique element identifier.
        node_i (int): ID of the start node (resolved to Node in _schema_to_model).
        node_j (int): ID of the end node (resolved to Node in _schema_to_model).
        type (str): Element type string -- one of "bar", "beam", "frame".
        material (str): Name of the material defined in the materials block.

    Notes:
        String-to-enum conversion and referential integrity are handled in
        _schema_to_model, not here, so this schema accepts any non-empty string.
    """

    id: int
    node_i: int
    node_j: int
    type: str
    material: str


class _BCSchema(BaseModel):
    """Pydantic schema for a single boundary condition entry.

    Args:
        node_id (int): ID of the constrained node.
        type (str): BC type string -- one of "fixed_u", "fixed_v", "fixed_theta",
            "fixed_all", "pin", "roller".

    Notes:
        String-to-enum conversion and node_id validation happen in _schema_to_model.
    """

    node_id: int
    type: str


class _NodalLoadSchema(BaseModel):
    """Pydantic schema for a single nodal load entry.

    Args:
        node_id (int): ID of the loaded node.
        type (str): Load type string -- one of "point_force_x", "point_force_y",
            "point_moment".
        magnitude (float): Load magnitude in N for forces, N*m for moments.

    Notes:
        String-to-enum conversion and node_id validation happen in _schema_to_model.
    """

    node_id: int
    type: str
    magnitude: float


class _DistributedLoadFunctionSchema(BaseModel):
    """Pydantic schema for a function-based distributed load entry.

    Args:
        element_ids (list[int] | Literal["all"]): Target element IDs. "all"
            applies the load to every element in the mesh; an explicit list
            restricts application to only those element IDs.
        expression (str): Python expression for load intensity as a function of
            position. 'x' is the node's raw (input-unit) x coordinate. Math
            builtins and all 'parameters' keys are in scope.
        parameters (dict[str, float]): Named float constants injected into the
            expression namespace. Must be in the same units as the node
            coordinates (i.e., the YAML input length unit). Defaults to {}.

    Notes:
        Referential integrity (element ID existence) and expression evaluation
        happen in _schema_to_model. Always resolves to LoadType.DISTRIBUTED_LINEAR
        internally, producing one DistributedLoad per targeted element.
    """

    element_ids: list[int] | Literal["all"]
    expression: str
    parameters: dict[str, float] = Field(default_factory=dict)


class _LoadsSchema(BaseModel):
    """Pydantic schema for the YAML loads block.

    Args:
        nodal (list[_NodalLoadSchema]): List of nodal load entries. Defaults to [].
        distributed (list[_DistributedLoadFunctionSchema]): List of function-based
            distributed load entries. Defaults to [].

    Notes:
        Both sublists default to empty so the 'loads' key may be omitted entirely
        from the YAML file without error.
    """

    nodal: list[_NodalLoadSchema] = Field(default_factory=list)
    distributed: list[_DistributedLoadFunctionSchema] = Field(default_factory=list)


class _MeshSchema(BaseModel):
    """Pydantic schema for the YAML mesh block.

    Args:
        nodes (list[_NodeSchema]): List of node entries.
        elements (list[_ElementSchema]): List of element entries.

    Notes:
        Both lists are required (no default). An empty mesh is technically valid
        at schema level but will fail later during DOF map construction.
    """

    nodes: list[_NodeSchema]
    elements: list[_ElementSchema]


class _FEAModelSchema(BaseModel):
    """Top-level Pydantic schema for the entire YAML case file.

    Args:
        label (str): Human-readable model label. Defaults to "" (overridden from
            filename in load_model_from_yaml if empty).
        unit_system (str): Canonical unit system -- "SI" or "empirical". Default "SI".
            Determines the target canonical system for all converted values.
        units (_UnitsSchema): Optional per-quantity-type input unit overrides.
            Defaults to an empty schema (all quantities use canonical system defaults).
        mesh (_MeshSchema): Mesh block containing nodes and elements.
        materials (dict[str, _MaterialSchema]): Named material property map.
        boundary_conditions (list[_BCSchema]): List of boundary condition entries.
            Defaults to [].
        loads (_LoadsSchema): Loads block. Defaults to empty _LoadsSchema.

    Notes:
        model_validate() is called once in load_model_from_yaml. Field-level errors
        (wrong types, E <= 0) raise pydantic.ValidationError here. Referential
        integrity errors and unknown unit strings are deferred to _schema_to_model.
    """

    label: str = ""
    unit_system: str = "SI"
    units: _UnitsSchema = Field(default_factory=_UnitsSchema)
    mesh: _MeshSchema
    materials: dict[str, _MaterialSchema]
    boundary_conditions: list[_BCSchema] = Field(default_factory=list)
    loads: _LoadsSchema = Field(default_factory=_LoadsSchema)


class _SolutionEntrySchema(BaseModel):
    """Pydantic schema for one entry in the top-level 'solutions' list.

    Each entry is a self-contained mesh definition for one refinement level.
    The top-level unit_system and units block are shared across all entries
    and are injected during conversion in _parse_multi_solution.

    Args:
        label (str): Short solution label (e.g. "coarse", "fine"). Default "".
        mesh (_MeshSchema): Mesh block for this solution.
        materials (dict[str, _MaterialSchema]): Named material properties.
        boundary_conditions (list[_BCSchema]): Kinematic constraints. Default [].
        loads (_LoadsSchema): Applied loads block. Defaults to empty.

    Notes:
        Extra keys (e.g. 'description') are silently ignored by Pydantic v2
        because model_config is not set to 'forbid' -- this allows YAML authors
        to include documentation fields without parse errors.
        unit_system and units are intentionally absent; they are inherited from
        the parent _MultiSolutionFileSchema and merged in _parse_multi_solution.
    """

    label: str = ""
    mesh: _MeshSchema
    materials: dict[str, _MaterialSchema]
    boundary_conditions: list[_BCSchema] = Field(default_factory=list)
    loads: _LoadsSchema = Field(default_factory=_LoadsSchema)

    model_config = {"extra": "ignore"}


class _MultiSolutionFileSchema(BaseModel):
    """Top-level Pydantic schema for multi-solution YAML files.

    Selected when the raw YAML dict contains a 'solutions' key rather than a
    top-level 'mesh' key. The unit_system and units fields are shared across
    all solution entries.

    Args:
        label (str): Top-level problem label. Defaults to "".
        unit_system (str): Shared canonical unit system ("SI" or "empirical"). Default "SI".
        units (_UnitsSchema): Shared input unit overrides. Defaults to empty.
        solutions (list[_SolutionEntrySchema]): One or more solution entries.
            Must be non-empty (min_length=1 enforced by Pydantic).

    Notes:
        Extra keys (e.g. 'description') are silently ignored.
        Validated by _parse_multi_solution via model_validate().
    """

    label: str = ""
    unit_system: str = "SI"
    units: _UnitsSchema = Field(default_factory=_UnitsSchema)
    solutions: list[_SolutionEntrySchema] = Field(min_length=1)

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Schema -> FEA dataclass conversion
# ---------------------------------------------------------------------------


def _schema_to_model(schema: _FEAModelSchema, label: str) -> FEAModel:
    """Convert a validated _FEAModelSchema into a fully checked FEAModel.

    Args:
        schema (_FEAModelSchema): Pydantic-validated schema object from model_validate().
        label (str): Model label (may come from YAML or filename stem).

    Returns:
        FEAModel: Fully constructed and referentially validated finite element model.
            All numeric field values are stored in the canonical unit system.

    Raises:
        ValueError: If unit_system string is not "SI" or "empirical", if any specified
            unit string is not recognised for its quantity type, if node IDs are not
            unique, element IDs are not unique, an element references an undefined node
            or material, a BC or load references an unknown node, a distributed load
            references an unknown element, any enum string is unrecognised, or any
            element has non-positive length.

    Notes:
        Field-level validation (E > 0, A > 0) was already enforced by Pydantic.
        This function handles unit conversion, cross-entity referential integrity,
        and enum resolution.
        Enum string lookup uses _ELEMENT_TYPE_MAP, _BC_TYPE_MAP, _LOAD_TYPE_MAP.
        Unit conversion pipeline: input unit -> canonical SI -> canonical Empirical (if needed).
    """
    # --- Unit system and converter ---
    unit_system_str = schema.unit_system.strip()
    try:
        unit_system = UnitSystem(unit_system_str)
    except ValueError:
        raise ValueError(
            f"Unknown unit_system '{unit_system_str}'. Expected 'SI' or 'empirical'."
        )

    # Build units_spec: start from canonical defaults, override with YAML units block.
    units_spec: dict[str, str] = dict(CANONICAL_UNITS[unit_system])
    units_overrides: dict[str, str | None] = {
        "length":        schema.units.length,
        "force":         schema.units.force,
        "modulus":       schema.units.modulus,
        "area":          schema.units.area,
        "second_moment": schema.units.second_moment,
        "distributed":   schema.units.distributed,
        "moment":        schema.units.moment,
    }
    for qty, unit_str in units_overrides.items():
        if unit_str is not None:
            validate_unit(qty, unit_str)
            units_spec[qty] = unit_str

    conv = UnitConverter(unit_system=unit_system, units=units_spec)

    # --- Nodes ---
    nodes_list: list[Node] = [Node(id=n.id, x=conv.convert(n.x, "length")) for n in schema.mesh.nodes]
    node_ids_list = [n.id for n in nodes_list]
    if len(node_ids_list) != len(set(node_ids_list)):
        raise ValueError(f"Duplicate node IDs detected: {node_ids_list}")
    nodes_by_id: dict[int, Node] = {n.id: n for n in nodes_list}
    node_ids: set[int] = set(node_ids_list)

    # --- Materials ---
    materials: dict[str, MaterialProperties] = {
        name: MaterialProperties(
            E=conv.convert(m.E, "modulus"),
            A=conv.convert(m.A, "area"),
            I=conv.convert(m.I, "second_moment"),
            label=name,
        )
        for name, m in schema.materials.items()
    }

    # --- Elements ---
    elements_list: list[Element] = []
    for e in schema.mesh.elements:
        if e.node_i not in nodes_by_id:
            raise ValueError(f"Element {e.id}: node_i={e.node_i} not found")
        if e.node_j not in nodes_by_id:
            raise ValueError(f"Element {e.id}: node_j={e.node_j} not found")
        if e.material not in materials:
            raise ValueError(f"Element {e.id}: material '{e.material}' not defined")
        etype_str = e.type.lower()
        if etype_str not in _ELEMENT_TYPE_MAP:
            raise ValueError(f"Element {e.id}: unknown type '{etype_str}'")
        elem = Element(
            id=e.id,
            node_i=nodes_by_id[e.node_i],
            node_j=nodes_by_id[e.node_j],
            element_type=_ELEMENT_TYPE_MAP[etype_str],
            material=materials[e.material],
        )
        if elem.length <= 0.0:
            raise ValueError(f"Element {e.id}: length must be > 0, got {elem.length}")
        elements_list.append(elem)
    elem_ids_list = [e.id for e in elements_list]
    if len(elem_ids_list) != len(set(elem_ids_list)):
        raise ValueError(f"Duplicate element IDs: {elem_ids_list}")
    element_ids: set[int] = set(elem_ids_list)

    # --- Mesh ---
    mesh = Mesh(nodes=tuple(nodes_list), elements=tuple(elements_list))

    # --- Boundary Conditions ---
    bcs: list[BoundaryCondition] = []
    for bc in schema.boundary_conditions:
        if bc.node_id not in node_ids:
            raise ValueError(f"BC references unknown node_id={bc.node_id}")
        bc_type_str = bc.type.lower()
        if bc_type_str not in _BC_TYPE_MAP:
            raise ValueError(f"Unknown BC type: '{bc_type_str}'")
        bcs.append(BoundaryCondition(node_id=bc.node_id, bc_type=_BC_TYPE_MAP[bc_type_str]))

    # --- Nodal Loads ---
    _MOMENT_LOAD_TYPES = {"point_moment"}
    nodal_loads: list[NodalLoad] = []
    for ld in schema.loads.nodal:
        if ld.node_id not in node_ids:
            raise ValueError(f"Nodal load references unknown node_id={ld.node_id}")
        lt_str = ld.type.lower()
        if lt_str not in _LOAD_TYPE_MAP:
            raise ValueError(f"Unknown load type: '{lt_str}'")
        qty = "moment" if lt_str in _MOMENT_LOAD_TYPES else "force"
        nodal_loads.append(NodalLoad(
            node_id=ld.node_id,
            load_type=_LOAD_TYPE_MAP[lt_str],
            magnitude=conv.convert(ld.magnitude, qty),
        ))

    # --- Distributed Loads ---
    # Build raw (pre-conversion) node x lookup and raw element schema lookup for
    # expression evaluation. Expression uses input-unit x so parameter values in
    # the YAML match the mesh coordinates as written by the user.
    raw_node_x_by_id: dict[int, float] = {n.id: n.x for n in schema.mesh.nodes}
    elem_schema_by_id: dict[int, _ElementSchema] = {e.id: e for e in schema.mesh.elements}

    dist_loads: list[DistributedLoad] = []
    for ld in schema.loads.distributed:
        # Resolve element_ids: "all" -> every element; explicit list -> validate each
        if ld.element_ids == "all":
            target_ids: list[int] = sorted(element_ids)
        else:
            for eid in ld.element_ids:
                if eid not in element_ids:
                    raise ValueError(f"Distributed load references unknown element_id={eid}")
            target_ids = list(ld.element_ids)

        for eid in target_ids:
            es = elem_schema_by_id[eid]
            x_i = raw_node_x_by_id[es.node_i]
            x_j = raw_node_x_by_id[es.node_j]
            w_i = conv.convert(
                _evaluate_expression(ld.expression, x_i, ld.parameters), "distributed"
            )
            w_j = conv.convert(
                _evaluate_expression(ld.expression, x_j, ld.parameters), "distributed"
            )
            dist_loads.append(DistributedLoad(
                element_id=eid,
                load_type=LoadType.DISTRIBUTED_LINEAR,
                w_i=w_i,
                w_j=w_j,
            ))

    return FEAModel(
        mesh=mesh,
        boundary_conditions=tuple(bcs),
        nodal_loads=tuple(nodal_loads),
        distributed_loads=tuple(dist_loads),
        label=label,
        unit_system=unit_system,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model_from_yaml(path: Path) -> FEAModel:
    """Parse a YAML config file into a validated FEAModel.

    Args:
        path (Path): Filesystem path to YAML case definition file.

    Returns:
        FEAModel: Fully parsed and validated finite element model.

    Raises:
        FileNotFoundError: If path does not exist.
        pydantic.ValidationError: If the YAML is structurally invalid -- wrong field
            types, missing required keys, or constraint violations (E <= 0, A <= 0).
        ValueError: If referential integrity fails -- duplicate IDs, undefined node/
            material/element references, unknown enum strings, or zero-length elements.

    Notes:
        Parsing is a two-stage pipeline:
          1. _FEAModelSchema.model_validate(raw) -- Pydantic handles field types and
             numeric constraints.
          2. _schema_to_model(schema, label) -- resolves cross-entity references and
             builds frozen FEA dataclasses.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    label = str(raw.get("label", path.stem))
    logger.info("Parsing model from: %s", path)

    schema = _FEAModelSchema.model_validate(raw)
    model = _schema_to_model(schema, label)

    logger.info("Model '%s' loaded: %d nodes, %d elements", label,
                len(model.mesh.nodes), len(model.mesh.elements))
    return model


def load_models_from_yaml(path: Path) -> list[FEAModel]:
    """Parse a YAML config file into one or more validated FEAModel instances.

    Detects whether the file uses the single-solution format (top-level 'mesh' key)
    or the multi-solution format (top-level 'solutions' key) and dispatches accordingly.
    Both formats return a list, so callers are uniform regardless of input format.

    Args:
        path (Path): Filesystem path to YAML case definition file.

    Returns:
        list[FEAModel]: One FEAModel per solution. For multi-solution files, labels
            follow the pattern "{top_label}/{solution_label}". For single-solution
            files, the label is the value of the top-level 'label' key (or file stem).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the YAML has neither a 'mesh' nor a 'solutions' key at top level,
            or if referential integrity fails within any solution.
        pydantic.ValidationError: If any solution entry is structurally invalid (wrong
            field types, E <= 0, A <= 0, empty solutions list).

    Notes:
        For multi-solution files, unit_system and units are shared across all solutions.
        The existing load_model_from_yaml function is reused for the single-solution path
        to avoid any duplication of parsing and conversion logic.
        Composite labels use '/' as a separator (e.g. "problem_1/coarse"). Callers that
        need filesystem-safe strings should replace '/' with '_'.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    top_label = str(raw.get("label", path.stem))

    if "solutions" in raw:
        return _parse_multi_solution(raw, top_label, path)
    elif "mesh" in raw:
        logger.info("Single-solution YAML detected: %s", path)
        return [load_model_from_yaml(path)]
    else:
        raise ValueError(
            f"YAML file '{path}' must have either a top-level 'mesh' key "
            f"(single-solution) or a top-level 'solutions' key (multi-solution)."
        )


def _parse_multi_solution(
    raw: dict[str, Any],
    top_label: str,
    path: Path,
) -> list[FEAModel]:
    """Parse a multi-solution YAML dict into a list of FEAModel instances.

    Args:
        raw (dict[str, Any]): Full parsed YAML dict loaded from file.
        top_label (str): Top-level label extracted from the YAML or file stem.
        path (Path): Source file path, used only for log messages.

    Returns:
        list[FEAModel]: One FEAModel per solution entry. Labels are composite:
            "{top_label}/{solution_label}" where solution_label defaults to the
            1-based index string if the solutions entry has no label.

    Raises:
        pydantic.ValidationError: If any solution entry fails Pydantic validation
            or the solutions list is empty.
        ValueError: If _schema_to_model raises referential integrity errors for
            any solution entry.

    Notes:
        Constructs a _FEAModelSchema for each solution by merging the shared
        unit_system/units from the top-level schema with the per-solution
        mesh/materials/boundary_conditions/loads.
        Calls _schema_to_model unchanged for each merged schema.
    """
    file_schema = _MultiSolutionFileSchema.model_validate(raw)
    logger.info("Parsing multi-solution file '%s': %d solutions", path, len(file_schema.solutions))

    models: list[FEAModel] = []
    for i, sol in enumerate(file_schema.solutions):
        sol_label_part = sol.label if sol.label else str(i + 1)
        composite_label = f"{top_label}/{sol_label_part}"

        merged_schema = _FEAModelSchema(
            label=composite_label,
            unit_system=file_schema.unit_system,
            units=file_schema.units,
            mesh=sol.mesh,
            materials=sol.materials,
            boundary_conditions=sol.boundary_conditions,
            loads=sol.loads,
        )

        model = _schema_to_model(merged_schema, composite_label)
        logger.info(
            "Solution '%s' loaded: %d nodes, %d elements",
            composite_label,
            len(model.mesh.nodes),
            len(model.mesh.elements),
        )
        models.append(model)

    return models
