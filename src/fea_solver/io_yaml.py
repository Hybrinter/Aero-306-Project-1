"""YAML input file parser -- converts config files into FEAModel instances.

Uses Pydantic schema models for structured parsing and field-level validation,
then converts validated schemas into frozen FEA dataclasses.

Expected YAML schema:
  label: str
  mesh:
    nodes: [{id: int, x: float}, ...]
    elements: [{id: int, node_i: int, node_j: int, type: str, material: str}, ...]
  materials:
    <name>: {E: float, A: float, I: float}
  boundary_conditions: [{node_id: int, type: str}, ...]
  loads:
    nodal: [{node_id: int, type: str, magnitude: float}, ...]
    distributed: [{element_id: int, type: str, w_i: float, w_j: float}, ...]

_NodeSchema:           Pydantic schema for a single node entry.
_MaterialSchema:       Pydantic schema for material properties; enforces E > 0, A > 0, I >= 0.
_ElementSchema:        Pydantic schema for a single element entry (refs resolved later).
_BCSchema:             Pydantic schema for a single boundary condition entry.
_NodalLoadSchema:      Pydantic schema for a single nodal load entry.
_DistributedLoadSchema: Pydantic schema for a single distributed load entry.
_LoadsSchema:          Pydantic schema for the loads block; defaults both sublists to empty.
_MeshSchema:           Pydantic schema for the mesh block.
_FEAModelSchema:       Top-level Pydantic schema for the entire YAML file.
_schema_to_model:      Converts a validated _FEAModelSchema into a FEAModel dataclass,
                       resolving referential integrity (node refs, duplicate IDs, enum lookups).
load_model_from_yaml:  Public API -- reads YAML, validates via Pydantic, returns FEAModel.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
# Pydantic schema models  (private -- YAML structure only, no FEA logic)
# ---------------------------------------------------------------------------


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


class _DistributedLoadSchema(BaseModel):
    """Pydantic schema for a single distributed load entry.

    Args:
        element_id (int): ID of the loaded element.
        type (str): Load type string -- "distributed_y" or "distributed_linear".
        w_i (float): Load intensity at node i in N/m.
        w_j (float): Load intensity at node j in N/m.

    Notes:
        String-to-enum conversion and element_id validation happen in _schema_to_model.
    """

    element_id: int
    type: str
    w_i: float
    w_j: float


class _LoadsSchema(BaseModel):
    """Pydantic schema for the YAML loads block.

    Args:
        nodal (list[_NodalLoadSchema]): List of nodal load entries. Defaults to [].
        distributed (list[_DistributedLoadSchema]): List of distributed load entries.
            Defaults to [].

    Notes:
        Both sublists default to empty so the 'loads' key may be omitted entirely
        from the YAML file without error.
    """

    nodal: list[_NodalLoadSchema] = Field(default_factory=list)
    distributed: list[_DistributedLoadSchema] = Field(default_factory=list)


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
        mesh (_MeshSchema): Mesh block containing nodes and elements.
        materials (dict[str, _MaterialSchema]): Named material property map.
        boundary_conditions (list[_BCSchema]): List of boundary condition entries.
            Defaults to [].
        loads (_LoadsSchema): Loads block. Defaults to empty _LoadsSchema.

    Notes:
        model_validate() is called once in load_model_from_yaml. Field-level errors
        (wrong types, E <= 0) raise pydantic.ValidationError here. Referential
        integrity errors are deferred to _schema_to_model.
    """

    label: str = ""
    mesh: _MeshSchema
    materials: dict[str, _MaterialSchema]
    boundary_conditions: list[_BCSchema] = Field(default_factory=list)
    loads: _LoadsSchema = Field(default_factory=_LoadsSchema)


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

    Raises:
        ValueError: If node IDs are not unique, element IDs are not unique, an element
            references an undefined node or material, a BC or load references an unknown
            node, a distributed load references an unknown element, any enum string is
            unrecognised, or any element has non-positive length.

    Notes:
        Field-level validation (E > 0, A > 0) was already enforced by Pydantic.
        This function handles cross-entity referential integrity only.
        Enum string lookup uses _ELEMENT_TYPE_MAP, _BC_TYPE_MAP, _LOAD_TYPE_MAP.
    """
    # --- Nodes ---
    nodes_list: list[Node] = [Node(id=n.id, x=n.x) for n in schema.mesh.nodes]
    node_ids_list = [n.id for n in nodes_list]
    if len(node_ids_list) != len(set(node_ids_list)):
        raise ValueError(f"Duplicate node IDs detected: {node_ids_list}")
    nodes_by_id: dict[int, Node] = {n.id: n for n in nodes_list}
    node_ids: set[int] = set(node_ids_list)

    # --- Materials ---
    materials: dict[str, MaterialProperties] = {
        name: MaterialProperties(E=m.E, A=m.A, I=m.I, label=name)
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
    nodal_loads: list[NodalLoad] = []
    for ld in schema.loads.nodal:
        if ld.node_id not in node_ids:
            raise ValueError(f"Nodal load references unknown node_id={ld.node_id}")
        lt_str = ld.type.lower()
        if lt_str not in _LOAD_TYPE_MAP:
            raise ValueError(f"Unknown load type: '{lt_str}'")
        nodal_loads.append(NodalLoad(
            node_id=ld.node_id,
            load_type=_LOAD_TYPE_MAP[lt_str],
            magnitude=ld.magnitude,
        ))

    # --- Distributed Loads ---
    dist_loads: list[DistributedLoad] = []
    for ld in schema.loads.distributed:
        if ld.element_id not in element_ids:
            raise ValueError(f"Distributed load references unknown element_id={ld.element_id}")
        lt_str = ld.type.lower()
        if lt_str not in _LOAD_TYPE_MAP:
            raise ValueError(f"Unknown distributed load type: '{lt_str}'")
        dist_loads.append(DistributedLoad(
            element_id=ld.element_id,
            load_type=_LOAD_TYPE_MAP[lt_str],
            w_i=ld.w_i,
            w_j=ld.w_j,
        ))

    return FEAModel(
        mesh=mesh,
        boundary_conditions=tuple(bcs),
        nodal_loads=tuple(nodal_loads),
        distributed_loads=tuple(dist_loads),
        label=label,
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
