"""YAML input file parser — converts config files into FEAModel instances.

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
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

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
# String → Enum mappings
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
# Private parsers
# ---------------------------------------------------------------------------


def _parse_nodes(raw: list[dict[str, Any]]) -> tuple[Node, ...]:
    nodes = tuple(Node(id=int(n["id"]), x=float(n["x"])) for n in raw)
    ids = [n.id for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate node IDs detected: {ids}")
    return nodes


def _parse_elements(
    raw: list[dict[str, Any]],
    nodes_by_id: dict[int, Node],
    materials_by_name: dict[str, MaterialProperties],
) -> tuple[Element, ...]:
    elements: list[Element] = []
    for e in raw:
        eid = int(e["id"])
        ni_id, nj_id = int(e["node_i"]), int(e["node_j"])
        if ni_id not in nodes_by_id:
            raise ValueError(f"Element {eid}: node_i={ni_id} not found")
        if nj_id not in nodes_by_id:
            raise ValueError(f"Element {eid}: node_j={nj_id} not found")
        mat_name = str(e["material"])
        if mat_name not in materials_by_name:
            raise ValueError(f"Element {eid}: material '{mat_name}' not defined")
        etype_str = str(e["type"]).lower()
        if etype_str not in _ELEMENT_TYPE_MAP:
            raise ValueError(f"Element {eid}: unknown type '{etype_str}'")
        elem = Element(
            id=eid,
            node_i=nodes_by_id[ni_id],
            node_j=nodes_by_id[nj_id],
            element_type=_ELEMENT_TYPE_MAP[etype_str],
            material=materials_by_name[mat_name],
        )
        if elem.length <= 0.0:
            raise ValueError(f"Element {eid}: length must be > 0, got {elem.length}")
        elements.append(elem)
    ids = [e.id for e in elements]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate element IDs: {ids}")
    return tuple(elements)


def _parse_materials(raw: dict[str, dict[str, Any]]) -> dict[str, MaterialProperties]:
    result: dict[str, MaterialProperties] = {}
    for name, props in raw.items():
        E = float(props["E"])
        A = float(props["A"])
        I = float(props.get("I", 0.0))
        if E <= 0:
            raise ValueError(f"Material '{name}': E must be > 0")
        if A <= 0:
            raise ValueError(f"Material '{name}': A must be > 0")
        result[name] = MaterialProperties(E=E, A=A, I=I, label=name)
    return result


def _parse_boundary_conditions(
    raw: list[dict[str, Any]],
    node_ids: set[int],
) -> tuple[BoundaryCondition, ...]:
    bcs: list[BoundaryCondition] = []
    for bc in raw:
        nid = int(bc["node_id"])
        if nid not in node_ids:
            raise ValueError(f"BC references unknown node_id={nid}")
        bc_type_str = str(bc["type"]).lower()
        if bc_type_str not in _BC_TYPE_MAP:
            raise ValueError(f"Unknown BC type: '{bc_type_str}'")
        bcs.append(BoundaryCondition(node_id=nid, bc_type=_BC_TYPE_MAP[bc_type_str]))
    return tuple(bcs)


def _parse_nodal_loads(
    raw: list[dict[str, Any]],
    node_ids: set[int],
) -> tuple[NodalLoad, ...]:
    loads: list[NodalLoad] = []
    for ld in raw:
        nid = int(ld["node_id"])
        if nid not in node_ids:
            raise ValueError(f"Nodal load references unknown node_id={nid}")
        lt_str = str(ld["type"]).lower()
        if lt_str not in _LOAD_TYPE_MAP:
            raise ValueError(f"Unknown load type: '{lt_str}'")
        loads.append(NodalLoad(
            node_id=nid,
            load_type=_LOAD_TYPE_MAP[lt_str],
            magnitude=float(ld["magnitude"]),
        ))
    return tuple(loads)


def _parse_distributed_loads(
    raw: list[dict[str, Any]],
    element_ids: set[int],
) -> tuple[DistributedLoad, ...]:
    loads: list[DistributedLoad] = []
    for ld in raw:
        eid = int(ld["element_id"])
        if eid not in element_ids:
            raise ValueError(f"Distributed load references unknown element_id={eid}")
        lt_str = str(ld["type"]).lower()
        if lt_str not in _LOAD_TYPE_MAP:
            raise ValueError(f"Unknown distributed load type: '{lt_str}'")
        loads.append(DistributedLoad(
            element_id=eid,
            load_type=_LOAD_TYPE_MAP[lt_str],
            w_i=float(ld["w_i"]),
            w_j=float(ld["w_j"]),
        ))
    return tuple(loads)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model_from_yaml(path: Path) -> FEAModel:
    """Parse a YAML config file into a validated FEAModel.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the config is structurally invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    label = str(raw.get("label", path.stem))
    logger.info("Parsing model from: %s", path)

    materials = _parse_materials(raw.get("materials", {}))

    mesh_raw = raw["mesh"]
    nodes = _parse_nodes(mesh_raw["nodes"])
    nodes_by_id = {n.id: n for n in nodes}

    elements = _parse_elements(mesh_raw["elements"], nodes_by_id, materials)
    mesh = Mesh(nodes=nodes, elements=elements)

    node_ids = {n.id for n in nodes}
    element_ids = {e.id for e in elements}

    bcs = _parse_boundary_conditions(raw.get("boundary_conditions", []), node_ids)

    loads_raw = raw.get("loads", {})
    nodal_loads = _parse_nodal_loads(loads_raw.get("nodal", []), node_ids)
    dist_loads = _parse_distributed_loads(loads_raw.get("distributed", []), element_ids)

    model = FEAModel(
        mesh=mesh,
        boundary_conditions=bcs,
        nodal_loads=nodal_loads,
        distributed_loads=dist_loads,
        label=label,
    )
    logger.info("Model '%s' loaded: %d nodes, %d elements", label,
                len(nodes), len(elements))
    return model
