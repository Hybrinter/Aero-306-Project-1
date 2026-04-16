"""Results reporting using rich for pretty-printed tables.

Provides functions for displaying:
  - DOF table (node IDs, DOF types, global indices)
  - Nodal displacements after solving
  - Reaction forces
  - Element force summaries

_lbl: Returns unit-label dict for a model's unit system.
"""
from __future__ import annotations

import logging

import numpy as np
from rich.console import Console
from rich.table import Table

from fea_solver.models import (
    DOFMap,
    DOFType,
    ElementResult,
    ElementType,
    FEAModel,
    SolutionResult,
)
from fea_solver.units import UNIT_LABELS

logger = logging.getLogger(__name__)
_console = Console()


def _mesh_is_2d(model: FEAModel) -> bool:
    """Return True if any node has a non-zero y-coordinate (2D truss problem).

    Args:
        model (FEAModel): The finite element model.

    Returns:
        bool: True if the mesh spans 2D space (any node.y != 0.0), False for 1D.

    Notes:
        Used to decide whether to show the y-coordinate column in output tables.
        1D bar/beam/frame problems have all nodes at y=0.0.
    """
    return any(n.y != 0.0 for n in model.mesh.nodes)


def _lbl(model: FEAModel) -> dict[str, str]:
    """Return the unit-label dictionary for the model's canonical unit system.

    Args:
        model (FEAModel): The finite element model.

    Returns:
        dict[str, str]: Maps quantity type keys (e.g. "length", "force") to label strings
            (e.g. "m", "N") for use in table column headers and report text.

    Notes:
        Delegates to UNIT_LABELS keyed by model.unit_system.
    """
    return UNIT_LABELS[model.unit_system]


def print_dof_table(model: FEAModel, dof_map: DOFMap) -> None:
    """Pretty-print the DOF table using rich.

    Shows: Node ID | x [<length unit>] | DOF Type | Global Index
    This is called BEFORE solving to inspect the system layout.

    Args:
        model (FEAModel): The finite element model containing mesh and metadata.
        dof_map (DOFMap): Degree-of-freedom mapping (node_id, DOF_type) -> global index.

    Returns:
        None

    Notes:
        Output is printed to console via rich.console.Console.
        Rows sorted by global DOF index.
    """
    lbl = _lbl(model)
    has_y = _mesh_is_2d(model)
    table = Table(title=f"DOF Table -- {model.label}", show_header=True, header_style="bold cyan")
    table.add_column("Node ID", justify="right")
    table.add_column(f"x [{lbl['length']}]", justify="right")
    if has_y:
        table.add_column(f"y [{lbl['length']}]", justify="right")
    table.add_column("DOF Type", justify="center")
    table.add_column("Global Index", justify="right")

    nodes_by_id = {n.id: n for n in model.mesh.nodes}

    for (node_id, dof_type), idx in sorted(dof_map.mapping.items(), key=lambda kv: kv[1]):
        node = nodes_by_id[node_id]
        row = [str(node_id), f"{node.x:.4f}"]
        if has_y:
            row.append(f"{node.y:.4f}")
        row += [dof_type.value, str(idx)]
        table.add_row(*row)

    _console.print(table)


def print_nodal_results(result: SolutionResult) -> None:
    """Pretty-print nodal displacements after solving.

    Args:
        result (SolutionResult): Solution containing displacements, DOF map, and model.

    Returns:
        None

    Notes:
        Output is printed to console via rich.console.Console.
        Displays u, v, and theta displacements for each node in the model's canonical units.
        Constrained DOFs shown as "--" if not present in the DOF map.
    """
    model = result.model
    dof_map = result.dof_map
    u = result.displacements

    lbl = _lbl(model)
    has_y = _mesh_is_2d(model)
    table = Table(title=f"Nodal Results -- {model.label}", show_header=True,
                  header_style="bold green")
    table.add_column("Node ID", justify="right")
    table.add_column(f"x [{lbl['length']}]", justify="right")
    if has_y:
        table.add_column(f"y [{lbl['length']}]", justify="right")
    table.add_column(f"u [{lbl['displacement']}]", justify="right")
    table.add_column(f"v [{lbl['displacement']}]", justify="right")
    table.add_column("theta [rad]", justify="right")

    for node in sorted(model.mesh.nodes, key=lambda n: n.id):
        u_val = (f"{u[dof_map.index(node.id, DOFType.U)]:.6e}"
                 if dof_map.has_dof(node.id, DOFType.U) else "--")
        v_val = (f"{u[dof_map.index(node.id, DOFType.V)]:.6e}"
                 if dof_map.has_dof(node.id, DOFType.V) else "--")
        t_val = (f"{u[dof_map.index(node.id, DOFType.THETA)]:.6e}"
                 if dof_map.has_dof(node.id, DOFType.THETA) else "--")
        row = [str(node.id), f"{node.x:.4f}"]
        if has_y:
            row.append(f"{node.y:.4f}")
        row += [u_val, v_val, t_val]
        table.add_row(*row)

    _console.print(table)


def print_reaction_forces(result: SolutionResult) -> None:
    """Pretty-print reaction forces at constrained DOFs.

    Args:
        result (SolutionResult): Solution containing reactions, DOF map, and model.

    Returns:
        None

    Notes:
        Output is printed to console via rich.console.Console.
        Identifies constrained DOFs and looks up corresponding reactions.
        Force/moment units match the model's canonical unit system.
    """
    model = result.model
    dof_map = result.dof_map

    # Reconstruct which DOFs are constrained (same logic as constraints.py)
    from fea_solver.constraints import get_constrained_dof_indices
    constrained = get_constrained_dof_indices(model, dof_map)

    lbl = _lbl(model)
    table = Table(title=f"Reaction Forces -- {model.label}", show_header=True,
                  header_style="bold red")
    table.add_column("Node ID", justify="right")
    table.add_column("DOF Type", justify="center")
    table.add_column("Global Index", justify="right")
    table.add_column(f"Reaction [{lbl['force']} or {lbl['moment']}]", justify="right")

    reverse_mapping = {idx: key for key, idx in dof_map.mapping.items()}
    for i, global_idx in enumerate(constrained):
        result_tuple = reverse_mapping.get(global_idx)
        if result_tuple is None:
            raise KeyError(f"No DOF found for global index {global_idx}")
        node_id, dof_type = result_tuple
        table.add_row(
            str(node_id),
            dof_type.value,
            str(global_idx),
            f"{result.reactions[i]:.6e}",
        )

    _console.print(table)


def print_element_forces(element_results: list[ElementResult], model: FEAModel) -> None:
    """Print per-element force summary.

    Args:
        element_results (list[ElementResult]): List of post-processed element results.
        model (FEAModel): Model whose unit_system determines column header labels.

    Returns:
        None

    Notes:
        Output is printed to console via rich.console.Console.
        Shows axial force, maximum shear, and maximum bending moment per element.
        Unit labels derived from model.unit_system.
    """
    lbl = _lbl(model)
    table = Table(title="Element Force Summary", show_header=True,
                  header_style="bold yellow")
    table.add_column("Elem ID", justify="right")
    table.add_column(f"Axial N [{lbl['force']}]", justify="right")
    table.add_column(f"Max |V| [{lbl['force']}]", justify="right")
    table.add_column(f"Max |M| [{lbl['moment']}]", justify="right")

    for er in element_results:
        max_v = float(np.max(np.abs(er.shear_forces))) if len(er.shear_forces) > 0 else 0.0
        max_m = float(np.max(np.abs(er.bending_moments))) if len(er.bending_moments) > 0 else 0.0
        table.add_row(
            str(er.element_id),
            f"{er.axial_force:.6e}",
            f"{max_v:.6e}",
            f"{max_m:.6e}",
        )

    _console.print(table)


def generate_report(
    model: FEAModel,
    dof_map: DOFMap,
    result: SolutionResult,
    element_results: list[ElementResult],
) -> str:
    """Generate and print a complete text report. Returns the report as a string.

    Args:
        model (FEAModel): The finite element model.
        dof_map (DOFMap): Degree-of-freedom mapping.
        result (SolutionResult): Solution containing displacements and reactions.
        element_results (list[ElementResult]): Post-processed element internal forces.

    Returns:
        str: The formatted report text.

    Notes:
        Output is both printed to console via rich and logged to file.
        Report includes node/element counts, nodal displacements, and element forces.
    """
    lbl = _lbl(model)
    lines: list[str] = [
        "=" * 60,
        f"FEA SOLVER REPORT -- {model.label}",
        f"Unit system: {model.unit_system.value}",
        "=" * 60,
        f"Nodes: {len(model.mesh.nodes)}",
        f"Elements: {len(model.mesh.elements)}",
        f"Total DOFs: {dof_map.total_dofs}",
        "",
        "--- Nodal Displacements ---",
    ]

    has_y = _mesh_is_2d(model)
    u = result.displacements
    for node in sorted(model.mesh.nodes, key=lambda n: n.id):
        coord = f"x={node.x:.3f}"
        if has_y:
            coord += f", y={node.y:.3f}"
        parts: list[str] = [f"  Node {node.id} ({coord} {lbl['length']}):"]
        if dof_map.has_dof(node.id, DOFType.U):
            parts.append(f"u={u[dof_map.index(node.id, DOFType.U)]:.6e} {lbl['displacement']}")
        if dof_map.has_dof(node.id, DOFType.V):
            parts.append(f"v={u[dof_map.index(node.id, DOFType.V)]:.6e} {lbl['displacement']}")
        if dof_map.has_dof(node.id, DOFType.THETA):
            parts.append(f"theta={u[dof_map.index(node.id, DOFType.THETA)]:.6e} rad")
        lines.append(" ".join(parts))

    lines += ["", "--- Element Forces ---"]
    for er in element_results:
        max_v = float(np.max(np.abs(er.shear_forces))) if len(er.shear_forces) else 0.0
        max_m = float(np.max(np.abs(er.bending_moments))) if len(er.bending_moments) else 0.0
        lines.append(
            f"  Elem {er.element_id}: N={er.axial_force:.4e} {lbl['force']}, "
            f"|V|_max={max_v:.4e} {lbl['force']}, |M|_max={max_m:.4e} {lbl['moment']}"
        )

    report = "\n".join(lines)
    logger.info("Report generated:\n%s", report)
    _console.print(report)
    return report
