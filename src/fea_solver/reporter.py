"""Results reporting using rich for pretty-printed tables.

Provides functions for displaying:
  - DOF table (node IDs, DOF types, global indices)
  - Nodal displacements after solving
  - Reaction forces
  - Element force summaries
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

logger = logging.getLogger(__name__)
_console = Console()


def print_dof_table(model: FEAModel, dof_map: DOFMap) -> None:
    """Pretty-print the DOF table using rich.

    Shows: Node ID | x [m] | DOF Type | Global Index
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
    table = Table(title=f"DOF Table -- {model.label}", show_header=True, header_style="bold cyan")
    table.add_column("Node ID", justify="right")
    table.add_column("x [m]", justify="right")
    table.add_column("DOF Type", justify="center")
    table.add_column("Global Index", justify="right")

    nodes_by_id = {n.id: n for n in model.mesh.nodes}

    for (node_id, dof_type), idx in sorted(dof_map.mapping.items(), key=lambda kv: kv[1]):
        node = nodes_by_id[node_id]
        table.add_row(
            str(node_id),
            f"{node.x:.4f}",
            dof_type.value,
            str(idx),
        )

    _console.print(table)


def print_nodal_results(result: SolutionResult) -> None:
    """Pretty-print nodal displacements after solving.

    Args:
        result (SolutionResult): Solution containing displacements, DOF map, and model.

    Returns:
        None

    Notes:
        Output is printed to console via rich.console.Console.
        Displays u, v, and theta displacements for each node.
        Constrained DOFs shown as "--" if not present in the DOF map.
    """
    model = result.model
    dof_map = result.dof_map
    u = result.displacements

    table = Table(title=f"Nodal Results -- {model.label}", show_header=True,
                  header_style="bold green")
    table.add_column("Node ID", justify="right")
    table.add_column("x [m]", justify="right")
    table.add_column("u [m]", justify="right")
    table.add_column("v [m]", justify="right")
    table.add_column("theta [rad]", justify="right")

    for node in sorted(model.mesh.nodes, key=lambda n: n.id):
        u_val = (f"{u[dof_map.index(node.id, DOFType.U)]:.6e}"
                 if dof_map.has_dof(node.id, DOFType.U) else "--")
        v_val = (f"{u[dof_map.index(node.id, DOFType.V)]:.6e}"
                 if dof_map.has_dof(node.id, DOFType.V) else "--")
        t_val = (f"{u[dof_map.index(node.id, DOFType.THETA)]:.6e}"
                 if dof_map.has_dof(node.id, DOFType.THETA) else "--")
        table.add_row(str(node.id), f"{node.x:.4f}", u_val, v_val, t_val)

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
        Units are N for forces and N*m for moments.
    """
    model = result.model
    dof_map = result.dof_map

    # Reconstruct which DOFs are constrained (same logic as constraints.py)
    from fea_solver.constraints import get_constrained_dof_indices
    constrained = get_constrained_dof_indices(model, dof_map)

    table = Table(title=f"Reaction Forces -- {model.label}", show_header=True,
                  header_style="bold red")
    table.add_column("Node ID", justify="right")
    table.add_column("DOF Type", justify="center")
    table.add_column("Global Index", justify="right")
    table.add_column("Reaction [N or N*m]", justify="right")

    nodes_by_id = {n.id: n for n in model.mesh.nodes}
    for i, global_idx in enumerate(constrained):
        # Find (node_id, dof_type) for this global index
        result_tuple = next(
            (nid_dof for nid_dof, idx in dof_map.mapping.items() if idx == global_idx),
            None
        )
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


def print_element_forces(element_results: list[ElementResult]) -> None:
    """Print per-element force summary.

    Args:
        element_results (list[ElementResult]): List of post-processed element results.

    Returns:
        None

    Notes:
        Output is printed to console via rich.console.Console.
        Shows axial force, maximum shear, and maximum bending moment per element.
        Units are N for forces and N*m for moments.
    """
    table = Table(title="Element Force Summary", show_header=True,
                  header_style="bold yellow")
    table.add_column("Elem ID", justify="right")
    table.add_column("Axial N [N]", justify="right")
    table.add_column("Max |V| [N]", justify="right")
    table.add_column("Max |M| [N*m]", justify="right")

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
    lines: list[str] = [
        f"=" * 60,
        f"FEA SOLVER REPORT -- {model.label}",
        f"=" * 60,
        f"Nodes: {len(model.mesh.nodes)}",
        f"Elements: {len(model.mesh.elements)}",
        f"Total DOFs: {dof_map.total_dofs}",
        "",
        "--- Nodal Displacements ---",
    ]

    nodes_by_id = {n.id: n for n in model.mesh.nodes}
    u = result.displacements
    for node in sorted(model.mesh.nodes, key=lambda n: n.id):
        parts: list[str] = [f"  Node {node.id} (x={node.x:.3f}):"]
        if dof_map.has_dof(node.id, DOFType.U):
            parts.append(f"u={u[dof_map.index(node.id, DOFType.U)]:.6e}")
        if dof_map.has_dof(node.id, DOFType.V):
            parts.append(f"v={u[dof_map.index(node.id, DOFType.V)]:.6e}")
        if dof_map.has_dof(node.id, DOFType.THETA):
            parts.append(f"theta={u[dof_map.index(node.id, DOFType.THETA)]:.6e}")
        lines.append(" ".join(parts))

    lines += ["", "--- Element Forces ---"]
    for er in element_results:
        max_v = float(np.max(np.abs(er.shear_forces))) if len(er.shear_forces) else 0.0
        max_m = float(np.max(np.abs(er.bending_moments))) if len(er.bending_moments) else 0.0
        lines.append(
            f"  Elem {er.element_id}: N={er.axial_force:.4e} N, "
            f"|V|_max={max_v:.4e} N, |M|_max={max_m:.4e} N*m"
        )

    report = "\n".join(lines)
    logger.info("Report generated:\n%s", report)
    _console.print(report)
    return report
