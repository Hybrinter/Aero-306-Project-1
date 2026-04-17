"""Problem definition for geometry optimization.

GeometryOptimizationProblem captures everything that does not change during
the optimization: the baseline FEAModel (mesh + connectivity), which node
ids are free vs frozen, the box bounds, the load magnitude, and the
constraint thresholds. apply_x_to_model rebuilds an FEAModel from a
12-vector by overwriting only the free node positions and the single
nodal load.

GeometryOptimizationProblem.from_baseline:  Validate inputs and build a frozen problem.
GeometryOptimizationProblem.n_vars:         Number of design variables (== 2 * len(free_node_ids)).
GeometryOptimizationProblem.box_bounds:     Tuple of (lo, hi) per design variable.
apply_x_to_model:                            Rebuild FEAModel with free nodes overwritten and
                                             nodal load magnitude set to -F_magnitude on the
                                             single load node (assumed unique in the baseline).
baseline_x:                                  Extract baseline coordinates of free nodes as a 12-vector.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fea_solver.models import FEAModel, Mesh, Node, NodalLoad


@dataclass(frozen=True, slots=True)
class GeometryOptimizationProblem:
    """Immutable problem definition for the geometry optimizer.

    Fields:
        baseline_model (FEAModel): Original FEA model providing connectivity,
            materials, BCs, and frozen node positions.
        free_node_ids (tuple[int, ...]): Node ids whose (x, y) are decision variables.
        frozen_node_ids (tuple[int, ...]): Node ids that must keep their baseline (x, y).
        box_bounds (tuple[tuple[float, float], ...]): One (lo, hi) per design variable,
            in the order (x_free[0], y_free[0], x_free[1], y_free[1], ...).
        F_magnitude (float): Magnitude of the downward tip load applied at the load node.
        sigma_max (float): Stress limit in MPa.
        L_min (float): Minimum allowed element length in the model's length unit.
        load_node_id (int): Node id where the single nodal load is applied
            (resolved from baseline_model at construction time).

    Notes:
        Frozen and slotted. Construct via GeometryOptimizationProblem.from_baseline,
        which validates inputs.
    """

    baseline_model: FEAModel
    free_node_ids: tuple[int, ...]
    frozen_node_ids: tuple[int, ...]
    box_bounds: tuple[tuple[float, float], ...]
    F_magnitude: float
    sigma_max: float
    L_min: float
    load_node_id: int

    @property
    def n_vars(self) -> int:
        """Return number of design variables (2 per free node).

        Returns:
            int: 2 * len(free_node_ids).
        """
        return 2 * len(self.free_node_ids)

    @classmethod
    def from_baseline(
        cls,
        model: FEAModel,
        free_node_ids: Sequence[int],
        frozen_node_ids: Sequence[int],
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        F_magnitude: float,
        sigma_max: float,
        L_min: float,
    ) -> "GeometryOptimizationProblem":
        """Validate inputs and build a frozen problem instance.

        Args:
            model (FEAModel): Baseline FEA model loaded from problem_7.yaml.
            free_node_ids (Sequence[int]): Free node ids; order is preserved
                in the design vector.
            frozen_node_ids (Sequence[int]): Frozen node ids.
            x_bounds (tuple[float, float]): (lo, hi) box for x of every free node.
            y_bounds (tuple[float, float]): (lo, hi) box for y of every free node.
            F_magnitude (float): Magnitude of the tip load (applied as -F in y).
            sigma_max (float): Stress limit (MPa).
            L_min (float): Minimum element length (length unit of the model).

        Returns:
            GeometryOptimizationProblem: Validated immutable problem.

        Raises:
            ValueError: If free and frozen node id sets overlap, if any id is
                missing from the model, if bounds are inverted, or if the
                baseline model does not contain exactly one nodal load.
        """
        free = tuple(free_node_ids)
        frozen = tuple(frozen_node_ids)
        overlap = set(free) & set(frozen)
        if overlap:
            raise ValueError(f"free and frozen node ids overlap: {sorted(overlap)}")
        all_ids = {n.id for n in model.mesh.nodes}
        missing = (set(free) | set(frozen)) - all_ids
        if missing:
            raise ValueError(f"node ids not in model: {sorted(missing)}")
        if x_bounds[0] >= x_bounds[1]:
            raise ValueError(f"x bounds must be (lo < hi), got {x_bounds}")
        if y_bounds[0] >= y_bounds[1]:
            raise ValueError(f"y bounds must be (lo < hi), got {y_bounds}")
        if len(model.nodal_loads) != 1:
            raise ValueError(
                f"baseline model must have exactly one nodal load, got {len(model.nodal_loads)}"
            )
        load_node_id = model.nodal_loads[0].node_id
        bounds = tuple(
            (x_bounds if i % 2 == 0 else y_bounds) for i in range(2 * len(free))
        )
        return cls(
            baseline_model=model,
            free_node_ids=free,
            frozen_node_ids=frozen,
            box_bounds=bounds,
            F_magnitude=float(F_magnitude),
            sigma_max=float(sigma_max),
            L_min=float(L_min),
            load_node_id=load_node_id,
        )


def baseline_x(problem: GeometryOptimizationProblem) -> NDArray[np.float64]:
    """Return the design vector that reproduces the baseline model geometry.

    Args:
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        NDArray[np.float64]: Shape (n_vars,) with (x, y) of each free node
            in the order specified by problem.free_node_ids.
    """
    nodes_by_id = {n.id: n for n in problem.baseline_model.mesh.nodes}
    out = np.empty(problem.n_vars, dtype=np.float64)
    for i, node_id in enumerate(problem.free_node_ids):
        node = nodes_by_id[node_id]
        out[2 * i] = node.x
        out[2 * i + 1] = node.y
    return out


def apply_x_to_model(
    x: NDArray[np.float64],
    problem: GeometryOptimizationProblem,
) -> FEAModel:
    """Rebuild an FEAModel by overwriting free node positions and the load magnitude.

    Args:
        x (NDArray[np.float64]): Design vector, shape (n_vars,).
        problem (GeometryOptimizationProblem): Problem definition.

    Returns:
        FEAModel: New frozen FEAModel with the same connectivity, materials,
            and boundary conditions as the baseline. Free node positions are
            overwritten with x; the single nodal load magnitude is set to
            -problem.F_magnitude.

    Notes:
        Element objects hold direct references to Node instances, so we must
        rebuild the elements with the new node objects. Frozen nodes keep
        their baseline positions.
    """
    if x.shape != (problem.n_vars,):
        raise ValueError(f"x must have shape ({problem.n_vars},), got {x.shape}")
    free_lookup = {
        node_id: (float(x[2 * i]), float(x[2 * i + 1]))
        for i, node_id in enumerate(problem.free_node_ids)
    }
    new_nodes_by_id: dict[int, Node] = {}
    for node in problem.baseline_model.mesh.nodes:
        if node.id in free_lookup:
            new_nodes_by_id[node.id] = Node(id=node.id, pos=free_lookup[node.id])
        else:
            new_nodes_by_id[node.id] = node
    new_nodes = tuple(new_nodes_by_id[n.id] for n in problem.baseline_model.mesh.nodes)
    new_elements = tuple(
        replace(
            e,
            node_i=new_nodes_by_id[e.node_i.id],
            node_j=new_nodes_by_id[e.node_j.id],
        )
        for e in problem.baseline_model.mesh.elements
    )
    new_mesh = Mesh(nodes=new_nodes, elements=new_elements)
    baseline_load = problem.baseline_model.nodal_loads[0]
    new_load = NodalLoad(
        node_id=baseline_load.node_id,
        load_type=baseline_load.load_type,
        magnitude=-problem.F_magnitude,
    )
    return replace(
        problem.baseline_model,
        mesh=new_mesh,
        nodal_loads=(new_load,),
    )
