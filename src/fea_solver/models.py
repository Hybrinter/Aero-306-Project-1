"""Core data models for the FEA solver.

All dataclasses are frozen (immutable after construction) following Rust-istic
design principles. Enums provide type-safe dispatch throughout the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ElementType(Enum):
    """Governs DOF layout per node."""

    BAR = auto()    # 1 DOF/node: u (axial)
    BEAM = auto()   # 2 DOF/node: v (transverse), theta (rotation)
    FRAME = auto()  # 3 DOF/node: u, v, theta


class DOFType(Enum):
    """Physical meaning of a single degree of freedom."""

    U = "u"        # axial displacement [m]
    V = "v"        # transverse displacement [m]
    THETA = "theta"  # rotation [rad]


class BoundaryConditionType(Enum):
    """Kinematic constraint applied at a node."""

    FIXED_U = auto()      # constrain axial DOF only
    FIXED_V = auto()      # constrain transverse DOF only
    FIXED_THETA = auto()  # constrain rotation DOF only
    FIXED_ALL = auto()    # constrain all DOFs present at node
    PIN = auto()          # constrain u and v, leave theta free
    ROLLER = auto()       # constrain v only


class LoadType(Enum):
    """Type of applied load."""

    POINT_FORCE_X = auto()       # nodal force in x direction [N]
    POINT_FORCE_Y = auto()       # nodal force in y direction [N]
    POINT_MOMENT = auto()        # nodal moment [N·m]
    DISTRIBUTED_Y = auto()       # uniform transverse distributed load [N/m]
    DISTRIBUTED_LINEAR = auto()  # linearly varying transverse distributed load [N/m]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """A point in 1D space."""

    id: int
    x: float  # position along beam axis [m]


# ---------------------------------------------------------------------------
# Material and Section
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialProperties:
    """Combined material and cross-section properties for an element."""

    E: float          # Young's modulus [Pa]
    A: float          # cross-sectional area [m²]
    I: float = 0.0    # second moment of area [m⁴] (0 for pure bar)
    label: str = "default"


# ---------------------------------------------------------------------------
# Structural Elements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Element:
    """A structural element connecting two nodes."""

    id: int
    node_i: Node
    node_j: Node
    element_type: ElementType
    material: MaterialProperties

    @property
    def length(self) -> float:
        """Element length computed from nodal coordinates."""
        return abs(self.node_j.x - self.node_i.x)


# ---------------------------------------------------------------------------
# Loads and Boundary Conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryCondition:
    """Kinematic constraint at a node."""

    node_id: int
    bc_type: BoundaryConditionType


@dataclass(frozen=True)
class NodalLoad:
    """Concentrated force or moment applied at a node."""

    node_id: int
    load_type: LoadType
    magnitude: float  # [N] for forces, [N·m] for moments


@dataclass(frozen=True)
class DistributedLoad:
    """Distributed load applied over an element."""

    element_id: int
    load_type: LoadType
    w_i: float  # intensity at node i [N/m]
    w_j: float  # intensity at node j [N/m]


# ---------------------------------------------------------------------------
# Model Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mesh:
    """Collection of nodes and elements."""

    nodes: tuple[Node, ...]
    elements: tuple[Element, ...]


@dataclass(frozen=True)
class FEAModel:
    """Complete FEA problem definition."""

    mesh: Mesh
    boundary_conditions: tuple[BoundaryCondition, ...]
    nodal_loads: tuple[NodalLoad, ...]
    distributed_loads: tuple[DistributedLoad, ...]
    label: str = "unnamed"


# ---------------------------------------------------------------------------
# DOF Map
# ---------------------------------------------------------------------------


@dataclass
class DOFMap:
    """Maps (node_id, DOFType) pairs to global DOF indices.

    This is the single source of truth for index mapping throughout
    assembly, constraint application, solving, and post-processing.

    DOF ordering convention:
      - Nodes are processed in ascending node id order.
      - Within each node, DOFs are assigned in canonical order: u, v, theta
        (only those applicable to the element type present at that node).
    """

    mapping: dict[tuple[int, DOFType], int] = field(default_factory=dict)
    total_dofs: int = 0

    def index(self, node_id: int, dof: DOFType) -> int:
        """Return global DOF index for the given node and DOF type."""
        return self.mapping[(node_id, dof)]

    def has_dof(self, node_id: int, dof: DOFType) -> bool:
        """Return True if this (node_id, dof) pair exists in the map."""
        return (node_id, dof) in self.mapping


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolutionResult:
    """Immutable container for solved nodal displacements and reactions."""

    displacements: NDArray[np.float64]  # shape (n_dofs,), full vector
    reactions: NDArray[np.float64]      # shape (n_constrained,)
    dof_map: DOFMap
    model: FEAModel


@dataclass(frozen=True)
class ElementResult:
    """Post-processed results for a single element."""

    element_id: int
    axial_force: float                      # N (0.0 for pure BEAM)
    shear_forces: NDArray[np.float64]       # shape (n_stations,) [N]
    bending_moments: NDArray[np.float64]    # shape (n_stations,) [N·m]
    x_stations: NDArray[np.float64]         # global x coords, shape (n_stations,)
