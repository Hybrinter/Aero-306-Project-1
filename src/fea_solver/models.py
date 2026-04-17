"""Core data models for the FEA solver.

All dataclasses are frozen (immutable after construction) following Rust-istic
design principles. Enums provide type-safe dispatch throughout the system.

Key types:
  Node, Element, Mesh, MaterialProperties: geometric and material data.
  LinearConstraint, NodalLoad, DistributedLoad: applied constraints and loads.
  FEAModel: complete problem definition; boundary conditions enforced via
      penalty method (penalty_alpha scales penalty stiffness).
  DOFMap: (node_id, DOFType) -> global DOF index mapping.
  SolutionResult: displacements and reactions from a single solve.
  ElementResult: internal forces, moments, and displacements at sampling stations.
  SolutionSeries: bundle of ElementResults + model for one named mesh refinement;
      used when overlaying multiple solutions on shared plot axes.
  MemberBuckling: per-element Euler buckling result (P_cr, axial force, ratio,
      buckled flag); tension members carry ratio = 0.0 to allow sign-based
      TENSION vs SAFE discrimination downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from fea_solver.units import UnitSystem

# Engineering notation exception: single-letter and compound uppercase variables
# (E, A, I, L, K, F, R, N, V, M, EI, K_ff, F_f) follow standard FEA/structural
# engineering conventions and are exempt from the snake_case rule throughout this
# module and the entire codebase.


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ElementType(Enum):
    """Enumeration governing the number and type of degrees of freedom per node.

    Fields:
        BAR: 1 DOF/node: u (axial displacement).
        BEAM: 2 DOF/node: v (transverse displacement), theta (rotation).
        FRAME: 3 DOF/node: u (axial), v (transverse), theta (rotation).
        TRUSS: 2 DOF/node: u (x-displacement), v (y-displacement).
            Used for 2D pin-jointed trusses; stiffness is assembled in global
            coordinates via coordinate transformation.

    Notes:
        Determines how many DOFs are allocated at each node during DOFMap construction
        and how elements are assembled into the global stiffness matrix.
    """

    BAR = auto()    # 1 DOF/node: u (axial)
    BEAM = auto()   # 2 DOF/node: v (transverse), theta (rotation)
    FRAME = auto()  # 3 DOF/node: u, v, theta
    TRUSS = auto()  # 2 DOF/node: u (x-disp), v (y-disp); 2D pin-jointed


class DOFType(Enum):
    """Enumeration of physical degree of freedom types.

    Fields:
        U: Axial displacement along element axis [m].
        V: Transverse displacement perpendicular to element axis [m].
        THETA: Rotation (slope) in radians [rad].

    Notes:
        Used as part of the (node_id, DOFType) mapping key in DOFMap for
        ordering DOFs during assembly and constraint application.
    """

    U = "u"        # axial displacement [canonical length unit]
    V = "v"        # transverse displacement [canonical length unit]
    THETA = "theta"  # rotation [rad]


class LoadType(Enum):
    """Enumeration of applied load types.

    Fields:
        POINT_FORCE_X: Concentrated force in axial (x) direction [N].
        POINT_FORCE_Y: Concentrated force in transverse (y) direction [N].
        POINT_MOMENT: Concentrated moment [N*m].
        DISTRIBUTED_Y: Uniform transverse distributed load [N/m].
        DISTRIBUTED_LINEAR: Linearly varying transverse distributed load [N/m].

    Notes:
        Nodal loads and distributed loads use these types to specify the
        nature of the applied loading for assembly and post-processing.
    """

    POINT_FORCE_X = auto()       # nodal force in x direction [N]
    POINT_FORCE_Y = auto()       # nodal force in y direction [N]
    POINT_MOMENT = auto()        # nodal moment [N*m]
    DISTRIBUTED_Y = auto()       # uniform transverse distributed load [N/m]
    DISTRIBUTED_LINEAR = auto()  # linearly varying transverse distributed load [N/m]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Node:
    """Immutable point in 2D space representing a structural node.

    Fields:
        id (int): Unique node identifier. Must be positive.
        pos (tuple[float, float]): (x, y) coordinates in metres.

    Notes:
        Frozen and slotted for efficient storage in large meshes. Nodes are
        typically created in ascending id order during mesh construction.
        The x and y properties provide named access to pos[0] and pos[1]
        for convenience and backward compatibility with 1D element code.
        For 1D (bar/beam/frame) problems, y is 0.0 and pos = (x, 0.0).
    """

    id: int
    pos: tuple[float, float]  # (x, y) coordinates [m]

    @property
    def x(self) -> float:
        """Return x-coordinate (pos[0]).

        Returns:
            float: Global x-coordinate in metres.
        """
        return self.pos[0]

    @property
    def y(self) -> float:
        """Return y-coordinate (pos[1]).

        Returns:
            float: Global y-coordinate in metres.
        """
        return self.pos[1]


# ---------------------------------------------------------------------------
# Material and Section
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaterialProperties:
    """Combined material and cross-section properties for an element.

    Fields:
        E (float): Young's modulus in Pascals [Pa].
        A (float): Cross-sectional area [m^2].
        I (float): Second moment of area [m^4]. Default 0.0 for pure bar elements.
        label (str): Optional descriptive label. Default "default".

    Notes:
        Frozen and slotted. E and A are required for all element types. I is
        required for BEAM and FRAME elements; set to 0.0 for BAR elements.
    """

    E: float          # Young's modulus [Pa]
    A: float          # cross-sectional area [m^2]
    I: float = 0.0    # second moment of area [m^4] (0 for pure bar)
    label: str = "default"


# ---------------------------------------------------------------------------
# Structural Elements
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Element:
    """A structural element connecting two nodes.

    Fields:
        id (int): Unique element identifier. Must be positive.
        node_i (Node): Start node of the element.
        node_j (Node): End node of the element.
        element_type (ElementType): Type of element (BAR, BEAM, or FRAME).
        material (MaterialProperties): Material and cross-section properties.

    Notes:
        Frozen and slotted. The length property computes Euclidean distance
        between the two end nodes on every call; for repeated use, cache the value.
    """

    id: int
    node_i: Node
    node_j: Node
    element_type: ElementType
    material: MaterialProperties

    @property
    def length(self) -> float:
        """Compute 2D Euclidean distance between the two end nodes.

        Returns:
            float: Element length in metres. Always positive.

        Notes:
            Computed on every call; not cached. For large meshes call once and store.
            For 1D elements (y=0), this equals abs(node_j.x - node_i.x).
            For 2D truss elements, this equals sqrt(Dx^2 + Dy^2).
        """
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Loads and Boundary Conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearConstraint:
    """One scalar linear constraint equation applied at a node.

    Encodes the constraint: a_U*u + a_V*v + a_THETA*theta = rhs

    Fields:
        node_id (int): Node at which the constraint is applied.
        coefficients (tuple[float, float, float]): Constraint direction vector
            in [U, V, THETA] DOF order (global coordinates). Typically normalized
            to unit length by the YAML parser (_schema_to_model) so that
            compute_constraint_residuals returns physically meaningful reaction
            magnitudes. Non-zero components for DOFs absent at the node raise
            ValueError during constraint application.
        rhs (float): Prescribed displacement or rotation value. Units are metres
            (for U/V constraints) or radians (for THETA constraints). Default 0.0
            (homogeneous constraint).

    Notes:
        Applied via the penalty method: adds k_penalty * outer(g, g) to K
        and k_penalty * rhs * g to F, where g is the global-length coefficient
        vector: g[dof_index] = coefficients[i] for each DOF type present at the
        node, zeros elsewhere.
    """

    node_id: int
    coefficients: tuple[float, float, float]
    rhs: float = 0.0


@dataclass(frozen=True, slots=True)
class NodalLoad:
    """Concentrated force or moment applied at a node.

    Fields:
        node_id (int): Unique identifier of the loaded node.
        load_type (LoadType): Type of load (force or moment).
        magnitude (float): Load magnitude in [N] for forces, [N*m] for moments.

    Notes:
        Frozen and slotted. Applied directly to the global force vector during
        assembly. Load type determines which DOF the load acts upon.
    """

    node_id: int
    load_type: LoadType
    magnitude: float  # [N] for forces, [N*m] for moments


@dataclass(frozen=True, slots=True)
class DistributedLoad:
    """Distributed load applied over an element.

    Fields:
        element_id (int): Unique identifier of the loaded element.
        load_type (LoadType): Type of distributed load (DISTRIBUTED_Y or DISTRIBUTED_LINEAR).
        w_i (float): Load intensity at node i [N/m].
        w_j (float): Load intensity at node j [N/m].

    Notes:
        Frozen and slotted. Converted to consistent nodal forces and moments via
        integration with Hermite shape functions. For uniform loads, w_i == w_j.
    """

    element_id: int
    load_type: LoadType
    w_i: float  # intensity at node i [N/m]
    w_j: float  # intensity at node j [N/m]


# ---------------------------------------------------------------------------
# Model Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Mesh:
    """Collection of nodes and elements that form the spatial discretization.

    Fields:
        nodes (tuple[Node, ...]): Immutable collection of all nodes.
        elements (tuple[Element, ...]): Immutable collection of all elements.

    Notes:
        Frozen and slotted. Forms the spatial backbone of the FEAModel. Nodes
        should be stored in ascending id order for efficient DOF map construction.
    """

    nodes: tuple[Node, ...]
    elements: tuple[Element, ...]


@dataclass(frozen=True, slots=True)
class FEAModel:
    """Complete FEA problem definition containing geometry, loads, and constraints.

    Fields:
        mesh (Mesh): Spatial discretization with nodes and elements.
        boundary_conditions (tuple[LinearConstraint, ...]): Penalty-enforced
            kinematic constraints. Each entry is one scalar linear equation
            in the node's [U, V, THETA] DOF space.
        nodal_loads (tuple[NodalLoad, ...]): Concentrated forces and moments.
        distributed_loads (tuple[DistributedLoad, ...]): Distributed loads over elements.
        label (str): Optional descriptive label. Default "unnamed".
        unit_system (UnitSystem): Canonical unit system all numeric values are stored in.
            Default UnitSystem.SI. Determines reporter column-header labels.
        penalty_alpha (float): Dimensionless scale factor for penalty stiffness. The penalty
            parameter used during constraint enforcement is computed as
            penalty_alpha * max(abs(diag(K))). Default 1e8.

    Notes:
        Frozen and slotted. Immutable after construction; use dataclasses.replace()
        to create modified copies. Forms the complete problem specification that
        is assembled into K and F matrices, solved, and post-processed.
        All field values (node coordinates, material properties, loads) must already
        be expressed in the canonical units for unit_system before construction.
    """

    mesh: Mesh
    boundary_conditions: tuple[LinearConstraint, ...]
    nodal_loads: tuple[NodalLoad, ...]
    distributed_loads: tuple[DistributedLoad, ...]
    label: str = "unnamed"
    unit_system: UnitSystem = UnitSystem.SI
    penalty_alpha: float = 1e8


# ---------------------------------------------------------------------------
# DOF Map
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DOFMap:
    """Maps (node_id, DOFType) pairs to global DOF indices.

    This is the single source of truth for index mapping throughout
    assembly, constraint application, solving, and post-processing.

    DOF ordering convention:
      - Nodes are processed in ascending node id order.
      - Within each node, DOFs are assigned in canonical order: u, v, theta
        (only those applicable to the element type present at that node).

    Fields:
        mapping (dict[tuple[int, DOFType], int]): Maps (node_id, dof_type) to global index.
        total_dofs (int): Total number of DOFs in the system.

    Notes:
        Mutable during construction (not frozen) to allow builder patterns.
        Once populated, should not be modified; all downstream code assumes
        consistent ordering.
    """

    mapping: dict[tuple[int, DOFType], int] = field(default_factory=dict)
    total_dofs: int = 0

    def index(self, node_id: int, dof: DOFType) -> int:
        """Return the global DOF index for the given node and DOF type.

        Args:
            node_id (int): Unique node identifier.
            dof (DOFType): Degree of freedom type (U, V, or THETA).

        Returns:
            int: Global DOF index in the system.

        Notes:
            Raises KeyError if the (node_id, dof) pair is not in the mapping.
        """
        return self.mapping[(node_id, dof)]

    def has_dof(self, node_id: int, dof: DOFType) -> bool:
        """Return True if the given (node_id, dof) pair exists in the map.

        Args:
            node_id (int): Unique node identifier.
            dof (DOFType): Degree of freedom type (U, V, or THETA).

        Returns:
            bool: True if the pair is in the mapping, False otherwise.

        Notes:
            Used during constraint application to ensure only valid DOFs are constrained.
        """
        return (node_id, dof) in self.mapping


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolutionResult:
    """Immutable container for solved nodal displacements and reaction forces.

    Fields:
        displacements (NDArray[np.float64]): Full displacement vector, shape (n_dofs,).
        reactions (NDArray[np.float64]): Reaction forces at constrained DOFs, shape (n_constrained,).
        dof_map (DOFMap): DOF mapping used for index lookups.
        model (FEAModel): Original FEA model for reference and post-processing.

    Notes:
        Frozen and slotted. Displacements are indexed by global DOF index;
        use dof_map to find the index for a specific (node_id, DOFType) pair.
    """

    displacements: NDArray[np.float64]  # shape (n_dofs,), full vector
    reactions: NDArray[np.float64]      # shape (n_constrained,)
    dof_map: DOFMap
    model: FEAModel


@dataclass(frozen=True, slots=True)
class ElementResult:
    """Post-processed internal forces, moments, and displacements for a single element.

    Fields:
        element_id (int): Unique element identifier.
        axial_force (float): Constant axial force (0.0 for pure BEAM elements).
        shear_forces (NDArray[np.float64]): Shear force at stations, shape (n_stations,).
        bending_moments (NDArray[np.float64]): Bending moment at stations, shape (n_stations,).
        x_stations (NDArray[np.float64]): Global x-coordinates of evaluation stations, shape (n_stations,).
        transverse_displacements (NDArray[np.float64]): Transverse displacement v(x) at stations,
            shape (n_stations,). Zeros for BAR elements (no bending DOFs).
        axial_displacements (NDArray[np.float64]): Axial displacement u(x) at stations,
            shape (n_stations,). Zeros for pure BEAM elements (no axial DOF).
        rotations (NDArray[np.float64]): Cross-section rotation theta(x) at stations [rad],
            shape (n_stations,). Zeros for BAR elements.

    Notes:
        Frozen and slotted. Internal forces are recovered from nodal displacements using
        Hermite shape function derivatives. Displacements use the shape functions directly.
        All values are in the canonical unit system of the parent FEAModel.
    """

    element_id: int
    axial_force: float                               # 0.0 for pure BEAM
    shear_forces: NDArray[np.float64]                # shape (n_stations,)
    bending_moments: NDArray[np.float64]             # shape (n_stations,)
    x_stations: NDArray[np.float64]                  # global x coords, shape (n_stations,)
    transverse_displacements: NDArray[np.float64]    # v(x), shape (n_stations,); zeros for BAR
    axial_displacements: NDArray[np.float64]         # u(x), shape (n_stations,); zeros for BEAM
    rotations: NDArray[np.float64]                   # theta(x) [rad], shape (n_stations,); zeros for BAR


@dataclass(frozen=True, slots=True)
class SolutionSeries:
    """Bundle of post-processed results for one named mesh refinement.

    Groups element-level results with their parent model so that plotting
    functions can render multiple solutions on shared axes without receiving
    parallel lists of disparate types.

    Fields:
        label (str): Short human-readable name for this solution (e.g. "coarse", "fine").
            Used for legend entries and per-solution max/min annotations.
        element_results (tuple[ElementResult, ...]): Immutable sequence of post-processed
            element results for all elements in this solution's mesh.
        model (FEAModel): The FEA model that produced these results; used to retrieve
            unit labels for axis annotations.
        result (SolutionResult): Full solution result carrying the displacement vector
            and DOF map. Required by truss deformed-shape plots to recover nodal (U, V).

    Notes:
        Frozen and slotted, matching the style of all other result containers.
        element_results is a tuple (not list) to satisfy the frozen invariant.
        At call sites, wrap a list with tuple(): tuple(element_results_list).
        The model is carried here (not recovered from ElementResult) because
        ElementResult does not hold a model reference. result carries the full
        displacement vector needed by truss deformed-shape plots to recover
        nodal (U, V) translations.
    """

    label: str
    element_results: tuple[ElementResult, ...]
    model: FEAModel
    result: SolutionResult


# ---------------------------------------------------------------------------
# Buckling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemberBuckling:
    """Euler buckling result for one TRUSS member.

    Fields:
        element_id (int): Identifier of the TRUSS element this result belongs to.
        P_cr (float): Euler critical load pi^2 * E * I / L^2 [force units].
            Always positive.
        axial_force (float): Signed axial force [force units] copied from
            ElementResult.axial_force. Positive = tension, negative = compression.
        ratio (float): abs(axial_force) / P_cr when axial_force < 0; 0.0 when
            axial_force >= 0. Dimensionless.
        is_buckled (bool): True iff axial_force < 0 and abs(axial_force) >= P_cr.

    Notes:
        Frozen and slotted; immutable after construction. Produced only for
        elements with element_type == TRUSS. Tension members are included in
        the returned tuple with ratio = 0.0 and is_buckled = False so that the
        downstream reporter can distinguish TENSION from SAFE compressive
        members by the sign of axial_force.
    """

    element_id: int
    P_cr: float
    axial_force: float
    ratio: float
    is_buckled: bool
