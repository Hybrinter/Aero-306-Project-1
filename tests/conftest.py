"""Shared pytest fixtures for the FEA solver test suite."""

from __future__ import annotations

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Material fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def steel() -> MaterialProperties:
    """Standard steel: E=200 GPa, A=0.01 m^2, I=1e-4 m^4."""
    return MaterialProperties(E=200.0e9, A=0.01, I=1.0e-4, label="steel")


@pytest.fixture
def unit_bar_material() -> MaterialProperties:
    """Unit bar material: E=1, A=1 -> EA/L = 1 for L=1."""
    return MaterialProperties(E=1.0, A=1.0, I=0.0, label="unit_bar")


@pytest.fixture
def unit_beam_material() -> MaterialProperties:
    """Unit beam material: E=1, I=1 -> EI=1."""
    return MaterialProperties(E=1.0, A=1.0, I=1.0, label="unit_beam")


# ---------------------------------------------------------------------------
# Simple model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_node_bar_model(unit_bar_material: MaterialProperties) -> FEAModel:
    """Single bar element: node 1 at x=0, node 2 at x=1. Fixed at node 1."""
    n1 = Node(id=1, pos=(0.0, 0.0))
    n2 = Node(id=2, pos=(1.0, 0.0))
    elem = Element(id=1, node_i=n1, node_j=n2,
                   element_type=ElementType.BAR, material=unit_bar_material)
    mesh = Mesh(nodes=(n1, n2), elements=(elem,))
    bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_U)
    load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_X, magnitude=1.0)
    return FEAModel(mesh=mesh, boundary_conditions=(bc,),
                    nodal_loads=(load,), distributed_loads=(), label="two_node_bar")


@pytest.fixture
def cantilever_beam_model(unit_beam_material: MaterialProperties) -> FEAModel:
    """Single beam element cantilever: fixed at node 1, tip load at node 2."""
    n1 = Node(id=1, pos=(0.0, 0.0))
    n2 = Node(id=2, pos=(1.0, 0.0))
    elem = Element(id=1, node_i=n1, node_j=n2,
                   element_type=ElementType.BEAM, material=unit_beam_material)
    mesh = Mesh(nodes=(n1, n2), elements=(elem,))
    bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_ALL)
    load = NodalLoad(node_id=2, load_type=LoadType.POINT_FORCE_Y, magnitude=-1.0)
    return FEAModel(mesh=mesh, boundary_conditions=(bc,),
                    nodal_loads=(load,), distributed_loads=(), label="cantilever_beam")
