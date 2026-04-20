"""Integration test for 45-degree inclined roller using the penalty method.

This test validates the primary new capability of LinearConstraint: constraints
in arbitrary directions that cannot be expressed with the old keyword system.

Triangle truss: nodes at (0,0), (2,0), (1,1). EA=1 for all members.
  - Node 1: pin (fix U and V)
  - Node 2: 45-degree inclined roller (constrain direction [1/sqrt(2), 1/sqrt(2)])
  - Node 3: applied downward load F_y = -1.0 N
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fea_solver.assembler import (
    assemble_global_force_vector,
    assemble_global_stiffness,
    build_dof_map,
)
from fea_solver.io_yaml import load_model_from_yaml
from fea_solver.models import DOFType
from fea_solver.solver import run_solve_pipeline

_CONFIG = Path(__file__).parent.parent.parent / "config"


@pytest.fixture(scope="module")
def inclined_roller_solution():
    """Solve the inclined roller test case once for the module."""
    yaml_path = _CONFIG / "example_case_09_inclined_roller.yaml"
    if not yaml_path.exists():
        pytest.skip("example_case_09_inclined_roller.yaml not found")
    model = load_model_from_yaml(yaml_path)
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    result = run_solve_pipeline(model, dof_map, K, F)
    return model, dof_map, result


class TestInclinedRoller45Deg:
    """Tests for the 45-degree inclined roller triangle truss."""

    def test_solves_without_error(self, inclined_roller_solution) -> None:
        """Model solves without raising any exception."""
        model, dof_map, result = inclined_roller_solution
        assert result.displacements is not None

    def test_inclined_constraint_satisfied(self, inclined_roller_solution) -> None:
        """Inclined roller constraint is satisfied to within penalty tolerance.

        The constraint enforces U_2/sqrt(2) + V_2/sqrt(2) = 0,
        i.e. U_2 = -V_2. Checks that the constraint residual is near zero.
        """
        model, dof_map, result = inclined_roller_solution
        u2 = result.displacements[dof_map.index(2, DOFType.U)]
        v2 = result.displacements[dof_map.index(2, DOFType.V)]
        n = 1.0 / math.sqrt(2.0)
        constraint_violation = abs(n * u2 + n * v2)
        # Violation should be on the order of 1/penalty_alpha
        assert constraint_violation < 1e-5

    def test_loaded_node_deflects_downward(self, inclined_roller_solution) -> None:
        """Node 3 (loaded node) deflects downward under downward applied load."""
        model, dof_map, result = inclined_roller_solution
        v3 = result.displacements[dof_map.index(3, DOFType.V)]
        assert v3 < 0.0

    def test_reactions_shape(self, inclined_roller_solution) -> None:
        """reactions array has one entry per constraint (3 total)."""
        model, dof_map, result = inclined_roller_solution
        assert result.reactions.shape == (3,)

    def test_vertical_equilibrium(self, inclined_roller_solution) -> None:
        """Sum of vertical reaction components balances applied load.

        Constraint 0 (node 1, [1,0,0]): pure horizontal reaction.
        Constraint 1 (node 1, [0,1,0]): pure vertical reaction R1_V.
        Constraint 2 (node 2, [n,n,0]): mixed reaction; vertical component = R2 * n.

        Equilibrium: R1_V + R2 * n = 1.0 (upward = -F_y applied)
        """
        model, dof_map, result = inclined_roller_solution
        n = 1.0 / math.sqrt(2.0)
        r1_v = result.reactions[1]   # pure V at node 1
        r2 = result.reactions[2]     # inclined roller magnitude
        total_vertical = r1_v + r2 * n
        assert abs(total_vertical) == pytest.approx(1.0, rel=0.01)

    def test_displacements_non_zero(self, inclined_roller_solution) -> None:
        """At least one DOF has non-zero displacement under applied load."""
        _, _, result = inclined_roller_solution
        assert not np.allclose(result.displacements, 0.0)
