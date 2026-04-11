"""Tests for matplotlib plotting functions -- TDD first."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
from fea_solver.models import (
    BoundaryCondition, BoundaryConditionType, Element, ElementResult, ElementType,
    FEAModel, MaterialProperties, Mesh, Node, UnitSystem,
)
from fea_solver.plotter import (
    plot_axial_displacement,
    plot_bending_moment_diagram,
    plot_rotation,
    plot_shear_force_diagram,
    plot_transverse_displacement,
)


def _make_model(unit_system: UnitSystem = UnitSystem.SI) -> FEAModel:
    """Build a minimal two-node beam FEAModel for plotter tests."""
    mat = MaterialProperties(E=1.0, A=1.0, I=1.0)
    n1, n2 = Node(1, 0.0), Node(2, 1.0)
    elem = Element(id=1, node_i=n1, node_j=n2, element_type=ElementType.BEAM, material=mat)
    bc = BoundaryCondition(node_id=1, bc_type=BoundaryConditionType.FIXED_ALL)
    return FEAModel(
        mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
        boundary_conditions=(bc,),
        nodal_loads=(),
        distributed_loads=(),
        label="test",
        unit_system=unit_system,
    )


def make_element_result(
    element_id: int = 1,
    x_start: float = 0.0,
    x_end: float = 1.0,
    n: int = 50,
) -> ElementResult:
    """Create an ElementResult with given parameters."""
    x = np.linspace(x_start, x_end, n)
    V = np.ones(n) * -1.0
    M = np.linspace(-1.0, 0.0, n)
    v = np.linspace(0.0, -0.01, n)
    u = np.linspace(0.0, 0.001, n)
    theta = np.linspace(0.0, -0.005, n)
    return ElementResult(
        element_id=element_id,
        axial_force=0.0,
        shear_forces=V,
        bending_moments=M,
        x_stations=x,
        transverse_displacements=v,
        axial_displacements=u,
        rotations=theta,
    )


class TestPlotShearForceDiagram:
    """Tests for plot_shear_force_diagram."""
    def test_returns_figure(self) -> None:
        """SFD plot returns matplotlib Figure."""
        model = _make_model()
        fig = plot_shear_force_diagram([make_element_result()], model, title="Test SFD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_multiple_elements(self) -> None:
        """SFD plot handles multiple elements."""
        model = _make_model()
        ers = [make_element_result(i + 1, float(i), float(i + 1)) for i in range(3)]
        fig = plot_shear_force_diagram(ers, model, title="Multi-element SFD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """SFD plot saves to file correctly."""
        model = _make_model()
        out = tmp_path / "sfd.png"
        fig = plot_shear_force_diagram([make_element_result()], model,
                                       title="Save Test", output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_empirical_ylabel(self) -> None:
        """SFD y-axis label uses empirical units when model is EMPIRICAL."""
        model = _make_model(UnitSystem.EMPIRICAL)
        fig = plot_shear_force_diagram([make_element_result()], model)
        ax = fig.axes[0]
        assert "lb" in ax.get_ylabel()
        plt.close(fig)

    def test_si_ylabel(self) -> None:
        """SFD y-axis label uses SI units when model is SI."""
        model = _make_model(UnitSystem.SI)
        fig = plot_shear_force_diagram([make_element_result()], model)
        ax = fig.axes[0]
        assert "N" in ax.get_ylabel()
        plt.close(fig)


class TestPlotBendingMomentDiagram:
    """Tests for plot_bending_moment_diagram."""
    def test_returns_figure(self) -> None:
        """BMD plot returns matplotlib Figure."""
        model = _make_model()
        fig = plot_bending_moment_diagram([make_element_result()], model, title="Test BMD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_multiple_elements(self) -> None:
        """BMD plot handles multiple elements."""
        model = _make_model()
        ers = [make_element_result(i + 1, float(i), float(i + 1)) for i in range(3)]
        fig = plot_bending_moment_diagram(ers, model, title="Multi-element BMD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """BMD plot saves to file correctly."""
        model = _make_model()
        out = tmp_path / "bmd.png"
        fig = plot_bending_moment_diagram([make_element_result()], model,
                                          title="Save Test", output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_empirical_moment_label(self) -> None:
        """BMD y-axis uses in-lb for empirical problems."""
        model = _make_model(UnitSystem.EMPIRICAL)
        fig = plot_bending_moment_diagram([make_element_result()], model)
        ax = fig.axes[0]
        assert "in-lb" in ax.get_ylabel()
        plt.close(fig)


class TestPlotTransverseDisplacement:
    """Tests for plot_transverse_displacement."""
    def test_returns_figure(self) -> None:
        """Transverse displacement plot returns matplotlib Figure."""
        model = _make_model()
        fig = plot_transverse_displacement([make_element_result()], model)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ylabel_has_displacement_unit(self) -> None:
        """ylabel contains displacement unit."""
        model = _make_model(UnitSystem.EMPIRICAL)
        fig = plot_transverse_displacement([make_element_result()], model)
        ax = fig.axes[0]
        assert "in" in ax.get_ylabel()
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """Transverse displacement plot saves to file correctly."""
        model = _make_model()
        out = tmp_path / "v.png"
        fig = plot_transverse_displacement([make_element_result()], model, output_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotAxialDisplacement:
    """Tests for plot_axial_displacement."""
    def test_returns_figure(self) -> None:
        """Axial displacement plot returns matplotlib Figure."""
        model = _make_model()
        fig = plot_axial_displacement([make_element_result()], model)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """Axial displacement plot saves to file correctly."""
        model = _make_model()
        out = tmp_path / "u.png"
        fig = plot_axial_displacement([make_element_result()], model, output_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotRotation:
    """Tests for plot_rotation."""
    def test_returns_figure(self) -> None:
        """Rotation plot returns matplotlib Figure."""
        model = _make_model()
        fig = plot_rotation([make_element_result()], model)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ylabel_contains_rad(self) -> None:
        """Rotation y-axis label contains 'rad'."""
        model = _make_model()
        fig = plot_rotation([make_element_result()], model)
        ax = fig.axes[0]
        assert "rad" in ax.get_ylabel()
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """Rotation plot saves to file correctly."""
        model = _make_model()
        out = tmp_path / "theta.png"
        fig = plot_rotation([make_element_result()], model, output_path=out)
        assert out.exists()
        plt.close(fig)
