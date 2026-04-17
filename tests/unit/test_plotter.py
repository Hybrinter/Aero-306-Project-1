"""Tests for matplotlib plotting functions -- TDD first."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from fea_solver.assembler import build_dof_map
from fea_solver.models import (
    DOFMap, DOFType, Element, ElementResult, ElementType,
    FEAModel, LinearConstraint, MaterialProperties, Mesh, Node,
    SolutionResult, SolutionSeries, UnitSystem,
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
    n1, n2 = Node(1, (0.0, 0.0)), Node(2, (1.0, 0.0))
    elem = Element(id=1, node_i=n1, node_j=n2, element_type=ElementType.BEAM, material=mat)
    c_v = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0))
    c_t = LinearConstraint(node_id=1, coefficients=(0.0, 0.0, 1.0))
    return FEAModel(
        mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
        boundary_conditions=(c_v, c_t),
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


def _make_series(
    label: str = "test",
    x_start: float = 0.0,
    x_end: float = 1.0,
    unit_system: UnitSystem = UnitSystem.SI,
) -> SolutionSeries:
    """Build a SolutionSeries with a single ElementResult for plotter tests."""
    model = _make_model(unit_system)
    dof_map = build_dof_map(model)
    result_stub = SolutionResult(
        displacements=np.zeros(dof_map.total_dofs),
        reactions=np.zeros(0),
        dof_map=dof_map,
        model=model,
    )
    er = make_element_result(x_start=x_start, x_end=x_end)
    return SolutionSeries(label=label, element_results=(er,), model=model, result=result_stub)


def _make_truss_model() -> FEAModel:
    """Build a minimal two-node TRUSS FEAModel for truss plotter tests."""
    mat = MaterialProperties(E=200e9, A=0.01, I=0.0)
    n1 = Node(1, (0.0, 0.0))
    n2 = Node(2, (3.0, 4.0))
    elem = Element(id=1, node_i=n1, node_j=n2, element_type=ElementType.TRUSS, material=mat)
    c1 = LinearConstraint(node_id=1, coefficients=(1.0, 0.0, 0.0))
    c2 = LinearConstraint(node_id=1, coefficients=(0.0, 1.0, 0.0))
    return FEAModel(
        mesh=Mesh(nodes=(n1, n2), elements=(elem,)),
        boundary_conditions=(c1, c2),
        nodal_loads=(),
        distributed_loads=(),
        label="test_truss",
    )


def _make_truss_series(axial_force: float = 50.0) -> SolutionSeries:
    """Build a SolutionSeries with a single TRUSS ElementResult."""
    model = _make_truss_model()
    dof_map = build_dof_map(model)
    # TRUSS 2-node: DOFs are (node1,U)=0, (node1,V)=1, (node2,U)=2, (node2,V)=3
    u = np.zeros(dof_map.total_dofs)
    u[dof_map.index(2, DOFType.U)] = 0.001
    u[dof_map.index(2, DOFType.V)] = -0.002
    result = SolutionResult(
        displacements=u,
        reactions=np.zeros(0),
        dof_map=dof_map,
        model=model,
    )
    er = ElementResult(
        element_id=1,
        axial_force=axial_force,
        shear_forces=np.zeros(5),
        bending_moments=np.zeros(5),
        x_stations=np.linspace(0.0, 3.0, 5),
        transverse_displacements=np.zeros(5),
        axial_displacements=np.zeros(5),
        rotations=np.zeros(5),
    )
    return SolutionSeries(label="test", element_results=(er,), model=model, result=result)


class TestTrussHelpers:
    """Tests for private truss plotting helpers."""

    def test_colormap_norm_symmetric_range(self) -> None:
        """Norm vmin and vmax are symmetric about zero."""
        from fea_solver.plotter import _truss_colormap_norm
        cmap, norm = _truss_colormap_norm([10.0, -5.0, 3.0])
        assert norm.vmax == pytest.approx(10.0)
        assert norm.vmin == pytest.approx(-10.0)
        assert norm.vcenter == pytest.approx(0.0)

    def test_colormap_norm_all_zero_fallback(self) -> None:
        """All-zero values use fallback norm vmin=-1, vmax=1."""
        from fea_solver.plotter import _truss_colormap_norm
        cmap, norm = _truss_colormap_norm([0.0, 0.0])
        assert norm.vmax == pytest.approx(1.0)
        assert norm.vmin == pytest.approx(-1.0)

    def test_colormap_returns_coolwarm(self) -> None:
        """Returned colormap is coolwarm."""
        from fea_solver.plotter import _truss_colormap_norm
        cmap, _ = _truss_colormap_norm([1.0])
        assert "coolwarm" in cmap.name

    def test_node_displacements_keys(self) -> None:
        """Node displacement dict has an entry for every node."""
        from fea_solver.plotter import _truss_node_displacements
        sol = _make_truss_series()
        disps = _truss_node_displacements(sol)
        assert set(disps.keys()) == {1, 2}

    def test_node_displacements_values(self) -> None:
        """Extracted displacements match what was put into the displacement vector."""
        from fea_solver.plotter import _truss_node_displacements
        sol = _make_truss_series()
        disps = _truss_node_displacements(sol)
        assert disps[1] == pytest.approx((0.0, 0.0))
        assert disps[2][0] == pytest.approx(0.001)
        assert disps[2][1] == pytest.approx(-0.002)


class TestPlotTrussForces:
    """Tests for plot_truss_forces."""

    def test_returns_figure(self) -> None:
        """plot_truss_forces returns a matplotlib Figure."""
        from fea_solver.plotter import plot_truss_forces
        fig = plot_truss_forces(_make_truss_series())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """plot_truss_forces saves a PNG when output_path is given."""
        from fea_solver.plotter import plot_truss_forces
        out = tmp_path / "forces.png"
        fig = plot_truss_forces(_make_truss_series(), output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_has_colorbar(self) -> None:
        """plot_truss_forces figure contains a colorbar axes."""
        from fea_solver.plotter import plot_truss_forces
        fig = plot_truss_forces(_make_truss_series())
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_title_set(self) -> None:
        """Custom title is applied to the axes."""
        from fea_solver.plotter import plot_truss_forces
        fig = plot_truss_forces(_make_truss_series(), title="My Forces")
        assert fig.axes[0].get_title() == "My Forces"
        plt.close(fig)

    def test_compression_series_no_crash(self) -> None:
        """plot_truss_forces handles all-compression (negative) forces."""
        from fea_solver.plotter import plot_truss_forces
        fig = plot_truss_forces(_make_truss_series(axial_force=-80.0))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTrussDeformed:
    """Tests for plot_truss_deformed."""

    def test_returns_figure(self) -> None:
        """plot_truss_deformed returns a matplotlib Figure."""
        from fea_solver.plotter import plot_truss_deformed
        fig = plot_truss_deformed(_make_truss_series())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """plot_truss_deformed saves PNG when output_path given."""
        from fea_solver.plotter import plot_truss_deformed
        out = tmp_path / "deformed.png"
        fig = plot_truss_deformed(_make_truss_series(), output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_scale_factor_in_title(self) -> None:
        """Scale factor string appears in the plot title."""
        from fea_solver.plotter import plot_truss_deformed
        fig = plot_truss_deformed(_make_truss_series(), title="Deformed")
        title_text = fig.axes[0].get_title()
        assert "scale" in title_text
        assert "x" in title_text
        plt.close(fig)

    def test_has_colorbar(self) -> None:
        """plot_truss_deformed figure contains a colorbar axes."""
        from fea_solver.plotter import plot_truss_deformed
        fig = plot_truss_deformed(_make_truss_series())
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_zero_displacement_no_crash(self) -> None:
        """plot_truss_deformed handles zero displacement (scale fallback to 1.0)."""
        from fea_solver.plotter import plot_truss_deformed
        model = _make_truss_model()
        dof_map = build_dof_map(model)
        result_zero = SolutionResult(
            displacements=np.zeros(dof_map.total_dofs),
            reactions=np.zeros(0),
            dof_map=dof_map,
            model=model,
        )
        er = ElementResult(
            element_id=1,
            axial_force=0.0,
            shear_forces=np.zeros(5),
            bending_moments=np.zeros(5),
            x_stations=np.linspace(0.0, 3.0, 5),
            transverse_displacements=np.zeros(5),
            axial_displacements=np.zeros(5),
            rotations=np.zeros(5),
        )
        sol = SolutionSeries(label="zero", element_results=(er,), model=model, result=result_zero)
        fig = plot_truss_deformed(sol)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTrussStress:
    """Tests for plot_truss_stress."""

    def test_returns_figure(self) -> None:
        """plot_truss_stress returns a matplotlib Figure."""
        from fea_solver.plotter import plot_truss_stress
        fig = plot_truss_stress(_make_truss_series())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """plot_truss_stress saves PNG when output_path given."""
        from fea_solver.plotter import plot_truss_stress
        out = tmp_path / "stress.png"
        fig = plot_truss_stress(_make_truss_series(), output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_has_colorbar(self) -> None:
        """plot_truss_stress figure contains a colorbar axes."""
        from fea_solver.plotter import plot_truss_stress
        fig = plot_truss_stress(_make_truss_series())
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_colorbar_label_contains_stress_unit(self) -> None:
        """Colorbar label contains force and length units for SI model."""
        from fea_solver.plotter import plot_truss_stress
        fig = plot_truss_stress(_make_truss_series())
        colorbar_ax = fig.axes[1]
        ylabel = colorbar_ax.get_ylabel()
        assert "N" in ylabel
        assert "m" in ylabel
        plt.close(fig)

    def test_title_set(self) -> None:
        """Custom title is applied."""
        from fea_solver.plotter import plot_truss_stress
        fig = plot_truss_stress(_make_truss_series(), title="My Stress")
        assert fig.axes[0].get_title() == "My Stress"
        plt.close(fig)


class TestPlotShearForceDiagram:
    """Tests for plot_shear_force_diagram."""

    def test_returns_figure(self) -> None:
        """SFD plot returns matplotlib Figure."""
        fig = plot_shear_force_diagram([_make_series()], title="Test SFD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_multiple_elements(self) -> None:
        """SFD plot handles multiple elements in a series."""
        model = _make_model()
        dof_map = build_dof_map(model)
        result_stub = SolutionResult(
            displacements=np.zeros(dof_map.total_dofs), reactions=np.zeros(0),
            dof_map=dof_map, model=model,
        )
        ers = tuple(make_element_result(i + 1, float(i), float(i + 1)) for i in range(3))
        series = SolutionSeries(label="multi", element_results=ers, model=model, result=result_stub)
        fig = plot_shear_force_diagram([series], title="Multi-element SFD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """SFD plot saves to file correctly."""
        out = tmp_path / "sfd.png"
        fig = plot_shear_force_diagram([_make_series()], title="Save Test", output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_empirical_ylabel(self) -> None:
        """SFD y-axis label uses empirical units when model is EMPIRICAL."""
        fig = plot_shear_force_diagram([_make_series(unit_system=UnitSystem.EMPIRICAL)])
        ax = fig.axes[0]
        assert "lb" in ax.get_ylabel()
        plt.close(fig)

    def test_si_ylabel(self) -> None:
        """SFD y-axis label uses SI units when model is SI."""
        fig = plot_shear_force_diagram([_make_series(unit_system=UnitSystem.SI)])
        ax = fig.axes[0]
        assert "N" in ax.get_ylabel()
        plt.close(fig)


class TestPlotBendingMomentDiagram:
    """Tests for plot_bending_moment_diagram."""

    def test_returns_figure(self) -> None:
        """BMD plot returns matplotlib Figure."""
        fig = plot_bending_moment_diagram([_make_series()], title="Test BMD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_multiple_elements(self) -> None:
        """BMD plot handles multiple elements."""
        model = _make_model()
        dof_map = build_dof_map(model)
        result_stub = SolutionResult(
            displacements=np.zeros(dof_map.total_dofs), reactions=np.zeros(0),
            dof_map=dof_map, model=model,
        )
        ers = tuple(make_element_result(i + 1, float(i), float(i + 1)) for i in range(3))
        series = SolutionSeries(label="multi", element_results=ers, model=model, result=result_stub)
        fig = plot_bending_moment_diagram([series], title="Multi-element BMD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """BMD plot saves to file correctly."""
        out = tmp_path / "bmd.png"
        fig = plot_bending_moment_diagram([_make_series()], title="Save Test", output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_empirical_moment_label(self) -> None:
        """BMD y-axis uses in-lb for empirical problems."""
        fig = plot_bending_moment_diagram([_make_series(unit_system=UnitSystem.EMPIRICAL)])
        ax = fig.axes[0]
        assert "in-lb" in ax.get_ylabel()
        plt.close(fig)


class TestPlotTransverseDisplacement:
    """Tests for plot_transverse_displacement."""

    def test_returns_figure(self) -> None:
        """Transverse displacement plot returns matplotlib Figure."""
        fig = plot_transverse_displacement([_make_series()])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ylabel_has_displacement_unit(self) -> None:
        """ylabel contains displacement unit."""
        fig = plot_transverse_displacement([_make_series(unit_system=UnitSystem.EMPIRICAL)])
        ax = fig.axes[0]
        assert "in" in ax.get_ylabel()
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """Transverse displacement plot saves to file correctly."""
        out = tmp_path / "v.png"
        fig = plot_transverse_displacement([_make_series()], output_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotAxialDisplacement:
    """Tests for plot_axial_displacement."""

    def test_returns_figure(self) -> None:
        """Axial displacement plot returns matplotlib Figure."""
        fig = plot_axial_displacement([_make_series()])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """Axial displacement plot saves to file correctly."""
        out = tmp_path / "u.png"
        fig = plot_axial_displacement([_make_series()], output_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotRotation:
    """Tests for plot_rotation."""

    def test_returns_figure(self) -> None:
        """Rotation plot returns matplotlib Figure."""
        fig = plot_rotation([_make_series()])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ylabel_contains_rad(self) -> None:
        """Rotation y-axis label contains 'rad'."""
        fig = plot_rotation([_make_series()])
        ax = fig.axes[0]
        assert "rad" in ax.get_ylabel()
        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """Rotation plot saves to file correctly."""
        out = tmp_path / "theta.png"
        fig = plot_rotation([_make_series()], output_path=out)
        assert out.exists()
        plt.close(fig)


class TestMultiSeriesPlots:
    """Tests for overlaid multi-series plotting behavior."""

    def test_sfd_two_series_returns_figure(self) -> None:
        """SFD with two series returns a Figure."""
        s1 = _make_series(label="coarse")
        s2 = _make_series(label="fine")
        fig = plot_shear_force_diagram([s1, s2])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_sfd_two_series_legend_contains_both_labels(self) -> None:
        """SFD legend text contains both series labels."""
        s1 = _make_series(label="coarse")
        s2 = _make_series(label="fine")
        fig = plot_shear_force_diagram([s1, s2])
        ax = fig.axes[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        legend_combined = " ".join(legend_texts)
        assert "coarse" in legend_combined
        assert "fine" in legend_combined
        plt.close(fig)

    def test_sfd_single_series_has_fill_between(self) -> None:
        """Single-series SFD preserves fill_between (PolyCollection present)."""
        fig = plot_shear_force_diagram([_make_series()])
        ax = fig.axes[0]
        poly_collections = [a for a in ax.collections if isinstance(a, PolyCollection)]
        assert len(poly_collections) > 0, "Expected fill_between for single series"
        plt.close(fig)

    def test_sfd_two_series_no_fill_between(self) -> None:
        """Multi-series SFD suppresses fill_between (no PolyCollection)."""
        s1 = _make_series(label="coarse")
        s2 = _make_series(label="fine")
        fig = plot_shear_force_diagram([s1, s2])
        ax = fig.axes[0]
        poly_collections = [a for a in ax.collections if isinstance(a, PolyCollection)]
        assert len(poly_collections) == 0, "Expected no fill_between for multi series"
        plt.close(fig)

    def test_sfd_empty_series_raises(self) -> None:
        """Empty series list raises ValueError."""
        with pytest.raises(ValueError):
            plot_shear_force_diagram([])

    def test_colors_are_distinct_for_two_series(self) -> None:
        """Two series main lines are plotted with distinct colors."""
        s1 = _make_series(label="coarse")
        s2 = _make_series(label="fine")
        fig = plot_shear_force_diagram([s1, s2])
        ax = fig.axes[0]
        # Filter for main plot lines only (label starts with "V(x)"), not extreme markers
        main_lines = [
            ln for ln in ax.get_lines()
            if ln.get_label().startswith("V(x)")
        ]
        assert len(main_lines) >= 2
        assert main_lines[0].get_color() != main_lines[1].get_color()
        plt.close(fig)

    def test_bmd_two_series_returns_figure(self) -> None:
        """BMD with two series returns a Figure."""
        s1 = _make_series(label="coarse")
        s2 = _make_series(label="fine")
        fig = plot_bending_moment_diagram([s1, s2])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_transverse_two_series_returns_figure(self) -> None:
        """Transverse displacement with two series returns a Figure."""
        s1 = _make_series(label="coarse")
        s2 = _make_series(label="fine")
        fig = plot_transverse_displacement([s1, s2])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axial_two_series_returns_figure(self) -> None:
        """Axial displacement with two series returns a Figure."""
        s1 = _make_series(label="coarse")
        s2 = _make_series(label="fine")
        fig = plot_axial_displacement([s1, s2])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_rotation_two_series_returns_figure(self) -> None:
        """Rotation with two series returns a Figure."""
        s1 = _make_series(label="coarse")
        s2 = _make_series(label="fine")
        fig = plot_rotation([s1, s2])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_linestyles_are_distinct_for_two_series(self) -> None:
        """Two series main lines use distinct line styles (solid vs dashed)."""
        s1 = _make_series(label="coarse")
        s2 = _make_series(label="fine")
        fig = plot_shear_force_diagram([s1, s2])
        ax = fig.axes[0]
        main_lines = [ln for ln in ax.get_lines() if ln.get_label().startswith("V(x)")]
        assert len(main_lines) >= 2
        assert main_lines[0].get_linestyle() != main_lines[1].get_linestyle()
        plt.close(fig)

    def test_first_series_is_solid(self) -> None:
        """First series uses a solid line style."""
        fig = plot_shear_force_diagram([_make_series(label="coarse"), _make_series(label="fine")])
        ax = fig.axes[0]
        first_line = next(ln for ln in ax.get_lines() if ln.get_label().startswith("V(x)"))
        assert first_line.get_linestyle() == "-"
        plt.close(fig)


class TestSolutionSeriesResult:
    """Tests that SolutionSeries carries a SolutionResult."""

    def test_solution_series_has_result_field(self) -> None:
        """SolutionSeries constructed with result field is accessible via .result."""
        model = _make_model()
        dof_map = build_dof_map(model)
        result_stub = SolutionResult(
            displacements=np.zeros(dof_map.total_dofs),
            reactions=np.zeros(0),
            dof_map=dof_map,
            model=model,
        )
        er = make_element_result()
        sol = SolutionSeries(
            label="test",
            element_results=(er,),
            model=model,
            result=result_stub,
        )
        assert sol.result is result_stub
