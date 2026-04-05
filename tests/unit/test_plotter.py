"""Tests for matplotlib plotting functions — TDD first."""
from __future__ import annotations
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
from fea_solver.models import ElementResult
from fea_solver.plotter import (
    plot_shear_force_diagram,
    plot_bending_moment_diagram,
    plot_deformed_shape,
)


def make_element_result(
    element_id: int = 1,
    x_start: float = 0.0,
    x_end: float = 1.0,
    n: int = 50,
) -> ElementResult:
    x = np.linspace(x_start, x_end, n)
    V = np.ones(n) * -1.0
    M = np.linspace(-1.0, 0.0, n)
    return ElementResult(
        element_id=element_id,
        axial_force=0.0,
        shear_forces=V,
        bending_moments=M,
        x_stations=x,
    )


class TestPlotShearForceDiagram:
    def test_returns_figure(self):
        er = [make_element_result()]
        fig = plot_shear_force_diagram(er, title="Test SFD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_multiple_elements(self):
        ers = [make_element_result(i+1, float(i), float(i+1)) for i in range(3)]
        fig = plot_shear_force_diagram(ers, title="Multi-element SFD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        er = [make_element_result()]
        out = tmp_path / "sfd.png"
        fig = plot_shear_force_diagram(er, title="Save Test", output_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotBendingMomentDiagram:
    def test_returns_figure(self):
        er = [make_element_result()]
        fig = plot_bending_moment_diagram(er, title="Test BMD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_multiple_elements(self):
        ers = [make_element_result(i+1, float(i), float(i+1)) for i in range(3)]
        fig = plot_bending_moment_diagram(ers, title="Multi-element BMD")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        er = [make_element_result()]
        out = tmp_path / "bmd.png"
        fig = plot_bending_moment_diagram(er, title="Save Test", output_path=out)
        assert out.exists()
        plt.close(fig)


class TestPlotDeformedShape:
    def test_returns_figure(self):
        er = [make_element_result()]
        fig = plot_deformed_shape(er)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
