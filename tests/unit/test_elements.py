"""Tests for element stiffness matrices and load vectors -- TDD first."""
from __future__ import annotations
import numpy as np
import pytest
from fea_solver.models import (
    Element, ElementType, MaterialProperties, Node, DistributedLoad, LoadType
)
from fea_solver.elements import (
    bar_stiffness_matrix, beam_stiffness_matrix, frame_stiffness_matrix,
    element_stiffness_matrix, beam_consistent_load_vector, element_load_vector,
)

def make_bar(L: float = 1.0, E: float = 1.0, A: float = 1.0) -> Element:
    """Create a bar element with given parameters."""
    mat = MaterialProperties(E=E, A=A, I=0.0)
    return Element(id=1, node_i=Node(1, 0.0), node_j=Node(2, L),
                   element_type=ElementType.BAR, material=mat)

def make_beam(L: float = 1.0, E: float = 1.0, I: float = 1.0) -> Element:
    """Create a beam element with given parameters."""
    mat = MaterialProperties(E=E, A=1.0, I=I)
    return Element(id=1, node_i=Node(1, 0.0), node_j=Node(2, L),
                   element_type=ElementType.BEAM, material=mat)

def make_frame(L: float = 1.0, E: float = 1.0, A: float = 1.0, I: float = 1.0) -> Element:
    """Create a frame element with given parameters."""
    mat = MaterialProperties(E=E, A=A, I=I)
    return Element(id=1, node_i=Node(1, 0.0), node_j=Node(2, L),
                   element_type=ElementType.FRAME, material=mat)


class TestBarStiffnessMatrix:
    """Tests for bar_stiffness_matrix computation."""
    def test_shape_is_2x2(self) -> None:
        """Bar stiffness matrix is 2x2."""
        k = bar_stiffness_matrix(make_bar())
        assert k.shape == (2, 2)

    def test_symmetry(self) -> None:
        """Bar stiffness matrix is symmetric."""
        k = bar_stiffness_matrix(make_bar(L=2.0, E=3.0, A=5.0))
        np.testing.assert_allclose(k, k.T)

    def test_known_values_unit(self) -> None:
        """Bar stiffness with E=A=L=1 gives known values."""
        # E=1, A=1, L=1 -> EA/L = 1
        k = bar_stiffness_matrix(make_bar(L=1.0, E=1.0, A=1.0))
        expected = np.array([[1, -1], [-1, 1]], dtype=float)
        np.testing.assert_allclose(k, expected)

    def test_known_values_scaled(self) -> None:
        """Bar stiffness with scaled parameters gives correct values."""
        # E=200e9, A=0.01, L=0.5 -> EA/L = 4e9
        k = bar_stiffness_matrix(make_bar(L=0.5, E=200.0e9, A=0.01))
        c = 4.0e9
        expected = np.array([[c, -c], [-c, c]])
        np.testing.assert_allclose(k, expected, rtol=1e-10)

    def test_row_sum_is_zero(self) -> None:
        """Bar stiffness matrix rows sum to zero."""
        # Each row must sum to zero (rigid body mode)
        k = bar_stiffness_matrix(make_bar(L=2.0, E=5.0, A=3.0))
        np.testing.assert_allclose(k.sum(axis=1), np.zeros(2), atol=1e-12)


class TestBeamStiffnessMatrix:
    """Tests for beam_stiffness_matrix computation."""
    def test_shape_is_4x4(self) -> None:
        """Beam stiffness matrix is 4x4."""
        k = beam_stiffness_matrix(make_beam())
        assert k.shape == (4, 4)

    def test_symmetry(self) -> None:
        """Beam stiffness matrix is symmetric."""
        k = beam_stiffness_matrix(make_beam(L=2.0, E=3.0, I=5.0))
        np.testing.assert_allclose(k, k.T, atol=1e-12)

    def test_known_values_unit(self) -> None:
        """Beam stiffness with E=I=L=1 gives known values."""
        # E=1, I=1, L=1 -> EI/L^3 = 1
        # DOF order: [v_i, theta_i, v_j, theta_j]
        k = beam_stiffness_matrix(make_beam(L=1.0, E=1.0, I=1.0))
        # k = EI/L^3 * [[12, 6L, -12, 6L],
        #                [6L, 4L^2, -6L, 2L^2],
        #                [-12, -6L, 12, -6L],
        #                [6L, 2L^2, -6L, 4L^2]]
        expected = np.array([
            [ 12,  6, -12,  6],
            [  6,  4,  -6,  2],
            [-12, -6,  12, -6],
            [  6,  2,  -6,  4],
        ], dtype=float)
        np.testing.assert_allclose(k, expected, atol=1e-12)

    def test_cantilever_tip_deflection(self) -> None:
        """Beam cantilever gives correct tip deflection formula."""
        # Fixed-free cantilever with P=1 at tip: v_tip = PL^3/3EI
        # With L=1, E=1, I=1: v_tip = 1/3
        # Free DOFs: v_j (index 2), theta_j (index 3) with fixed v_i=theta_i=0
        k = beam_stiffness_matrix(make_beam(L=1.0, E=1.0, I=1.0))
        k_ff = k[2:4, 2:4]  # free-free partition
        f_f = np.array([-1.0, 0.0])  # P=-1 (downward) at v_j, no moment
        u_f = np.linalg.solve(k_ff, f_f)
        np.testing.assert_allclose(u_f[0], -1.0 / 3.0, rtol=1e-10)


class TestFrameStiffnessMatrix:
    """Tests for frame_stiffness_matrix computation."""
    def test_shape_is_6x6(self) -> None:
        """Frame stiffness matrix is 6x6."""
        k = frame_stiffness_matrix(make_frame())
        assert k.shape == (6, 6)

    def test_symmetry(self) -> None:
        """Frame stiffness matrix is symmetric."""
        k = frame_stiffness_matrix(make_frame(L=2.0))
        np.testing.assert_allclose(k, k.T, atol=1e-12)

    def test_axial_block_matches_bar(self) -> None:
        """Frame axial block matches bar stiffness."""
        # DOF order: [u_i, v_i, theta_i, u_j, v_j, theta_j]
        # Axial block rows/cols at indices [0, 3]
        e = make_frame(L=1.0, E=1.0, A=1.0, I=1.0)
        k_frame = frame_stiffness_matrix(e)
        e_bar = Element(id=1, node_i=e.node_i, node_j=e.node_j,
                        element_type=ElementType.BAR, material=e.material)
        k_bar = bar_stiffness_matrix(e_bar)
        np.testing.assert_allclose(k_frame[np.ix_([0,3],[0,3])], k_bar, atol=1e-12)

    def test_bending_block_matches_beam(self) -> None:
        """Frame bending block matches beam stiffness."""
        # Bending block rows/cols at indices [1,2,4,5]
        e = make_frame(L=1.0, E=1.0, A=1.0, I=1.0)
        k_frame = frame_stiffness_matrix(e)
        e_beam = Element(id=1, node_i=e.node_i, node_j=e.node_j,
                         element_type=ElementType.BEAM, material=e.material)
        k_beam = beam_stiffness_matrix(e_beam)
        bending_idx = [1, 2, 4, 5]
        np.testing.assert_allclose(k_frame[np.ix_(bending_idx, bending_idx)], k_beam, atol=1e-12)


class TestElementStiffnessMatrixDispatch:
    """Tests for element_stiffness_matrix dispatcher."""
    def test_dispatches_bar(self) -> None:
        """Dispatcher routes bar elements correctly."""
        e = make_bar()
        k = element_stiffness_matrix(e)
        assert k.shape == (2, 2)

    def test_dispatches_beam(self) -> None:
        """Dispatcher routes beam elements correctly."""
        e = make_beam()
        k = element_stiffness_matrix(e)
        assert k.shape == (4, 4)

    def test_dispatches_frame(self) -> None:
        """Dispatcher routes frame elements correctly."""
        e = make_frame()
        k = element_stiffness_matrix(e)
        assert k.shape == (6, 6)


class TestBeamConsistentLoadVector:
    """Tests for beam_consistent_load_vector computation."""
    def test_uniform_load_shape(self) -> None:
        """Uniform load vector has shape 4."""
        e = make_beam(L=1.0)
        load = DistributedLoad(element_id=1, load_type=LoadType.DISTRIBUTED_Y, w_i=-1.0, w_j=-1.0)
        f = beam_consistent_load_vector(e, load)
        assert f.shape == (4,)

    def test_uniform_load_known_values(self) -> None:
        """Uniform load vector has known values."""
        # w=-1 N/m uniform, L=1: f = (wL/12)*[6, L, 6, -L]
        # = (-1*1/12)*[6, 1, 6, -1] = [-0.5, -1/12, -0.5, 1/12]
        e = make_beam(L=1.0)
        load = DistributedLoad(element_id=1, load_type=LoadType.DISTRIBUTED_Y, w_i=-1.0, w_j=-1.0)
        f = beam_consistent_load_vector(e, load)
        w, L = -1.0, 1.0
        expected = np.array([
            w * L / 2.0,
            w * L**2 / 12.0,
            w * L / 2.0,
            -w * L**2 / 12.0,
        ])
        np.testing.assert_allclose(f, expected, rtol=1e-10)

    def test_resultant_equals_total_load(self) -> None:
        """Load vector resultant equals total distributed load."""
        # Sum of transverse forces (indices 0, 2) must equal total load w*L
        w, L = -2000.0, 3.0
        e = make_beam(L=L)
        load = DistributedLoad(element_id=1, load_type=LoadType.DISTRIBUTED_Y, w_i=w, w_j=w)
        f = beam_consistent_load_vector(e, load)
        np.testing.assert_allclose(f[0] + f[2], w * L, rtol=1e-10)
