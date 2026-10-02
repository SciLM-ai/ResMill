"""The thin-barrier factor of resmill.layers.drapes, which the mud drapes and the lobes' thin interlobe caps share."""
import numpy as np
import pytest

from resmill.layers.drapes import MULTIPLIER_FLOOR, drape_faces, thin_barrier


def test_thin_barrier_is_the_series_factor_of_a_mud_layer_replacing_some_of_the_rock():
    """M = 1 / (1 + (t / h)(ks / kd - 1)): 0.3 m of 0.01 mD mud between two 1 m cells whose harmonic mean is 100 mD is
    1 / (1 + 0.3 x 9999) = 3.3337e-4; a layer thicker than the cell counts as the cell (1 / 10000 = 1e-4)."""
    assert thin_barrier(0.3, 1.0, 100.0, 0.01) == pytest.approx(1.0 / (1.0 + 0.3 * 9999.0))
    assert thin_barrier(5.0, 1.0, 100.0, 0.01) == pytest.approx(1.0e-4)
    assert thin_barrier(0.0, 1.0, 100.0, 0.01) == 1.0


def test_thin_barrier_is_kept_between_the_floor_and_one_and_takes_arrays():
    """A seal of 1e-18 is held at the floor of 1e-12, mud more permeable than the rock (ks < kd) at 1, and arrays of
    thicknesses and permeabilities broadcast."""
    assert thin_barrier(1.0, 1.0, 1.0e6, 1.0e-12) == MULTIPLIER_FLOOR
    assert thin_barrier(0.5, 1.0, 1.0, 10.0) == 1.0
    out = thin_barrier(np.array([0.1, 0.5, 2.0]), 1.0, np.array([10.0, 10.0, 10.0]), 0.1)
    np.testing.assert_allclose(out, 1.0 / (1.0 + np.array([0.1, 0.5, 1.0]) * 99.0))


def test_a_drape_face_is_the_thin_barrier_of_its_cells():
    """Two sand cells one above the other, the upper of the younger storey and its column draped: the multiplier on the
    face between them is the barrier of the drape's 0.2 m of 1e-3 mD in 2 m cells of 300 and 100 mD across the face
    (harmonic mean 150 mD), and no other face (lateral, or the one below the lower cell) is touched."""
    facies = np.full((1, 1, 2), 4, dtype=int)
    storey = np.array([[[0, 1]]])                        # k up: the cell above (k = 1) is the younger storey
    draped = np.ones((2, 1, 1), dtype=bool)
    kz = np.array([[[300.0, 100.0]]])
    mult_x, mult_y, mult_z = drape_faces(facies, storey, draped, (kz, kz, kz), (50.0, 50.0, 2.0), 0.2, 1.0e-3)
    assert mult_z[0, 0, 1] == pytest.approx(thin_barrier(0.2, 2.0, 150.0, 1.0e-3)) and mult_z[0, 0, 0] == 1.0
    assert (mult_x == 1.0).all() and (mult_y == 1.0).all()
