"""Mud drapes at the bases of channel storeys: face transmissibility multipliers, off by default.

The numbers the tests are held against do not come from the code: Barton et al. (2010)'s 17 outcrop means, the series
resistance of a thin barrier, Sheppard's formula for the correlation of a thresholded Gaussian field, and the layer's
own flow-event bookkeeping read back cell by cell.
"""
import functools
import math

import numpy as np
import pytest
from scipy.special import betainc

from resmill.layers.channel import ChannelLayer
from resmill.layers.drapes import (COVERAGE_BETA, COVERAGE_RANGE, HOLE_RANGE_WIDTHS, MARGIN_BIAS_MAX, check_drapes,
                                   coverage_from_unit, draped_columns, drape_faces, sample_drapes, storey_map)


# Barton et al. 2010, AAPG Memoir 92 Fig. 10: mean share of an element base covered by a drape, one value per outcrop
# (17 outcrops, 154 elements), read off the figure.
BARTON_MEANS = np.array([0, 5, 29, 38, 49, 50, 53, 55, 60, 65, 70, 71, 72, 75, 79, 83, 92]) / 100.0


# --------------------------------------------------------------------------------------------------------------
# The settings of a reservoir
# --------------------------------------------------------------------------------------------------------------

def test_the_coverage_beta_has_the_moments_of_bartons_17_outcrops():
    """Beta(1.52, 1.21) has mean a / (a + b) and variance ab / ((a + b)^2 (a + b + 1)); they are the mean (0.556) and
    standard deviation (0.257) of the 17 outcrop means."""
    a, b = COVERAGE_BETA
    assert a / (a + b) == pytest.approx(BARTON_MEANS.mean(), abs=0.002)
    assert math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))) == pytest.approx(BARTON_MEANS.std(ddof=1), abs=0.002)


def test_sampled_coverage_is_that_beta_cut_to_the_span_of_the_outcrops():
    """Draws lie in 0.05-0.92 (the outcrops' span), each by the inverse CDF of the truncated Beta, never clipped. Their
    moments are those of the truncated density (integrated numerically), and still within 0.03 of Barton's."""
    a, b = COVERAGE_BETA
    u = (np.arange(40000) + 0.5) / 40000
    c = coverage_from_unit(u)
    assert COVERAGE_RANGE == (0.05, 0.92)
    assert c.min() >= 0.05 and c.max() <= 0.92 and np.all(np.diff(c) > 0)
    x = np.linspace(0.05, 0.92, 400001)
    w = x ** (a - 1) * (1 - x) ** (b - 1)
    mean = (x * w).sum() / w.sum()
    sd = math.sqrt(((x - mean) ** 2 * w).sum() / w.sum())
    assert c.mean() == pytest.approx(mean, abs=2e-3) and c.std() == pytest.approx(sd, abs=2e-3)
    assert abs(c.mean() - BARTON_MEANS.mean()) < 0.03 and abs(c.std() - BARTON_MEANS.std(ddof=1)) < 0.03
    assert (c < 0.3).mean() == pytest.approx(3 / 17, abs=0.05)            # three of the 17 outcrops lie below 0.3


def test_sample_drapes_draws_the_three_settings_of_a_reservoir():
    """Coverage as above, the margin bias uniform from none to 0.55 (which the axis-ness of real layers turns into Vento's
    0.33 against 0.67 between the axis and margin thirds), thickness log-uniform 0.1-1.5 m; three random numbers a
    reservoir, the same for the same generator state."""
    draws = [sample_drapes(np.random.default_rng(s)) for s in range(4000)]
    bias = np.array([d["margin_bias"] for d in draws])
    thick = np.log(np.array([d["thickness"] for d in draws]))
    assert MARGIN_BIAS_MAX == 0.55
    assert bias.min() >= 0.0 and bias.max() <= MARGIN_BIAS_MAX
    assert bias.mean() == pytest.approx(MARGIN_BIAS_MAX / 2, abs=0.01)
    assert bias.std() == pytest.approx(MARGIN_BIAS_MAX / math.sqrt(12), abs=0.005)
    assert thick.min() >= math.log(0.1) and thick.max() <= math.log(1.5)
    assert thick.mean() == pytest.approx(0.5 * (math.log(0.1) + math.log(1.5)), abs=0.05)
    assert all(COVERAGE_RANGE[0] <= d["coverage"] <= COVERAGE_RANGE[1] for d in draws)
    rng = np.random.default_rng(7)
    sample_drapes(rng)
    assert rng.random() == np.random.default_rng(7).random(4)[3]          # exactly three numbers drawn
    assert sample_drapes(np.random.default_rng(5)) == sample_drapes(np.random.default_rng(5))


def test_sample_drapes_can_also_draw_the_hole_size_in_channel_widths():
    """Opt in with ``hole_widths``: the owner's G-S11 draws the holes' practical range log-uniform over 0.25-4 channel widths
    per reservoir, with one more random number; the other three settings are those of the same generator without it."""
    assert HOLE_RANGE_WIDTHS == (0.25, 4.0)
    draws = [sample_drapes(np.random.default_rng(s), hole_widths=HOLE_RANGE_WIDTHS) for s in range(4000)]
    holes = np.log([d["hole_range_widths"] for d in draws])
    assert holes.min() >= math.log(0.25) and holes.max() <= math.log(4.0)
    assert holes.mean() == pytest.approx(0.5 * (math.log(0.25) + math.log(4.0)), abs=0.05)
    assert holes.std() == pytest.approx((math.log(4.0) - math.log(0.25)) / math.sqrt(12), abs=0.05)
    rng = np.random.default_rng(7)
    got = sample_drapes(rng, hole_widths=HOLE_RANGE_WIDTHS)
    assert rng.random() == np.random.default_rng(7).random(5)[4]                  # four numbers drawn, not three
    assert {k: v for k, v in got.items() if k != "hole_range_widths"} == sample_drapes(np.random.default_rng(7))
    assert "hole_range_widths" not in sample_drapes(np.random.default_rng(7))


# --------------------------------------------------------------------------------------------------------------
# Storeys
# --------------------------------------------------------------------------------------------------------------

def test_storey_map_is_the_level_of_each_cells_flow_event():
    """A cell's storey is the level its event pair maps to in ``event_group``; cells outside the mask, and events the
    map does not know, are -1 (as the per-cell lookup it replaces gave)."""
    rng = np.random.default_rng(0)
    pairs = [(np.float32(p), np.float32(q)) for p, q in zip(rng.normal(1, 0.04, 9), rng.normal(0, 0.1, 9))]
    group = {pair: level // 3 for level, pair in enumerate(pairs[:8])}      # the ninth event is unknown
    pick = rng.integers(0, 9, (7, 5, 4))
    pm = np.array([p[0] for p in pairs], dtype=np.float32)[pick]
    po = np.array([p[1] for p in pairs], dtype=np.float32)[pick]
    mask = rng.random(pick.shape) < 0.7
    expected = np.full(pick.shape, -1)
    for idx in zip(*np.nonzero(mask)):
        expected[idx] = group.get((pm[idx], po[idx]), -1)
    got = storey_map(mask, pm, po, group)
    assert np.array_equal(got, expected)
    assert (got[pick == 8] == -1).all() and (got[~mask] == -1).all() and got.max() == 2
    assert (storey_map(mask, pm, po, {}) == -1).all() and (storey_map(mask, pm, po, None) == -1).all()
    assert np.array_equal(storey_map(mask, pm.astype(np.float64), po.astype(np.float64), group), expected)   # float32 keys
    pairs_in_c_order = np.asfortranarray(pm), np.asfortranarray(po)                                         # any memory layout
    assert np.array_equal(storey_map(mask, *pairs_in_c_order, group), expected)


# --------------------------------------------------------------------------------------------------------------
# Which columns of a storey's base are draped
# --------------------------------------------------------------------------------------------------------------

def _one_storey(nx=120, ny=120, axis=None):
    """One storey whose fill is one cell thick over the whole map, with its axis-ness ``axis`` (0 margin, 1 axis)."""
    fill = np.zeros((nx, ny, 3), dtype=bool)
    fill[:, :, 1] = True
    dn = np.full(fill.shape, 0.5, dtype=np.float32)
    if axis is not None:
        dn[:, :, 1] = axis
    return fill, np.where(fill, 0, -1), dn


@pytest.mark.parametrize("coverage", [0.05, 0.3, 0.6, 0.92])
def test_coverage_is_the_share_of_the_footprint_that_is_draped(coverage):
    """With no margin bias the draped share of a storey's base is the coverage, to one column (ranks of a smooth
    field below the coverage), whatever the hole size; a storey that does not exist has no drape."""
    fill, storey, dn = _one_storey()
    for seed, sigma in ((0, (1.0, 1.0)), (1, (6.0, 6.0)), (2, (15.0, 15.0))):
        draped = draped_columns(fill, storey, dn, coverage, 0.0, sigma, np.random.default_rng(seed))
        assert draped.shape == (1, 120, 120)
        assert draped[0].mean() == pytest.approx(coverage, abs=1.0 / 14400)
    fill[:, :60] = False
    draped = draped_columns(fill, np.where(fill, 0, -1), dn, coverage, 0.0, (3.0, 3.0), np.random.default_rng(3))
    assert not draped[0][:, :60].any() and draped[0][:, 60:].mean() == pytest.approx(coverage, abs=1.0 / 7200)


def _many_storeys(levels, nx=24, ny=24):
    """``levels`` storeys, each one fill cell thick over the whole map, one above the other."""
    fill = np.ones((nx, ny, levels), dtype=bool)
    return fill, np.broadcast_to(np.arange(levels), fill.shape), np.full(fill.shape, 0.5, dtype=np.float32)


def test_each_storeys_coverage_is_a_beta_draw_that_holds_the_mean():
    """With ``concentration`` k each storey's own share is drawn from Beta(c k, (1 - c) k): mean c, variance
    c (1 - c) / (k + 1). Barton et al. (2010): drapes continuous in about 25 % of the channel elements, at their
    mean coverage of 0.5565 (read as: coverage of 0.95 or more) is k = 0.71. A very large k leaves every storey at c."""
    fill, storey, dn = _many_storeys(800)
    c, k = 0.5565, 0.71
    share = draped_columns(fill, storey, dn, c, 0.0, (1.0, 1.0), np.random.default_rng(0), concentration=k).mean(axis=(1, 2))
    assert share.mean() == pytest.approx(c, abs=0.04)
    assert share.var() == pytest.approx(c * (1 - c) / (k + 1), abs=0.02)
    assert (share >= 0.95).mean() == pytest.approx(1 - betainc(c * k, (1 - c) * k, 0.95), abs=0.04) == pytest.approx(0.25, abs=0.05)
    assert (share <= 0.05).mean() > 0.1 and share.min() == 0.0 and share.max() == 1.0       # bare and sealed storeys both occur
    tight = draped_columns(fill, storey, dn, c, 0.0, (1.0, 1.0), np.random.default_rng(1), concentration=1e9).mean(axis=(1, 2))
    assert np.abs(tight - c).max() < 2.0 / 576
    for edge in (0.0, 1.0):                                             # no spread to draw: the Beta would have a parameter 0
        shares = draped_columns(fill, storey, dn, edge, 0.0, (1.0, 1.0), np.random.default_rng(2), concentration=k).mean(axis=(1, 2))
        assert np.all(shares == edge)


def test_without_a_concentration_nothing_more_is_drawn_and_every_storey_gets_the_coverage():
    """Off (None) the stream is what it was: one noise field per storey and nothing else, so the next number is the one
    a fresh generator gives after that field; every storey is covered to the share asked for."""
    fill, storey, dn = _one_storey(40, 30)
    rng, ref = np.random.default_rng(3), np.random.default_rng(3)
    draped = draped_columns(fill, storey, dn, 0.4, 0.0, (3.0, 3.0), rng)
    ref.standard_normal((40, 30))
    assert rng.random() == ref.random() and draped[0].mean() == pytest.approx(0.4, abs=1.0 / 1200)
    rng = np.random.default_rng(3)
    draped_columns(fill, storey, dn, 0.4, 0.0, (3.0, 3.0), rng, concentration=1.0)
    assert rng.random() != ref.random()


def test_the_margin_bias_moves_the_drape_from_the_axis_to_the_margin():
    """With bias b, the covered share of columns near the axis falls below that near the margin by b times the
    difference of their mean axis-ness: 0.8 b between the bands below 0.2 and above 0.8 of a uniform axis-ness. The
    mean coverage stays what was asked, also where the shares are clipped at 0 and 1."""
    nx = 200
    axis = np.broadcast_to(np.linspace(0.0, 1.0, nx)[:, None], (nx, 100))    # axis-ness grows along x
    fill, storey, dn = _one_storey(nx, 100, axis=axis)
    hi, lo = np.zeros((nx, 100), dtype=bool), np.zeros((nx, 100), dtype=bool)
    hi[160:], lo[:40] = True, True
    for coverage, bias in ((0.5, 0.3), (0.3, 0.0), (0.5, 0.0), (0.9, 0.34), (0.1, 0.34)):
        rows = []
        for seed in range(40):
            d = draped_columns(fill, storey, dn, coverage, bias, (0.8, 0.8), np.random.default_rng(seed))[0]
            rows.append((d.mean(), d[lo].mean() - d[hi].mean()))
        mean, contrast = np.mean(rows, axis=0)
        assert mean == pytest.approx(coverage, abs=0.005)
        if (coverage, bias) == (0.5, 0.3):
            assert contrast == pytest.approx(0.8 * bias, abs=0.02)
        elif bias == 0.0:
            assert contrast == pytest.approx(0.0, abs=0.02)
        else:
            assert 0.0 < contrast <= 0.8 * bias + 0.02                      # clipping can only flatten it


def test_holes_have_the_set_range():
    """The smooth field behind the map has the Gaussian shape exp(-h^2 / 4 s^2) with s the practical range (the
    distance where correlation is 5 %) over 2 sqrt 3. At coverage 0.5 the drape is the field's sign, whose correlation
    at lag h is (2 / pi) arcsin of the field's (Sheppard)."""
    n, rng_cells = 360, 24.0
    fill = np.ones((n, n, 1), dtype=bool)
    storey = np.zeros((n, n, 1), dtype=int)
    dn = np.full(fill.shape, 0.5, dtype=np.float32)
    s = rng_cells / (2.0 * math.sqrt(3.0))
    got, expected = {}, {}
    for lag in (6, 12, 24):
        vals = []
        for seed in range(6):
            d = draped_columns(fill, storey, dn, 0.5, 0.0, (s, s), np.random.default_rng(seed))[0].astype(float)
            vals.append(np.corrcoef(d[:-lag].ravel(), d[lag:].ravel())[0, 1])
        got[lag] = np.mean(vals)
        expected[lag] = (2.0 / math.pi) * math.asin(math.exp(-lag ** 2 / (4.0 * s ** 2)))
        assert got[lag] == pytest.approx(expected[lag], abs=0.04)
    assert got[6] > got[12] > got[24] and got[24] < 0.1


# --------------------------------------------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------------------------------------------

SIZE = (10.0, 10.0, 2.0)
KS, KD, T = 100.0, 0.01, 0.5


def _series(k1, k2, kd, t, h):
    """Transmissibility of the face of two cells of size h with a drape of thickness t and permeability kd, over the
    one without: resistances in series, half a cell each side less half the drape."""
    return (h / 2 / k1 + h / 2 / k2) / ((h - t) / 2 / k1 + (h - t) / 2 / k2 + t / kd)


def _perms(shape, k=KS):
    return tuple(np.full(shape, k) for _ in range(3))


def _stack_cube():
    """Six columns by four rows by six layers (k up). Storey 0: sand of layers 0-1 everywhere but column 5, plus
    layer 2 in columns 0-1. Storey 1: channel fill of layers 2-3 in columns 2-5. Everything else is mud."""
    f = np.full((6, 4, 6), -1, dtype=np.int8)
    s = np.full(f.shape, -1)
    f[:5, :, :2], s[:5, :, :2] = 4, 0
    f[:2, :, 2], s[:2, :, 2] = 4, 0
    f[2:, :, 2:4], s[2:, :, 2:4] = 4, 1
    return f, s


def test_a_drape_sits_only_on_the_faces_between_a_younger_fill_and_older_sand():
    """Full coverage: the base of the younger fill over older sand (columns 2-4, layer 2, the lower face) and its wall
    against older sand (column 1 against 2, layers 2) get the thin-barrier multiplier; faces inside one storey, sand
    over mud (column 5), mud against sand, and the older sand's own faces stay 1."""
    f, s = _stack_cube()
    draped = np.ones((2, 6, 4), dtype=bool)
    mx, my, mz = drape_faces(f, s, draped, _perms(f.shape), SIZE, T, KD)
    m_z = _series(KS, KS, KD, T, 2.0)
    m_x = _series(KS, KS, KD, T, 10.0)
    expect_z = np.ones(f.shape)
    expect_z[2:5, :, 2] = m_z                         # lower face of layer 2 over the older slab (not column 5: mud)
    expect_x = np.ones(f.shape)
    expect_x[1, :, 2] = m_x                           # the +x face of column 1 (older) against column 2 (younger)
    assert np.allclose(mz, expect_z, rtol=1e-5) and np.allclose(mx, expect_x, rtol=1e-5)
    assert np.all(my == 1.0)
    assert mx.dtype == mz.dtype == np.float32 and mx.shape == f.shape
    assert m_z == pytest.approx(1 / (1 + (T / 2.0) * (KS / KD - 1)), rel=1e-12)       # the closed form, by hand
    assert m_z == pytest.approx(3.9988e-4, rel=1e-4) and m_x == pytest.approx(1.9962e-3, rel=1e-4)


def test_a_younger_levee_or_splay_makes_no_drape_and_coverage_zero_none():
    """A drape belongs to the base of a channel fill: a younger levee (or splay) on older sand leaves every face 1;
    so does a map with no covered column."""
    f, s = _stack_cube()
    levee = f.copy()
    levee[2:, :, 2:4] = 2                              # the younger body is a levee now
    out = drape_faces(levee, s, np.ones((2, 6, 4), dtype=bool), _perms(f.shape), SIZE, T, KD)
    assert all(np.all(a == 1.0) for a in out)
    out = drape_faces(f, s, np.zeros((2, 6, 4), dtype=bool), _perms(f.shape), SIZE, T, KD)
    assert all(np.all(a == 1.0) for a in out)


def test_only_the_base_of_a_younger_fill_is_a_base_and_unknown_storeys_make_no_drape():
    """A younger fill that lies below older sand has its top, not its base, against it: no vertical drape (its walls
    against older sand are still draped). Sand of an unknown storey (-1) makes no drape on either side."""
    f = np.full((3, 1, 6), -1, dtype=np.int8)
    s = np.full(f.shape, -1)
    f[0, :, 2:4], s[0, :, 2:4] = 4, 0                  # older sand above ...
    f[0, :, 0:2], s[0, :, 0:2] = 4, 1                  # ... a younger fill in column 0
    f[1, :, 0:2], s[1, :, 0:2] = 4, 0                  # older sand beside the younger fill (a wall) in column 1
    f[2, :, 0:2], s[2, :, 0:2] = 4, -1                 # sand of an unknown storey in column 2, beside column 1 only
    mx, _, mz = drape_faces(f, s, np.ones((2, 3, 1), dtype=bool), _perms(f.shape), SIZE, T, KD)
    assert np.all(mz == 1.0)                            # the younger fill's top meets older sand: not a base
    assert np.all(mx[0, 0, 0:2] < 2e-3) and np.all(mx[0, 0, 2:] == 1.0)    # its wall against column 1 is draped
    assert np.all(mx[1] == 1.0)                         # unknown beside known: nothing


def test_without_a_known_storey_nothing_is_draped_and_nothing_fails():
    """With no event record every storey is unknown (-1), so no cell is a younger fill over older sand: no column is draped
    and no face gets a multiplier (it used to raise an IndexError on the empty storey axis)."""
    f, _ = _stack_cube()
    unknown = np.full(f.shape, -1)
    draped = draped_columns((f == 3) | (f == 4), unknown, np.full(f.shape, 0.5, dtype=np.float32), 0.5, 0.0, (1.0, 1.0),
                            np.random.default_rng(0))
    out = drape_faces(f, unknown, draped, _perms(f.shape), SIZE, T, KD)
    assert not draped.any() and all(np.all(a == 1.0) for a in out)


def test_each_draped_column_decides_for_its_own_storey():
    """The drape of a face is read off the map of the younger cell's storey and column: with the map covering only
    columns 3-4, the base of columns 2 stays open while 3 and 4 are draped, and the wall against column 1 (column 2's
    own map) is open too."""
    f, s = _stack_cube()
    draped = np.zeros((2, 6, 4), dtype=bool)
    draped[1, 3:5] = True
    mx, my, mz = drape_faces(f, s, draped, _perms(f.shape), SIZE, T, KD)
    assert np.all(mz[2, :, 2] == 1.0) and np.all(mz[3:5, :, 2] < 1e-3)
    assert np.all(mx == 1.0)


def test_a_later_storey_erodes_the_drape_it_cuts_through():
    """A third storey cuts into the second where it lies on the first (columns 2-3): the second storey's drape is gone
    with its cells there, and the new contact takes the third storey's map -- open where that map is, draped where it
    is. Where the second storey survives (columns 0-1) its own drape stays."""
    f = np.full((6, 1, 6), -1, dtype=np.int8)
    s = np.full(f.shape, -1)
    f[:, :, :2], s[:, :, :2] = 4, 0                    # storey 0, a slab
    f[:4, :, 2:4], s[:4, :, 2:4] = 4, 1                # storey 1, columns 0-3
    f[2:, :, 2:5], s[2:, :, 2:5] = 4, 2                # storey 2 cuts down columns 2-3 and spreads over 4-5
    draped = np.zeros((3, 6, 1), dtype=bool)
    draped[1], draped[2, 4:] = True, True              # storey 1 draped everywhere, storey 2 only in columns 4-5
    _, _, mz = drape_faces(f, s, draped, _perms(f.shape), SIZE, T, KD)
    assert np.all(mz[:2, 0, 2] < 1e-3)                 # storey 1's drape survives on storey 0 (columns 0-1)
    assert np.all(mz[2:4, 0, 2] == 1.0)                # eroded: storey 2's map is open there
    assert np.all(mz[4:, 0, 2] < 1e-3)                 # storey 2's own drape on storey 0
    mx, _, _ = drape_faces(f, s, np.ones((3, 6, 1), dtype=bool), _perms(f.shape), SIZE, T, KD)
    assert np.all(mx[1, 0, 2:4] < 2e-3)                # wall of storey 2 (column 2) against storey 1 (column 1)
    _, _, mz = drape_faces(f, s, np.ones((3, 6, 1), dtype=bool), _perms(f.shape), SIZE, T, KD)
    assert np.all(mz[:, 0, 2] < 1e-3)


def test_the_multiplier_is_the_thin_barrier_between_the_two_cells_permeabilities():
    """Across a face, cells of 100 and 400 mD, size h and a drape of thickness t and kd: the series resistance of the
    cells with t of rock replaced by the drape; monotone in t (up to h), kd and the cell size; capped when the drape
    is thicker than the cell; kept in [1e-12, 1]; the permeability across the face is that along its axis."""
    f, s = _stack_cube()
    perms = (np.full(f.shape, 100.0), np.full(f.shape, 100.0), np.full(f.shape, 100.0))
    perms[2][:, :, 2:] = 400.0                                                # PERMZ differs above layer 2
    perms[0][2:, :, :] = 400.0
    draped = np.ones((2, 6, 4), dtype=bool)
    mx, _, mz = drape_faces(f, s, draped, perms, SIZE, T, KD)
    assert mz[3, 0, 2] == pytest.approx(_series(400.0, 100.0, KD, T, 2.0), rel=1e-4)        # PERMZ above, PERMZ below
    assert mx[1, 0, 2] == pytest.approx(_series(100.0, 400.0, KD, T, 10.0), rel=1e-4)       # PERMX of columns 1 and 2
    m_t = [drape_faces(f, s, draped, perms, SIZE, t, KD)[2][3, 0, 2] for t in (0.1, 0.5, 1.0, 2.0, 5.0)]
    assert m_t[:4] == sorted(m_t[:4], reverse=True) and len(set(m_t[:4])) == 4        # thicker drape, tighter face
    assert m_t[4] == pytest.approx(m_t[3])                                            # thicker than the cell: capped
    assert m_t[3] == pytest.approx(KD / (2 * 400.0 * 100.0 / 500.0), rel=1e-5)        # t = h: kd over the harmonic ks
    ms = [drape_faces(f, s, draped, perms, SIZE, T, kd)[2][3, 0, 2] for kd in (1e-6, 1e-3, 1e-1, 10.0)]
    assert ms == sorted(ms) and ms[0] < ms[-1] < 1.0                                  # a leakier drape, a leakier face
    assert drape_faces(f, s, draped, perms, SIZE, T, 1e-30)[2][3, 0, 2] == pytest.approx(1e-12)    # the floor
    for kd in (160.0, 1e9):                          # a drape as permeable as the sand or more: open, never above 1
        assert np.allclose(drape_faces(f, s, draped, perms, SIZE, T, kd)[2][2:5, :, 2], 1.0, atol=1e-6)


def test_the_sealing_statement_holds_for_vertical_faces_and_only_chokes_wide_lateral_ones():
    """Against sand of 400 mD, a mud of 4e-6 to 1e-3 mD gives M below 2e-4 on a vertical face for any thickness from 0.1 m in
    cells up to 5 m; across a lateral face of a 100 m cell a drape of 0.1 m gives 1.5e-4 for the anchor's FF mud (6e-5 mD) and
    2.5e-3 for 1e-3 mD, a four hundred fold choke that is not a seal."""
    f, s = _stack_cube()
    draped = np.ones((2, 6, 4), dtype=bool)
    for kd in (4e-6, 6e-5, 1e-3):
        for t, h in ((0.1, 5.0), (0.1, 1.0), (1.5, 5.0), (0.5, 0.5)):
            assert drape_faces(f, s, draped, _perms(f.shape, 400.0), (10.0, 10.0, h), t, kd)[2][2:5, :, 2].max() < 2e-4
    for kd, expected in ((6e-5, 1.5e-4), (1e-3, 2.5e-3)):
        mx = drape_faces(f, s, draped, _perms(f.shape, 400.0), (100.0, 100.0, 2.0), 0.1, kd)[0]
        assert mx[1, :, 2] == pytest.approx(expected, rel=0.02)


# --------------------------------------------------------------------------------------------------------------
# On a channel layer
# --------------------------------------------------------------------------------------------------------------

GEOLOGY = dict(nlevel=6, ntime=6, ntime_per_level=True, level_inherit=1.0, probAvulOutside=0.0, probAvulInside=0.0,
               mCHdepth=3.0, mCHwdratio=16.0, NTGtarget=0.9, mdistMigrate=12.0, mLVwidth=100.0, mCSnum=0.5,
               stdevCHsource=40.0)
MUD = {-1: {"poro": 0.12, "log10_perm": -3.0}}


def _build(seed=5, **kw):
    layer = ChannelLayer(nx=100, ny=70, nz=12, x_len=1000.0, y_len=700.0, z_len=12.0, top_depth=2000.0)
    layer.create_geology(seed=seed, **{"facies_props": MUD, **GEOLOGY, **kw})
    return layer


@functools.lru_cache(maxsize=None)
def _layer(seed=5, **drapes):
    """A small deepwater-like layer, built once per setting; without drapes when none are given."""
    return _build(seed=seed, **({"drapes": drapes} if drapes else {}))


def _storeys(layer, levels=None):
    """Every sand cell's level, looked up cell by cell in an event record (the engine's own unless ``levels`` is given;
    the slow, plain way)."""
    eng = layer._engine
    levels = eng.event_levels if levels is None else levels
    pm, po = np.asarray(eng.poro_mult_field), np.asarray(eng.log_perm_offset_field)
    storey = np.full(pm.shape, -1)
    for idx in zip(*np.nonzero(layer.facies >= 1)):
        storey[idx] = levels.get((pm[idx], po[idx]), -1)
    return storey


def _global_state():
    """numpy's global generator state in full: the key array, the position in it and the Gaussian cache."""
    _, key, pos, has_gauss, cached = np.random.get_state()
    return key.tobytes(), pos, has_gauss, cached


EVENT_SDS = dict(event_poro_sd=2.0, event_log_perm_sd=0.01)       # nearly every event clips to a corner pair: pairs repeat


def test_a_cells_storey_is_looked_up_once_for_levee_fading_and_for_drapes(monkeypatch):
    """Levee fading and drapes both need every sand cell's storey; the lookup (seconds on a million-cell layer) runs once when
    either is on, also when both are, and not at all when neither is."""
    from resmill.layers import channel
    calls, real = [], channel.storey_map
    monkeypatch.setattr(channel, "storey_map", lambda *a, **k: calls.append(1) or real(*a, **k))
    levee = dict(levee_ntg_decay_m=150.0, facies_props={**MUD, 2: {"ntg": 0.55, "bed_poro": 0.27, "bed_log10_perm": 2.0}})
    drapes = dict(drapes=dict(coverage=0.5))
    for kw, expected in (({}, 0), (drapes, 1), (levee, 1), ({**levee, **drapes}, 1)):
        calls.clear()
        _build(**kw)
        assert len(calls) == expected, kw


def test_distinct_events_give_every_sand_cell_the_level_of_the_event_that_stamped_it(monkeypatch):
    """An event's (poro_mult, log_perm_offset) pair is clipped at +-2 sd in both components, so now and then two events
    draw the same pair and the record (pair -> level) keeps the later one's level: the cells of the earlier event, of
    another level, then carry the wrong storey. ``distinct_events`` makes each event's pair its own and changes nothing
    else. The reference is the same build with a counter added to every event's poro_mult (steps far above a float32
    ulp): each cell then holds the pair of the one event that stamped it, whose level the counter's own record gives."""
    from resmill.layers._fluvial import fluvial
    draw, exact = fluvial._draw_event_mults, {}

    def counted(self):
        pm, po = draw(self)
        pm += 1e-3 * (len(exact) + 1)
        exact[(np.float32(pm), np.float32(po))] = self._level
        return pm, po

    default = _build(**EVENT_SDS)
    state = _global_state()
    monkeypatch.setattr(fluvial, "_draw_event_mults", counted)
    reference = _build(**EVENT_SDS)
    monkeypatch.undo()
    distinct = _build(distinct_events=True, **EVENT_SDS)
    sand = np.asarray(reference.facies) >= 1
    assert np.array_equal(default.facies, reference.facies) and np.array_equal(distinct.facies, reference.facies)
    assert _global_state() == state                                            # no random number drawn differently
    truth = _storeys(reference, exact)[sand]
    assert truth.min() >= 0 and truth.max() >= 4                               # every cell's event is known; many levels
    assert len(distinct._engine.event_levels) == len(exact) > len(default._engine.event_levels)   # one entry per event
    assert np.array_equal(_storeys(distinct)[sand], truth)
    assert (_storeys(default)[sand] != truth).sum() > 100                      # without it, many cells carry another level


def test_drapes_are_off_by_default_and_change_no_rock():
    """Without ``drapes`` there are no multipliers; with them the facies, porosity and permeability are the same arrays
    and the global random stream ends where it would have, so a run with and without drapes shares one geology."""
    off = _build()
    after_off = _global_state()
    on = _build(drapes=dict(coverage=0.6, margin_bias=0.2))
    after_on = _global_state()
    assert off.mult_x is off.mult_y is off.mult_z is None
    assert on.mult_x.shape == on.mult_y.shape == on.mult_z.shape == on.facies.shape
    for name in ("facies", "poro_mat", "perm_mat", "active", "kvkh_mat", "kx_mult", "ky_mult"):
        a, b = getattr(off, name), getattr(on, name)
        assert (a is None and b is None) or np.array_equal(a, b), name
    assert after_off == after_on                     # key, position and Gaussian cache: not one number drawn more
    again = _build(drapes=dict(coverage=0.6, margin_bias=0.2))
    assert all(np.array_equal(getattr(again, k), getattr(on, k)) for k in ("mult_x", "mult_y", "mult_z"))
    assert np.array_equal(_layer(coverage=0.6, margin_bias=0.2).mult_z, on.mult_z)


def test_without_a_seed_the_drapes_follow_numpys_global_state_and_leave_it_alone():
    """``seed=None`` makes the engine follow numpy's global generator, so a caller who seeds that gets the same sand; the
    drapes then repeat too (they used to draw fresh entropy), and leave the generator where the same build without drapes
    leaves it."""
    def built(**kw):
        np.random.seed(7)
        return _build(seed=None, **kw), _global_state()

    (a, state_a), (b, state_b), (off, state_off) = built(drapes=dict(coverage=0.6)), built(drapes=dict(coverage=0.6)), built()
    assert np.array_equal(a.facies, b.facies) and np.array_equal(a.facies, off.facies)
    assert all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("mult_x", "mult_y", "mult_z"))
    assert (np.asarray(a.mult_z) < 1.0).sum() > 100
    assert state_a == state_b == state_off


def test_full_coverage_drapes_every_contact_of_a_younger_fill_with_older_sand_and_nothing_else():
    """The multipliers are below 1 exactly on the faces between a younger fill cell (CH or LA) and an older-storey sand
    cell: the lower face of an upper younger cell, or either side wall of a younger cell. Storeys from the engine's own
    event record."""
    layer = _layer(coverage=1.0)
    fac, storey = np.asarray(layer.facies), _storeys(layer)
    fill = (fac == 3) | (fac == 4)
    sand = fac >= 1
    n_faces = {}
    for name, axis in (("mult_x", 0), ("mult_y", 1), ("mult_z", 2)):
        a, b = [[slice(None)] * 3 for _ in range(2)]
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        both = sand[a] & sand[b] & (storey[a] >= 0) & (storey[b] >= 0) & (storey[a] != storey[b])
        young_hi = (storey[b] > storey[a]) & fill[b]
        young_lo = (storey[a] > storey[b]) & fill[a]
        flag = both & (young_hi if axis == 2 else (young_hi | young_lo))
        expected = np.zeros(fac.shape, dtype=bool)
        expected[b if axis == 2 else a] = flag                       # a z face is held by the cell above (k up)
        assert np.array_equal(np.asarray(getattr(layer, name)) < 1.0, expected), name
        n_faces[name] = int(flag.sum())
    assert n_faces["mult_z"] > 100 and n_faces["mult_x"] + n_faces["mult_y"] > 100       # both kinds of contact exist
    assert np.asarray(layer.mult_z)[:, :, 0].min() == 1.0                          # nothing under the layer's floor
    assert min(float(np.asarray(getattr(layer, k)).min()) for k in ("mult_x", "mult_y", "mult_z")) >= 1e-12


def test_a_draped_faces_multiplier_follows_the_floodplain_muds_permeability():
    """The drape takes the FF mud's permeability (``facies_props[-1]``, 1e-3 mD here) unless ``perm`` says otherwise:
    a draped face is the thin barrier between its two cells' permeabilities across it (series resistance), so it is
    sealed (far below 0.01). A silty drape of 1 mD only chokes it."""
    layer = _layer(coverage=1.0, thickness=0.5)
    kz = np.asarray(layer.perm_mat, dtype=float) * float(layer.kzkx)    # no kvkh here: PERMZ = kzkx x PERMX
    mz = np.asarray(layer.mult_z, dtype=float)
    faces = np.argwhere(mz < 1.0)
    assert len(faces) > 100
    for i, j, k in faces[::max(1, len(faces) // 40)]:
        assert mz[i, j, k] == pytest.approx(_series(kz[i, j, k], kz[i, j, k - 1], 1e-3, 0.5, layer.dz), rel=1e-4)
    assert mz[mz < 1.0].max() < 0.01
    silty = np.asarray(_layer(coverage=1.0, thickness=0.5, perm=1.0).mult_z, dtype=float)
    assert np.array_equal(silty < 1.0, mz < 1.0) and silty[silty < 1.0].min() > mz[mz < 1.0].max()


def test_drapes_cover_about_the_coverage_of_the_contacts():
    """A storey's drape covers the coverage of its base columns; of the faces where a younger fill meets older sand
    (all of them draped at coverage 1) about that share is draped, and only those faces."""
    layer, full = _layer(coverage=0.6, hole_range_m=40.0), _layer(coverage=1.0)
    draped = np.asarray(layer.mult_z) < 1.0
    possible = np.asarray(full.mult_z) < 1.0
    assert not (draped & ~possible).any()
    assert draped.sum() / possible.sum() == pytest.approx(0.6, abs=0.05)          # 0.59-0.61 over four seeds of this layer


def test_the_largest_margin_bias_reaches_vento_between_the_axis_and_margin_thirds_on_a_real_layer():
    """At coverage 0.5 the draped share of the base contacts near the axis (the top third of the engine's axis-ness) falls
    below that near the margin (the bottom third) by the bias times the thirds' difference in mean axis-ness (0.6 or so),
    so MARGIN_BIAS_MAX gives about Vento's 0.33 against 0.67 (a contrast of 0.34, 0.28-0.35 over four layers); with no bias
    the thirds are alike. The share of all contacts stays near 0.5."""
    possible = np.asarray(_layer(coverage=1.0).mult_z) < 1.0
    axis_ness = np.asarray(_layer(coverage=1.0)._engine.depth_norm)[possible]
    low, high = np.quantile(axis_ness, [1 / 3, 2 / 3])
    spread = axis_ness[axis_ness > high].mean() - axis_ness[axis_ness <= low].mean()
    contrast = {}
    for bias in (0.0, MARGIN_BIAS_MAX):
        draped = (np.asarray(_layer(coverage=0.5, margin_bias=bias, hole_range_m=30.0).mult_z) < 1.0)[possible]
        contrast[bias] = draped[axis_ness <= low].mean() - draped[axis_ness > high].mean()
        assert draped.mean() == pytest.approx(0.5, abs=0.05)
    assert abs(contrast[0.0]) < 0.05
    assert contrast[MARGIN_BIAS_MAX] == pytest.approx(MARGIN_BIAS_MAX * spread, abs=0.04)
    assert 0.25 < contrast[MARGIN_BIAS_MAX] < 0.40


def _exported(path, keyword, shape):
    """One array of a written GRDECL as an (nx, ny, nz) cube in the layer's own order (k up)."""
    words = path.read_text().split()
    start = words.index(keyword) + 1
    return np.array(words[start:words.index("/", start)], dtype=float).reshape(shape, order="F")[:, :, ::-1]


def test_the_multipliers_use_the_permeabilities_the_exporter_writes_and_the_cell_size_across_each_face(tmp_path):
    """Cells of 10 x 20 x 1 m, so dx differs from dy, and kx / ky of 4 along the channel, so PERMX differs from PERMY: the
    multiplier of every draped face is the thin barrier between the PERMX, PERMY or PERMZ of the GRDECL written for its two
    cells, over the cell size across it (dx, dy, dz). A swap of dx and dy, or of the kx and ky multipliers, changes it."""
    layer = ChannelLayer(nx=100, ny=35, nz=12, x_len=1000.0, y_len=700.0, z_len=12.0, top_depth=2000.0)
    layer.create_geology(seed=5, **{**GEOLOGY, "facies_props": {**MUD, 3: {"kxky": 4.0}, 4: {"kxky": 4.0}}},
                         drapes=dict(coverage=1.0, thickness=0.5))
    path = layer.to_grdecl(tmp_path / "m.grdecl")
    perm = [_exported(path, key, layer.facies.shape) for key in ("PERMX", "PERMY", "PERMZ")]
    assert (layer.dx, layer.dy, layer.dz) == (10.0, 20.0, 1.0) and not np.allclose(perm[0], perm[1], rtol=0.05)
    for axis, (name, h) in enumerate((("mult_x", layer.dx), ("mult_y", layer.dy), ("mult_z", layer.dz))):
        mult = np.asarray(getattr(layer, name), dtype=float)
        faces = np.argwhere(mult < 1.0)
        assert len(faces) > 50, name
        for idx in faces[::max(1, len(faces) // 80)]:
            neighbour = idx.copy()
            neighbour[axis] += -1 if axis == 2 else 1             # a z face is held by the upper cell, its partner is below
            k1, k2 = perm[axis][tuple(idx)], perm[axis][tuple(neighbour)]
            assert mult[tuple(idx)] == pytest.approx(_series(k1, k2, 1e-3, 0.5, h), rel=1e-4), (name, idx)


@pytest.mark.parametrize("name", ["PV_SHOESTRING", "CB_JIGSAW", "CB_LABYRINTH", "SH_DISTAL", "SH_PROXIMAL", "MEANDER_OXBOW"])
def test_drapes_change_no_rock_and_draw_nothing_on_each_preset(name):
    """On each of the six presets (a small grid): without drapes there are no multipliers; with them the facies, porosity and
    permeability are the arrays of the same build, there are draped faces, and numpy's global state ends where it did."""
    from resmill.layers import channel
    built = []
    for drapes in (None, dict(coverage=0.6)):
        layer = ChannelLayer(nx=40, ny=30, nz=12, x_len=800.0, y_len=600.0, z_len=12.0, top_depth=2000.0)
        layer.create_geology(seed=4, drapes=drapes, **getattr(channel, name))
        built.append((layer, _global_state()))
    (off, state_off), (on, state_on) = built
    assert off.mult_x is off.mult_y is off.mult_z is None and state_off == state_on
    for attr in ("facies", "poro_mat", "perm_mat", "active", "kvkh_mat", "kx_mult", "ky_mult"):
        a, b = getattr(off, attr), getattr(on, attr)
        assert (a is None and b is None) or np.array_equal(a, b), attr
    assert (np.asarray(on.mult_z) < 1.0).sum() > 50


def test_a_delta_layer_refuses_drapes_where_it_would_otherwise_ignore_them():
    """Only the channel layer has drapes: a DeltaLayer's generations each keep their own event record, so it says so."""
    from resmill.layers.delta import DeltaLayer
    with pytest.raises(TypeError, match="drapes"):
        DeltaLayer(nx=10, ny=10, nz=4, x_len=100.0, y_len=100.0, z_len=8.0, top_depth=0.0).create_geology(
            seed=1, drapes=dict(coverage=0.5))


def test_unknown_or_bad_drape_settings_are_refused_before_any_geology_is_made():
    """A typo in a setting name, a missing coverage or a value out of range would silently change the geology; so would
    a number that is not finite (an infinite hole range failed in a filter after the engine had run) or a bool."""
    inf, nan = math.inf, math.nan
    bad_settings = [dict(coverge=0.5), dict(margin_bias=0.1), dict(coverage=1.2), dict(coverage=-0.1), dict(coverage=True),
                    dict(coverage=nan), dict(coverage="0.5"), dict(coverage=None)]
    for name, values in dict(thickness=(0.0, -1.0, inf, nan, True), perm=(-1.0, 0.0, inf, nan), margin_bias=(1.5, -1.5, nan, True),
                             hole_range_m=(0.0, -3.0, inf, nan), hole_range_widths=(0.0, -3.0, inf, nan),
                             coverage_concentration=(0.0, -1.0, inf, nan, True)).items():
        bad_settings += [dict(coverage=0.5, **{name: v}) for v in values]
    for bad in bad_settings:
        with pytest.raises(ValueError):
            ChannelLayer(nx=4, ny=4, nz=4, x_len=40.0, y_len=40.0, z_len=4.0, top_depth=0.0).create_geology(drapes=bad)
    check_drapes(dict(coverage=np.float32(0.5), thickness=np.int64(1), coverage_concentration=0.71, hole_range_widths=2))   # fine


def test_the_hole_range_may_be_given_in_channel_widths():
    """``hole_range_widths`` is a multiple of the element's width (depth x width/depth ratio = 48 m here): 2 widths give
    the drapes of ``hole_range_m`` 96, half a width those of the default, and both together are refused."""
    a, b = (_layer(coverage=0.6, **kw) for kw in (dict(hole_range_widths=2.0), dict(hole_range_m=96.0)))
    assert np.array_equal(a.mult_z, b.mult_z)
    assert np.array_equal(_layer(coverage=0.6, hole_range_widths=0.5).mult_z, _layer(coverage=0.6).mult_z)
    assert not np.array_equal(a.mult_z, _layer(coverage=0.6).mult_z)
    with pytest.raises(ValueError):
        ChannelLayer(nx=4, ny=4, nz=4, x_len=40.0, y_len=40.0, z_len=4.0, top_depth=0.0).create_geology(
            drapes=dict(coverage=0.5, hole_range_m=10.0, hole_range_widths=1.0))


def test_the_hole_range_defaults_to_half_the_channel_width():
    """Without ``hole_range_m`` the holes are half the element's width (depth x width/depth ratio = 48 m here): the same
    drapes as giving 24 m, and not those of a full width."""
    a, b, c = (_layer(coverage=0.6, **kw) for kw in ({}, dict(hole_range_m=24.0), dict(hole_range_m=48.0)))
    assert np.array_equal(a.mult_z, b.mult_z) and not np.array_equal(a.mult_z, c.mult_z)
