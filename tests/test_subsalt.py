"""Subsalt traps and the salt labels: the base of salt as an erosion surface (cells above it are salt), traps truncated
against it or folded beneath a cover, the thickness draw (N30), and the labels an episode carries (mask, thickness, base
of salt)."""
import math
import signal
from contextlib import contextmanager

import numpy as np
import pytest

from resmill import salt as sl
from resmill import structure as st
from resmill.export import _build_geometry
from resmill.fault_seal import Seal, face_multipliers, fault_blocks
from resmill.layers.base import Layer

NX, NY, DX, TOP = 80, 60, 50.0, 2000.0
CENTER = (0.5 * NX * DX, 0.5 * NY * DX)


def layer(nz=60, dz=2.0):                                                # 120 m of reservoir, more than the 80 m relief
    L = Layer(NX, NY, nz, NX * DX, NY * DX, nz * dz, top_depth=TOP, kzkx=0.1)
    L.poro_mat = np.full((NX, NY, nz), 0.2)
    L.perm_mat = np.full((NX, NY, nz), 100.0)
    return L


def fold(**kw):
    return st.closure(area=4e6, height=80.0, aspect=1.5, center=CENTER, **kw)


def top_map(zc, act):
    """Depth of each column's first active cell's top (m), NaN where nothing is active."""
    cell = 0.25 * (zc[0::2, 0::2] + zc[1::2, 0::2] + zc[0::2, 1::2] + zc[1::2, 1::2])
    k = act.argmax(axis=2)
    return np.where(act.any(axis=2), np.take_along_axis(cell[..., :-1], k[..., None], axis=2)[..., 0], np.nan)


@contextmanager
def finishes_within(seconds):
    """Fail instead of hanging: a flood that never ends (the dead columns of a salt wall once did that, with memory
    growing by tens of MB a second) raises TimeoutError after ``seconds``."""
    if not hasattr(signal, "SIGALRM"):                                       # no alarm on this platform: unguarded
        yield
        return

    def stop(*_):
        raise TimeoutError(f"did not finish within {seconds} s")

    old = signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def blocks_of(zc, act):
    perm = np.full(act.shape, 100.0)
    mults = face_multipliers([], zc, act, np.full(act.shape, 0.1), (perm, perm, perm), DX, DX, Seal())
    with finishes_within(30):
        return fault_blocks(zc, act, [], mults, DX, DX)


def test_the_spill_flood_ends_on_a_wall_of_dead_columns():
    """A salt wall is a line of columns with no active cell, which fault_blocks gives infinite depth. The flood once kept
    two such neighbours on its queue for ever (memory grew by 13 GB in five minutes): it must end, give the pocket behind
    a closed wall an infinite level, and leave every level on the near side as it is without the wall."""
    n = 21
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    depth = 2000.0 + 0.5 * np.hypot(i - 10, j - 10) ** 2                       # a bowl turned over: a crest in the middle
    plain = st._spill_levels(depth)
    across = depth.copy()
    across[14:16, :] = np.inf                                                  # a wall two columns wide across the map
    with finishes_within(5):
        levels = st._spill_levels(across)
    assert np.isinf(levels[14:16, :]).all() and np.array_equal(levels[:14], plain[:14])
    assert np.isfinite(levels[16:]).all()                                      # the far side still reaches the edge
    pocket = depth.copy()
    pocket[4:7, 4:7] = np.inf                                                  # a closed pocket of dead cells inside
    with finishes_within(5):
        assert np.array_equal(st._spill_levels(pocket)[:3], plain[:3])


def test_base_of_salt_is_a_plane_at_the_depth_and_dip_plus_roughness_and_a_high():
    base = sl.base_of_salt(3000.0, dip=20.0, azimuth=90.0, center=(1000.0, 1000.0))       # deepening along +x
    assert base(1000.0, 1000.0) == pytest.approx(3000.0)
    assert base(1100.0, 1000.0) == pytest.approx(3000.0 + 100.0 * math.tan(math.radians(20.0)))
    assert base(1000.0, 5000.0) == pytest.approx(3000.0)                                  # constant along the strike
    rough, high = st.dome(-60.0, 400.0, center=(1000.0, 1000.0)), st.dome(150.0, 300.0, center=(0.0, 0.0))
    both = sl.base_of_salt(3000.0, rough=rough, high=high)
    assert both(1234.0, 987.0) == pytest.approx(3000.0 + rough(1234.0, 987.0) + high(1234.0, 987.0))
    assert isinstance(base, st.Structure)
    lazy = sl.base_of_salt(3000.0, dip=10.0, azimuth=0.0)                                 # pivot: the middle of the map
    y = np.array([0.0, 500.0, 1000.0])
    assert lazy(np.zeros(3), y) == pytest.approx(3000.0 + (y - 500.0) * math.tan(math.radians(10.0)))


def test_the_thickness_draw_is_the_log_normal_of_the_seven_published_values():
    published = np.array([302.0, 338.0, 515.0, 1006.0, 2118.0, 2438.0, 3353.0])           # N30, m
    draws = sl.salt_thickness(np.random.default_rng(0), size=20000)
    assert np.median(draws) == pytest.approx(np.exp(np.log(published).mean()), rel=0.04)    # about 1.0 km
    assert np.median(draws) == pytest.approx(1000.0, rel=0.1)
    assert 0.3 * published.mean() < draws.mean() < 2.0 * published.mean()
    assert draws.max() <= 4600.0 and draws.min() >= 100.0                                   # the canopy's 4,572 m cap
    assert (draws > 3400.0).mean() > 0.04                                                    # the tail past the 3.4 km sample
    assert sl.salt_thickness(np.random.default_rng(1)) == sl.salt_thickness(np.random.default_rng(1))


def test_cells_above_the_base_of_salt_are_inactive_and_the_cut_cells_are_truncated():
    L = layer(nz=10, dz=1.0)
    base = sl.base_of_salt(TOP + 4.5, thickness=900.0)                   # a flat base through the 5th cell
    _, _, zc, act = _build_geometry([L], erode_above=base)
    assert (act[:, :, :4] == 0).all() and (act[:, :, 4:] == 1).all()
    assert np.allclose(zc[:, :, 4], TOP + 4.5, atol=1e-6)                # cell k = 4 starts at the base ...
    assert np.allclose(zc[:, :, 5] - zc[:, :, 4], 0.5)                   # ... and is cut to 0.5 m against it
    assert np.allclose(zc[:, :, 6] - zc[:, :, 5], 1.0)


def test_a_cut_of_half_the_relief_makes_a_three_way_trap_against_the_salt():
    """Cutting the crest at half the relief leaves a trap 0.5 of the height whose crest is the base of salt and whose
    spill is the fold's, and closure_stats of the effective top and fault_blocks agree on both (research test 6)."""
    L, f = layer(), fold(seed=1, warp=0.2)
    _, _, zc, act = _build_geometry([L], structure=f)
    tm = top_map(zc, act)
    stats = st.closure_stats(tm, DX, DX)
    relief = stats["height"]
    cut = stats["spill_depth"] - 0.5 * relief                            # crest + 0.5 relief
    _, _, zc2, act2 = _build_geometry([L], structure=f, erode_above=sl.base_of_salt(cut))
    eff = top_map(zc2, act2)
    assert np.nanmin(eff) == pytest.approx(cut, abs=1e-3)
    stats2 = st.closure_stats(np.where(np.isnan(eff), 1e9, eff), DX, DX)
    assert stats2["spill_depth"] == pytest.approx(stats["spill_depth"], abs=1e-6)
    assert stats2["height"] == pytest.approx(0.5 * relief, rel=1e-3)
    assert stats2["area"] == pytest.approx(stats["area"], rel=1e-9)      # the same outline at the spill level
    assert (~np.isnan(eff)).all() and np.isclose(eff, cut).any()         # the stack is thicker than the relief: no hole
    block = blocks_of(zc2, act2)[0]
    assert block["spill_depth"] == pytest.approx(stats2["spill_depth"], abs=1e-6)
    assert block["crest_depth"] == pytest.approx(cut, abs=1e-3)
    assert block["height"] == pytest.approx(0.5 * relief, rel=1e-3) and block["spill_depth"] > block["crest_depth"]


def test_a_dipping_base_of_salt_cuts_one_side_of_the_crest():
    L, f = layer(), fold()
    _, _, zc, act = _build_geometry([L], structure=f)
    tm = top_map(zc, act)
    stats = st.closure_stats(tm, DX, DX)
    base = sl.base_of_salt(stats["spill_depth"] - 0.6 * stats["height"], dip=25.0, azimuth=0.0, center=CENTER)
    _, _, zc2, act2 = _build_geometry([L], structure=f, erode_above=base)
    eff = top_map(zc2, act2)
    cut = ~np.isclose(eff, tm, atol=1e-6)
    ys = (np.arange(NY) + 0.5) * DX
    assert cut.any() and not cut.all()
    assert ys[np.nonzero(cut)[1]].mean() > CENTER[1]                     # the plane deepens toward +y: that is the side cut
    assert np.nanmin(eff) < np.nanmax(eff[cut])                          # and it cuts at different depths (it dips)


def test_a_fold_under_a_cover_is_untouched_when_the_base_of_salt_lies_above_it():
    L, f = layer(), fold()
    _, _, zc, act = _build_geometry([L], structure=f)
    cover = sl.base_of_salt(np.nanmin(top_map(zc, act)) - 120.0, thickness=1500.0)
    _, _, zc2, act2 = _build_geometry([L], structure=f, erode_above=cover)
    assert (act2 == act).all() and np.allclose(zc2, zc)


def test_a_salt_stock_inside_the_trap_is_a_wall_the_trap_does_not_leak_through():
    """Research test 7: the columns of a stock in the closure are impassable; the trap is the ring round it, closed at
    the fold's spill depth, and its area is the closure's less the salt's."""
    L, f = layer(), fold()
    _, _, zc, act = _build_geometry([L], structure=f)
    stats = st.closure_stats(top_map(zc, act), DX, DX)
    cx, cy = (CENTER[0] + f.crest_offset[0], CENTER[1] + f.crest_offset[1])
    body = sl.salt_body((cx, cy), (250.0, 250.0))
    _, _, zc2, act2 = _build_geometry([L], structure=f, salt=body)
    dead = ~act2.any(axis=2)
    block = blocks_of(zc2, act2)[0]
    assert dead.sum() > 0 and not (block["mask"] & dead).any()
    assert block["spill_depth"] == pytest.approx(stats["spill_depth"], abs=1e-6)       # the salt gave no way out
    assert block["area"] == pytest.approx(stats["area"] - dead.sum() * DX ** 2, rel=0.01)
    assert block["height"] == pytest.approx(block["spill_depth"] - block["crest_depth"])
    assert block["crest_depth"] > stats["spill_depth"] - stats["height"] + 1.0            # the crest went into the salt


def test_a_salt_wall_across_the_closure_splits_it_into_two_traps():
    L, f = layer(), fold()
    _, _, zc, act = _build_geometry([L], structure=f)
    wall = sl.salt_body(CENTER, (3000.0, 60.0), azimuth=90.0)                           # along y through the middle
    _, _, zc2, act2 = _build_geometry([L], structure=f, salt=wall)
    traps = [b for b in blocks_of(zc2, act2) if b["area"] > 0.3e6]
    assert len(traps) == 2
    crest_x = sorted(b["crest"][0] for b in traps)
    assert crest_x[0] < NX // 2 < crest_x[1]


# --- the labels --------------------------------------------------------------------------------------------------

def test_labels_of_a_body_are_its_cells_the_salt_in_the_model_and_the_depth_below_it():
    L = layer(nz=10, dz=5.0)
    body = sl.salt_body(CENTER, (300.0, 300.0))
    out = sl.salt_labels(L, salt=body)
    xs, ys = np.meshgrid((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, indexing="ij")
    inside = (xs - CENTER[0]) ** 2 + (ys - CENTER[1]) ** 2 < 300.0 ** 2
    assert out["mask"].dtype == np.uint8 and out["mask"].shape == (NX, NY, 10)
    assert (out["mask"] == inside[:, :, None]).all()
    assert np.allclose(out["thickness"][inside], 50.0) and (out["thickness"][~inside] == 0.0).all()
    assert np.allclose(out["base"][inside], TOP + 50.0) and np.isnan(out["base"][~inside]).all()
    assert out["volume_fraction"] == pytest.approx(inside.mean())
    assert out["thickness"].dtype == np.float32 and out["base"].dtype == np.float32


def test_labels_of_an_overhang_give_the_underside_as_the_base_of_the_salt():
    nz, dz, flare = 40, 5.0, -0.3
    L = layer(nz=nz, dz=dz)
    body = sl.salt_body(CENTER, (300.0, 300.0), z_ref=TOP + 100.0, flare=flare)
    out = sl.salt_labels(L, salt=body)
    xs, ys = np.meshgrid((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, indexing="ij")
    rho = np.hypot(xs - CENTER[0], ys - CENTER[1])
    cells = np.argwhere((rho > 305.0) & (rho < 325.0))                                  # salt on top only
    assert len(cells) >= 2
    mid = TOP + (np.arange(nz) + 0.5) * dz
    for i, j in cells:
        contact = TOP + 100.0 - (rho[i, j] - 300.0) / -flare                            # radius 300 - flare (z_ref - z) = rho
        n_salt = int((mid < contact).sum())                                             # cell centres above the contact
        assert 0 < n_salt < nz
        assert out["mask"][i, j, :n_salt].all() and not out["mask"][i, j, n_salt:].any()
        assert out["thickness"][i, j] == pytest.approx(n_salt * dz) and out["base"][i, j] == pytest.approx(TOP + n_salt * dz)


def test_labels_of_a_base_of_salt_are_the_surface_the_cut_cells_and_the_drawn_thickness():
    L = layer(nz=10, dz=1.0)
    cut = sl.base_of_salt(TOP + 4.5, dip=0.0, thickness=1100.0)
    out = sl.salt_labels(L, erode_above=cut)
    assert (out["mask"][:, :, :4] == 1).all() and (out["mask"][:, :, 4:] == 0).all()
    assert (out["thickness"] == 1100.0).all() and np.allclose(out["base"], TOP + 4.5)
    assert out["volume_fraction"] == pytest.approx(0.4)
    under = sl.salt_labels(L, erode_above=sl.base_of_salt(TOP - 300.0, thickness=st.Structure(lambda x, y: 800.0 + 0 * x)))
    assert under["mask"].sum() == 0                                                      # a cover: no cell is salt
    assert (under["thickness"] == 800.0).all() and np.allclose(under["base"], TOP - 300.0)
    assert np.isnan(sl.salt_labels(L, erode_above=sl.base_of_salt(TOP + 4.5))["thickness"]).all()   # none drawn: unknown
    plain = sl.salt_labels(L)
    assert plain["mask"].sum() == 0 and np.isnan(plain["base"]).all() and (plain["thickness"] == 0.0).all()
