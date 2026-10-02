"""The simulation outline: structure.outline, to_grdecl(outline=) and fold_trap(rim=) (ResSimMill step 7, G6)."""
import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial import cKDTree

from resmill import structure as st
from resmill.export import to_grdecl
from resmill.fault_seal import Capillary, Seal, block_labels, blocks_at
from resmill.fold_traps import FOLD_STYLES, fold_trap
from resmill.structure import closure_stats, outline
from tests.test_export import prop_cube, read_grdecl
from tests.test_fold_traps import DX, NX, NY, NZ, THICKNESS, TOP, X_LEN, Y_LEN, args, layer

REFINE = 4                                  # the reference polygons are read on cells this many times finer


def actnum_of(path):
    return prop_cube(read_grdecl(path), "ACTNUM", NX, NY, NZ).astype(int)


def without_actnum(path):
    """The file's text without its ACTNUM block."""
    text = path.read_text()
    start = text.index("\nACTNUM\n")
    return text[:start] + text[text.index("\n/\n", start) + 3:]


def test_the_outline_is_the_trap_dilated_by_the_rim_from_centre_to_centre():
    nx, ny, dx, dy = 60, 40, 100.0, 50.0
    x, y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    trap = ((x - 3000.0) / 900.0) ** 2 + ((y - 1000.0) / 400.0) ** 2 <= 1.0
    tree = cKDTree(np.column_stack([x[trap], y[trap]]))
    near = tree.query(np.column_stack([x.ravel(), y.ravel()]))[0].reshape(nx, ny)           # an independent distance
    for rim in (0.0, 150.0, 700.0, 2500.0):
        assert (outline(trap, dx, dy, rim) == (near <= rim)).all()
    assert (outline(trap, dx, dy, 0.0) == trap).all() and outline(trap, dx, dy, 1e6).all()
    with pytest.raises(ValueError, match="no column"):
        outline(np.zeros((nx, ny), dtype=bool), dx, dy, 100.0)
    with pytest.raises(ValueError, match="rim"):
        outline(trap, dx, dy, -1.0)


def footprint():
    """An ellipse of columns about the middle of the map, a closure's stand-in."""
    x, y = np.meshgrid((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, indexing="ij")
    return ((x - 0.5 * X_LEN) / 1200.0) ** 2 + ((y - 0.5 * Y_LEN) / 700.0) ** 2 <= 1.0


def test_to_grdecl_cuts_the_actnum_block_and_nothing_else(tmp_path):
    """The cut file is the whole map's with its ACTNUM replaced: ZCORN, the seal's multipliers, FAULTS and the report
    are those of the whole map."""
    built = fold_trap("faulted_anticline", X_LEN, Y_LEN, DX, TOP, THICKNESS, **args("faulted_anticline"))
    keep = outline(footprint(), DX, DX, 1000.0)
    assert built["kwargs"]["faults"] and 0 < keep.sum() < keep.size
    seal = Seal(vsh=np.full((NX, NY, NZ), 0.3), seed=1)
    plain, cut, whole, kept = tmp_path / "a.grdecl", tmp_path / "b.grdecl", {}, {}
    to_grdecl(layer(), plain, seal=seal, report=whole, **built["kwargs"])
    to_grdecl(layer(), cut, seal=seal, report=kept, outline=keep, **built["kwargs"])
    assert without_actnum(cut) == without_actnum(plain) and cut.read_bytes() != plain.read_bytes()
    assert (actnum_of(cut) == actnum_of(plain) * keep[:, :, None]).all()     # a cell a fault collapsed stays inactive
    assert kept["fault_names"] == whole["fault_names"] and kept["faults"] == whole["faults"]
    assert np.array_equal(kept["block_inputs"]["depth"], whole["block_inputs"]["depth"], equal_nan=True)
    for key in ("alive", "face_records"):
        assert np.array_equal(kept["block_inputs"][key], whole["block_inputs"][key])
    assert all(np.array_equal(a, b) for a, b in zip(kept["block_inputs"]["split"], whole["block_inputs"]["split"]))


def test_an_outline_that_keeps_every_column_or_none_is_what_it_says(tmp_path):
    built = fold_trap("four_way", X_LEN, Y_LEN, DX, TOP, THICKNESS, **args("four_way"))
    plain, everything, none = tmp_path / "a.grdecl", tmp_path / "b.grdecl", tmp_path / "c.grdecl"
    to_grdecl(layer(), plain, **built["kwargs"])
    to_grdecl(layer(), everything, outline=np.ones((NX, NY), dtype=bool), **built["kwargs"])
    to_grdecl(layer(), none, outline=None, **built["kwargs"])
    assert plain.read_bytes() == everything.read_bytes() == none.read_bytes()
    for bad, message in ((np.ones((NX, NY + 1), dtype=bool), "shape"), (np.zeros((NX, NY), dtype=bool), "keeps no")):
        with pytest.raises(ValueError, match=message):
            to_grdecl(layer(), tmp_path / "d.grdecl", outline=bad, **built["kwargs"])
    assert not (tmp_path / "d.grdecl").exists()                                     # refused before a file is written


def build(style, rim=None, **closure):
    return fold_trap(style, X_LEN, Y_LEN, DX, TOP, THICKNESS, **args(style, closure=closure), rim=rim)


def test_without_a_rim_fold_trap_returns_what_it_always_did(tmp_path):
    for style in FOLD_STYLES:
        built = build(style)
        assert set(built["kwargs"]) == {"structure", "faults"}
        assert set(built["meta"]) == {"style", "area", "height", "crest", "n_faults", "fault_sets"}
    a = args("faulted_anticline")
    fold = st.closure(center=(0.5 * X_LEN, 0.5 * Y_LEN), **a["closure"])
    rough = st.roughness(a["roughness"]["sd"], a["roughness"]["range_m"], X_LEN, Y_LEN, seed=a["roughness"]["seed"])
    built = build("faulted_anticline")
    to_grdecl(layer(), tmp_path / "a.grdecl", **built["kwargs"])
    to_grdecl(layer(), tmp_path / "b.grdecl", structure=fold + rough, faults=built["kwargs"]["faults"])
    assert (tmp_path / "a.grdecl").read_bytes() == (tmp_path / "b.grdecl").read_bytes()


def ellipse(x, y, closure, center):
    """The columns of the closure as drawn: an ellipse of its area, ``aspect`` times longer along ``azimuth``."""
    az, aspect = np.radians(closure["azimuth"]), closure["aspect"]
    half = np.sqrt(closure["area"] / (np.pi * aspect))
    along = (x - center[0]) * np.cos(az) - (y - center[1]) * np.sin(az)
    across = (x - center[0]) * np.sin(az) + (y - center[1]) * np.cos(az)
    return (along / (aspect * half)) ** 2 + (across / half) ** 2 <= 1.0


def fine_distance(built, closure):
    """The distance (m) from each cell centre to the closure's polygon, an independent read: the ellipse drawn and
    the traps of the fold and of the built surface on cells ``REFINE`` times finer, joined, and a distance transform."""
    h = DX / REFINE
    x, y = np.meshgrid((np.arange(NX * REFINE) + 0.5) * h, (np.arange(NY * REFINE) + 0.5) * h, indexing="ij")
    fold = st.closure(center=(0.5 * X_LEN, 0.5 * Y_LEN), **closure)
    ci, cj = (int(v // h) for v in built["meta"]["crest"])
    reach = 2 * REFINE
    trap = ellipse(x, y, closure, (0.5 * X_LEN, 0.5 * Y_LEN))                            # the closure drawn
    for surface in (fold, built["kwargs"]["structure"]):
        depth = TOP + surface(x, y)
        window = depth[ci - reach:ci + reach + 1, cj - reach:cj + reach + 1]
        at = np.unravel_index(int(np.argmin(window)), window.shape)
        trap |= closure_stats(depth, h, h, crest=(ci - reach + at[0], cj - reach + at[1]))["mask"]
    d = ndimage.distance_transform_edt(~trap, sampling=h)
    mid = slice(REFINE // 2 - 1, REFINE // 2 + 1)                       # the four fine pixels around a cell's centre
    return d.reshape(NX, REFINE, NY, REFINE)[:, mid, :, mid].mean(axis=(1, 3))


def closure_as_it_came_out(built):
    """The trap of the built surface at the cell centres around its shallowest column near the crest."""
    x, y = np.meshgrid((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, indexing="ij")
    depth = TOP + built["kwargs"]["structure"](x, y)
    ci, cj = (int(v // DX) for v in built["meta"]["crest"])
    window = depth[max(ci - 2, 0):ci + 3, max(cj - 2, 0):cj + 3]
    at = np.unravel_index(int(np.argmin(window)), window.shape)
    return closure_stats(depth, DX, DX, crest=(max(ci - 2, 0) + at[0], max(cj - 2, 0) + at[1]))["mask"]


@pytest.mark.parametrize("style", FOLD_STYLES)
def test_the_inactive_region_is_the_dilated_spill_contour_within_a_cell(style, tmp_path):
    """The columns the file leaves active are those within the rim of the closure's contour, read off an independent
    polygon on finer cells: none farther than a cell beyond the rim, none written inactive within a cell of it on the
    closure's side (columns a fault had collapsed are inactive anyway); and the closure as it came out, the fold plus
    its roughness, lies inside the outline whatever the rim."""
    distance = fine_distance(build(style, seed=3), args(style, closure=dict(seed=3))["closure"])
    to_grdecl(layer(), tmp_path / "whole.grdecl", **build(style, seed=3)["kwargs"])
    rock = actnum_of(tmp_path / "whole.grdecl").any(axis=2)
    for rim in (500.0, 1500.0):
        built = build(style, rim=rim, seed=3)
        to_grdecl(layer(), tmp_path / "m.grdecl", **built["kwargs"])
        active = actnum_of(tmp_path / "m.grdecl").any(axis=2)
        assert (active == (built["kwargs"]["outline"] & rock)).all()
        assert not (active & (distance > rim + DX)).any()
        assert not (rock & ~active & (distance < rim - DX)).any()
        assert abs(int((active & rock).sum()) - int(((distance <= rim) & rock).sum())) <= 0.03 * active.sum()
        assert built["meta"]["outline_columns"] == built["kwargs"]["outline"].sum() and built["meta"]["rim"] == rim
        came_out = closure_as_it_came_out(built)
        assert came_out.any() and not (came_out & ~built["kwargs"]["outline"]).any()


@pytest.mark.parametrize("rim", [500.0, 1000.0, 2000.0])
def test_the_fault_blocks_of_a_cut_model_are_those_of_the_whole_map(rim, tmp_path):
    """Read as the file says, a cut model's rock-free columns are walls and a flood from the map's edge cannot reach a
    trap inside the outline; the report is the whole map's, so the trap, its contact and its block labels are what the
    uncut model gives, for any rim."""
    built = build("faulted_anticline", rim=rim)
    keep = built["kwargs"].pop("outline")
    seal = Seal(vsh=np.full((NX, NY, NZ), 0.3), scatter=0.5, offset=-0.75, seed=2)
    reports = []
    for name, kwargs in (("whole", {}), ("cut", {"outline": keep})):
        reports.append({})
        to_grdecl(layer(), tmp_path / f"{name}.grdecl", seal=seal, report=reports[-1], **kwargs, **built["kwargs"])
    cap = Capillary(delta_rho=300.0)
    whole, cut = (blocks_at(report["block_inputs"], DX, DX, cap) for report in reports)
    assert len(whole) == len(cut) > 1
    for a, b in zip(whole, cut):
        assert {k: v for k, v in a.items() if k != "mask"} == {k: v for k, v in b.items() if k != "mask"}
        assert (a["mask"] == b["mask"]).all()
    assert (block_labels(reports[0]["block_inputs"]) == block_labels(reports[1]["block_inputs"])).all()
    kept, rock = actnum_of(tmp_path / "cut.grdecl"), actnum_of(tmp_path / "whole.grdecl")
    assert kept.sum() == (rock * keep[:, :, None]).sum() < rock.sum()


def test_read_as_walls_the_cut_columns_seal_the_trap_which_is_why_the_report_is_the_whole_maps(tmp_path):
    """Why ``report`` is not the cut model's own: with the columns outside the outline as rock-free walls, a flood from
    the map's edge cannot enter an outline that does not touch it, so the block is sealed and fills to its deepest
    column -- a larger trap than the closure's, whatever contact the spill gives."""
    built = build("four_way", rim=500.0, tilt=0.1, satellites=0)
    built["kwargs"]["faults"], keep = [], built["kwargs"].pop("outline")
    report = {}
    to_grdecl(layer(), tmp_path / "m.grdecl", report=report, **built["kwargs"])
    inputs, cap = report["block_inputs"], Capillary(delta_rho=300.0)
    (whole,) = blocks_at(inputs, DX, DX, cap)
    (walled,) = blocks_at(dict(inputs, alive=inputs["alive"] & keep), DX, DX, cap)
    assert whole["limited_by"] == "spill" and walled["limited_by"] == "sealed"
    assert walled["area"] > 1.5 * whole["area"] and walled["height"] > whole["height"]


def test_a_closure_the_map_does_not_hold_keeps_its_drawn_ellipse_and_its_rim():
    """A tilt carries the crest of a large closure to the map's edge (a sampled plan: 9.1 km2, tilt 0.33, on 4.9 x 5.9
    km): the trap on the grid is a dozen columns and the model must not shrink to a disc about its crest."""
    closure = dict(area=9143024.0, height=36.4, aspect=1.585, azimuth=115.35, limb_ratio=1.39, tilt=0.326, satellites=0,
                   warp=0.398, seed=236493098)
    nx, ny = 49, 59
    built = fold_trap("turtle", nx * DX, ny * DX, DX, TOP, THICKNESS, closure, rim=905.0)
    x, y = np.meshgrid((np.arange(nx) + 0.5) * DX, (np.arange(ny) + 0.5) * DX, indexing="ij")
    ci, cj = (int(v // DX) for v in built["meta"]["crest"])
    trap = closure_stats(TOP + built["kwargs"]["structure"](x, y), DX, DX, crest=(ci, cj))["mask"]
    keep, drawn = built["kwargs"]["outline"], ellipse(x, y, closure, (0.5 * nx * DX, 0.5 * ny * DX))
    assert trap.sum() < 0.05 * drawn.sum()                                  # the map does not hold the closure
    assert keep[drawn].all() and keep.sum() > 2000                          # its ellipse and the rim round it are kept
    assert built["meta"]["outline_columns"] == keep.sum()
