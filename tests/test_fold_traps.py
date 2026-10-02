"""fold_trap: the six fold styles composed from ResMill's closure, roughness and fold_faults (ResSimMill step 7)."""
import numpy as np
import pytest

from resmill import structure as st
from resmill.export import to_grdecl
from resmill.fault_patterns import STYLES, fold_faults
from resmill.fault_seal import Capillary, Seal, block_labels, blocks_at
from resmill.fold_traps import FOLD_STYLES, fold_trap
from resmill.layers.base import Layer
from resmill.structure import closure_stats

NX, NY, NZ, DX, DZ, TOP = 80, 60, 4, 100.0, 5.0, 2000.0
X_LEN, Y_LEN, THICKNESS = NX * DX, NY * DX, NZ * DZ

# What a tree of each style is built from (the values the sampler draws, written out): the closure's keywords, a
# roughness and the faults.
CLOSURE = dict(area=5e6, height=60.0, aspect=1.8, azimuth=20.0, limb_ratio=1.2, tilt=0.1, satellites=1, warp=0.2,
               seed=3)
STYLE_CLOSURE = {
    "four_way": {},
    "turtle": dict(satellites=0),
    "faulted_anticline": {},
    "fold_belt": dict(limb_ratio=2.0, satellites=0),
    "fault_bounded": dict(tilt=0.3, satellites=0),
    "low_relief": dict(height=15.0, satellites=0),
}


def args(style, **override):
    """fold_trap's arguments for ``style``: the plan's blocks as the sampler holds them."""
    closure = {**CLOSURE, **STYLE_CLOSURE[style], **override.pop("closure", {})}
    rough = dict(sd=3.0 if style == "low_relief" else 8.0, range_m=2500.0, seed=11)
    faults = dict(density=1.0, over_salt=style == "four_way", seed=17)
    out = dict(roughness=rough, faults=faults, closure=closure)
    out.update(override)
    return out


def build(style, **override):
    return fold_trap(style, X_LEN, Y_LEN, DX, TOP, THICKNESS, **args(style, **override))


def layer():
    model = Layer(NX, NY, NZ, X_LEN, Y_LEN, THICKNESS, top_depth=TOP, kzkx=0.1)
    model.poro_mat = np.full((NX, NY, NZ), 0.2)
    model.perm_mat = np.full((NX, NY, NZ), 100.0)
    return model


def cell_depths(structure):
    """The top surface's depth at the cell centres, as an independent read of the structure."""
    x, y = np.meshgrid((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, indexing="ij")
    return TOP + structure(x, y)


def test_the_six_fold_styles_are_the_fault_patterns_fold_styles():
    assert set(FOLD_STYLES) == {"four_way", "turtle", "faulted_anticline", "fold_belt", "fault_bounded", "low_relief"}
    assert set(FOLD_STYLES) <= set(STYLES)


@pytest.mark.parametrize("style", FOLD_STYLES)
def test_every_fold_style_gives_a_model_the_grid_export_takes(style, tmp_path):
    built = build(style)
    assert set(built["kwargs"]) == {"structure", "faults"} and built["meta"]["style"] == style
    report = {}
    path = to_grdecl(layer(), tmp_path / "m.grdecl", seal=Seal(vsh=np.full((NX, NY, NZ), 0.3), seed=1), report=report,
                     **built["kwargs"])
    n = built["meta"]["n_faults"]
    assert n == len(built["kwargs"]["faults"]) == len(report["fault_names"])
    assert ("FAULTS" in path.read_text()) == (n > 0)
    assert built["meta"]["fault_sets"] == sorted({fault.kind for fault in built["kwargs"]["faults"]})


@pytest.mark.parametrize("style", FOLD_STYLES)
def test_the_closure_is_the_one_asked_for_by_closure_stats(style):
    """A fold without roughness or faults closes over the asked area and relief when read off the cells (closure_stats'
    own flood, on a grid 100 m across): the crest is the closure's, found as the trap's top."""
    built = build(style, roughness=None, faults=None)
    depth = cell_depths(built["kwargs"]["structure"])
    c = CLOSURE | STYLE_CLOSURE[style]
    ci, cj = (int(round(v / DX - 0.5)) for v in built["meta"]["crest"])
    stats = closure_stats(depth, DX, DX, crest=(ci, cj))
    top = depth[stats["mask"]].min()
    assert stats["area"] == pytest.approx(c["area"], rel=0.03)
    assert stats["spill_depth"] - top == pytest.approx(c["height"], rel=0.02)
    assert built["meta"]["area"] == c["area"] and built["meta"]["height"] == c["height"]
    where = np.unravel_index(int(np.argmin(np.where(stats["mask"], depth, np.inf))), depth.shape)
    assert np.hypot(*(np.array(where) - (ci, cj))) <= 2.0                       # the trap's top is the reported crest


def test_the_closure_is_centred_in_the_map_unless_told_otherwise():
    middle = build("four_way", closure=dict(satellites=0, warp=0.0, tilt=0.0, limb_ratio=1.0), roughness=None,
                   faults=None)
    assert middle["meta"]["crest"] == pytest.approx((0.5 * X_LEN, 0.5 * Y_LEN), abs=0.02 * X_LEN)
    off = build("four_way", closure=dict(satellites=0, warp=0.0, tilt=0.0, limb_ratio=1.0, center=(3000.0, 2500.0)),
                roughness=None, faults=None)
    assert off["meta"]["crest"] == pytest.approx((3000.0, 2500.0), abs=0.02 * X_LEN)


def test_a_composition_is_the_primitives_called_as_the_sampler_does():
    """The model is closure + roughness with the faults drawn on the smooth fold: the same numbers as calling the three
    primitives with the plan's values and seeds."""
    a = args("faulted_anticline")
    fold = st.closure(center=(0.5 * X_LEN, 0.5 * Y_LEN), **a["closure"])
    rough = st.roughness(a["roughness"]["sd"], a["roughness"]["range_m"], X_LEN, Y_LEN, seed=a["roughness"]["seed"])
    faults = fold_faults("faulted_anticline", fold, X_LEN, Y_LEN, DX, a["faults"]["density"], TOP, THICKNESS,
                         seed=a["faults"]["seed"], over_salt=a["faults"]["over_salt"])
    built = build("faulted_anticline")
    assert faults and built["kwargs"]["faults"] == faults
    x, y = np.meshgrid(np.linspace(0.0, X_LEN, 33), np.linspace(0.0, Y_LEN, 27), indexing="ij")
    assert (built["kwargs"]["structure"](x, y) == (fold + rough)(x, y)).all()


def test_a_model_is_a_function_of_its_arguments_and_leaves_the_global_stream_alone():
    np.random.seed(5)
    state = np.random.get_state()
    a, b = build("fault_bounded"), build("fault_bounded")
    after = np.random.get_state()
    assert after[0] == state[0] and (after[1] == state[1]).all() and after[2:] == state[2:]
    x, y = np.meshgrid(np.linspace(0.0, X_LEN, 33), np.linspace(0.0, Y_LEN, 27), indexing="ij")
    assert (a["kwargs"]["structure"](x, y) == b["kwargs"]["structure"](x, y)).all()
    assert a["kwargs"]["faults"] == b["kwargs"]["faults"] and a["meta"] == b["meta"]
    other = build("fault_bounded", faults=dict(density=1.0, over_salt=False, seed=18))
    assert other["kwargs"]["faults"] != a["kwargs"]["faults"]


def test_the_options_that_are_off_draw_and_call_nothing(monkeypatch):
    """No roughness, no faults: the model is the bare closure, and neither primitive is called."""
    def refuse(*a, **k):
        raise AssertionError("called")

    bare = st.closure(center=(0.5 * X_LEN, 0.5 * Y_LEN), **args("four_way")["closure"])
    monkeypatch.setattr("resmill.fold_traps.st.roughness", refuse)
    monkeypatch.setattr("resmill.fold_traps.fold_faults", refuse)
    built = build("four_way", roughness=None, faults=None)
    x, y = np.meshgrid(np.linspace(0.0, X_LEN, 33), np.linspace(0.0, Y_LEN, 27), indexing="ij")
    assert (built["kwargs"]["structure"](x, y) == bare(x, y)).all()
    assert built["kwargs"]["faults"] == [] and built["meta"]["n_faults"] == 0 and built["meta"]["fault_sets"] == []
    with pytest.raises(AssertionError):
        build("four_way")                                            # and with them on, both are


def test_the_roughness_is_the_asked_surface_added_to_the_fold():
    rough = build("four_way", faults=None)["kwargs"]["structure"]
    smooth = build("four_way", roughness=None, faults=None)["kwargs"]["structure"]
    x, y = np.meshgrid(np.linspace(0.0, X_LEN, 161), np.linspace(0.0, Y_LEN, 121), indexing="ij")
    residual = rough(x, y) - smooth(x, y)
    expected = st.roughness(8.0, 2500.0, X_LEN, Y_LEN, seed=11)(x, y)
    assert np.allclose(residual, expected) and 4.0 < residual.std() < 12.0


@pytest.mark.parametrize("style", ["rollover", "tilted_blocks", "salt_flank", "no_such_style"])
def test_a_style_that_is_not_a_fold_style_is_refused_before_anything_is_built(style, monkeypatch):
    monkeypatch.setattr("resmill.fold_traps.st.closure", lambda **kw: pytest.fail("built"))
    with pytest.raises(ValueError, match="style must be one of"):
        fold_trap(style, X_LEN, Y_LEN, DX, TOP, THICKNESS, **args("four_way"))


def test_the_chain_to_the_fault_blocks_reads_the_surface_the_closure_made(tmp_path):
    """Without faults the exported grid's tops are the structure averaged over each cell's corners (an independent
    evaluation), the one block of blocks_at is closure_stats' trap on them, and it is the closure asked for."""
    built = build("fold_belt", roughness=None, faults=None)
    report = {}
    to_grdecl(layer(), tmp_path / "m.grdecl", report=report, **built["kwargs"])
    inputs = report["block_inputs"]
    x, y = np.meshgrid(np.arange(NX + 1) * DX, np.arange(NY + 1) * DX, indexing="ij")
    node = TOP + built["kwargs"]["structure"](x, y)
    tops = 0.25 * (node[:-1, :-1] + node[1:, :-1] + node[:-1, 1:] + node[1:, 1:])
    assert np.allclose(inputs["depth"], tops, rtol=0.0, atol=1e-9)
    blocks = blocks_at(inputs, DX, DX, Capillary(delta_rho=300.0))
    stats = closure_stats(tops, DX, DX)
    assert len(blocks) == 1 and (block_labels(inputs) == 0).all() and blocks[0]["limited_by"] == "spill"
    assert blocks[0]["area"] == stats["area"] and blocks[0]["height"] == pytest.approx(stats["height"])
    assert blocks[0]["area"] == pytest.approx(CLOSURE["area"], rel=0.03)
    assert blocks[0]["height"] == pytest.approx(CLOSURE["height"], rel=0.02)
