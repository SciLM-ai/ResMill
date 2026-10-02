"""Backward compatibility as a test: what the options added since the published ResMill (``salt=``, ``upturn=``,
``radial_rate=``, the pass levels of the spill flood, the wave helper) leave of every default output.

The golden digests are SHA-256 of outputs made with default arguments, from layers whose properties come from no random draw:

* ``EXPORT``: ``to_grdecl`` text (less its version line) of five models, written by bc6bc24, the ResMill that the published
  dataset was built from;
* ``PATTERNS``: ``fold_faults`` of the six styles that existed before salt (two folds and settings each), their faults' fields
  rounded to 9 significant digits, and ``FAULTED``: a faulted model with a fault seal exported, both written by 9b23681,
  the commit step 5 began from; ``CLOSURE``: a closure and a roughness surface evaluated on a grid (rounded likewise).

A change that moves one of them is a change of the default outputs and has to be an opt-in; regenerate deliberately with
``python tests/test_defaults_unchanged.py`` from the commit that is to be the new reference.
"""
import dataclasses
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from resmill import structure as st
from resmill.export import to_grdecl
from resmill.layers.base import Layer
from resmill.reservoir import Reservoir

EXPORT = {
    "dome": "efa0311f240e41d31857b25c66a97c7c3c8eeb4528b32ac119f228ebeafcdce5",
    "fault": "2bb980fc768499410faae98c6118b895a255c677c1a6f45e1fe2f93006180cf1",
    "stack": "06c588dd08c493fc748c9dfabf5437bb765f0b8a40ab2b478aa2d55880ab3fd9",
    "per_layer": "69686c2ed1c83e08ae3dd447a2e35988d731b659b9754f8f67c8b3cf8a0e086e",
    "surface": "fdc00d0faf52d2ef7b962d4d43de302cba9a468c7e9585ef0d0ef2e36e3fdacb",
}
PATTERNS = {
    "four_way|0": "593d177c8c312cb7f077630270e0786ce8ca4016bf7fe1ff5742281c8f60343c",
    "four_way|1": "79017604a1f1c289eaac3f88f1459052cfbd0d864e998f3627e50c366085feab",
    "turtle|0": "f6e351bde87fab9c92bd3aef805a941e935619a7efed61021f55eaa601aab400",
    "turtle|1": "79017604a1f1c289eaac3f88f1459052cfbd0d864e998f3627e50c366085feab",
    "faulted_anticline|0": "ec6d7b83ac6ffaa8c83896aaef44a14820cc3c709dfde78135cf1e5dbc92a4fe",
    "faulted_anticline|1": "79017604a1f1c289eaac3f88f1459052cfbd0d864e998f3627e50c366085feab",
    "fold_belt|0": "78c68da687dfb790f380ac5983ce8b9ce73f42dd8a57412f797c6dcc8ea0d310",
    "fold_belt|1": "0cb1663849fd65e4da7b46431dfcd135e036e338f074095c80f9fd5f6997c4e2",
    "fault_bounded|0": "53317b43940c99a3b8610ab9e0a7a668599bc32b85c591d701f3505f75069591",
    "fault_bounded|1": "7b76ba4c69d447432e405c39dce9a469eef1fccb0c65e11e9408da00233bb500",
    "low_relief|0": "2fbf9fbbb49ebe1586c492d5a2dfc0c8795cdc3c74934b4b20c775d705a8b65a",
    "low_relief|1": "a66935efc2024d458c69a96cac82c85d8a595bd244ac8427682a209b43de0716",
}
FAULTED = {
    "faulted_anticline": "4da19bb8e8e971f7a911f6aadb5c3a5cde16b4a366b70262399e0d65981d5975",
}
CLOSURE = {
    "closure0": "e538fe39e0a157ed33c16eb37137f5e94c1045439d5719f517a7a97ca45f8c03",
    "closure1": "33812589a7183672e9b756ba0b2ee70e7820957a90227b61229dfa704e2a84fc",
    "roughness": "a592b27b7476e24a7905c7333760c1b30086a66f2eb4e14870e5f0e9634ce24c",
}


def layer(top=2000.0, nx=30, ny=20, nz=6, shift=0):
    """A layer of nz x 5 m whose properties are integers mapped onto the porosity and permeability ranges (no random draw)."""
    L = Layer(nx, ny, nz, nx * 50.0, ny * 50.0, nz * 5.0, top_depth=top, kzkx=0.1)
    n = (np.arange(nx * ny * nz).reshape(nx, ny, nz) + shift)
    L.poro_mat = 0.05 + 0.25 * (n % 17) / 16.0
    L.perm_mat = 10.0 ** (-1.0 + 4.0 * (n % 23) / 22.0)
    return L


def grdecl_digest(model, **kw):
    with tempfile.TemporaryDirectory() as d:
        to_grdecl(model, Path(d) / "m.grdecl", **kw)
        return hashlib.sha256((Path(d) / "m.grdecl").read_bytes().split(b"\n", 1)[1]).hexdigest()


def exports():
    stack = lambda: Reservoir([layer(shift=1), layer(top=2030.0, shift=2)])
    return {
        "dome": grdecl_digest(layer(), structure=st.dome(50.0, 600.0, aspect=1.5, azimuth=30.0)),
        "fault": grdecl_digest(layer(), structure=st.fault(15.0, x0=700.0, y0=500.0, azimuth=60.0), poro_floor=0.06, perm_floor=0.5),
        "stack": grdecl_digest(stack(), structure=st.anticline(40.0, 1500.0, azimuth=20.0), erode_above=st.dome(10.0, 500.0)),
        "per_layer": grdecl_digest(stack(), structure=[st.ramp(2.0, azimuth=45.0), st.dome(20.0, 400.0)]),
        "surface": grdecl_digest(layer(), structure=st.surface(np.outer(np.linspace(0, 20, 7), np.linspace(1, 2, 5)), 1500.0, 1000.0)),
    }


def rounded(value):
    """``value`` with every float rounded to 9 significant digits (so the digest does not depend on the last bits)."""
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def digest(obj):
    return hashlib.sha256(json.dumps(rounded(obj), sort_keys=True, default=str).encode()).hexdigest()


def fold(k=0):
    aspect, tilt, limb = ((1.0, 0.0, 1.0), (2.0, 0.2, 1.5))[k]
    return st.closure(area=20e6, height=150.0, aspect=aspect, azimuth=20.0, center=(8000.0, 6000.0), limb_ratio=limb, tilt=tilt,
                      satellites=1, warp=0.2, seed=3 + k)


def patterns():
    from resmill.fault_patterns import STYLES, fold_faults
    out = {}
    for style in (s for s in STYLES if s != "salt_flank"):
        for k, extra in ((0, {}), (1, {"regional": (1.2, 35.0)})):
            faults = fold_faults(style, fold(k), 16000.0, 12000.0, 100.0, 1.0, 2000.0, 60.0, seed=k, **extra)
            out[f"{style}|{k}"] = digest([dataclasses.asdict(f) for f in faults])
    return out


def faulted():
    from resmill.fault_patterns import fold_faults
    from resmill.fault_seal import Seal
    nx, ny, nz = 80, 60, 6
    L = layer(nx=nx, ny=ny, nz=nz)
    faults = fold_faults("faulted_anticline", fold(1), nx * 50.0, ny * 50.0, 50.0, 2.0, 2000.0, 30.0, seed=5)
    return {"faulted_anticline": grdecl_digest(L, structure=fold(1), faults=faults, seal=Seal(vsh=np.full((nx, ny, nz), 0.3)))}


def closures():
    x, y = np.meshgrid(np.linspace(0.0, 8000.0, 41), np.linspace(0.0, 6000.0, 31), indexing="ij")
    out = {}
    for k, kw in enumerate((dict(), dict(warp=0.2, satellites=2, tilt=0.2, limb_ratio=1.3))):
        f = st.closure(area=9e6, height=120.0, aspect=1.4, azimuth=20.0, center=(4000.0, 3000.0), seed=7 + k, **kw)
        out[f"closure{k}"] = digest([f(x, y).tolist(), list(f.crest_offset)])
    out["roughness"] = digest(st.roughness(9.0, 1500.0, 8000.0, 6000.0, seed=4)(x, y).tolist())
    return out


def test_the_exports_of_the_published_resmill_are_unchanged():
    assert exports() == EXPORT


def test_the_fault_patterns_that_existed_before_salt_are_unchanged():
    assert patterns() == PATTERNS


def test_a_faulted_export_with_a_fault_seal_is_unchanged():
    assert faulted() == FAULTED


def test_the_closure_and_the_roughness_are_unchanged():
    assert closures() == CLOSURE


if __name__ == "__main__":
    print(json.dumps({"EXPORT": exports(), "PATTERNS": patterns() if "--no-patterns" not in sys.argv else {},
                      "FAULTED": faulted() if "--no-patterns" not in sys.argv else {}, "CLOSURE": closures()}, indent=1))
