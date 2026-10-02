"""Helpers shared by the stratigraphic-trap tests: flat layers, a disc outline, the fine analytic map of a built trap."""
import numpy as np

from resmill import structure as st
from resmill.layers.base import Layer
from resmill.trap_measure import _traps


def _disc(cx, cy, radius):
    """An outline of a disc: negative inside, as ``closure`` is (a distance, so its footprint is exact)."""
    return st.Structure(lambda x, y: np.hypot(np.asarray(x) - cx, np.asarray(y) - cy) - radius)


def _layers(x_len, y_len, dx, thicknesses, top=2000.0, dz=2.0):
    """Flat layers, top to bottom, ``thicknesses`` (m) thick, of cells ``dx`` m wide and about ``dz`` m high."""
    layers, depth = [], top
    for t in thicknesses:
        layers.append(Layer(int(round(x_len / dx)), int(round(y_len / dx)), max(int(round(t / dz)), 1), x_len, y_len, t,
                            depth))
        depth += t
    return layers


def _length(area, aspect):
    """The research note's length along dip of an elliptical trap of ``area`` and ``aspect`` (strike over dip)."""
    return 2.0 * np.sqrt(area / (np.pi * aspect))


def _fine_traps(built, top, x_len, y_len, n=401):
    """The traps of the geometry a generator drew, on a fine analytic map of the top of its sand (n x n points): the
    structure and the thickness factor or erosion surface it returned, and the flood of the cells' report. It reads
    neither the corner-point grid nor which cells ACTNUM keeps, so it is what the cells' trap is checked against."""
    kw, thick = built["kwargs"], built["meta"]["thickness"]
    x, y = np.meshgrid(np.linspace(0.0, x_len, n), np.linspace(0.0, y_len, n), indexing="ij")
    bed = top + kw["structure"](x, y)                              # the top of the beds
    if "erode_above" in kw:                                        # truncation: the surface is the top, where it cuts
        depth, present = kw["erode_above"](x, y), kw["erode_above"](x, y) < bed + thick - 1e-9
    elif "erode_below" in kw:                                      # onlap: the layers follow the top, to the surface
        depth, present = bed, kw["erode_below"](x, y) > bed + 1e-9
    else:
        depth, present = bed, kw["isochore"][0](x, y) > 0.0
    return sorted(_traps(np.where(present, depth, np.inf), x_len / (n - 1), y_len / (n - 1)),
                  key=lambda trap: -trap["height"])


NOSE = dict(area=6.0e6, height=60.0, aspect=0.7)       # 6 km2, 60 m of relief, longer along dip than across
