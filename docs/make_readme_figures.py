"""Regenerate the README preset figures.

Each figure is a 1x3 panel — XY map view, XZ section, YZ section — through
the mid-planes of a model built on the standard 64x64x32 tutorial grid.
Colouring matches what ``rm.plot_slices`` / ``rm.plot_cube_slices`` produce
(``origin='lower'`` — vertical cell index Z increases upward, so the top of
the deposited interval is at the top of each section):

  * channel / delta — multi-class Alluvsim facies (CH/LA/LV/CS/FFCH/FF)
  * lobe            — binary sand / non-reservoir facies
  * gaussian        — continuous SGS porosity field

Run from the repo root:  python docs/make_readme_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

import resmill as rm
from resmill.layers.channel import (
    PV_SHOESTRING, CB_JIGSAW, CB_LABYRINTH, SH_DISTAL, SH_PROXIMAL, MEANDER_OXBOW,
)
from resmill.plotting import (
    alluvsim_cmap, ALLUVSIM_FACIES_NAMES, ALLUVSIM_FACIES_COLORS, RESMILL_CMAP,
)

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

GRID = dict(nx=64, ny=64, nz=32, x_len=640, y_len=640, z_len=32, top_depth=0)
SEED = 42


def _panels(arr, fname, title, cmap, norm, legend_handles=None, cbar_label=None):
    nx, ny, nz = arr.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))

    im = axes[0].imshow(arr[:, :, nz // 2].T, origin="lower",
                        cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    axes[0].set_title("XY  (map view, mid Z)")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")

    axes[1].imshow(arr[:, ny // 2, :].T, origin="lower",
                   cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    axes[1].set_title("XZ  (section, mid Y)")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Z  (↑ top of interval)")

    axes[2].imshow(arr[nx // 2, :, :].T, origin="lower",
                   cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    axes[2].set_title("YZ  (section, mid X)")
    axes[2].set_xlabel("Y"); axes[2].set_ylabel("Z  (↑ top of interval)")

    fig.suptitle(title, fontsize=13, weight="bold")

    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="lower center", ncol=6,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
        rect = [0, 0.05, 1, 0.93]
    else:
        cbar = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.04, shrink=0.9)
        cbar.set_label(cbar_label or "")
        rect = [0, 0, 0.96, 0.93]

    fig.tight_layout(rect=rect)
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


_FACIES_HANDLES = [
    Patch(facecolor=ALLUVSIM_FACIES_COLORS[k], edgecolor="none",
          label=ALLUVSIM_FACIES_NAMES[k])
    for k in sorted(ALLUVSIM_FACIES_NAMES)
]


def facies_fig(layer, fname, title):
    cmap, norm = alluvsim_cmap()
    _panels(layer.facies, fname, title, cmap, norm, legend_handles=_FACIES_HANDLES)


def poro_fig(layer, fname, title):
    arr = np.where(layer.active > 0, layer.poro_mat, np.nan)
    cmap = plt.get_cmap(RESMILL_CMAP).copy()
    cmap.set_bad("#e8e8e8")
    vmax = float(np.nanmax(arr))
    norm = plt.Normalize(vmin=0, vmax=vmax)
    _panels(arr, fname, title, cmap, norm, cbar_label="porosity")


_BINARY_COLORS = ["#e3e3e3", "#c98a3c"]  # 0 = non-reservoir, 1 = sand
_BINARY_HANDLES = [
    Patch(facecolor=_BINARY_COLORS[0], edgecolor="none", label="non-reservoir"),
    Patch(facecolor=_BINARY_COLORS[1], edgecolor="none", label="sand (reservoir)"),
]


def binary_fig(layer, fname, title):
    """Binary sand / non-reservoir facies view (uses ``layer.active``)."""
    cmap = ListedColormap(_BINARY_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    arr = (layer.active > 0).astype(int)
    _panels(arr, fname, title, cmap, norm, legend_handles=_BINARY_HANDLES)


# ---- Lobe -----------------------------------------------------------------
lobe = rm.LobeLayer(**GRID)
lobe.create_geology(poro_ave=0.20, perm_ave=1.5, poro_std=0.03, perm_std=0.5,
                    ntg=0.7, r_ave=180, r_std=30, asp=1.5, upthinning=True)
binary_fig(lobe, "lobe.png", "LobeLayer — turbidite lobes (sand facies)")

# ---- Gaussian -------------------------------------------------------------
gauss = rm.GaussianLayer(**GRID)
gauss.create_geology(poro_ave=0.18, perm_ave=1.5, poro_std=0.04, perm_std=0.5, ntg=0.6)
poro_fig(gauss, "gaussian.png", "GaussianLayer — SGS porosity field")

# ---- Channel presets ------------------------------------------------------
for preset, fname, title in [
    (PV_SHOESTRING, "channel_pv_shoestring.png", "ChannelLayer — PV_SHOESTRING (paleo-valley shoestring)"),
    (CB_JIGSAW,     "channel_cb_jigsaw.png",     "ChannelLayer — CB_JIGSAW (channel-and-bar jigsaw)"),
    (CB_LABYRINTH,  "channel_cb_labyrinth.png",  "ChannelLayer — CB_LABYRINTH (labyrinthine channel bodies)"),
    (SH_DISTAL,     "channel_sh_distal.png",     "ChannelLayer — SH_DISTAL (distal sand sheet)"),
    (SH_PROXIMAL,   "channel_sh_proximal.png",   "ChannelLayer — SH_PROXIMAL (proximal sand sheet)"),
    (MEANDER_OXBOW, "channel_meander_oxbow.png", "ChannelLayer — MEANDER_OXBOW (meander belt with oxbow mud plugs)"),
]:
    ch = rm.ChannelLayer(**GRID)
    ch.create_geology(seed=SEED, **preset)
    facies_fig(ch, fname, title)

# ---- Delta ----------------------------------------------------------------
delta = rm.DeltaLayer(**GRID)
delta.create_geology(seed=3)
facies_fig(delta, "delta_fan.png", "DeltaLayer — DELTA_FAN (distributary delta)")

print("done")
