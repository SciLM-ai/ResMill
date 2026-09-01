"""Angular unconformity: four folded layers eroded flat, two flat layers above.

The lower four layers share one cylindrical anticline; the upper two get no
structure. Where the fold lifts the older interfaces above the flat base of
the upper package, the exporter's younger-truncates-older stitch clamps them
to it: the fold crest is beveled off, successively older layers subcrop
against the unconformity, and the eroded cells collapse to zero thickness
(written inactive). No erosion surface needs to be passed; the base of the
flat package is the erosion surface.

Run from the repo root:

    python examples/angular_unconformity.py           # writes .grdecl + .png
    python examples/angular_unconformity.py --show    # + interactive 3-D (needs pyvista)
"""
import sys

import matplotlib.pyplot as plt

import resmill as rm
from resmill import structure as st

GRID = dict(nx=64, ny=64, x_len=640, y_len=640)


def gaussian(top_depth, thickness, poro):
    layer = rm.GaussianLayer(nz=thickness, z_len=thickness, top_depth=top_depth, **GRID)
    layer.create_geology(poro_ave=poro, perm_ave=1.5, poro_std=0.02,
                        perm_std=0.3, ntg=1.0)
    return layer


# Two flat post-unconformity layers (1900-1912 m) over four layers that will
# be folded (1912-1944 m). Contrasting porosity makes the bands read clearly.
upper = [gaussian(1900, 6, 0.10), gaussian(1906, 6, 0.21)]
folded = [gaussian(1912, 8, 0.27), gaussian(1920, 8, 0.13),
          gaussian(1928, 8, 0.24), gaussian(1936, 8, 0.17)]
res = rm.Reservoir(upper + folded)

# One full cylindrical fold across the domain, crest lifted 22 m: enough to
# erode the top two folded layers completely at the crest and bite into the
# third, leaving a subcrop pattern under the flat cover.
fold = st.anticline(amplitude=22, wavelength=640, azimuth=90)
shifts = [None, None, fold, fold, fold, fold]   # one entry per layer, top->bottom

path = res.to_grdecl("angular_unconformity.grdecl", structure=shifts)
print(f"wrote {path}")

fig, ax = plt.subplots(figsize=(9, 4))
rm.plot_section(res, structure=shifts, ax=ax,
                title="Angular unconformity — folded layers beveled flat "
                      "under two undeformed layers")
fig.tight_layout()
fig.savefig("angular_unconformity.png", dpi=110)
print("wrote angular_unconformity.png")

if "--show" in sys.argv:
    import pyvista as pv

    grid = rm.to_pyvista(res, structure=shifts)
    p = pv.Plotter()
    p.add_mesh(grid, scalars="PORO", cmap="resmill",
               show_edges=True, edge_color="#555555", line_width=0.4)
    p.set_scale(zscale=5)
    p.show()
