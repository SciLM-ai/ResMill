"""A meandering river folded into an anticline, exported for Petrel/Eclipse.

Run from the repo root:

    python examples/anticline_meander.py           # writes .grdecl + .png
    python examples/anticline_meander.py --show    # + interactive 3-D (needs pyvista)
"""
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import resmill as rm
from resmill import structure as st
from resmill.layers.channel import MEANDER_OXBOW
from resmill.plotting import ALLUVSIM_FACIES_COLORS, alluvsim_cmap

# A 640 x 640 m meander belt, 16 m thick, buried at 2000 m.
river = rm.ChannelLayer(nx=64, ny=64, nz=16, x_len=640, y_len=640,
                        z_len=16, top_depth=2000)
river.create_geology(seed=7, **MEANDER_OXBOW)

# A doubly plunging anticline (four-way dip closure): crest lifted 55 m,
# long axis along y, dip falling away from the crest in every direction —
# the classic structural trap. The river (flowing +x) crosses the crest.
trap = st.dome(amplitude=55, radius=170, aspect=2.2, azimuth=90)

path = river.to_grdecl("anticline_meander.grdecl", structure=trap, facies=True)
print(f"wrote {path}")

fig, axes = plt.subplots(3, 1, figsize=(9, 10), height_ratios=[1.4, 1, 1])
cmap, norm = alluvsim_cmap()
axes[0].imshow(river.facies[:, :, river.nz // 2].T, origin="lower", cmap=cmap,
               norm=norm, interpolation="nearest",
               extent=(0, river.x_len, 0, river.y_len))
axes[0].set_title("Map view — facies at mid depth (river flows +x, trap long axis runs y)")
axes[0].set_xlabel("X (m)")
axes[0].set_ylabel("Y (m)")
rm.plot_section(river, structure=trap, axis="y", ax=axes[1],
                title="XZ section through the crest — closes in x")
rm.plot_section(river, structure=trap, axis="x", ax=axes[2],
                title="YZ section along the long axis — plunges at both ends (closes in y)")
fig.tight_layout()
fig.savefig("anticline_meander.png", dpi=110)
print("wrote anticline_meander.png")

if "--show" in sys.argv:
    import pyvista as pv

    grid = rm.to_pyvista(river, structure=trap)
    codes = sorted(ALLUVSIM_FACIES_COLORS)
    fac_cmap = ListedColormap([ALLUVSIM_FACIES_COLORS[c] for c in codes])
    # Petrel-style corner-point rendering: every cell drawn from its own
    # eight corners, with the cell edges visible.
    p = pv.Plotter()
    p.add_mesh(grid, scalars="FACIES", cmap=fac_cmap,
               clim=(codes[0] - 0.5, codes[-1] + 0.5),
               show_edges=True, edge_color="grey", line_width=0.5)
    p.set_scale(zscale=6)   # 6x vertical exaggeration: the layer is 16 m in a 640 m box
    p.show()
    # Tip: to see only the sand geobodies (facies >= 1, like a Petrel
    # facies filter), plot grid.threshold(0.5, scalars="FACIES") instead.
