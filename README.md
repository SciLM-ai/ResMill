# ResMill

Rule-based 3D geological reservoir modeling.

## Install

```bash
pip install resmill
```

For development:

```bash
git clone https://github.com/IlgarBaghishov/ResMill.git
cd ResMill
pip install -e ".[dev]"
```

## Quick Start

```python
import resmill as rm

# Create a turbidite lobe layer
lobe = rm.LobeLayer(nx=100, ny=100, nz=50, x_len=3000, y_len=3000, z_len=100, top_depth=5000)
lobe.create_geology(poro_ave=0.20, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)

# Create a fluvial channel layer (Alluvsim-faithful meandering/avulsing system)
channel = rm.ChannelLayer(nx=100, ny=100, nz=50, x_len=3000, y_len=3000, z_len=100, top_depth=5100)
channel.create_geology(NTGtarget=0.10, mCHdepth=4.0, mCHwdratio=10.0, seed=0)

# Stack into a reservoir
reservoir = rm.Reservoir([lobe, channel])

# Visualize
rm.plot_cube_slices(reservoir.poro_mat, title="Porosity")
```

## Layer Types

- **LobeLayer** — Turbidite lobe deposition with compensational stacking
- **GaussianLayer** — Sequential Gaussian simulation for heterogeneous facies
- **ChannelLayer** — Fluvial channel systems with meandering, migration, avulsion, neck cutoffs, levees, and crevasse splays
- **DeltaLayer** — Delta/distributary systems built on the fluvial engine

## License

MIT
