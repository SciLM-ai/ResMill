"""The dataset's training window: one 64 x 64 x 32 window per stored volume.

``window_origin`` is the rule ``examples/dataset_generation/crop_windows.py``
applies to every volume of the dataset, kept in the package so that anything
that must reproduce a dataset sample from the engine's output (the ResBench
reference builders, for instance) uses the same draw.

The origin of a sample comes from ``numpy.random.default_rng([crop_seed, seed])``
with the dataset's ``crop_seed`` (42) and the sample's own ``seed``: ``x0``
uniform on ``[0, nx - wx]``, ``y0`` on ``[0, ny - wy]``, ``z0`` on
``[z_min, nz - wz - 1]`` (``z_min`` 1, so the window never contains the engine's
floor or roof, where the level ladder is anchored). When ``facies`` is given the
draw is repeated from the same stream while the window holds fewer than
``min_sand_cells`` sand cells (default 1), so no window is mud only.
"""
from __future__ import annotations

import numpy as np

WINDOW = (64, 64, 32)
CROP_SEED = 42
Z_MIN = 1
MIN_SAND_CELLS = 1


def window_origin(crop_seed: int, sample_seed: int, shape, win=WINDOW, z_min: int = Z_MIN,
                  facies=None, min_sand_cells: int = MIN_SAND_CELLS, max_tries: int = 64):
    """Origin ``(x0, y0, z0)`` of the window of one sample; see the module doc."""
    nx, ny, nz = shape
    wx, wy, wz = win
    rng = np.random.default_rng([int(crop_seed), int(sample_seed)])
    for _ in range(max_tries):
        x0 = int(rng.integers(0, nx - wx + 1))
        y0 = int(rng.integers(0, ny - wy + 1))
        z0 = int(rng.integers(z_min, nz - wz))      # high is exclusive: z0 <= nz - wz - 1
        if facies is None or min_sand_cells <= 0:
            break
        if int(np.count_nonzero(facies[x0:x0 + wx, y0:y0 + wy, z0:z0 + wz])) >= min_sand_cells:
            break
    return x0, y0, z0


def cut(volume, origin, win=WINDOW):
    """The window of ``volume`` at ``origin``, as a contiguous array."""
    x0, y0, z0 = origin
    wx, wy, wz = win
    return np.ascontiguousarray(volume[x0:x0 + wx, y0:y0 + wy, z0:z0 + wz])


def dataset_window(facies, sample_seed: int, crop_seed: int = CROP_SEED, win=WINDOW):
    """Origin and facies window of an engine volume exactly as the dataset
    would store it for a sample with this seed."""
    origin = window_origin(crop_seed, sample_seed, facies.shape, win, Z_MIN, facies, MIN_SAND_CELLS)
    return origin, cut(facies, origin, win)
