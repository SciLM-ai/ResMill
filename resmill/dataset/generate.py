"""Generate one reservoir sample and package outputs for storage.

Wraps a ResMill Layer construction and ``create_geology`` call with
deterministic seeding, binarises the resulting facies / active array,
and casts porosity and permeability to float16 for on-disk compactness.

``layer_type`` strings exposed by this module:

* ``lobe``     — :py:class:`resmill.LobeLayer`
* ``gaussian`` — :py:class:`resmill.GaussianLayer`
* ``channel``  — :py:class:`resmill.ChannelLayer` (Alluvsim-faithful
  fluvial engine; supports the canonical Pyrcz 2004 presets plus
  MEANDER_OXBOW via the ``preset`` config field — see
  :func:`_apply_preset`)
* ``delta``    — :py:class:`resmill.DeltaLayer` (fluvial-engine delta
  with trunk-length / progradation / branch-spread / mouth-bar
  controls; see ``DELTA_FAN`` baseline)
"""

import numpy as np

import resmill as rm
from .captions import caption_for


_LAYER_FACTORY = {
    "lobe": rm.LobeLayer,
    "gaussian": rm.GaussianLayer,
    "channel": rm.ChannelLayer,
    "delta": rm.DeltaLayer,
}


def _apply_preset(params: dict) -> dict:
    """Resolve any ``"preset"`` entry in ``params`` to a baseline dict.

    If ``params["preset"]`` is set to one of the known preset names,
    the matching preset dict is loaded and ``params`` (excluding
    ``"preset"`` itself) is merged on top — i.e. user values override
    the preset baseline. Allows config-side syntax like::

        "params": {
            "preset": {"choices": ["PV_SHOESTRING", "CB_JIGSAW"]},
            "mCHsinu": {"range": [1.1, 1.6]}
        }

    so the dataset can mix five canonical Alluvsim architectures with
    only a handful of varying knobs on top.
    """
    preset_name = params.pop("preset", None)
    if preset_name is None:
        return params
    from resmill.layers.channel import (
        PV_SHOESTRING, CB_JIGSAW, CB_LABYRINTH, SH_DISTAL, SH_PROXIMAL,
        MEANDER_OXBOW,
    )
    from resmill.layers.delta import DELTA_FAN
    PRESETS = {
        "PV_SHOESTRING": PV_SHOESTRING,
        "CB_JIGSAW": CB_JIGSAW,
        "CB_LABYRINTH": CB_LABYRINTH,
        "SH_DISTAL": SH_DISTAL,
        "SH_PROXIMAL": SH_PROXIMAL,
        "MEANDER_OXBOW": MEANDER_OXBOW,
        "DELTA_FAN": DELTA_FAN,
    }
    if preset_name not in PRESETS:
        raise ValueError(
            f"unknown preset {preset_name!r}; "
            f"expected one of {sorted(PRESETS)}")
    out = dict(PRESETS[preset_name])
    out.update(params)
    return out


def generate_sample(job: dict, grid_cfg: dict):
    """Generate one sample.

    If ``grid_cfg`` contains a ``crop`` field (a dict like
    ``{"x": "8:-8", "y": "8:-8", "z": "9:-9"}``), each axis is sliced
    using Python slice semantics so the layer is built on the larger
    grid and the interior is kept free of boundary artifacts. Slice
    strings are ``"start:stop"`` with negative indices allowed
    (``"8:-8"`` symmetric, ``"4:-12"`` asymmetric, ``":"`` no crop on
    that axis). A 2-list ``[start, stop]`` is also accepted. The
    realized NTG is recomputed from the cropped facies and stored as
    ``meta["ntg"]``; the input NTG (``params["ntg"]`` or
    ``params["NTGtarget"]``) is preserved as ``meta["requested_ntg"]``
    so downstream flow-matching that needs the post-crop value to
    match the data.

    Returns ``(facies, poro, perm, facies_alluvsim, meta)``:

    - ``facies`` — int8 array in {0, 1}, shape ``(nx, ny, nz)`` (binary
      sand vs mud).
    - ``poro``   — float16 array, shape ``(nx, ny, nz)``
    - ``perm``   — float16 array (mD), shape ``(nx, ny, nz)``
    - ``facies_alluvsim`` — int8 array, shape ``(nx, ny, nz)``, raw
      Alluvsim 6-class codes (-1=FF, 0=FFCH, 1=CS, 2=LV, 3=LA, 4=CH).
      Lets downstream tools render the jigsaw/labyrinth signature
      that's invisible in the binarized ``facies``.
    - ``meta``   — dict with ``layer_type``, ``seed``, ``caption``,
      every sampled parameter, plus ``ntg`` (post-crop, realized) and
      ``requested_ntg`` (the value passed into ``create_geology``).
    """
    seed = int(job["seed"])
    np.random.seed(seed)
    layer_type = job["layer_type"]
    params = _apply_preset(dict(job["params"]))

    # Strip ``crop`` from grid_cfg before passing to the layer
    # constructor — it's a generator-only knob, not a Layer kwarg.
    # Also strip any ``_comment_*`` keys (config inline documentation
    # that the JSON parser keeps but Layer.__init__ doesn't accept).
    crop_spec = grid_cfg.get("crop")
    layer_grid_kw = {k: v for k, v in grid_cfg.items()
                     if k != "crop" and not k.startswith("_comment_")}

    layer = _LAYER_FACTORY[layer_type](**layer_grid_kw)
    layer.create_geology(**params)

    facies = _binarize(layer, layer_type).astype(np.int8)
    # Raw 6-class Alluvsim facies (codes -1..4). For lobe / gaussian
    # this is just (-1, 3) — same info as the binary ``facies`` — but
    # for channel / delta layers it preserves CH/LA/LV/CS/FFCH/FF
    # distinctions that the jigsaw / labyrinth visual signature lives
    # on. ``GaussianLayer`` does build a ``facies`` attr in our ports,
    # so this is always present.
    facies_alluvsim = np.asarray(
        getattr(layer, "facies", layer.active), dtype=np.int8
    )
    # Clip to float16-safe ranges before casting: poro to [0, 1], perm to
    # [0, 60000] mD. This also silently coerces any residual +inf from
    # extreme perm_std draws into 60000 (float16 max is 65504).
    poro = np.clip(
        np.asarray(layer.poro_mat, dtype=np.float32), 0.0, 1.0
    ).astype(np.float16)
    perm = np.clip(
        np.asarray(layer.perm_mat, dtype=np.float32), 0.0, 6e4
    ).astype(np.float16)

    if crop_spec is not None:
        sl = _build_crop_slices(crop_spec)
        facies = facies[sl]
        poro = poro[sl]
        perm = perm[sl]
        facies_alluvsim = facies_alluvsim[sl]

    realized_ntg = realized_stats(facies, poro, perm)["ntg"]

    # Pre-compute physical (m) and cell-count equivalents for the size
    # parameters that show up in captions / parquet. Cell count of a
    # length L (m) along a unit direction (a, b) is
    #     L / sqrt((a*dx)^2 + (b*dy)^2)
    # — which collapses to L/dx for an isotropic grid (dx=dy) but stays
    # correct for anisotropic dx≠dy at any azimuth.
    dx = float(layer_grid_kw["x_len"]) / int(layer_grid_kw["nx"])
    dy = float(layer_grid_kw["y_len"]) / int(layer_grid_kw["ny"])
    dz = float(layer_grid_kw["z_len"]) / int(layer_grid_kw["nz"])
    az_rad = np.deg2rad(float(params.get("azimuth", 0.0)))
    cs_az = np.cos(az_rad); sn_az = np.sin(az_rad)
    # cell size measured along the channel-flow / lobe-major axis direction
    cell_along  = float(np.sqrt((cs_az * dx) ** 2 + (sn_az * dy) ** 2))
    # cell size measured perpendicular (channel-width / lobe-minor)
    cell_perp   = float(np.sqrt((sn_az * dx) ** 2 + (cs_az * dy) ** 2))

    size_meta: dict = {}
    # Channel-family geometry: keep both _m (full parquet) and bare
    # cell-units (slim parquet) versions.
    if "mCHdepth" in params:
        depth_m = float(params["mCHdepth"])
        size_meta["mCHdepth_m"]     = depth_m
        size_meta["mCHdepth_cells"] = depth_m / dz
        if "mCHwdratio" in params:
            width_m = depth_m * float(params["mCHwdratio"])
            size_meta["mCHwidth_m"]     = width_m
            size_meta["mCHwidth_cells"] = width_m / cell_perp
    if "r_ave" in params:
        # Lobe semi-minor radius (m) → cells; semi-major derivable from asp.
        r_m = float(params["r_ave"])
        asp = float(params.get("asp", 1.5))
        size_meta["r_ave_m"]        = r_m
        size_meta["r_ave_cells"]    = r_m / cell_perp                # semi-minor (cells)
        size_meta["r_major_m"]      = r_m * asp
        size_meta["r_major_cells"]  = (r_m * asp) / cell_along       # semi-major (cells)
    if "dh_ave" in params:
        size_meta["dh_ave_m"]     = float(params["dh_ave"])
        size_meta["dh_ave_cells"] = float(params["dh_ave"]) / dz

    # Slim-parquet unified geometry: ``width_cells`` = full lateral extent
    # of the depositional unit in cells; ``depth_cells`` = thickness.
    # Lobe full-minor diameter = 2 × semi-minor (cells); channels already
    # store full mCHwidth, so use as-is. Vertical: lobe dh_ave (cells) ≡
    # channel mCHdepth (cells).
    if "r_ave_cells" in size_meta:
        size_meta["width_cells"] = 2.0 * size_meta["r_ave_cells"]
        if "dh_ave_cells" in size_meta:
            size_meta["depth_cells"] = size_meta["dh_ave_cells"]
    if "mCHwidth_cells" in size_meta:
        size_meta["width_cells"] = size_meta["mCHwidth_cells"]
        size_meta["depth_cells"] = size_meta["mCHdepth_cells"]

    # Realized poro/perm averages (active cells only) — flow-matching
    # condition. Computed post-crop so they match the saved arrays.
    size_meta.update(realized_stats(facies, poro, perm))

    # Caption should describe the realized data, not the request, so
    # text-conditioned downstream models train on faithful (data ↔ text)
    # pairs. Pass a copy of params with the realized NTG patched in.
    # Re-attach the preset name (popped by ``_apply_preset``) so the
    # caption template can lead with the architectural identity.
    caption_params = dict(params)
    caption_params.update(size_meta)
    preset_name = job["params"].get("preset")
    if preset_name is not None:
        caption_params["preset"] = preset_name
    caption_params["ntg"] = realized_ntg

    # Encoded layer_type for slim parquet: channels carry the architecture
    # name as a suffix so the slim parquet can drop the standalone
    # ``preset`` column and still distinguish the 6 channel architectures.
    if layer_type == "channel" and preset_name is not None:
        encoded_layer_type = f"channel:{preset_name}"
    else:
        encoded_layer_type = layer_type

    meta = {
        "layer_type": encoded_layer_type,
        "seed": seed,
        "caption": caption_for(layer_type, caption_params),
        **params,
        **size_meta,
    }
    # Re-attach preset for parquet/meta so the caption template can be
    # re-rendered downstream without losing architectural identity
    # (``_apply_preset`` pops ``preset`` from ``params`` before passing
    # to the layer kwargs).
    if preset_name is not None:
        meta["preset"] = preset_name
    # Preserve the input value as ``requested_ntg`` and overwrite ``ntg``
    # with the realized fraction so downstream consumers always see the
    # value that matches the saved facies array.
    if "ntg" in params:
        meta["requested_ntg"] = params["ntg"]
    elif "NTGtarget" in params:
        meta["requested_ntg"] = params["NTGtarget"]
    meta["ntg"] = realized_ntg
    return facies, poro, perm, facies_alluvsim, meta


def realized_stats(facies, poro, perm) -> dict:
    """Realised reservoir statistics of one stored cube, as recorded in the
    parquet: ``ntg`` = sand fraction of the binary facies, ``poro_ave`` = mean
    porosity and ``perm_ave`` = mean log10 permeability over sand cells (None
    when there is no sand). Shared by ``generate_sample`` and
    ``examples/dataset_generation/crop_windows.py`` so a window is described
    exactly like a volume."""
    facies = np.asarray(facies)
    out = {"ntg": float(facies.mean())}
    active_mask = facies > 0
    if active_mask.any():
        # poro/perm are float16 cubes
        poro_f32 = np.asarray(poro).astype(np.float32)
        perm_f32 = np.maximum(np.asarray(perm).astype(np.float32), 1e-3)  # log-safe floor
        out["poro_ave"] = float(poro_f32[active_mask].mean())
        out["perm_ave"] = float(np.log10(perm_f32[active_mask]).mean())
    else:
        out["poro_ave"] = None
        out["perm_ave"] = None
    return out


def _parse_axis_slice(spec):
    """Parse one axis spec (string ``"a:b"`` or 2-list) into a slice.

    Empty start/stop means "no bound": ``":"`` → ``slice(None, None)``,
    ``":-9"`` → ``slice(None, -9)``. Negative indices count from the
    end the same way Python slicing does, so ``"8:-8"`` is symmetric
    and ``"4:-12"`` is asymmetric.
    """
    if isinstance(spec, str):
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"crop slice spec must be 'start:stop', got {spec!r}"
            )
        a = int(parts[0]) if parts[0] else None
        b = int(parts[1]) if parts[1] else None
        return slice(a, b)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        a, b = spec
        return slice(
            int(a) if a is not None else None,
            int(b) if b is not None else None,
        )
    raise ValueError(
        f"crop spec must be 'start:stop' string or [start, stop] list, "
        f"got {spec!r}"
    )


def _build_crop_slices(crop_spec: dict):
    """Build a 3-tuple of slices ``(sx, sy, sz)`` from a config crop dict."""
    if not isinstance(crop_spec, dict):
        raise ValueError(
            f"crop must be a dict with keys 'x', 'y', 'z'; got {crop_spec!r}"
        )
    return tuple(
        _parse_axis_slice(crop_spec.get(ax, ":"))
        for ax in ("x", "y", "z")
    )


def _binarize(layer, layer_type: str) -> np.ndarray:
    # GaussianLayer has no ``facies`` attribute; use ``active`` (0/1).
    # All other layer types: fold any ``facies > 0`` into binary 1
    # (LobeLayer encodes the lobe generation index as int, not 0/1).
    if layer_type == "gaussian":
        return np.asarray(layer.active) > 0
    return np.asarray(layer.facies) > 0
