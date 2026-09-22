"""Parameter-space sampling for dataset generation.

Reads the ``layers`` section of a dataset config and returns a
memory-efficient :class:`JobList` that lazily materialises one
``{layer_type, params, seed}`` dict per indexed access. At 10M samples the
list-of-dicts representation would cost ~8 GB per worker, which does not
fit 128 concurrent ranks on an HPC CPU node; the compact storage
here keeps the per-rank overhead under ~1 GB even at tens of millions
of samples.

Per-parameter specs supported:

- ``{"range": [lo, hi]}``                       continuous float uniform
- ``{"range": [lo, hi], "scale": "log"}``       log-uniform in [lo, hi]
- ``{"range": [lo, hi], "type": "int"}``        integer in [lo, hi] inclusive
- ``{"choices": [...]}``                         categorical (int/str/bool)
- ``{"value": v}``                               fixed scalar
- ``{"levels": N}`` (grid sampling only)        number of grid steps on that axis
- ``{"fraction_of": "X", "value": k}``          derived: k * params["X"]
- ``{"linear_of": "X", "slope": a, "intercept": b}``  derived: a*X + b
  (intercept defaults to 0)
- ``{"inverse_of": "X", "scale": s, "min": m, "max": M, "type": "int"}``
  derived: clip(s / params["X"], m, M). Use this to auto-tune e.g.
  ``nlevel`` so deeper channels stack fewer levels.
- ``{"levels_from_ratio": "mCHdepth", "ratio": [lo, hi], "column": H,
  "base": b, "min": m, "max": M}``   number of channel levels from a
  sampled aggradation ratio: the ratio r (level spacing over channel
  depth) is drawn uniformly in [lo, hi] on this param's own unit-cube
  column, then ``round((H - b) / (r * depth)) + 1`` clipped to [m, M].
  ``H`` is the layer's ``z_len`` (the engine spreads the levels over
  the whole column) and ``b`` the height of the lowest level top, which
  defaults to the depth (bottom channel base on the floor, as both
  ``ChannelLayer`` and ``DeltaLayer`` anchor it); pass ``"base"`` only
  for a layer anchored elsewhere. r below 1 makes successive levels cut into each other,
  above 1 leaves floodplain between them, at most (r - 1) x depth.
- ``{"range": [...], "shared": "tag"}``         couple params by tying them
  to the same Sobol/LHS coordinate — every param with the same tag uses
  the same unit-cube draw, so they are 100%-rank-correlated. Use this
  to avoid pathological corners (e.g. small lobes paired with high NTG).
- ``{"range": [...], "shared": "tag", "jitter": j}`` — same as above
  but mix in a fraction ``j ∈ [0, 1]`` of independent noise. ``j=0``
  is pure shared (corr=1), ``j=0.5`` gives corr=0.5, ``j=1`` is fully
  independent. Implemented as
  ``u_effective = (1-j)*u_shared + j*u_indep`` where ``u_indep`` is
  this param's own private Sobol column.

Sampling strategies:

- ``"sobol"``   — ``scipy.stats.qmc.Sobol(scramble=True)``
- ``"lhs"``     — ``scipy.stats.qmc.LatinHypercube``
- ``"grid"``    — full factorial Cartesian product of per-param levels
- ``"uniform"`` — IID uniform via ``numpy.random.default_rng``
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc


def _is_fixed(spec: dict) -> bool:
    return "value" in spec and not _is_derived(spec)


@dataclass
class _LayerState:
    name: str
    variable_names: list[str]
    unit_col_map: list[int]      # primary column for each variable_name
    jitter_col: list[int]        # extra independent column when jitter>0, else -1
    jitter_amt: list[float]      # 0.0 when no jitter
    derived: list[tuple]         # (name, spec) resolved per-sample after variables
    fixed: dict[str, Any]
    unit: np.ndarray             # (count, n_unit_cols) float32 unit-cube samples
    grid_combos: list[tuple] | None   # only for sampling="grid"


class JobList:
    """Shuffled list of jobs with ``O(N)`` memory and lazy materialisation.

    Indexing ``job_list[i]`` returns a dict
    ``{"layer_type": str, "params": dict, "seed": int}`` equivalent to
    what the previous eager implementation produced — same content, just
    computed on demand.
    """

    def __init__(
        self,
        layer_names: list[str],
        layer_type_ids: np.ndarray,
        within_idx: np.ndarray,
        seeds: np.ndarray,
        layer_states: dict[str, _LayerState],
        layers_cfg: dict,
    ):
        self._layer_names = layer_names
        self._layer_type_ids = layer_type_ids
        self._within_idx = within_idx
        self._seeds = seeds
        self._layer_states = layer_states
        self._layers_cfg = layers_cfg

    def __len__(self) -> int:
        return int(self._layer_type_ids.shape[0])

    def __getitem__(self, i: int) -> dict:
        lt = self._layer_names[int(self._layer_type_ids[i])]
        within = int(self._within_idx[i])
        seed = int(self._seeds[i])
        state = self._layer_states[lt]
        params_cfg = self._layers_cfg[lt]["params"]
        params = dict(state.fixed)

        if state.grid_combos is not None:
            combo = state.grid_combos[within]
            params.update(dict(zip(state.variable_names, combo)))
        else:
            unit_row = state.unit[within]
            deferred = []
            for name, pcol, jcol, jamt in zip(
                state.variable_names, state.unit_col_map,
                state.jitter_col, state.jitter_amt,
            ):
                u_primary = float(unit_row[pcol])
                if jcol < 0:
                    u = u_primary
                else:
                    u = (1.0 - jamt) * u_primary + jamt * float(unit_row[jcol])
                if "levels_from_ratio" in params_cfg[name]:
                    deferred.append((name, u))
                    continue
                params[name] = _map_unit_value(u, params_cfg[name])
            # Level counts need the channel depth, so resolve them once
            # every plain variable is in place.
            for name, u in deferred:
                params[name] = _levels_from_ratio(u, params_cfg[name], params)

        # Derived params (fraction_of / linear_of) reference already-
        # resolved variable / fixed params, so resolve them last.
        for name, spec in state.derived:
            params[name] = _resolve_derived(spec, params)

        return {"layer_type": lt, "params": params, "seed": seed}


def build_jobs(layers_cfg: dict, master_seed: int) -> JobList:
    """Build a shuffled :class:`JobList` over every layer section."""
    layer_names = list(layers_cfg.keys())
    counts = np.array([int(cfg["count"]) for cfg in layers_cfg.values()],
                      dtype=np.int64)
    total_n = int(counts.sum())

    layer_type_ids = np.empty(total_n, dtype=np.int8)
    within_idx = np.empty(total_n, dtype=np.int32)
    seeds = np.empty(total_n, dtype=np.uint32)

    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    layer_states: dict[str, _LayerState] = {}

    for lt_id, name in enumerate(layer_names):
        cfg = layers_cfg[name]
        n = int(counts[lt_id])
        section_seed = int((master_seed + 1) * 10007 + lt_id)

        params_cfg = cfg["params"]
        # Three buckets: fixed (resolved at config-load), derived
        # (resolved per-sample from other params), variable (consume
        # unit-cube columns).
        fixed: dict[str, Any] = {}
        derived: list[tuple] = []
        variable_names: list[str] = []
        for k, s in params_cfg.items():
            if _is_derived(s):
                derived.append((k, s))
            elif _is_fixed(s):
                fixed[k] = s["value"]
            else:
                variable_names.append(k)

        # Validate derived references — must point at a fixed or
        # variable param, otherwise we'd hit KeyError per-sample.
        _known = set(fixed) | set(variable_names)
        for vn in variable_names:
            vspec = params_cfg[vn]
            if "levels_from_ratio" in vspec:
                ref = vspec["levels_from_ratio"]
                if ref not in _known or ref == vn or "levels_from_ratio" in params_cfg.get(ref, {}):
                    raise ValueError(
                        f"param {vn!r}: levels_from_ratio must reference a "
                        f"fixed or plain variable param, got {ref!r}"
                    )
                if cfg.get("sampling", "sobol") == "grid":
                    raise ValueError(
                        f"param {vn!r}: levels_from_ratio is not supported "
                        "with grid sampling"
                    )
        for dname, dspec in derived:
            ref = (dspec.get("fraction_of") or dspec.get("linear_of")
                   or dspec.get("inverse_of"))
            if ref not in _known:
                raise ValueError(
                    f"derived param {dname!r} references unknown param "
                    f"{ref!r} (known: {sorted(_known)})"
                )

        # Assign unit-cube columns to variable params, honouring
        # ``shared`` tags so multiple params can ride the same Sobol /
        # LHS coordinate, and ``jitter`` which allocates an extra
        # independent column blended in at fraction ``j``.
        tag_to_col: dict[str, int] = {}
        unit_col_map: list[int] = []
        jitter_col: list[int] = []
        jitter_amt: list[float] = []
        n_cols = 0
        for vn in variable_names:
            spec = params_cfg[vn]
            tag = spec.get("shared")
            j = float(spec.get("jitter", 0.0))
            if not (0.0 <= j <= 1.0):
                raise ValueError(
                    f"param {vn!r}: jitter must be in [0, 1], got {j}"
                )
            if tag is None:
                if j > 0.0:
                    raise ValueError(
                        f"param {vn!r}: 'jitter' only makes sense with "
                        "'shared' (an unshared param is already independent)"
                    )
                unit_col_map.append(n_cols)
                n_cols += 1
                jitter_col.append(-1)
                jitter_amt.append(0.0)
            else:
                if tag not in tag_to_col:
                    tag_to_col[tag] = n_cols
                    n_cols += 1
                unit_col_map.append(tag_to_col[tag])
                if j > 0.0:
                    jitter_col.append(n_cols)
                    n_cols += 1
                    jitter_amt.append(j)
                else:
                    jitter_col.append(-1)
                    jitter_amt.append(0.0)

        sampling = cfg.get("sampling", "sobol")
        grid_combos = None
        if sampling == "grid":
            # Grid is a full factorial of the variable params at fixed
            # ``levels``; it doesn't use a unit cube, so ``shared`` and
            # ``jitter`` are silently ignored. ``fraction_of`` /
            # ``linear_of`` derived params are resolved per-sample in
            # ``__getitem__`` after the combo is unpacked, so they work
            # in grid mode just like they do in sobol/lhs/uniform.
            grid_combos = _grid_combos(variable_names, params_cfg, n)
            unit = np.empty((n, 0), dtype=np.float32)
        elif n_cols == 0:
            unit = np.empty((n, 0), dtype=np.float32)
        elif sampling == "sobol":
            unit = _sample_sobol(n_cols, n, section_seed)
        elif sampling == "lhs":
            unit = _sample_lhs(n_cols, n, section_seed)
        elif sampling == "uniform":
            unit = np.random.default_rng(section_seed).uniform(
                size=(n, n_cols)
            ).astype(np.float32)
        else:
            raise ValueError(f"unknown sampling strategy {sampling!r}")

        layer_states[name] = _LayerState(
            name=name,
            variable_names=variable_names,
            unit_col_map=unit_col_map,
            jitter_col=jitter_col,
            jitter_amt=jitter_amt,
            derived=derived,
            fixed=fixed,
            unit=unit.astype(np.float32, copy=False),
            grid_combos=grid_combos,
        )

        a, b = int(offsets[lt_id]), int(offsets[lt_id + 1])
        layer_type_ids[a:b] = lt_id
        within_idx[a:b] = np.arange(n, dtype=np.int32)
        seeds[a:b] = np.random.default_rng(section_seed).integers(
            1, 2**31 - 1, size=n, dtype=np.uint32
        )

    # Global shuffle — mixes layer types across the index order so ranks
    # get a statistically uniform slice of costs.
    perm = np.random.default_rng(master_seed).permutation(total_n)
    layer_type_ids = layer_type_ids[perm]
    within_idx = within_idx[perm]
    seeds = seeds[perm]

    return JobList(layer_names, layer_type_ids, within_idx, seeds,
                   layer_states, layers_cfg)


def _sample_sobol(d: int, n: int, seed: int) -> np.ndarray:
    # Sobol is balanced at powers of 2; request next power up and truncate.
    m = int(np.ceil(np.log2(max(n, 2))))
    total = 2**m
    return qmc.Sobol(d=d, scramble=True, seed=seed).random(total)[:n]


def _sample_lhs(d: int, n: int, seed: int) -> np.ndarray:
    return qmc.LatinHypercube(d=d, seed=seed).random(n)


def _resolve_derived(spec: dict, params: dict):
    if "fraction_of" in spec:
        return float(spec["value"]) * float(params[spec["fraction_of"]])
    if "linear_of" in spec:
        slope = float(spec.get("slope", 1.0))
        intercept = float(spec.get("intercept", 0.0))
        return slope * float(params[spec["linear_of"]]) + intercept
    if "inverse_of" in spec:
        # value = scale / params[ref], optionally clipped to [min, max]
        # and cast to int. Use this to auto-size nlevel so deeper
        # channels stack fewer levels and shallower ones stack more.
        scale = float(spec["scale"])
        ref = float(params[spec["inverse_of"]])
        if ref <= 0:
            raise ValueError(
                f"inverse_of param requires positive ref value, got {ref}"
            )
        v = scale / ref
        if "min" in spec:
            v = max(float(spec["min"]), v)
        if "max" in spec:
            v = min(float(spec["max"]), v)
        if spec.get("type") == "int":
            return int(round(v))
        return float(v)
    raise ValueError(f"unknown derived spec: {spec}")


def _levels_from_ratio(u: float, spec: dict, params: dict) -> int:
    """Number of levels for a sampled aggradation ratio (see module doc)."""
    lo, hi = spec["ratio"]
    ratio = float(lo) + u * (float(hi) - float(lo))
    depth = float(params[spec["levels_from_ratio"]])
    if depth <= 0 or ratio <= 0:
        raise ValueError(
            f"levels_from_ratio needs positive depth and ratio, got "
            f"depth={depth}, ratio={ratio}"
        )
    column = float(spec["column"])
    base = float(spec.get("base", depth))
    n = int(round((column - base) / (ratio * depth))) + 1
    n = max(int(spec.get("min", 1)), n)
    if "max" in spec:
        n = min(int(spec["max"]), n)
    return n


def _is_derived(spec: dict) -> bool:
    return ("fraction_of" in spec or "linear_of" in spec
            or "inverse_of" in spec)


def _map_unit_value(u: float, spec: dict):
    if "choices" in spec:
        choices = spec["choices"]
        k = min(int(u * len(choices)), len(choices) - 1)
        return choices[k]
    lo, hi = spec["range"]
    if spec.get("scale") == "log":
        val = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    else:
        val = float(lo + u * (hi - lo))
    if spec.get("type") == "int":
        return int(round(val))
    return val


def _grid_combos(names: list, params_cfg: dict, count: int) -> list[tuple]:
    axes: list[list] = []
    for name in names:
        spec = params_cfg[name]
        if "choices" in spec:
            axes.append(list(spec["choices"]))
            continue
        n_levels = spec.get("levels")
        if n_levels is None:
            raise ValueError(f"grid sampling requires 'levels' on param {name!r}")
        us = np.linspace(0, 1, int(n_levels))
        axes.append([_map_unit_value(float(u), spec) for u in us])
    full = list(itertools.product(*axes))
    if len(full) != count:
        raise ValueError(
            f"grid sampling over {names} produced {len(full)} combinations "
            f"but config declares count={count}"
        )
    return full
