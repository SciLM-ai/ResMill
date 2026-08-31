"""Corner-point grid export (Petrel/Eclipse GRDECL) and 3-D geometry assembly.

The model stays in flat stratigraphic space; structural shape is applied
here, at export time, by shifting the layer-interface depth surfaces. See
:mod:`resmill.structure` for the deformation fields.

Conventions: depth is in meters, positive down; the grid pillars are
vertical; array axes are ``(x, y, z)`` with ``indexing='ij'``; within a
layer the k index increases upward (package convention), while the written
file uses Eclipse order (I fastest, then J, then K counting down from the
top), so properties are flipped per layer during assembly.
"""

from pathlib import Path

import numpy as np

from .structure import Structure, surface as _surface_field

# A cell whose corner-pair thickness never exceeds this is fully collapsed
# (eroded or pinched out) and is written with ACTNUM = 0.
_MIN_THICKNESS = 1e-6
# Corners are nudged toward cell centers by this fraction of a cell so a
# discontinuous structure (a fault) assigns each cell to its own side of
# the trace; continuous fields are snapped back to exact node values.
_EPS = 1e-9
_SNAP_TOL = 1e-3  # 1 mm


def _field(obj, x_len, y_len):
    """Coerce None | Structure | callable | 2-D array | scalar to a Structure."""
    if obj is None or isinstance(obj, Structure):
        return obj
    if callable(obj):
        return Structure(obj)
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == 2:
        return _surface_field(arr, x_len, y_len)
    if arr.ndim == 0:
        return Structure(lambda x, y: float(arr))
    raise ValueError(f"cannot interpret a {arr.ndim}-D array as a surface")


def _dbl(a, axis):
    """Expand a node-sampled axis (n+1,) to doubled cell corners (2n,)."""
    idx = np.repeat(np.arange(a.shape[axis]), 2)[1:-1]
    return np.take(a, idx, axis=axis)


def _corner_axis(n, d):
    """1-D corner coordinates (2n,): cell edges nudged inward by _EPS."""
    centers = (np.arange(n) + 0.5) * d
    half = (0.5 - _EPS) * d
    out = np.empty(2 * n)
    out[0::2] = centers - half
    out[1::2] = centers + half
    return out


def _build_geometry(layers, structure=None, top=None, base=None,
                    erode_above=None, erode_below=None):
    """Assemble deformed interface depths on the doubled corner grid.

    Returns ``(Xc, Yc, Zc, actnum)``: corner coordinates ``(2nx, 2ny)``,
    interface depths ``(2nx, 2ny, nz+1)`` ordered top-down, and the
    ``(nx, ny, nz)`` activity mask in the same top-down k order.
    """
    L0 = layers[0]
    nx, ny = L0.nx, L0.ny
    x_len, y_len = L0.x_len, L0.y_len
    Xn, Yn = L0.X, L0.Y  # (nx+1, ny+1) node grids

    # Stratigraphic interface stack, top-down: layer boundaries plus every
    # internal k-surface, conformable to each layer's (possibly dipping) z1.
    surfs = [L0.z1]
    for L in layers:
        for m in range(1, L.nz + 1):
            surfs.append(L.z1 + m * L.dz)
    Zc = _dbl(_dbl(np.stack(surfs, axis=-1), 0), 1)  # (2nx, 2ny, nz+1)

    Xc1, Yc1 = _corner_axis(nx, L0.dx), _corner_axis(ny, L0.dy)
    Xc, Yc = np.meshgrid(Xc1, Yc1, indexing='ij')

    def ev(obj):
        fld = _field(obj, x_len, y_len)
        if fld is None:
            return None
        Fc = fld(Xc, Yc)
        Fn = _dbl(_dbl(fld(Xn, Yn), 0), 1)
        return np.where(np.abs(Fc - Fn) < _SNAP_TOL, Fn, Fc)

    # 1. Conform to absolute surfaces, in the stratigraphic frame: both
    # given squeezes proportionally; one given drapes with thickness kept.
    T, B = ev(top), ev(base)
    if T is not None and B is not None:
        if not np.all(B > T):
            raise ValueError("base must lie below top everywhere")
        t0, b0 = Zc[:, :, :1], Zc[:, :, -1:]
        Zc = T[:, :, None] + (Zc - t0) / (b0 - t0) * (B - T)[:, :, None]
    elif T is not None:
        Zc = Zc + (T - Zc[:, :, 0])[:, :, None]
    elif B is not None:
        Zc = Zc + (B - Zc[:, :, -1])[:, :, None]

    # 2. Structural shift: one field for the whole stack, or one per layer.
    # A layer's shift covers its internal surfaces and its base, so a
    # shared contact carries the shift of the layer above (younger) it.
    if structure is not None:
        if isinstance(structure, (list, tuple)):
            if len(structure) != len(layers):
                raise ValueError(
                    f"structure list needs one entry per layer "
                    f"({len(layers)}), got {len(structure)}")
            shifts = [ev(s) for s in structure]
            s0 = shifts[0]
            if s0 is not None:
                Zc[:, :, 0] += s0
            m = 1
            for L, s in zip(layers, shifts):
                if s is not None:
                    Zc[:, :, m:m + L.nz] += s[:, :, None]
                m += L.nz
        else:
            Zc = Zc + ev(structure)[:, :, None]

    # 3. Stitch: younger truncates older. A no-op for a shared shift;
    # with per-layer shifts it clamps older interfaces down to any
    # younger base that cuts them (angular unconformities, incision).
    Zc = np.maximum.accumulate(Zc, axis=2)

    # 4. Erosion clips (present-day surfaces), applied last.
    E = ev(erode_above)
    if E is not None:
        Zc = np.maximum(Zc, E[:, :, None])
    Eb = ev(erode_below)
    if Eb is not None:
        Zc = np.minimum(Zc, Eb[:, :, None])

    if not np.isfinite(Zc).all():
        raise ValueError("non-finite depths after applying structure/top/base/erosion")

    thick = Zc[:, :, 1:] - Zc[:, :, :-1]
    cell_thick = thick.reshape(nx, 2, ny, 2, -1).max(axis=(1, 3))
    actnum = (cell_thick > _MIN_THICKNESS).astype(np.int32)
    return Xc, Yc, Zc, actnum


def _interleave(Zc):
    """Interface stack (2nx, 2ny, nz+1) to per-cell faces (2nx, 2ny, 2nz)."""
    nz = Zc.shape[2] - 1
    out = np.empty(Zc.shape[:2] + (2 * nz,))
    out[:, :, 0::2] = Zc[:, :, :-1]
    out[:, :, 1::2] = Zc[:, :, 1:]
    return out


def _stack_prop(layers, name):
    """Concatenate a per-layer (nx, ny, nz) attribute into Eclipse K-down order."""
    mats = []
    for L in layers:
        a = getattr(L, name, None)
        if a is None:
            raise ValueError(
                f"{type(L).__name__} has no '{name}' array; call create_geology() first")
        mats.append(np.asarray(a)[:, :, ::-1])
    return np.concatenate(mats, axis=2)


def _write_array(f, keyword, values, fmt, per_line):
    f.write(keyword + "\n")
    v = np.asarray(values).ravel()
    n_full = v.size // per_line
    if n_full:
        np.savetxt(f, v[:n_full * per_line].reshape(n_full, per_line), fmt=fmt)
    if v.size > n_full * per_line:
        f.write(" ".join(fmt % x for x in v[n_full * per_line:]) + "\n")
    f.write("/\n\n")


def _write_rle(f, keyword, values, per_line=12):
    """Write an integer array with GRDECL count*value run compression."""
    v = np.asarray(values).ravel()
    edges = np.flatnonzero(np.diff(v)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [v.size]))
    toks = [f"{e - s}*{v[s]:d}" if e - s > 1 else f"{v[s]:d}"
            for s, e in zip(starts, ends)]
    f.write(keyword + "\n")
    for i in range(0, len(toks), per_line):
        f.write(" ".join(toks[i:i + per_line]) + "\n")
    f.write("/\n\n")


def to_grdecl(model, path, structure=None, top=None, base=None,
              erode_above=None, erode_below=None, facies=False,
              poro_floor=None, perm_floor=None,
              fmt_z="%.2f", fmt_prop="%.6g"):
    """Write a self-contained Eclipse/Petrel corner-point file (GRDECL).

    The file carries SPECGRID, COORD, ZCORN, ACTNUM, PORO, PERMX, PERMY
    (= PERMX), PERMZ (= per-layer ``kzkx`` x PERMX) and optionally FACIES,
    and imports directly into Petrel ("ECLIPSE keywords (grid geometry and
    properties)"), ResInsight, tNavigator, or an Eclipse deck INCLUDE.

    Parameters
    ----------
    model : Layer or Reservoir
    path : str or Path
        Output file, conventionally ``.grdecl``.
    structure : Structure, callable, 2-D array, scalar, or a list of these
        Vertical shift field(s) in meters, positive down (see
        :mod:`resmill.structure`). A list gives one entry per layer
        (None allowed); layers then deform independently and younger
        layers truncate older ones where they collide, producing
        angular unconformities.
    top, base : Structure-like, optional
        Absolute depth surfaces. Both: the stack is squeezed
        proportionally between them (wedges, pinch-outs). One: the stack
        is draped onto it with thickness preserved.
    erode_above, erode_below : Structure-like, optional
        Absolute erosion surfaces applied last; cells entirely outside
        them collapse to zero thickness and get ACTNUM = 0.
    facies : bool
        Also write the layers' facies codes as a FACIES keyword
        (non-standard; Petrel imports it as a generic property).
    poro_floor, perm_floor : float, optional
        Lower clamps applied to the written PORO / PERMX+PERMZ arrays
        (GaussianLayer and LobeLayer zero out shale cells, which Eclipse
        would auto-deactivate as zero-pore-volume cells).
    fmt_z, fmt_prop : str
        printf formats for depth/coordinate and permeability values.
    """
    layers = list(getattr(model, "layers", [model]))
    L0 = layers[0]
    nx, ny = L0.nx, L0.ny
    nz = sum(L.nz for L in layers)

    Xc, Yc, Zc, actnum = _build_geometry(
        layers, structure, top, base, erode_above, erode_below)

    poro = _stack_prop(layers, "poro_mat").astype(float)
    permx = _stack_prop(layers, "perm_mat").astype(float)
    permz = np.concatenate(
        [np.asarray(L.perm_mat, dtype=float)[:, :, ::-1] * L.kzkx for L in layers],
        axis=2)
    if poro_floor is not None:
        poro = np.maximum(poro, poro_floor)
    if perm_floor is not None:
        permx = np.maximum(permx, perm_floor)
        permz = np.maximum(permz, perm_floor)

    zcorn = _interleave(Zc)

    # COORD: vertical pillars through the (nx+1, ny+1) nodes, ordered
    # i-fastest then j, 6 values per pillar (x y z_top x y z_bottom).
    Xn, Yn = L0.X, L0.Y
    zt = np.full_like(Xn, Zc.min() - 10.0)
    zb = np.full_like(Xn, Zc.max() + 10.0)
    coord = np.stack([Xn, Yn, zt, Xn, Yn, zb], axis=-1).transpose(1, 0, 2).ravel()

    from . import __version__

    path = Path(path)
    with open(path, "w") as f:
        f.write(f"-- Generated by ResMill v{__version__}\n")
        f.write(f"-- {nx} x {ny} x {nz} corner-point grid, metric units, "
                f"depth positive down\n\n")
        f.write("PINCH\n/\n\n")
        f.write("MAPUNITS\n METRES /\n\n")
        f.write("GRIDUNIT\n METRES /\n\n")
        f.write(f"SPECGRID\n {nx} {ny} {nz} 1 F /\n\n")
        _write_array(f, "COORD", coord, fmt_z, per_line=6)
        _write_array(f, "ZCORN", zcorn.ravel(order="F"), fmt_z, per_line=10)
        _write_rle(f, "ACTNUM", actnum.ravel(order="F"))
        _write_array(f, "PORO", poro.ravel(order="F"), "%.4f", per_line=14)
        _write_array(f, "PERMX", permx.ravel(order="F"), fmt_prop, per_line=10)
        _write_array(f, "PERMY", permx.ravel(order="F"), fmt_prop, per_line=10)
        _write_array(f, "PERMZ", permz.ravel(order="F"), fmt_prop, per_line=10)
        if facies:
            fac = _stack_prop(layers, "facies").astype(int)
            _write_rle(f, "FACIES", fac.ravel(order="F"))
    return path


def to_pyvista(model, structure=None, top=None, base=None,
               erode_above=None, erode_below=None):
    """Build a ``pyvista.ExplicitStructuredGrid`` of the deformed model.

    Cell data carries PORO, PERMX, ACTNUM and (where every layer has it)
    FACIES; elevation is ``-depth`` so structure reads the right way up.
    Requires the ``viz`` extra (``pip install resmill[viz]``).
    """
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            "to_pyvista requires pyvista: pip install pyvista "
            "(or resmill[viz])") from e

    layers = list(getattr(model, "layers", [model]))
    nx, ny = layers[0].nx, layers[0].ny
    Xc, Yc, Zc, actnum = _build_geometry(
        layers, structure, top, base, erode_above, erode_below)

    # pyvista wants the corners as a global F-order ravel of the doubled
    # corner arrays (the same layout as ZCORN). The grid's k axis points
    # up (elevation = -depth increases with k) so cells are right-handed;
    # cell data is flipped from Eclipse K-down to match.
    elev = -_interleave(Zc)[:, :, ::-1]
    Xg = np.broadcast_to(Xc[:, :, None], elev.shape)
    Yg = np.broadcast_to(Yc[:, :, None], elev.shape)
    corners = np.column_stack([Xg.ravel(order="F"), Yg.ravel(order="F"),
                               elev.ravel(order="F")])
    grid = pv.ExplicitStructuredGrid((nx + 1, ny + 1, Zc.shape[2]), corners)

    def kup(a):
        return np.ascontiguousarray(a[:, :, ::-1]).ravel(order="F")

    grid.cell_data["PORO"] = kup(_stack_prop(layers, "poro_mat"))
    grid.cell_data["PERMX"] = kup(_stack_prop(layers, "perm_mat"))
    grid.cell_data["ACTNUM"] = kup(actnum)
    if all(getattr(L, "facies", None) is not None for L in layers):
        grid.cell_data["FACIES"] = kup(_stack_prop(layers, "facies"))
    return grid
