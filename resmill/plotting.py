import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap, ListedColormap, BoundaryNorm

# ---------------------------------------------------------------------------
# Unified ResMill colormap: grey → yellow → orange/brown
#
# Maps linearly from reservoir-quality 0 (worst) to max (best):
#   0.0  →  grey    (#999999)   — shale / inactive / zero porosity
#   0.5  →  yellow  (#e8c840)   — intermediate sand / moderate porosity
#   1.0  →  brown   (#b85a18)   — best sand / high porosity
#
# Works for both:
#   • Continuous properties (porosity, permeability) with mask_zeros=True
#     — zero cells become NaN and render as light grey via set_bad()
#   • Discrete facies (0=shale, 1=bar, 2=channel) with mask_zeros=False
#     — 0 maps to grey, 1 to yellow, 2 to brown/orange
# ---------------------------------------------------------------------------
RESMILL_CMAP = LinearSegmentedColormap.from_list(
    'resmill',
    ['#999999', '#e8c840', '#b85a18'],
    N=256,
)

DEFAULT_CMAP = 'resmill'
# Register so plt.get_cmap('resmill') works everywhere
try:
    plt.colormaps.register(RESMILL_CMAP, name='resmill', force=True)
except AttributeError:
    plt.register_cmap(name='resmill', cmap=RESMILL_CMAP)


def plot_cube_slices(data, ix=None, iy=None, iz=None,
                     cmap=None, vmin=None, vmax=None, title=None,
                     ax=None, mask_zeros=True):
    """Plot 3 orthogonal slices arranged as faces of a cube.

    Mode is auto-detected from ``data``:
      * a ResMill ``Layer`` → uses ``layer.facies`` with the discrete
        Alluvsim 6-class colormap (every layer now exposes a uniform
        ``facies`` array)
      * an integer ndarray with values ⊂ {-1..4} → Alluvsim mode
      * an integer ndarray with values ⊂ {0, 1} → binary
      * a float ndarray → continuous (poro / perm)

    Parameters
    ----------
    data : (nx, ny, nz) ndarray  OR  a ResMill Layer
    ix, iy, iz : int or None
        Slice indices for the YZ / XZ / XY faces of the cube. Default
        is the **mid-slice** of each axis (``nx // 2``, etc.). Pass
        any integer to inspect a specific position.
    cmap, vmin, vmax : continuous-mode styling (ignored for facies
        modes — those use the Alluvsim discrete cmap).
    title, ax, mask_zeros : as before.

    Returns
    -------
    fig, ax
    """
    mode, data_3d = _detect_mode(data)

    if mode == 'alluvsim':
        mask_zeros = False
        cmap_obj, norm = alluvsim_cmap()
    elif mode == 'binary':
        mask_zeros = False
        cmap_obj = plt.get_cmap(cmap or DEFAULT_CMAP)
        norm = Normalize(vmin=0, vmax=1)
    else:
        cmap_obj = plt.get_cmap(cmap or DEFAULT_CMAP)
        norm = None

    nx, ny, nz = data_3d.shape
    if ix is None: ix = nx // 2
    if iy is None: iy = ny // 2
    if iz is None: iz = nz // 2

    # Mask zeros for better visualization (skip for categorical data)
    if mask_zeros:
        masked = np.where(data_3d > 0, data_3d, np.nan)
    else:
        masked = data_3d.astype(float)
    if norm is None:
        if vmin is None:
            vmin = np.nanmin(masked) if np.any(~np.isnan(masked)) else 0
        if vmax is None:
            vmax = np.nanmax(masked) if np.any(~np.isnan(masked)) else 1
        norm = Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    grey = np.array([0.8, 0.8, 0.8, 1.0])  # light grey for NaN/inactive
    nan_fill = float(norm.vmin) if hasattr(norm, 'vmin') else 0.0

    # Render full slices with proper z-ordering.
    #
    # Each slice plane is split into 4 quadrants by the other two slice
    # positions, giving 12 patches total. We then draw the patches in
    # back-to-front order (farthest from the camera first), so closer
    # slices naturally occlude the parts of the far slices that sit
    # behind them — yielding a balanced 3-slice view where each plane
    # is fully visible *except* where another slice is in front of it.
    #
    # ``matplotlib`` 3D ``plot_surface`` z-sorts surfaces as a single
    # unit, which is why drawing the three full planes directly leads
    # to the last-drawn plane covering everything. Sub-dividing into
    # smaller patches makes the per-patch z-sort give the right answer.

    # Camera direction approximation for the default (elev=25, azim=225).
    az_rad = np.radians(225.0)
    el_rad = np.radians(25.0)
    # depth metric: project centroid onto -camera_dir; larger ⇒ farther
    cam_x = -np.cos(el_rad) * np.cos(az_rad)
    cam_y = -np.cos(el_rad) * np.sin(az_rad)
    cam_z = -np.sin(el_rad)

    def _patch(plane, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi):
        if plane == 'yz':
            y_rng = np.arange(y_lo, y_hi + 1)
            z_rng = np.arange(z_lo, z_hi + 1)
            Zg, Yg = np.meshgrid(z_rng, y_rng)
            Xg = np.full_like(Yg, ix, dtype=float)
            sl = masked[ix, y_lo:y_hi + 1, z_lo:z_hi + 1]
        elif plane == 'xz':
            x_rng = np.arange(x_lo, x_hi + 1)
            z_rng = np.arange(z_lo, z_hi + 1)
            Zg, Xg = np.meshgrid(z_rng, x_rng)
            Yg = np.full_like(Xg, iy, dtype=float)
            sl = masked[x_lo:x_hi + 1, iy, z_lo:z_hi + 1]
        else:  # 'xy'
            y_rng = np.arange(y_lo, y_hi + 1)
            x_rng = np.arange(x_lo, x_hi + 1)
            Yg, Xg = np.meshgrid(y_rng, x_rng)
            Zg = np.full_like(Xg, iz, dtype=float)
            sl = masked[x_lo:x_hi + 1, y_lo:y_hi + 1, iz]
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi)
        cz = 0.5 * (z_lo + z_hi)
        return Xg, Yg, Zg, sl, cx, cy, cz

    patches = []
    # YZ slice at x=ix: 4 quadrants by y / z
    for y_lo, y_hi in [(0, iy), (iy, ny - 1)]:
        for z_lo, z_hi in [(0, iz), (iz, nz - 1)]:
            patches.append(_patch('yz', ix, ix, y_lo, y_hi, z_lo, z_hi))
    # XZ slice at y=iy: 4 quadrants by x / z
    for x_lo, x_hi in [(0, ix), (ix, nx - 1)]:
        for z_lo, z_hi in [(0, iz), (iz, nz - 1)]:
            patches.append(_patch('xz', x_lo, x_hi, iy, iy, z_lo, z_hi))
    # XY slice at z=iz: 4 quadrants by x / y
    for x_lo, x_hi in [(0, ix), (ix, nx - 1)]:
        for y_lo, y_hi in [(0, iy), (iy, ny - 1)]:
            patches.append(_patch('xy', x_lo, x_hi, y_lo, y_hi, iz, iz))

    # Sort back-to-front: the farther a patch's centroid is from the
    # camera, the earlier it's drawn (so closer patches paint over it).
    def _depth(p):
        cx, cy, cz = p[4], p[5], p[6]
        return cx * cam_x + cy * cam_y + cz * cam_z

    patches.sort(key=_depth, reverse=True)

    for Xg, Yg, Zg, sl, *_unused in patches:
        colors = cmap_obj(norm(np.nan_to_num(sl, nan=nan_fill)))
        colors[np.isnan(sl)] = grey
        ax.plot_surface(Xg, Yg, Zg, facecolors=colors, shade=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=25, azim=225)
    if title:
        ax.set_title(title)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, label=title or '')

    return fig, ax


# ---------------------------------------------------------------------------
# Alluvsim 6-class facies palette — used by plot_slices(facies='alluvsim').
# ---------------------------------------------------------------------------
ALLUVSIM_FACIES_NAMES = {
    -1: 'FF (floodplain)', 0: 'FFCH (mud plug)', 1: 'CS (splay)',
    2: 'LV (levee)',       3: 'LA (point bar)', 4: 'CH (channel)',
}
ALLUVSIM_FACIES_COLORS = {
    -1: '#b4b4b4', 0: '#5b4636', 1: '#f2d16b',
    2: '#e8a23a', 3: '#c89b5e', 4: '#7a3f14',
}


def alluvsim_cmap():
    """Discrete cmap+norm for the Alluvsim 6-class facies codes (-1..4)."""
    codes = sorted(ALLUVSIM_FACIES_COLORS.keys())
    cmap = ListedColormap([ALLUVSIM_FACIES_COLORS[c] for c in codes])
    bounds = [c - 0.5 for c in codes] + [codes[-1] + 0.5]
    return cmap, BoundaryNorm(bounds, cmap.N)


def alluvsim_legend_handles():
    """Return ``(handles, labels)`` for an Alluvsim-facies legend."""
    handles = [plt.Rectangle((0, 0), 1, 1, fc=ALLUVSIM_FACIES_COLORS[c])
               for c in sorted(ALLUVSIM_FACIES_NAMES.keys())]
    labels = [ALLUVSIM_FACIES_NAMES[c]
              for c in sorted(ALLUVSIM_FACIES_NAMES.keys())]
    return handles, labels


def _detect_mode(data):
    """Pick a render mode from a ResMill Layer or a raw ndarray.

    Returns ``(mode, arr)`` where ``mode`` is one of
    ``'alluvsim'`` (discrete 6-class), ``'binary'`` (0/1) or
    ``'continuous'`` (float poro/perm).

    A Layer always plots its ``.facies`` (Alluvsim 6-class). For a raw
    ndarray we look at the dtype + value set:
      * float → continuous
      * int with values ⊂ {0, 1} → binary
      * int with values ⊂ {-1..4} → alluvsim
      * everything else falls back to continuous
    """
    if not isinstance(data, np.ndarray) and hasattr(data, 'facies'):
        return 'alluvsim', np.asarray(data.facies)
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.floating):
        return 'continuous', arr
    uniq = set(int(v) for v in np.unique(arr).tolist())
    if uniq.issubset({0, 1}):
        return 'binary', arr
    if uniq.issubset({-1, 0, 1, 2, 3, 4}):
        return 'alluvsim', arr
    return 'continuous', arr.astype(float)


def plot_slices(data, axis=None, n_slices=8, ncols=None,
                cmap=None, vmin=None, vmax=None, title=None,
                mask_zeros=True):
    """Plot 2D slices of a 3D array or a ResMill Layer.

    The single multi-panel viewer for any ResMill layer or array.

    **Mode is auto-detected** from the input:

    * a ResMill ``Layer`` → uses ``layer.facies`` (Alluvsim 6-class
      array — every layer now exposes a uniform ``facies`` array) with
      the discrete Alluvsim colormap, embedded legend, and per-facies
      fraction breakdown printed below the figure.
    * an integer ndarray with values ⊂ {-1..4} → Alluvsim mode
      (e.g. ``layer.facies``)
    * an integer ndarray with values ⊂ {0, 1} → binary mode
      (e.g. ``layer.active``)
    * a float ndarray → continuous (e.g. ``layer.poro_mat``)

    Parameters
    ----------
    data : ndarray (nx, ny, nz)  OR  a ResMill Layer
    axis : 0 | 1 | 2 | None
        ``None`` (default): plot ``n_slices`` slices along **each of
        the three axes**. 0 / 1 / 2: plot ``n_slices`` slices along
        that single axis.
    n_slices : int (default 8)
    ncols : int | None
        Columns in single-axis layout (default 4). Ignored when
        ``axis=None`` (one row per axis, ``n_slices`` cols).
    cmap, vmin, vmax : continuous-mode styling. Ignored for the
        binary/alluvsim modes.
    title : figure suptitle.
    mask_zeros : bool — render zero cells as light grey (continuous only).

    Returns
    -------
    None — the figure is rendered via matplotlib_inline's ``flush_figures``
    post-execute hook, which displays every figure in ``Gcf`` once in
    creation order. Returning ``None`` (rather than the ``Figure``) keeps
    last-expression auto-display from showing the figure a second time
    out of creation order when multiple ``plot_slices`` calls share a cell.
    """
    mode, arr = _detect_mode(data)

    # Footer extras: alluvsim legend, or horizontal colorbar for continuous.
    footer = None
    if mode == 'alluvsim':
        cm, norm = alluvsim_cmap()
        plot_kw = dict(cmap=cm, norm=norm, interpolation='nearest', origin='lower')
        footer = ('legend', None)
    elif mode == 'binary':
        plot_kw = dict(cmap=RESMILL_CMAP, vmin=0, vmax=1, origin='lower',
                       interpolation='nearest')
    else:
        if cmap is None: cmap = DEFAULT_CMAP
        cm_obj = plt.get_cmap(cmap)
        try:
            cm_obj = cm_obj.copy()
        except AttributeError:
            pass
        cm_obj.set_bad(color='#cccccc')
        if mask_zeros:
            arr = np.where(arr > 0, arr, np.nan)
        else:
            arr = arr.astype(float)
        if vmin is None:
            vmin = np.nanmin(arr) if np.any(~np.isnan(arr)) else 0
        if vmax is None:
            vmax = np.nanmax(arr) if np.any(~np.isnan(arr)) else 1
        plot_kw = dict(cmap=cm_obj, vmin=vmin, vmax=vmax, origin='lower')
        footer = ('cbar', _continuous_label(data, arr, vmin, vmax))

    if axis is not None:
        fig = _plot_slices_one_axis(arr, axis, n_slices, ncols or 4, plot_kw,
                                     title, footer)
    else:
        fig = _plot_slices_all_axes(arr, n_slices, plot_kw, title, footer)

    # Per-facies fraction breakdown (only when a Layer was passed)
    if mode == 'alluvsim' and not isinstance(data, np.ndarray) and hasattr(data, 'active'):
        n = arr.size
        parts = [f'NTG={data.active.mean()*100:5.1f}%']
        for c in sorted(ALLUVSIM_FACIES_NAMES.keys()):
            frac = (arr == c).sum() / n * 100
            if frac > 0.05:
                parts.append(f'{ALLUVSIM_FACIES_NAMES[c].split()[0]}={frac:.1f}%')
        print('  '.join(parts))


# Each slice axis maps to a 2-D plane with its own (xlabel, ylabel) so the
# user can tell which side of the panel is X / Y / Z at a glance.
#   axis=0 (X-slice): YZ plane → xlabel='Y', ylabel='Z'
#   axis=1 (Y-slice): XZ plane → xlabel='X', ylabel='Z'
#   axis=2 (Z-slice): XY plane → xlabel='X', ylabel='Y'
_AXIS_LABELS = {0: ('Y', 'Z'), 1: ('X', 'Z'), 2: ('X', 'Y')}


def _continuous_label(data, arr, vmin, vmax):
    """Best-effort colorbar label for a continuous-mode array.

    Tries to infer porosity vs permeability from a Layer, otherwise
    falls back on the value range: 0 ≤ v ≤ 1 → ``'porosity'``;
    larger range → ``'permeability (mD)'``; otherwise ``'value'``.
    """
    # Layer attribute names give the most reliable hint.
    if not isinstance(data, np.ndarray):
        for attr_name, label in [('poro_mat', 'porosity'),
                                  ('perm_mat', 'permeability (mD)')]:
            if hasattr(data, attr_name) and np.shares_memory(arr, getattr(data, attr_name)):
                return label
    finite = np.isfinite(arr)
    if not finite.any():
        return 'value'
    if vmin >= 0 and vmax <= 1.0:
        return 'porosity'
    if vmax > 1.0:
        return 'permeability (mD)'
    return 'value'


def _add_footer(fig, footer):
    """Attach a horizontal alluvsim legend or continuous colorbar to ``fig``.

    ``footer`` is one of: ``None``, ``('legend', _)`` or
    ``('cbar', label)``. Both anchor at the bottom of the figure
    parallel to the suptitle on top.
    """
    if footer is None:
        return
    kind, payload = footer
    if kind == 'legend':
        handles, labels = alluvsim_legend_handles()
        fig.legend(handles, labels, loc='lower center', ncol=6,
                   bbox_to_anchor=(0.5, -0.005), fontsize=9, frameon=False)
        return
    if kind == 'cbar':
        # Use the cmap+vmin/vmax stamped on the first imshow of the figure.
        first_im = next((im for ax in fig.axes for im in ax.get_images()), None)
        if first_im is None:
            return
        cax = fig.add_axes([0.20, 0.005, 0.60, 0.012])
        cbar = fig.colorbar(first_im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        if payload:
            cbar.set_label(payload, fontsize=9, labelpad=2)


def _slice_aspect(shape, axis):
    """Height/width ratio of an imshow(slice.T) panel for the given axis."""
    nx_, ny_, nz_ = shape
    return {0: nz_ / ny_,    # YZ plane
            1: nz_ / nx_,    # XZ plane
            2: ny_ / nx_     # XY plane
            }[axis]


def _plot_slices_one_axis(arr, axis, n_slices, ncols, plot_kw, title, footer):
    n = arr.shape[axis]
    indices = np.linspace(0, n - 1, min(n_slices, n), dtype=int)
    nrows = (len(indices) + ncols - 1) // ncols

    panel_w = 2.6
    panel_h = panel_w * _slice_aspect(arr.shape, axis)
    title_pad = 0.5 if title else 0.1
    footer_pad = 0.45 if footer is not None else 0.0   # room for legend/cbar
    fig_w = panel_w * ncols
    fig_h = panel_h * nrows + title_pad + footer_pad + 0.20 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                              squeeze=False)
    fig.subplots_adjust(
        hspace=0.12, wspace=0.06,
        left=0.05, right=0.99,
        top=1 - (title_pad + 0.04) / fig_h,
        bottom=(footer_pad + 0.05) / fig_h,
    )
    xlabel, ylabel = _AXIS_LABELS[axis]
    axes_2d = axes
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        r, c = divmod(i, ncols)
        sl = np.take(arr, idx, axis=axis)
        axes[i].imshow(sl.T, aspect='equal', **plot_kw)
        axes[i].set_title(f'{"XYZ"[axis]}={idx}', fontsize=10, pad=2)
        if r == nrows - 1:
            axes[i].set_xlabel(xlabel, fontsize=8, labelpad=1)
        else:
            axes[i].set_xticklabels([])
        if c == 0:
            axes[i].set_ylabel(ylabel, fontsize=8, labelpad=1)
        else:
            axes[i].set_yticklabels([])
        axes[i].tick_params(labelsize=7, pad=1)
    for j in range(len(indices), len(axes)):
        axes[j].axis('off')
    _add_footer(fig, footer)
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    return fig


def _plot_slices_all_axes(arr, n_slices, plot_kw, title, footer):
    """Layout: 3 axes × ⌈n_slices/4⌉ rows × 4 cols.

    Each axis group's row height matches the data aspect, so XZ/YZ rows
    are shorter than XY rows and there is no internal whitespace. Inner
    panels suppress their tick labels for tightness; only the bottom
    row of each group keeps its xlabel and only the leftmost column
    keeps its ylabel.
    """
    nx_, ny_, nz_ = arr.shape
    rows_per_axis = (n_slices + 3) // 4   # 4 cols/row
    ncols = 4

    panel_w = 2.4
    # Build height ratios: rows_per_axis panels per axis group, plus one
    # tiny spacer row between groups to keep group titles from kissing
    # the previous group's xlabel/ticks.
    SPACER = 0.18  # relative units (panel-height fractions)
    row_specs = []   # list of (kind, ax_id, row_in_group); kind ∈ {'panel', 'spacer'}
    height_ratios = []
    for ax_id in range(3):
        ar = _slice_aspect(arr.shape, ax_id)
        for r_in in range(rows_per_axis):
            row_specs.append(('panel', ax_id, r_in))
            height_ratios.append(ar)
        if ax_id < 2:
            row_specs.append(('spacer', ax_id, None))
            height_ratios.append(SPACER)
    nrows = len(height_ratios)

    title_pad = 0.5 if title else 0.1
    footer_pad = 0.45 if footer is not None else 0.0
    fig_w = panel_w * ncols
    fig_h = (panel_w * sum(height_ratios) + title_pad + footer_pad
             + 0.20 * (3 * rows_per_axis))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows, ncols,
        height_ratios=height_ratios,
        hspace=0.12, wspace=0.06,
        left=0.05, right=0.99,
        top=1 - (title_pad + 0.04) / fig_h,
        bottom=(footer_pad + 0.04) / fig_h,
    )

    # Map (ax_id, r_in_group) → row index in the gridspec.
    row_idx = {}
    for r, (kind, ax_id, r_in) in enumerate(row_specs):
        if kind == 'panel':
            row_idx[(ax_id, r_in)] = r

    for ax_id, slabel in enumerate(['X', 'Y', 'Z']):
        n = arr.shape[ax_id]
        idxs = np.linspace(0, n - 1, min(n_slices, n), dtype=int)
        xlabel, ylabel = _AXIS_LABELS[ax_id]
        last_row_in_group = (len(idxs) - 1) // ncols
        for k, idx in enumerate(idxs):
            r_grp = k // ncols
            r = row_idx[(ax_id, r_grp)]
            c = k % ncols
            ax = fig.add_subplot(gs[r, c])
            sl = np.take(arr, idx, axis=ax_id)
            ax.imshow(sl.T, aspect='equal', **plot_kw)
            ax.set_title(f'{slabel}={idx}', fontsize=10, pad=2)
            if r_grp == last_row_in_group:
                ax.set_xlabel(xlabel, fontsize=8, labelpad=1)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8, labelpad=1)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7, pad=1)
    _add_footer(fig, footer)
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.997)
    return fig


def plot_layer(layer, prop='poro_mat', **kwargs):
    """Convenience: plot_cube_slices on a Layer's property."""
    data = getattr(layer, prop)
    return plot_cube_slices(data, **kwargs)


def plot_reservoir(reservoir, prop='poro_mat', **kwargs):
    """Convenience: plot_cube_slices on a Reservoir's property."""
    data = getattr(reservoir, prop)
    return plot_cube_slices(data, **kwargs)


def plot_section(model, prop='poro_mat', axis='y', index=None,
                 structure=None, top=None, base=None,
                 erode_above=None, erode_below=None,
                 cmap=None, vmin=None, vmax=None, ax=None, title=None):
    """True-depth vertical cross-section of a (possibly deformed) model.

    Draws every cell of one section as its own quadrilateral using the
    same corner geometry the GRDECL exporter writes, so folds, faults,
    variable thickness and erosion appear exactly as they will in
    Petrel/ResInsight. Depth is on the vertical axis, increasing down.

    Parameters
    ----------
    model : Layer or Reservoir
    prop : str
        Cell property to color by (``'poro_mat'``, ``'perm_mat'``, ...).
    axis : 'y' | 'x'
        ``'y'``: an XZ section at cell row ``index`` (default the middle
        row); ``'x'``: a YZ section at cell column ``index``.
    structure, top, base, erode_above, erode_below :
        Same shaping arguments as :func:`resmill.export.to_grdecl`.
    """
    from .export import _build_geometry, _stack_prop

    layers = list(getattr(model, 'layers', [model]))
    Xc, Yc, Zc, actnum = _build_geometry(
        layers, structure, top, base, erode_above, erode_below)
    vals = _stack_prop(layers, prop).astype(float)
    nz = vals.shape[2]

    if axis == 'y':
        j = vals.shape[1] // 2 if index is None else index
        H, Zs = Xc[:, 2 * j], Zc[:, 2 * j, :]
        C, A = vals[:, j, :], actnum[:, j, :]
        xlabel = 'X (m)'
    elif axis == 'x':
        i = vals.shape[0] // 2 if index is None else index
        H, Zs = Yc[2 * i, :], Zc[2 * i, :, :]
        C, A = vals[i, :, :], actnum[i, :, :]
        xlabel = 'Y (m)'
    else:
        raise ValueError("axis must be 'x' or 'y'")

    # One quad per cell from its own four section corners, so fault
    # throws and pinch-outs render without interpolation across cells.
    n_lat = C.shape[0]
    hl, hr = H[0::2], H[1::2]
    quads = np.empty((n_lat, nz, 4, 2))
    quads[..., 0, 0] = hl[:, None]; quads[..., 0, 1] = Zs[0::2, :-1]
    quads[..., 1, 0] = hr[:, None]; quads[..., 1, 1] = Zs[1::2, :-1]
    quads[..., 2, 0] = hr[:, None]; quads[..., 2, 1] = Zs[1::2, 1:]
    quads[..., 3, 0] = hl[:, None]; quads[..., 3, 1] = Zs[0::2, 1:]
    keep = A.ravel() > 0

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    shown = C.ravel()[keep]
    # Edges painted in the face color close the antialiasing seams that
    # would otherwise show as white hairlines between cells.
    pc = PolyCollection(quads.reshape(-1, 4, 2)[keep],
                        array=shown,
                        cmap=plt.get_cmap(cmap or DEFAULT_CMAP),
                        edgecolors='face', linewidths=0.3)
    pc.set_clim(vmin if vmin is not None else shown.min(),
                vmax if vmax is not None else shown.max())
    ax.add_collection(pc)
    ax.autoscale()
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Depth (m)')
    if title:
        ax.set_title(title)
    plt.colorbar(pc, ax=ax, label=_continuous_label(model, shown,
                                                    float(shown.min()),
                                                    float(shown.max())))
    return None
