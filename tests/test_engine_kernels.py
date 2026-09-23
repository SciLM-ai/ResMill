"""The Numba kernels that replaced Python loops in the fluvial engine must
reproduce the former arithmetic bit for bit (reference implementations below
are the former code, verbatim)."""
import numpy as np

from resmill.layers._genchannel import nearest_refined
from resmill.layers._fluvial import (_movwinsmooth, _curv_azimuth, _curv_from_azimuth,
                                     _dcds, _bank_velocity)


def _ref_refine(cx, cy, x_loc, y_loc, dd_initial, ndiscr=5):
    n_local = dd_initial.size; n_nodes = cx.size
    refined_dist = np.empty(n_local, dtype=np.float64)
    for myid in range(n_local):
        idis = int(dd_initial[myid]); lo = max(0, idis - 1); hi = min(n_nodes - 1, idis + 1)
        best = (cx[idis] - x_loc[myid])**2 + (cy[idis] - y_loc[myid])**2
        for sub in range(1, ndiscr):
            t = sub / float(ndiscr)
            xt = (1.0 - t) * cx[lo] + t * cx[hi]; yt = (1.0 - t) * cy[lo] + t * cy[hi]
            d = (xt - x_loc[myid])**2 + (yt - y_loc[myid])**2
            if d < best:
                best = d
        refined_dist[myid] = float(np.sqrt(max(best, 0.0)))
    return refined_dist


def _ref_movwin(arr, nwin):
    n = arr.size; out = np.empty_like(arr, dtype=np.float64)
    weights = np.array([(nwin - abs(i) + 1) / (nwin + 1.0) for i in range(-nwin, nwin + 1)], dtype=np.float64)
    for i in range(n):
        lo = max(0, i - nwin); hi = min(n, i + nwin + 1); wlo = lo - (i - nwin); whi = wlo + (hi - lo)
        w = weights[wlo:whi]; out[i] = float((arr[lo:hi] * w).sum() / w.sum())
    return out


def test_nearest_refined_matches_matrix_argmin_and_python_refine():
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = int(rng.integers(20, 400)); m = int(rng.integers(50, 3000))
        cx = np.cumsum(rng.normal(8, 3, n)); cy = np.cumsum(rng.normal(0, 6, n))
        lx = rng.uniform(cx.min() - 50, cx.max() + 50, m); ly = rng.uniform(cy.min() - 50, cy.max() + 50, m)
        idmat = np.sqrt((cx.reshape(n, 1) - lx)**2 + (cy.reshape(n, 1) - ly)**2)
        dd_ref = idmat.argmin(axis=0); dist_ref = _ref_refine(cx, cy, lx, ly, dd_ref)
        dd, dist = nearest_refined(cx, cy, lx, ly, 5, 2.0)
        assert np.array_equal(dd, dd_ref) and np.array_equal(dist, dist_ref)


def test_movwinsmooth_matches_numpy_reference():
    rng = np.random.default_rng(12)
    for _ in range(100):
        n = int(rng.integers(3, 500)); arr = rng.normal(0, 100, n) * rng.choice([1.0, 1e-3, 1e4])
        assert np.array_equal(_movwinsmooth(arr, 10), _ref_movwin(arr, 10))


def test_curvature_and_bank_velocity_match_python_loops():
    rng = np.random.default_rng(13)
    for _ in range(30):
        n = int(rng.integers(25, 500)); cx = np.cumsum(rng.normal(8, 3, n)); cy = np.cumsum(rng.normal(0, 6, n))
        dl = np.zeros(n); dl[1:] = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2); s = np.cumsum(dl)
        azi = np.zeros(n)
        for i in range(1, n):
            di = cx[i] - cx[i - 1]; dj = cy[i] - cy[i - 1]
            if di == 0.0: azi[i] = 0.0 if dj > 0 else 180.0
            elif di > 0.0 and dj >= 0.0: azi[i] = 90.0 - np.degrees(np.arctan(dj / di))
            elif di < 0.0: azi[i] = 270.0 - np.degrees(np.arctan(dj / di))
            else: azi[i] = 90.0 - np.degrees(np.arctan(dj / di))
        azi[0] = azi[1]
        assert np.array_equal(_curv_azimuth(cx, cy), azi)
        c = np.zeros(n)
        for i in range(1, n):
            ds = s[i] - s[i - 1]; d1 = azi[i] - azi[i - 1]; d2 = azi[i] - (azi[i - 1] + 360.0)
            c[i] = (d1 if abs(d1) < abs(d2) else d2) / max(ds, 1e-9)
        c[0] = c[1]
        assert np.array_equal(_curv_from_azimuth(azi, s), c)
        d = np.zeros(n)
        for i in range(1, n):
            d[i] = (c[i] - c[i - 1]) / max(s[i] - s[i - 1], 1e-9)
        d[0] = d[1]
        assert np.array_equal(_dcds(c, s), d)
        us0, Cf, h0, g, A, hw = 1.2, 0.0036, 5.0, 9.8, 10.0, 35.0
        part2 = hw * Cf / us0; part3 = us0 ** 4 / (g * h0 ** 2); part4 = (A + 2.0) * us0 ** 2 / h0
        usb = np.zeros(n)
        for idis in range(1, n):
            start = max(0, idis - 30); ds_cum = 0.0; inte = 0.0
            for j in range(idis, start - 1, -1):
                ds_cum += dl[j]; inte += np.exp(-2.0 * Cf * ds_cum / h0) * c[j]
            usb[idis] = -us0 * c[idis] + part2 * (part3 + part4) * inte
        assert np.array_equal(_bank_velocity(n, dl, c, us0, Cf, h0, part2, part3, part4), usb)
