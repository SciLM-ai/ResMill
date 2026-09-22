"""Dataset sampler: levels from a sampled aggradation ratio."""
import numpy as np
import pytest

from resmill.dataset.sampling import build_jobs


def _jobs(spec_extra=None, n=400, sampling="sobol"):
    spec = {"levels_from_ratio": "mCHdepth", "ratio": [0.7, 1.4],
            "column": 64.0, "min": 1, "max": 40}
    spec.update(spec_extra or {})
    layers = {"channel": {"count": n, "sampling": sampling, "params": {
        "mCHdepth": {"range": [3.0, 16.0], "scale": "log"},
        "nlevel": spec,
        "NTGtarget": {"value": 0.3},
    }}}
    return build_jobs(layers, 42)


def test_levels_span_the_column_at_the_sampled_ratio():
    jobs = _jobs()
    ratios = []
    for i in range(len(jobs)):
        p = jobs[i]["params"]
        d, n = p["mCHdepth"], p["nlevel"]
        assert isinstance(n, int) and n >= 2          # 16 m in 64 m: at least 2
        spacing = (64.0 - d) / (n - 1)                  # engine: linspace(d, z_len, n)
        ratios.append(spacing / d)
    ratios = np.array(ratios)
    # rounding of the level count moves the realised ratio a little
    assert ratios.min() > 0.55 and ratios.max() < 1.7
    assert 0.9 < np.median(ratios) < 1.2
    # the ratio is a Sobol coordinate of its own: it is not a function of depth
    depths = np.array([jobs[i]["params"]["mCHdepth"] for i in range(len(jobs))])
    assert abs(np.corrcoef(depths, ratios)[0, 1]) < 0.3


def test_levels_base_offset_and_bounds():
    jobs = _jobs({"base": 1.0, "max": 6})
    for i in range(50):
        p = jobs[i]["params"]
        assert 2 <= p["nlevel"] <= 6


def test_levels_from_ratio_is_reproducible():
    a = [(_jobs()[i]["params"]["nlevel"]) for i in range(30)]
    b = [(_jobs()[i]["params"]["nlevel"]) for i in range(30)]
    assert a == b


def test_levels_from_ratio_rejects_bad_reference_and_grid():
    with pytest.raises(ValueError):
        _jobs({"levels_from_ratio": "missing"})
    with pytest.raises(ValueError):
        build_jobs({"channel": {"count": 4, "sampling": "grid", "params": {
            "mCHdepth": {"range": [3.0, 16.0], "levels": 2},
            "nlevel": {"levels_from_ratio": "mCHdepth", "ratio": [0.7, 1.4],
                       "column": 64.0},
        }}}, 1)
