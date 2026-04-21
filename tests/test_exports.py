"""Tests for desktop CAD export helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moth_designer.config import DEFAULTS, LWL
from moth_designer.exports import _loft_station_xs
from moth_designer.geometry import beam_eval, build_ctrl


def test_loft_station_layout_includes_true_bow():
    xs = _loft_station_xs(LWL, 30)
    assert xs[0] == 0.0
    assert xs[-1] == float(LWL)
    assert np.all(np.diff(xs) > 0.0)


def test_first_loft_station_honors_bow_half_beam():
    params = {k: float(v) for k, v in DEFAULTS.items()}
    ctrl_x, beam_hb, *_ = build_ctrl(params)
    xs = _loft_station_xs(LWL, 30)
    bow_half_beam = float(params.get("bow_half_beam", 1.0))
    first_station_width = 2.0 * float(beam_eval([xs[0]], ctrl_x, beam_hb)[0])
    assert abs(first_station_width - 2.0 * bow_half_beam) < 1e-9
