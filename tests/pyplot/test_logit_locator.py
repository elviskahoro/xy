"""Direct and end-to-end coverage for ``LogitLocator`` probability-decade ticks.

``LogitLocator`` previously had no direct unit tests, which let an
unconditional clamp into ``[1e-7, 1 - 1e-7]`` ship: logit views lying entirely
inside ``(0, 1e-7)`` rendered with zero ticks, and views extending below
``1e-7`` silently dropped the low probability decades. These tests pin
Matplotlib's ``LogitLocator.nonsingular`` semantics the locator must match:
in-domain positive ``vmin`` below ``1e-7`` (and ``vmax`` above ``1 - 1e-7``)
are kept so the decade math can place ticks there, only out-of-domain ends
(``vmin <= 0`` / ``vmax >= 1``) are substituted with the ``1e-7`` minpos, and a
degenerate (``vmin == vmax``) interval is expanded to
``(0.1 * vmin, 1 - 0.1 * vmin)`` instead of collapsing to zero ticks.
"""

from __future__ import annotations

import numpy as np
import pytest

import xy.pyplot as plt
from xy.pyplot._ticker import LogitLocator


def teardown_function() -> None:
    plt.close("all")


def _ticks(vmin: float, vmax: float, *, minor: bool = False) -> np.ndarray:
    return np.asarray(LogitLocator(minor=minor).tick_values(vmin, vmax), dtype=float)


def test_keeps_sub_epsilon_decades_below_minpos() -> None:
    ticks = _ticks(1e-10, 0.5)
    assert 1e-10 in ticks
    assert 1e-9 in ticks
    assert 1e-8 in ticks
    assert 1e-7 in ticks
    assert 0.5 in ticks
    assert np.all((ticks > 0) & (ticks < 1))
    assert ticks.min() == pytest.approx(1e-10)


def test_does_not_collapse_when_view_lies_entirely_below_minpos() -> None:
    ticks = _ticks(8.9e-10, 1.1e-8)
    assert ticks.size > 0
    assert 1e-9 in ticks
    assert 1e-8 in ticks
    assert np.all((ticks > 0) & (ticks < 1))


def test_expands_degenerate_interval_near_half() -> None:
    assert list(_ticks(0.5, 0.5)) == [0.01, 0.1, 0.5, 0.9, 0.99]


def test_control_band_inside_minpos_is_unchanged() -> None:
    assert list(_ticks(0.001, 0.999)) == [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]


def test_substitutes_only_out_of_domain_low_end() -> None:
    ticks = _ticks(-5.0, 0.5)
    assert 0.5 in ticks
    assert 1e-7 in ticks
    assert np.all((ticks > 0) & (ticks < 1))


def test_substitutes_only_out_of_domain_high_end() -> None:
    ticks = _ticks(0.5, 5.0)
    assert 0.5 in ticks
    assert (1 - 1e-7) in ticks
    assert np.all((ticks > 0) & (ticks < 1))


def test_all_out_of_domain_below_zero_uses_initial_range() -> None:
    ticks = _ticks(-5.0, -1.0)
    assert ticks.size > 0
    assert 0.5 in ticks
    assert np.all((ticks > 0) & (ticks < 1))
    assert ticks.min() <= 1e-6
    assert ticks.max() >= 0.99


@pytest.mark.parametrize("limits", [(np.nan, np.inf), (np.inf, -np.inf), (np.nan, np.nan)])
def test_non_finite_inputs_use_initial_range(limits) -> None:
    ticks = _ticks(float(limits[0]), float(limits[1]))
    assert ticks.size > 0
    assert 0.5 in ticks
    assert np.all((ticks > 0) & (ticks < 1))
    assert ticks.min() <= 1e-6
    assert ticks.max() >= 0.99


def test_minor_returns_empty_only_when_zoomed_into_linear_region() -> None:
    assert _ticks(0.5, 0.5001, minor=True).size == 0


def test_major_zoomed_into_linear_region_falls_back_to_maxn() -> None:
    ticks = _ticks(0.5, 0.5001)
    assert ticks.size >= 2
    assert ticks.min() == pytest.approx(0.5, abs=1e-9)
    assert ticks.max() == pytest.approx(0.5001, abs=1e-9)


def _axis_payload(ax, width: int = 640, height: int = 480):
    return ax._build_chart(width, height).figure().axis_options


def test_logit_axis_renders_ticks_near_zero_data() -> None:
    _, ax = plt.subplots()
    ax.plot([0.0], [1e-9], "o")
    ax.set_yscale("logit")
    y_axis = _axis_payload(ax)["y"]
    assert len(y_axis["tick_values"]) > 0
    assert len(y_axis["tick_labels"]) > 0
    transform = ax.yaxis.get_transform()
    tick_data = transform.inverted().transform(y_axis["tick_values"])
    assert np.all((tick_data > 0.0) & (tick_data < 1.0))


def test_logit_user_xlim_below_minpos_keeps_low_decades() -> None:
    _, ax = plt.subplots()
    ax.plot([0.0, 1.0], [1e-9, 0.5])
    ax.set_yscale("logit")
    ax.set_ylim(1e-10, 0.5)
    y_axis = _axis_payload(ax)["y"]
    assert len(y_axis["tick_values"]) > 0
    transform = ax.yaxis.get_transform()
    tick_data = transform.inverted().transform(y_axis["tick_values"])
    assert tick_data.min() <= 1e-9
    assert 0.5 in tick_data


def test_logit_default_demo_range_still_matches_matplotlib_gallery() -> None:
    _, ax = plt.subplots()
    x = np.linspace(-10.0, 10.0, 1000)
    probability = np.arctan(x) / np.pi + 0.5
    ax.plot(x, probability)
    ax.set_yscale("logit", one_half="1/2", use_overline=True)
    ax.set_ylim(1e-5, 1 - 1e-5)
    y_axis = _axis_payload(ax)["y"]
    tick_data = ax.yaxis.get_transform().inverted().transform(y_axis["tick_values"])
    assert np.all((tick_data > 0.0) & (tick_data < 1.0))
    assert 0.5 in tick_data
    assert "1/2" in y_axis["tick_labels"]
