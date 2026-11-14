"""LogitLocator keeps decades below 1e-7 and expands degenerate views (mpl nonsingular)."""

import numpy as np

from xy.pyplot._ticker import LogitLocator


def test_keeps_sub_epsilon_decades():
    ticks = LogitLocator().tick_values(1e-10, 0.5)
    assert ticks[0] == 1e-10 and 1e-8 in ticks and ticks[-1] == 0.5


def test_view_entirely_below_epsilon_has_ticks():
    ticks = LogitLocator().tick_values(8.9e-10, 1.1e-8)
    np.testing.assert_allclose(ticks, [1e-10, 1e-9, 1e-8, 1e-7])


def test_degenerate_view_expanded():
    assert len(LogitLocator().tick_values(0.5, 0.5)) > 0
    assert len(LogitLocator().tick_values(1e-9, 1e-9)) > 0


def test_out_of_domain_views_still_have_ticks():
    for lo, hi in [
        (-0.5, -0.1),
        (1.2, 1.5),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 5e-9),
        (0.99999998, 1.1),
    ]:
        assert len(LogitLocator().tick_values(lo, hi)) > 0, (lo, hi)
