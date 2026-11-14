from __future__ import annotations

import pytest

import xy.pyplot as plt
from xy.pyplot._colors import is_color_like
from xy.pyplot._fmt import parse_fmt


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("r--o", ("r", "--", "o")),
        ("r--", ("r", "--", None)),
        ("--r", ("r", "--", None)),
        ("o", (None, None, "o")),
        ("go", ("g", None, "o")),
        ("-.", (None, "-.", None)),
        (":", (None, ":", None)),
        ("k", ("k", None, None)),
        ("C1-.s", ("C1", "-.", "s")),
        ("C9", ("C9", None, None)),
        ("bs-", ("b", "-", "s")),
        ("", (None, None, None)),
        ("D-", (None, "-", "D")),
        ("x:", (None, ":", "x")),
        ("2", (None, None, "2")),
        ("0.5", ("0.5", None, None)),
        ("orchid", ("orchid", None, None)),
        ("tab:purple", ("tab:purple", None, None)),
        ("xkcd:crimson", ("xkcd:crimson", None, None)),
        # matplotlib ``lineStyles[' ']`` is the ``_draw_nothing`` style: a
        # lone space is a valid "draw no line" linestyle token. ``parse_fmt``
        # must accept it (matching ``matplotlib.axes._base._process_plot_format``)
        # rather than raising ``ValueError``. See GH issue (stem markerfmt=' ').
        (" ", (None, " ", None)),
        ("k ", ("k", " ", None)),
        (" C1", ("C1", " ", None)),
        (" r", ("r", " ", None)),
    ],
)
def test_parse(fmt: str, expected: tuple) -> None:
    assert parse_fmt(fmt) == expected


def test_marker_one_is_not_a_linestyle_dash() -> None:
    # '1' is the tri_down marker, never part of a linestyle.
    assert parse_fmt("1") == (None, None, "1")


@pytest.mark.parametrize(
    "value",
    [
        (float("inf"), 0.0, 0.0),
        (-float("inf"), 0.0, 0.0, 1.0),
        (0.0, float("nan"), 0.0),
    ],
)
def test_nonfinite_rgba_is_not_color_like(value: tuple[float, ...]) -> None:
    assert is_color_like(value) is False


@pytest.mark.parametrize("bad", ["z", "r--q", "??"])
def test_rejects_unknown(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_fmt(bad)


def test_stem_with_space_markerfmt_does_not_crash() -> None:
    # ``stem(markerfmt=' ')`` is the standard community idiom for suppressing
    # stem markers; it previously raised an unhandled ``ValueError`` from
    # ``parse_fmt``. It must now complete without error.
    _fig, ax = plt.subplots()
    ax.stem([1.0, 2.0, 3.0, 4.0], markerfmt=" ")
    assert ax._entries  # stems still emitted
