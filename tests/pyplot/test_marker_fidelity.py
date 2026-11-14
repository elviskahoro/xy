from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pytest

import xy
import xy.pyplot as plt
from xy import _raster, _svg, marks


def teardown_function():
    plt.close("all")


def test_matplotlib_marker_family_keeps_distinct_symbols_in_payload():
    fig, ax = plt.subplots()
    markers = (
        "o",
        ".",
        ",",
        "x",
        "+",
        "|",
        "_",
        "v",
        "^",
        "<",
        ">",
        "s",
        "d",
        "D",
        "P",
        "X",
    )
    for index, marker in enumerate(markers):
        ax.plot([index], [index], marker=marker, linestyle="none")

    payload, _blob = ax._build_chart(640, 480).figure().build_payload()
    symbols = [
        trace["style"].get("symbol", "circle")
        for trace in payload["traces"]
        if trace["kind"] == "scatter"
    ]
    assert symbols == [
        "circle",
        "point",
        "pixel",
        "x_line",
        "plus_line",
        "vertical_line",
        "horizontal_line",
        "triangle_down",
        "triangle",
        "triangle_left",
        "triangle_right",
        "square",
        "thin_diamond",
        "diamond",
        "cross",
        "x",
    ]

    for format in ("png", "svg"):
        output = io.BytesIO()
        fig.savefig(output, format=format)
        assert output.tell() > 100


@pytest.mark.parametrize(
    ("symbol", "expected_width"),
    (("diamond", 2**0.5 * 10), ("thin_diamond", 0.6 * 2**0.5 * 10)),
)
def test_svg_diamond_markers_match_matplotlib_path_extents(
    symbol: str, expected_width: float
) -> None:
    path = _svg._SYMBOL_BUILDERS[symbol](10.0, 20.0, 5.0)
    coordinates = np.asarray(
        [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path)]
    ).reshape(-1, 2)

    assert np.ptp(coordinates[:, 0]) == pytest.approx(expected_width, abs=0.01)
    assert np.ptp(coordinates[:, 1]) == pytest.approx(2**0.5 * 10, abs=0.01)


@pytest.mark.parametrize(
    ("symbol", "expected_path"),
    (
        ("horizontal_line", '<path d="M 5 20 H 15" fill="none"'),
        ("vertical_line", '<path d="M 10 15 V 25" fill="none"'),
    ),
)
def test_svg_line_markers_have_one_point_space_orientation(
    symbol: str,
    expected_path: str,
) -> None:
    path = _svg._SYMBOL_BUILDERS[symbol](10.0, 20.0, 5.0)
    assert path == expected_path


def test_line_marker_codes_stay_aligned_across_renderers() -> None:
    root = Path(__file__).resolve().parents[2]
    chartview = (root / "js/src/50_chartview.ts").read_text(encoding="utf-8")
    shader = (root / "js/src/40_gl.ts").read_text(encoding="utf-8")
    raster = (root / "src/raster.rs").read_text(encoding="utf-8")

    assert marks._SYMBOL_CODES["horizontal_line"] == _raster._SYMBOLS["horizontal_line"] == 17
    assert marks._SYMBOL_CODES["vertical_line"] == _raster._SYMBOLS["vertical_line"] == 18
    assert "horizontal_line: 17" in chartview
    assert "vertical_line: 18" in chartview
    assert "symbol == 17 || symbol == 18" in shader
    assert "17 => (px.abs() - r).max(py.abs())" in raster
    assert "18 => px.abs().max(py.abs() - r)" in raster


def test_scatter_authored_markers_keep_distinct_renderer_specs_and_exports():
    fig, ax = plt.subplots()
    markers = (
        r"$\clubsuit$",
        [[-1, -1], [1, -1], [1, 1], [-1, -1]],
        (5, 0),
        (5, 1),
        (5, 2),
    )
    for index, marker in enumerate(markers):
        ax.scatter([index], [index], s=80, marker=marker, label=f"marker {index}")
    ax.legend()

    payload, _blob = ax._build_chart(640, 480).figure().build_payload()
    styles = [trace["style"] for trace in payload["traces"] if trace["kind"] == "scatter"]
    assert styles[0]["marker_glyph"] == "♣"
    assert styles[1]["marker_path"]["filled"] is True
    assert len(styles[1]["marker_path"]["contours"][0]) == 8
    assert len(styles[2]["marker_path"]["contours"][0]) == 12
    assert len(styles[3]["marker_path"]["contours"][0]) == 22
    assert styles[4]["marker_path"]["filled"] is False
    assert len(styles[4]["marker_path"]["contours"]) == 5
    assert (
        len({repr(style.get("marker_path") or style.get("marker_glyph")) for style in styles}) == 5
    )

    svg = ax._build_chart(640, 480).figure().to_svg()
    assert svg.count(">♣</text>") >= 2  # mark plus legend handle
    assert svg.count("<path d=") >= 8
    png = fig._to_png()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_raster_authored_marker_uses_explicit_edgecolor(monkeypatch):
    fig, ax = plt.subplots()
    ax.scatter(
        [0],
        [0],
        marker=(5, 0),
        s=200,
        c="#ff0000",
        edgecolors="#00ff00",
        linewidths=4,
    )

    core = ax._build_chart(320, 240).figure()
    spec, blob = core.build_payload()
    strokes = []
    original_stroke = _raster._Cmd.stroke

    def record_stroke(self, points, width, color, closed=False, dash=None, cap="round"):
        strokes.append((width, color, closed))
        return original_stroke(
            self,
            points,
            width,
            color,
            closed=closed,
            dash=dash,
            cap=cap,
        )

    monkeypatch.setattr(_raster._Cmd, "stroke", record_stroke)
    _raster.render_raster(spec, blob, scale=1)

    assert any(color == (0, 255, 0, 255) and closed for _width, color, closed in strokes)
    svg = core.to_svg()
    assert 'stroke="#00ff00"' in svg


def test_raster_authored_marker_match_fill_keeps_per_point_colors(monkeypatch):
    _, ax = plt.subplots()
    ax.scatter(
        [0, 1],
        [0, 1],
        marker=(5, 0),
        s=200,
        c=["#ff0000", "#0000ff"],
        edgecolors="face",
        linewidths=4,
    )

    spec, blob = ax._build_chart(320, 240).figure().build_payload()
    strokes = []
    original_stroke = _raster._Cmd.stroke

    def record_stroke(self, points, width, color, closed=False, dash=None, cap="round"):
        if closed:
            strokes.append(color)
        return original_stroke(
            self,
            points,
            width,
            color,
            closed=closed,
            dash=dash,
            cap=cap,
        )

    monkeypatch.setattr(_raster._Cmd, "stroke", record_stroke)
    _raster.render_raster(spec, blob, scale=1)

    assert (255, 0, 0, 255) in strokes
    assert (0, 0, 255, 255) in strokes


def test_constant_scatter_sizes_survive_automatic_legend_derivation(monkeypatch):
    fig, ax = plt.subplots()
    areas = np.asarray([100.0, 300.0, 500.0])
    for area in areas:
        ax.scatter([], [], s=area, c="black", linewidth=0, label=f"{area:g} km²")
    ax.legend(frameon=False, labelspacing=1, title="City Area")

    core = ax._build_chart(640, 480).figure()
    spec, blob = core.build_payload()
    expected_sizes = np.sqrt(areas) * (100 / 72)
    items = _svg.legend_items(spec["traces"])
    np.testing.assert_allclose(
        [item["style"]["size"] for item in items],
        expected_sizes,
    )

    svg = core.to_svg()
    radii = [float(value) for value in re.findall(r"<circle[^>]+r=\"([^\"]+)\"", svg)]
    np.testing.assert_allclose(radii, expected_sizes / 2, atol=0.01)

    raster_radii = []
    original_point = _raster._Cmd.point

    def record_point(self, x, y, radius, symbol, fill, stroke_width, stroke):
        raster_radii.append(radius)
        return original_point(self, x, y, radius, symbol, fill, stroke_width, stroke)

    monkeypatch.setattr(_raster._Cmd, "point", record_point)
    _raster.render_raster(spec, blob, scale=1)
    np.testing.assert_allclose(raster_radii, expected_sizes / 2)


def test_native_core_scatter_keeps_fixed_legend_swatch_semantics():
    chart = xy.scatter_chart(
        xy.scatter([], [], size=32, name="native"),
        xy.legend(),
    )
    spec, _ = chart.figure().build_payload()
    [item] = _svg.legend_items(spec["traces"])

    assert "_legend_trace_size" not in spec["traces"][0]["style"]
    assert "size" not in item["style"]


def test_authored_marker_grammar_fails_loudly_outside_bounded_contract():
    _, ax = plt.subplots()
    with pytest.raises(NotImplementedError, match="marker mathtext"):
        ax.scatter([0], [0], marker=r"$\frac{1}{2}$")
    with pytest.raises(ValueError, match="shape"):
        ax.scatter([0], [0], marker=[[0, 0, 1]])
    with pytest.raises(ValueError, match="style must be"):
        ax.scatter([0], [0], marker=(5, 7))


def test_numpy_custom_marker_vertices_do_not_require_scalar_truthiness():
    _, ax = plt.subplots()
    ax.scatter([0], [0], marker=np.asarray([[-1, -1], [1, -1], [0, 1]]))
    payload, _blob = ax._build_chart(320, 240).figure().build_payload()
    marker_path = payload["traces"][0]["style"]["marker_path"]
    assert marker_path["filled"] is True
    assert len(marker_path["contours"][0]) == 8


@pytest.mark.parametrize("method", ("axhline", "axvline"))
def test_axis_lines_render_direct_endpoint_markers(method: str):
    _, ax = plt.subplots()
    getattr(ax, method)(
        0.5,
        marker=".",
        ms=9,
        mfc="red",
        mec="black",
        mew=2,
        color="blue",
    )

    payload, _blob = ax._build_chart(640, 480).figure().build_payload()
    scatter = [trace for trace in payload["traces"] if trace["kind"] == "scatter"]
    assert len(scatter) == 1
    assert scatter[0]["style"]["symbol"] == "point"
    assert scatter[0]["color"]["color"] == "red"
    assert scatter[0]["style"]["stroke"] == "black"
    assert scatter[0]["n_marks"] == 2


@pytest.mark.parametrize(
    ("method", "coordinate"),
    (("axhline", "x"), ("axvline", "y")),
)
def test_axis_line_endpoint_markers_use_final_aspect_domain(
    method: str,
    coordinate: str,
):
    _, ax = plt.subplots(figsize=(8, 3))
    ax.plot([0, 1], [0, 4])
    ax.axis("equal")
    if method == "axhline":
        ax.axhline(2, xmin=0.25, xmax=0.75, marker=".")
    else:
        ax.axvline(0.5, ymin=0.25, ymax=0.75, marker=".")

    figure = ax._build_chart(800, 300).figure()
    payload, _blob = figure.build_payload()
    domain = np.asarray(payload[f"{coordinate}_axis"]["domain"], dtype=float)
    expected = domain[0] + np.asarray([0.25, 0.75]) * np.diff(domain)[0]
    [marker_trace] = [trace for trace in figure.traces if trace.kind == "scatter"]
    actual = np.asarray(getattr(marker_trace, coordinate).values, dtype=float)

    np.testing.assert_allclose(actual, expected)


# Matplotlib's documented "no marker" family — ``"none"``, ``"None"``, ``""``,
# ``" "`` — means "draw no markers".  These were rejected by the strict marker
# grammar with ``ValueError: unsupported marker: ...`` (commit 052da1a); the
# cases below pin the recognition of the family and its distinction from
# ``marker=None`` (which still means "use the default marker").
_NO_MARKER_SPELLINGS = ("none", "None", "", " ")


@pytest.mark.parametrize("spelling", _NO_MARKER_SPELLINGS)
def test_marker_render_spec_recognizes_no_marker_family(spelling: str):
    from xy.pyplot._markers import is_no_marker, marker_render_spec

    assert is_no_marker(spelling)
    # Case- and whitespace-insensitive, matching matplotlib's MarkerStyle nothing.
    assert is_no_marker("  NoNe ")
    assert marker_render_spec(spelling) == {"symbol": None}


def test_marker_render_spec_keeps_real_markers_and_no_marker_sentinel_distinct():
    from xy.pyplot._markers import is_no_marker, marker_render_spec

    # ``None`` is "use the default marker", NOT a no-marker sentinel — it must
    # stay outside the nothing family so scatter() still defaults to a circle
    # and plot() still omits the overlay.
    assert is_no_marker(None) is False
    assert is_no_marker("o") is False
    assert is_no_marker((5, 0)) is False
    assert marker_render_spec("o") == {"symbol": "circle"}


def test_plot_with_no_marker_family_draws_line_without_marker_overlay():
    _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], marker="none")

    payload, _blob = ax._build_chart(640, 480).figure().build_payload()
    kinds = [trace["kind"] for trace in payload["traces"]]
    assert "line" in kinds
    assert "scatter" not in kinds


def test_plot_with_explicit_marker_still_overlays_scatter():
    _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], marker="o")

    payload, _blob = ax._build_chart(640, 480).figure().build_payload()
    assert [trace["kind"] for trace in payload["traces"]] == ["line", "scatter"]


def test_scatter_with_no_marker_family_emits_no_scatter_trace():
    _, ax = plt.subplots()
    collection = ax.scatter([0, 1], [0, 1], marker="none", label="nothing")

    payload, _blob = ax._build_chart(640, 480).figure().build_payload()
    assert [trace for trace in payload["traces"] if trace["kind"] == "scatter"] == []
    # The handle remains usable for the public Collection API.
    np.testing.assert_array_equal(
        np.asarray(collection.get_offsets()), np.asarray([[0, 0], [1, 1]])
    )
    assert collection.get_label() == "nothing"


def test_scatter_with_no_marker_still_draws_default_circle():
    _, ax = plt.subplots()
    ax.scatter([0, 1], [0, 1], marker=None)

    payload, _blob = ax._build_chart(640, 480).figure().build_payload()
    assert [trace for trace in payload["traces"] if trace["kind"] == "scatter"]


def test_errorbar_with_no_marker_family_skips_marker_overlay():
    _, ax = plt.subplots()
    ax.errorbar([0, 1], [0, 1], yerr=0.1, marker="none")

    payload, _blob = ax._build_chart(640, 480).figure().build_payload()
    kinds = [trace["kind"] for trace in payload["traces"]]
    assert "errorbar" in kinds
    assert "scatter" not in kinds


@pytest.mark.parametrize("method", ("axhline", "axvline"))
def test_axis_lines_with_no_marker_family_skip_endpoint_markers(method: str):
    _, ax = plt.subplots()
    if method == "axhline":
        ax.axhline(0.5, marker="none")
    else:
        ax.axvline(0.5, marker="none")

    payload, _blob = ax._build_chart(640, 480).figure().build_payload()
    assert [trace for trace in payload["traces"] if trace["kind"] == "scatter"] == []


def test_unknown_marker_string_still_raises_loudly():
    # The strict grammar still rejects genuinely unknown strings — the nothing
    # family is an accepted exception, not a relaxation of validation.
    from xy.pyplot._markers import marker_render_spec

    with pytest.raises(ValueError, match="unsupported marker"):
        marker_render_spec("definitely-not-a-matplotlib-marker")
