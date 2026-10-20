"""Native PDF export (_pdf.py): structural validity (xref/trailer), vector
fidelity (text as text, shapes as paths, rasters as image XObjects, gradients
as axial shadings), the closed-subset drift guard, and determinism."""

from __future__ import annotations

import base64
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
import pytest

import xy
from xy._figure import Figure
from xy._pdf import svg_to_pdf


def _basic_figure() -> Figure:
    x = np.linspace(0.0, 10.0, 40)
    fig = Figure(title="combo", x_label="time", y_label="value")
    fig.scatter(x, np.cos(x), name="pts")
    fig.line(x, np.sin(x), name="ln")
    fig.bar(["a", "b", "c"], [1.0, 3.0, 2.0], name="bars")
    return fig


def _objects(pdf: bytes) -> dict[int, bytes]:
    """Every `N 0 obj ... endobj` body keyed by object number."""
    return {
        int(m.group(1)): m.group(2)
        for m in re.finditer(rb"(\d+) 0 obj\n(.*?)\nendobj\n", pdf, re.S)
    }


def _stream(obj: bytes) -> bytes:
    head, _, rest = obj.partition(b"stream\n")
    assert b"/FlateDecode" in head
    return zlib.decompress(rest.rsplit(b"\nendstream", 1)[0])


def _content(pdf: bytes) -> bytes:
    objs = _objects(pdf)
    page = next(body for body in objs.values() if re.search(rb"/Type /Page(?!s)", body))
    match = re.search(rb"/Contents (\d+) 0 R", page)
    assert match is not None
    return _stream(objs[int(match.group(1))])


def _text_runs(content: bytes) -> list[tuple[float, float, bytes]]:
    """One ``(tx, ty, pdf_string)`` per ``BT/ET`` glyph run in a content stream.

    The generator emits a single ``BT ... Tm ... Tj ... ET`` block per text
    run; the ``Tm`` operands are ``a b c d tx ty`` whose ``tx``/``ty`` are the
    run's baseline in SVG user space (the top-level ``cm`` y flip and the
    per-text ``Tm`` y flip compose to net-upright, so ``ty`` equals the SVG
    ``y``). Used to assert multi-line chrome lines render at distinct
    baselines instead of collapsing onto the page origin.
    """
    runs: list[tuple[float, float, bytes]] = []
    in_text = False
    tm: tuple[float, float] | None = None
    for op in content.decode("latin1").split("\n"):
        if op == "BT":
            in_text = True
            tm = None
        elif op == "ET":
            in_text = False
        elif in_text:
            if op.endswith(" Tm"):
                parts = op.split()
                tm = (float(parts[-3]), float(parts[-2]))
            elif op.endswith(" Tj") and tm is not None:
                runs.append((tm[0], tm[1], op[:-3].encode("latin1")))
    return runs


def _xref_offsets(pdf: bytes) -> dict[int, int]:
    """Parse the xref table, verifying each offset points at `N 0 obj`."""
    startxref = int(pdf[pdf.rindex(b"startxref") + len(b"startxref") :].split()[0])
    header = re.match(rb"xref\n0 (\d+)\n", pdf[startxref:])
    assert header is not None, "startxref must point at the xref keyword"
    size = int(header.group(1))
    table = pdf[startxref + header.end() : startxref + header.end() + 20 * size]
    assert table[:20] == b"0000000000 65535 f \n"
    offsets: dict[int, int] = {}
    for num in range(1, size):
        entry = table[20 * num : 20 * num + 20]
        assert entry[10:11] == b" " and entry[16:18] == b" n", entry
        offset = int(entry[:10])
        assert pdf[offset:].startswith(f"{num} 0 obj".encode()), (
            f"xref offset for object {num} is not byte-accurate"
        )
        offsets[num] = offset
    return offsets


def test_pdf_structure_xref_and_trailer() -> None:
    pdf = svg_to_pdf(_basic_figure().to_svg())
    assert pdf.startswith(b"%PDF-1.")
    assert len(re.findall(rb"/Type /Page(?![s])", pdf)) == 1  # single page
    # 900x420 px at 0.75 pt/px.
    assert b"/MediaBox [0 0 675 315]" in pdf

    offsets = _xref_offsets(pdf)
    root_match = re.search(rb"trailer\n<< /Size \d+ /Root (\d+) 0 R >>", pdf)
    assert root_match is not None
    root_num = int(root_match.group(1))
    catalog = pdf[offsets[root_num] : offsets[root_num] + 120]
    assert b"/Type /Catalog" in catalog  # /Root resolves


def test_pdf_content_is_vector_text_and_paths() -> None:
    svg = _basic_figure().to_svg()
    pdf = svg_to_pdf(svg)
    content = _content(pdf)

    # Text stays text: BT/ET blocks with Tj strings using the Helvetica family.
    assert b"BT" in content and b"ET" in content
    assert b"/BaseFont /Helvetica" in pdf
    root = ET.fromstring(svg)
    tick_labels = [
        el.text
        for el in root.iter()
        if el.tag.endswith("text") and el.get("text-anchor") == "end" and el.text
    ]
    assert tick_labels, "figure should carry y tick labels"
    assert any(f"({label}) Tj".encode() in content for label in tick_labels)

    # Vector shapes, not one big image: path construction + paint operators.
    for op in (b" re\n", b" m\n", b" l\n", b"\nf\n", b"\nS\n", b"W n"):
        assert op in content, f"missing content-stream op {op!r}"
    assert b"/Subtype /Image" not in pdf  # nothing was rasterized


def test_heatmap_embeds_matching_image_xobject() -> None:
    rng = np.random.default_rng(2)
    fig = Figure()
    fig.heatmap(rng.random((8, 6)))
    svg = fig.to_svg()
    pdf = svg_to_pdf(svg)

    root = ET.fromstring(svg)
    image = next(el for el in root.iter() if el.tag.endswith("image"))
    png = base64.b64decode(image.get("href").split(",", 1)[1])
    w, h = struct.unpack(">II", png[16:24])

    dims = [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(
            rb"/Subtype /Image /Width (\d+) /Height (\d+) /ColorSpace /DeviceRGB", pdf
        )
    ]
    assert (w, h) in dims, f"no /DeviceRGB image XObject with {w}x{h}, found {dims}"
    assert b"/Interpolate false" in pdf  # pixelated rendering intent


def test_gradient_becomes_axial_shading() -> None:
    x = np.linspace(0.0, 10.0, 30)
    fig = Figure()
    fig.area(x, np.abs(np.sin(x)) + 0.1, fill="linear-gradient(currentColor, transparent)")
    svg = fig.to_svg()
    if "linearGradient" not in svg:
        pytest.skip("generator emitted no gradient for this figure")
    pdf = svg_to_pdf(svg)
    assert b"/ShadingType 2" in pdf or b"/ShadingType 3" in pdf
    assert b"/SMask << /S /Luminosity" in pdf  # transparent stop -> soft mask
    assert b" sh" in _content(pdf)  # painted inside the geometry's clip
    assert b"W n" in _content(pdf)


def test_unsupported_svg_features_raise() -> None:
    ns = 'xmlns="http://www.w3.org/2000/svg"'
    with pytest.raises(ValueError, match="unsupported SVG feature"):
        svg_to_pdf(f'<svg {ns} width="100" height="50"><foreignObject/></svg>')
    # Attribute drift on a known element fails loudly too.
    with pytest.raises(ValueError, match="unsupported SVG feature"):
        svg_to_pdf(
            f'<svg {ns} width="100" height="50">'
            '<rect x="0" y="0" width="10" height="10" filter="url(#b)"/></svg>'
        )
    # ...as does a path command outside the generator's subset.
    with pytest.raises(ValueError, match="unsupported SVG feature"):
        svg_to_pdf(f'<svg {ns} width="100" height="50"><path d="M 0 0 Q 5 5 10 0"/></svg>')
    # ...and the multi-line tspan drift this bug fixed: the generator must
    # keep emitting absolute per-line <tspan y="..."> rather than <tspan dy>,
    # so the closed subset still refuses dy on a tspan (browsers would render
    # it, but the native PDF path must not silently accept the drift again).
    with pytest.raises(ValueError, match="unsupported SVG feature"):
        svg_to_pdf(
            f'<svg {ns} width="100" height="50">'
            '<text x="10" y="20" font-size="12"><tspan x="10" dy="12">x</tspan></text></svg>'
        )


def test_pdf_output_is_deterministic() -> None:
    svg = _basic_figure().to_svg()
    assert svg_to_pdf(svg) == svg_to_pdf(svg)

    rng = np.random.default_rng(3)
    fig = Figure()
    fig.heatmap(rng.random((6, 5)))
    heatmap_svg = fig.to_svg()
    assert svg_to_pdf(heatmap_svg) == svg_to_pdf(heatmap_svg)


def test_facet_grid_converts_with_clipped_panels() -> None:
    x = np.linspace(0.0, 10.0, 50)
    data = {
        "x": np.tile(x, 3),
        "y": np.concatenate([np.sin(x), np.cos(x), np.sin(2 * x)]),
        "g": np.repeat(["a", "b", "c"], 50),
    }
    chart = xy.facet_chart(xy.line(x="x", y="y"), by="g", cols=2, title="grid", data=data)
    svg = chart.to_svg()
    panels = svg.count("<svg") - 1  # nested panel viewports
    assert panels > 1

    content = _content(svg_to_pdf(svg))
    # Each panel viewport becomes a translated group clipped to its bounds.
    translated = re.findall(rb"\n1 0 0 1 [\d.]+ [\d.]+ cm\n", content)
    assert len(translated) == panels
    assert content.count(b"W n") >= panels


def test_round_trip_via_external_oracle(tmp_path: Path) -> None:
    pdf = svg_to_pdf(_basic_figure().to_svg())
    path = tmp_path / "figure.pdf"
    path.write_bytes(pdf)
    qpdf = shutil.which("qpdf")
    mutool = shutil.which("mutool")
    pdftoppm = shutil.which("pdftoppm")
    if qpdf:
        proc = subprocess.run([qpdf, "--check", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    elif mutool:
        out = tmp_path / "page.png"
        proc = subprocess.run(
            [mutool, "draw", "-o", str(out), str(path), "1"], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert out.exists() and out.stat().st_size > 0
    elif pdftoppm:
        proc = subprocess.run(
            [pdftoppm, "-png", "-r", "72", str(path), str(tmp_path / "page")],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert list(tmp_path.glob("page*.png"))
    else:
        pytest.skip("no external PDF oracle (qpdf/mutool/pdftoppm) on PATH")


def _title_root(svg: str, joined: str) -> ET.Element:
    root = ET.fromstring(svg)
    return next(
        el for el in root.iter() if el.tag.endswith("text") and "".join(el.itertext()) == joined
    )


def test_multiline_title_exports_pdf_with_one_tspan_per_line() -> None:
    """A multi-line title exports to PDF instead of raising on a ``<tspan>``
    ``dy`` attribute (regression: ``ValueError: unsupported SVG feature:
    <tspan> attribute 'dy'``).

    The generator now emits one ``<tspan x y>`` per line — no ``dy`` — so the
    PDF backend's whitelist accepts every line, and each line renders at its
    own baseline (the parent ``y`` offset by ``index * line_step``) instead
    of every line collapsing onto the page origin (``y=0``).
    """
    lines = ["First title line", "Second title line"]
    fig = Figure(width=640, height=360, title="\n".join(lines))
    fig.line([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    svg = fig.to_svg()

    # The chrome block is one <text> with one <tspan> per line, each carrying
    # an absolute <tspan y="..."> — never dy.
    assert "dy=" not in svg
    title = _title_root(svg, "".join(lines))
    tspans = [el for el in title if el.tag.endswith("tspan")]
    assert [el.text for el in tspans] == lines
    assert all("y" in el.attrib for el in tspans)
    svg_ys = [float(el.attrib["y"]) for el in tspans]
    assert len(svg_ys) == 2 and svg_ys[1] > svg_ys[0]

    pdf = svg_to_pdf(svg)
    assert pdf.startswith(b"%PDF-1.")
    assert b"/Type /Page" in pdf
    content = _content(pdf)
    # Both lines are present as distinct Tj glyph runs.
    for line in lines:
        assert f"({line}) Tj".encode("latin1") in content

    title_runs = [
        r for r in _text_runs(content) if any(li.encode("latin1") in r[2] for li in lines)
    ]
    assert len(title_runs) == 2
    tys = sorted(r[1] for r in title_runs)
    assert 0.0 not in tys  # the pre-fix failure mode stacked every line at y=0
    assert tys[1] - tys[0] == pytest.approx(svg_ys[1] - svg_ys[0], abs=0.01)

    # Determinism is preserved on the multi-line path.
    assert svg_to_pdf(svg) == pdf


def test_multiline_tick_labels_export_pdf() -> None:
    """A multi-line tick label splits into PDF-consumable per-line tspans and
    each line is a distinct Tj (exercises the tick-label ``_text_block_content``
    caller, separate from titles and axis titles)."""
    chart = xy.chart(
        xy.line([0.0, 1.0], [0.0, 1.0]),
        xy.x_axis(
            tick_values=(0.0, 1.0),
            tick_labels=("plain tick", "split\ntick"),
            tick_label_strategy="preserve",
        ),
        width=360,
        height=240,
    )
    svg = chart.figure().to_svg()
    assert "dy=" not in svg
    root = ET.fromstring(svg)
    split = next(
        el
        for el in root.iter()
        if el.tag.endswith("text") and "".join(el.itertext()) == "splittick"
    )
    tspans = [el for el in split if el.tag.endswith("tspan")]
    assert [el.text for el in tspans] == ["split", "tick"]
    assert all("y" in el.attrib for el in tspans)
    assert float(tspans[1].attrib["y"]) > float(tspans[0].attrib["y"])

    content = _content(svg_to_pdf(svg))
    assert b"(plain tick) Tj" in content  # single-line tick stays one direct run
    assert b"(split) Tj" in content
    assert b"(tick) Tj" in content


def test_pdf_tspan_inherits_parent_text_xy() -> None:
    """A ``<tspan>`` that omits ``x``/``y`` inherits the parent ``<text>``'s
    position (SVG semantics) instead of defaulting to ``0.0`` and stacking at
    the page origin. The generator emits absolute coordinates today, so this
    guards the subset against a future tspan that elides a coordinate.
    """
    ns = 'xmlns="http://www.w3.org/2000/svg"'
    pdf = svg_to_pdf(
        f'<svg {ns} width="100" height="50">'
        '<text x="10" y="20" font-size="12"><tspan>inherited</tspan></text></svg>'
    )
    (run,) = [r for r in _text_runs(_content(pdf)) if b"inherited" in r[2]]
    assert run[0] == pytest.approx(10.0, abs=0.01)
    assert run[1] == pytest.approx(20.0, abs=0.01)
