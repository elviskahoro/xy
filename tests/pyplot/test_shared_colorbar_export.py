import io

import numpy as np
import pytest

from xy import pyplot as plt


def _height(png):
    return int.from_bytes(png[20:24], "big")


def _fig(n):
    fig, axs = plt.subplots(1, n)
    axs = np.atleast_1d(axs)
    im = axs[0].imshow(np.arange(25.0).reshape(5, 5))
    fig.colorbar(im, ax=list(axs))
    return fig


@pytest.mark.parametrize("n", [1, 2])
def test_shared_colorbar_in_svg_html_png(n, tmp_path):
    fig = _fig(n)
    svg = tmp_path / "a.svg"
    html = tmp_path / "a.html"
    fig.savefig(svg)
    fig.savefig(html)
    assert 'data-xy-colorbar="shared"' in svg.read_text()
    assert 'data-xy-colorbar="shared"' in html.read_text()
    buf = io.BytesIO()
    base, _ = plt.subplots(1, n)
    fig.savefig(buf, format="png")
    plain = io.BytesIO()
    base.savefig(plain, format="png")
    assert buf.getvalue() != plain.getvalue()
    assert _height(buf.getvalue()) == _height(plain.getvalue()) + 52
    plt.close("all")


def test_shared_colorbar_absolute_layout_png():
    fig, axs = plt.subplots(1, 2)
    fig.subplots_adjust(left=0.2)
    im = axs[0].imshow(np.arange(25.0).reshape(5, 5))
    base = io.BytesIO()
    fig.savefig(base, format="png")
    fig.colorbar(im, ax=list(axs))
    out = io.BytesIO()
    fig.savefig(out, format="png")
    assert _height(out.getvalue()) == _height(base.getvalue()) + 52
    plt.close("all")


@pytest.mark.parametrize("adjust", [False, True])
def test_shared_colorbar_reserves_notebook_height(adjust):
    fig, axs = plt.subplots(1, 2)
    if adjust:
        fig.subplots_adjust(left=0.2)
    im = axs[0].imshow(np.arange(25.0).reshape(5, 5))
    before = fig._to_notebook_html()[2]
    fig.colorbar(im, ax=list(axs))
    assert fig._to_notebook_html()[2] == before + 52
    plt.close("all")
