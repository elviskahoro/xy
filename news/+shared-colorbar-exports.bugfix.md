Fix the figure-level shared colorbar (`Figure.colorbar(mappable, ax=[ax1, ...])`)
being silently dropped from SVG, HTML, and multi-panel PNG exports. The shared
colorbar was only plumbed through `_to_png -> stitch_png`'s grid branch, so
`fig.colorbar(im, ax=axs)` produced SVG/HTML files missing the colorbar
entirely and even multi-panel PNG (whose `stitch_png` absolute-layout branch
returned before painting it). The fix plumbs `_shared_colorbar` through
`compose_svg` and `compose_html`, makes `stitch_png`'s absolute branch honor
the colorbar, and adds `_shared_colorbar is None` to the `_single()` fast-path
gate so single-chart figures with a shared colorbar take the composed path.
