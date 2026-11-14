Scatter charts with per-point RGBA colors and a numeric size array no longer
export as a single-color PNG on linear axes. The native raster exporter's
affine fast path admitted `direct_rgba` color traces whenever the size
channel was `continuous`, but its opcode can only encode constant,
continuous, and categorical colors, so the per-point colors were silently
dropped and every point was painted with the trace's fallback fill; such
traces now defer to the expanded path that resolves per-point RGBA, matching
the SVG and WebGL renderers.
