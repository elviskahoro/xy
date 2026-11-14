Static HTML exports of per-point RGBA scatters with authored (mathtext/path)
markers no longer paint every marker in the flat fallback color. The standalone
hydration loop now merges the wire fields into the build-retained `g._cpu`
record instead of wholesale-replacing it, and re-attaches the color column
under `rgba` for `direct_rgba` (matching the kernel append path) — so the
Canvas2D authored-marker overlay reads the correct per-point colors instead of
falling back to the trace color.
