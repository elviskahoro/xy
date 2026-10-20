`Axes.grid()` no longer strands the grid permanently invisible after a
`grid(True) -> grid(False) -> grid(True)` round trip. The disable path
persisted a `"transparent"` sentinel into the per-axis style dict; the
re-enable read that sentinel back via `axis_style.get("grid_color", fallback)`,
so the rc/theme grid color (`#b0b0b0`) was never restored and every
renderer (WebGL canvas, SVG, native PNG) — which paint grid lines solely
from the persisted `grid_color` — drew nothing. The same poisoning arose
from the `else`-branch seed for an unselected axis (so
`grid(True, axis="x")` then `grid(True, axis="y")` silently lost the y
grid) and from `_rc_minor_axis_style`'s default `grid_color="transparent"`
(so a fresh `grid(True, which="minor")` with no explicit color read the
sentinel back). Re-enabling now drops the disable-time sentinel unless
the caller authored it via an explicit `color=`, then falls back to the
rc/theme color the way matplotlib's `grid(True)` does; an explicit
`color="transparent"` and a previously authored color are still honored.
