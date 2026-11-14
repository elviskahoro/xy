PDF export no longer fails for figures with a multi-line or auto-wrapped title,
axis label, or tick label. The static SVG generator emitted one `<tspan>` per
wrapped line with a relative `dy`, but the native PDF backend's closed
attribute subset accepts only `x`/`y` on a `<tspan`, so any chrome text that
wrapped to a second line raised `ValueError: unsupported SVG feature: <tspan>
attribute 'dy'` and `to_image(format="pdf")` produced no file at all — including
long single-line titles the chart wrapped automatically with no authored
newline. Multi-line chrome now uses the same absolute per-line `<tspan x y>`
form the annotation path already used, and the PDF backend inherits the parent
`<text>`'s position for a `<tspan>` that omits a coordinate instead of stacking
every line at the page origin.
