`error_band` now honors CSS `stroke` as an independent outline color (via
`line_color`, matching `area`) instead of repainting the whole band's fill
and rejecting `{"fill": ..., "stroke": ...}` with a spurious conflict.
An `error_band(line_color=...)` kwarg was added for parity with `area`.
