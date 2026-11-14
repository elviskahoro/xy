Area and `error_band` traces whose bottom edge (`base`/`lower`) moves
independently of the top edge (`y`/`upper`) no longer have their baseline
dips/spikes silently clipped when the decimation tier activates above
`DECIMATION_THRESHOLD` (10,000 rows). The M4 selection now keys on the sorted
union of the per-bucket extrema of **both** edges, so a within-bucket
baseline extremum that is not also a `y` first/min/max/last row survives into
the shipped `base` vertex buffer — fixing the `extrema-exact` M4 guarantee
(spec/design/lod-architecture.md, §5 Tier 1) for the baseline of an area
fill. The zoom re-decimation path (`Figure.decimate_view`) is fixed the same
way. Line traces (no `base` column) are unchanged.
