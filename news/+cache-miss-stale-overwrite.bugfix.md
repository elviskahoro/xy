Data-bound charts no longer go stale on a registry cache miss that races a
state-driven republish. The composite-plan rebuild used to re-run the
`@reflex_xy.data` method and then publish the resulting columns
unconditionally, so a rebuild whose state read predated a concurrent data
republish installed a newer version number carrying older columns,
overwriting the fresher generation. Subscribers of every mounted dependent
plan then received a stale figure stamped as the current version (client-side
staleness checks could not reject it), and the column cache kept serving the
stale data until the next state mutation republished. The cache-fill publish
is now conditional on the generation observed at the initial miss, mirroring
the compare-and-insert guard the figure tier already uses, so an older
rebuild that finishes late can never overwrite a newer one's columns.
