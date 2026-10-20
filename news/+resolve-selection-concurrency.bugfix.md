Cross-filter selections on streaming figures no longer silently drop or go
stale: `resolve_selection` now serializes its kernel read with the figure's
`sync_lock`, leases the generation, and discards results from a superseded
generation via `is_current`, matching `registry.append` and
`push_view_message`. A concurrent `append` previously surfaced a torn x/y
read as a spurious `None`, and a concurrent republish could return rows from
the old generation as if they were the live selection.
