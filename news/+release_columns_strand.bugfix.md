`@reflex_xy.data` charts no longer freeze on stale pixels when the data method
returns `None` and then columns again within the client's batching window.
`release_columns` now honors the same mounted-ness rule every other unmount
transition enforces, so a still-subscribed chart keeps its plan binding and
the next republish rebuilds and broadcasts to it; a chart left mounted
during the transient "no data" state also gets an `err {resync}` frame
instead of being silently stranded.
