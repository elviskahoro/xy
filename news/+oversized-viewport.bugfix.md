Wide charts whose backing store exceeds the GPU's `MAX_VIEWPORT_DIMS`
(typical on software renderers and some mobile/older integrated GPUs)
render again instead of going permanently blank. The shared WebGL host
used to throw an uncaught `RangeError` out of the `requestAnimationFrame`
draw callback whenever a chart's canvas backing store crossed that limit,
leaving it blank while still reporting `ready`/`live`. It now clamps the
render and pick passes to the GPU's viewport limit the way the native
per-chart path's driver does — painting the within-viewport portion and
keeping the chart live through context loss/restore instead of bricking
it with a misleading "WebGL2 context could not be restored." card.
