Backends served by uvicorn no longer abort startup when an `@reflex_xy.figure`
builder that explicitly opts into the compile probe is an `async def`. The probe
ran opted-in async builders via `asyncio.run(...)`, which raises
`RuntimeError: asyncio.run() cannot be called from a running event loop` whenever
`post_compile` fires inside the server's already-running event loop — exactly
what happens under `uvicorn --factory` (the path `reflex run` dispatches to, the
Windows prod backend, and any manual `uvicorn ... --factory` deployment). The
worker died during startup with a `FigureProbeError` that advised
`@reflex_xy.figure(probe=False)` — telling the user to disable the feature they
had explicitly opted into. The probe now drives opted-in async builders on a
fresh, private event loop in a worker thread when a foreign loop is already
running (the same-thread `run_until_complete` path is also blocked on Python 3.10+,
enforced in 3.13), and distinguishes a probe-infrastructure failure from a real
builder defect so the surfaced message no longer recommends opting out of a probe
the user turned on.