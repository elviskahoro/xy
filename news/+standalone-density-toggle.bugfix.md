Standalone density scatter charts no longer lose their point cloud when a
legend category is toggled. The legend category-toggle handler was emptying
the density cache to a bare `[]`, which is truthy and so defeated the sample
selector's `densityCache || fallback` path that reaches the surviving home
sample overlay — leaving the kernel-less standalone page with only the stale
aggregate texture and no points tier until a zoom-in re-bin followed by a
zoom-out home-restore. The live grid now stays in the rebuilt cache so the
filtered overlay keeps drawing at home zoom.
