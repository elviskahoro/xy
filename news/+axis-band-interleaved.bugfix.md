Multi-touch axis-band gestures no longer strand the zoom-span overlay or
drop the pan's end-phase `view_change` event. A second pointer landing on the
same band mid-gesture used to overwrite the in-flight drag record and orphan
it: its `end` handler returned early on the pointerId mismatch, so the grey
span rectangle stayed drawn indefinitely and an abandoned pan never emitted
its `pan_drag` `phase:"end"` event — the kernel-facing `view_state()` cache
kept reporting the pre-pan view while the chart showed the panned view, until
some later unrelated gesture shipped an end event. Per `spec/api/interaction.md`
§5, every capture-owning gesture must terminate deterministically with exactly
one end event; a second `pointerdown` on the same band now settles the prior
gesture first (releases capture, hides the span rect, commits the pan end at
its last in-frame view, rolls back a coordinate-dependent span) before starting
the new one.