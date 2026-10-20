Hidden legend traces and categories no longer reappear on the next streamed
data tick when the chart has animation configured. The animated streaming
`append` path now re-applies legend-toggle state after `updatePayload`'s full
rebuild, so hidden series/categories stay hidden (and un-hoverable, un-pickable,
and excluded from selections) while their legend rows still read toggled off,
and density traces keep their masked re-bin request armed instead of standing
on the unfiltered aggregate.
