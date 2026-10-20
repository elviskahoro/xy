"""Server-side resolution of bounded Reflex selection events."""

from __future__ import annotations

from typing import Any

from xy._figure import Selection

from .registry import registry

__all__ = ["resolve_selection"]


def resolve_selection(event: Any) -> Selection | None:
    """Re-resolve a ``select_end`` event against the live figure.

    Returns the complete, unbounded canonical selection for the event token.
    Call :meth:`Selection.rows` for deterministic JSON dictionaries. This is
    the fallback when the bounded event payload reports ``truncated``.
    Unknown tokens, cleared/empty selections, and malformed events return
    ``None``; client input never raises from this helper.

    The kernel read is serialized and generation-leased like every other
    runtime sync figure call site (``registry.append`` and
    ``push_view_message``): a concurrent append cannot tear the canonical
    columns mid-resolution, a republish that swaps the generation is
    discarded via ``is_current``, and the TTL sweep cannot evict the leased
    entry mid-call. A stale-generation resolution returns ``None`` — the
    same swallow-and-``None`` contract this helper uses for any client-facing
    miss, never stale rows.
    """
    try:
        if not isinstance(event, dict) or event.get("type") != "select_end":
            return None
        token = event.get("token")
        selection = event.get("selection")
        if not isinstance(token, str) or not isinstance(selection, dict):
            return None
        if selection.get("cleared") or selection.get("kind") == "clear":
            return None
        # Lease the current immutable generation so the TTL sweep skips it
        # mid-call and a republish is detectable via ``is_current``. Mirrors
        # ``registry.append`` / ``push_view_message``; this is the only other
        # runtime sync kernel call site in the package.
        entry = registry._acquire_operation(token)
        if entry is None:
            return None
        try:
            kind = selection.get("kind")
            # Serialize the kernel read against concurrent append mutation,
            # which extends the canonical columns in two statements and would
            # otherwise surface as a torn x/y ``ValueError`` (swallowed into a
            # spurious ``None`` by the broad catch below). ``sync_lock`` is a
            # plain threading.Lock taken briefly by sync callers only, so a
            # read hold cannot deadlock an async path.
            with entry.sync_lock:
                if kind == "box":
                    bounds = selection["data_bounds"]
                    per_trace = entry.figure.select_range(
                        bounds["x0"], bounds["x1"], bounds["y0"], bounds["y1"]
                    )
                elif kind == "lasso":
                    per_trace = entry.figure.select_polygon(selection["polygon"])
                else:
                    return None
            # A republish that swapped the generation during the kernel call
            # discards this result, as every other path does; returning stale
            # rows would violate the "live figure" contract.
            if not registry.is_current(token, entry):
                return None
            resolved = Selection(entry.figure, per_trace)
            return resolved if len(resolved) else None
        finally:
            registry._release_operation(entry)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
