Markered ``plot`` lines no longer leave "undead" markers behind after
``remove()``, ``set_visible(False)``, or ``set_alpha()``. The ``Line2D``
handle now owns its marker overlay the same way a ``Patch`` owns its
outline, so hiding, fading, recoloring, reordering, transforming, or
removing a line moves or takes down its markers too — instead of
orphaning a fully opaque ``_legend_skip`` scatter trace with no handle
left to remove it.
