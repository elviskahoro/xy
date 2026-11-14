``pyplot.parse_fmt`` now accepts a fmt string containing a single space
(``' '``) as the matplotlib ``_draw_nothing`` linestyle token, matching
``matplotlib.lines.lineStyles[' ']`` and
``matplotlib.axes._base._process_plot_format``. Previously ``parse_fmt(' ')``,
``parse_fmt('k ')`` and ``parse_fmt(' C1')`` raised
``ValueError: unrecognized character ' '``; because ``plot``/``stem``/
``triplot``/``boxplot``/``errorbar`` only guard with ``if fmt:`` (a space is
truthy), the exception propagated and the documented ``stem(markerfmt=' ')``
idiom for suppressing stem markers crashed. The downstream
``LINESTYLE_TO_DASH[' ']`` already mapped ``' '`` to the ``"none"``
sentinel, so accepting it in the parser routes the entry through the existing
no-line path (e.g. ``plot(..., 'k ')`` emits a marker-only ``scatter`` entry).
