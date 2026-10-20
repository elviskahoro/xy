Fixed the mathtext converter dropping interior spaces inside ``\text{...}``
wrappers. Multi-word ``\text{...}`` spans in axis labels, tick labels, titles,
legends, annotations and colorbar labels now render with their spaces preserved
(e.g. ``$\text{Water level}$`` → ``Water level`` instead of ``Waterlevel``),
matching matplotlib's text-mode ``\text`` (PR #22173). Math font groups
(``\mathrm``/``\textrm``/``\mathit``/``\mathbf``/``\operatorname``) still drop
unescaped spaces, matching matplotlib.