Hovering a decimated line or area chart above 10,000 points no longer
reports the wrong source row. The client's GPU pick speaks a shipped vertex
index; the kernel's exact-row readout now translates that index through the
recorded M4 decimation (and any NaN/log/baseline drop) before reading the
canonical row, instead of treating the shipped index as a canonical row and
returning an arbitrary unrelated row. Out-of-range shipped indices also
return `None` instead of reading arbitrary canonical rows.
