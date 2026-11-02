# Reported manuscript values — not reproduced outputs

CSV files in this directory transcribe the supplied manuscript's tables and
explicitly stated ablation averages. They are reference material only. No entry
is a locally rerun GPU result, random-seed sample or simulated benchmark score.

`table_*.csv` retain the original table labels with explicit machine-readable
column names. Citation/LaTeX styling and thousands separators are removed; blank
cells correspond to the manuscript's dashes. Numeric values are unchanged.
Loss, PPL, source `G` memory estimates, measured GB and step reduction stay in their
original separate tables. `ablation_figure_averages.csv` includes only aggregate
values stated in the prose for missing figures. Unknown domain cells are not filled.
The exact source segments are also saved under `table_*.tex` for an audit.

Use `scripts/export_tables.py` on a new suite's `results.json` to create measured
comparison tables elsewhere. The implementation never reads this directory to
return benchmark results.
