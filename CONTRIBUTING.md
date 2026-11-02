Run `python -m pytest -q` and `python -m ruff check src tests scripts` before a PR.
For mathematical changes, identify the manuscript equation and add a small
invariant/reference test. Preserve distinction between reported numbers and
measured results. Changes to a default, normalization, tie rule, or ablation
definition must update the method and reproduction notes.
