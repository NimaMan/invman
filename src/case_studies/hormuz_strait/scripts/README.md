# scripts

Scripts for rebuilding the Strait of Hormuz data assets.

Primary entrypoints:

```bash
python src/case_studies/hormuz_strait/scripts/fetch_and_build.py
python src/case_studies/hormuz_strait/scripts/run_month_ahead_simulation.py
```

`fetch_and_build.py`:

- downloads the current raw source files referenced in the manifest
- computes checksums
- builds processed CSV views from the EIA figure spreadsheets
- writes the node-set, scenario-parameter, and market-anchor tables

`run_month_ahead_simulation.py`:

- calls the Rust month-ahead Brent scenario engine through the Python extension
- verifies that scenario severity is monotone in the generated day-30 means
- writes JSON, Markdown, and daily-path CSV outputs under `results/`
