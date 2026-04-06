# results

Generated summary outputs for the one-year Hormuz backtest package.

Files:

- `one_year_backtest_summary.md`
  - human-readable summary of the key price and shipping results
- `one_year_backtest_summary.json`
  - machine-readable version of the same summary

These files are rebuilt by:

```bash
python rust/src/problems/hormuz_strait/history/scripts/fetch_and_build.py
```
