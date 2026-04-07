# results

Generated summary outputs for the Hormuz history package.

Files:

- `one_year_backtest_summary.md`
  - human-readable summary of the key price and shipping results
- `one_year_backtest_summary.json`
  - machine-readable version of the same summary
- `ten_year_market_context_summary.md`
  - human-readable summary of the ten-year Brent/WTI context plus crisis placement
- `ten_year_market_context_summary.json`
  - machine-readable version of the ten-year summary
- `twenty_year_market_context_summary.md`
  - human-readable summary of the twenty-year Brent/WTI context plus crisis placement
- `twenty_year_market_context_summary.json`
  - machine-readable version of the twenty-year summary

These files are rebuilt by:

```bash
python rust/src/problems/hormuz_strait/history/scripts/fetch_and_build.py
```
