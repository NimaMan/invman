# one_year_backtest_summary

Generated from the reproducible one-year Hormuz history package.

## Key results

- Brent moved from `$66.13/b` on `2025-04-07` to `$127.61/b` on `2026-04-02`, a `93.0%` increase over the backtest window.
- WTI moved from `$61.05/b` on `2025-04-07` to `$113.23/b` on `2026-04-02`, a `85.5%` increase.
- Brent averaged `$70.29/b` across the full year, with a pre-crisis 30-day average of `$70.95/b` and a crisis-window average of `$103.18/b`.
- The highest Brent observation in the crisis window was `$127.61/b` on `2026-04-02`.
- Observed AIS commercial transits through Hormuz fell from `148` on `2026-02-28` to a trough of `1` on `2026-03-03`, a `-99.3%` collapse.
- By `2026-03-17`, observed AIS commercial transits were still only `4`, a `-97.3%` gap versus the pre-hostilities snapshot.

## Interpretation

- The backtest window contains a clear regime break after `2026-02-28`: prices rise sharply while observed commercial transits collapse to single digits.
- This is enough evidence for a first-degree FlowNet to treat Hormuz transit capacity as a time-varying shock state rather than a static parameter.
- Shipping counts should be interpreted as lower-bound disruption indicators because JMIC repeatedly warns about AIS suppression, GNSS disruption, and possible dark transits.

## Rebuild

```bash
python rust/src/problems/hormuz_strait/history/scripts/fetch_and_build.py
```

Use `--refresh-downloads` only when intentionally refreshing the downloadable raw snapshots.
