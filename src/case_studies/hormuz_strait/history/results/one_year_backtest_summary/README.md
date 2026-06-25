# one_year_backtest_summary

Generated from the reproducible one-year backtest within the Hormuz history package.

## Key results

- Brent moved from `$66.13/b` on `2025-04-07` to `$127.61/b` on `2026-04-02`, a `93.0%` move across the `1.00`-year window.
- WTI moved from `$61.05/b` on `2025-04-07` to `$113.23/b` on `2026-04-02`, a `85.5%` move.
- Brent ranged from `$59.93/b` on `2025-12-16` to `$127.61/b` on `2026-04-02`. The latest Brent close sits at the `100.0` percentile of the window.
- WTI ranged from `$55.44/b` on `2025-12-16` to `$113.23/b` on `2026-04-02`. The latest WTI close sits at the `100.0` percentile of the window.
- Within this horizon, the current crisis window still stands out: Brent averaged `$70.95/b` in the 30 days before `2026-02-28` versus `$103.18/b` during the crisis segment, with a crisis peak of `$127.61/b` on `2026-04-02`.
- Observed AIS commercial transits through Hormuz fell from `148` on `2026-02-28` to a trough of `1` on `2026-03-03`, a `-99.3%` collapse.
- By `2026-03-17`, observed AIS commercial transits were still only `4`, a `-97.3%` gap versus the pre-hostilities snapshot.

## Interpretation

- Crisis-focused verification window used for first-degree FlowNet backtesting.
- The history contains a clear regime break after `2026-02-28`: prices rise sharply while observed commercial transits collapse to single digits.
- This is enough evidence for a first-degree FlowNet to treat Hormuz transit capacity as a time-varying shock state rather than a static parameter.
- Shipping counts should be interpreted as lower-bound disruption indicators because JMIC repeatedly warns about AIS suppression, GNSS disruption, and possible dark transits.

## Rebuild

```bash
python src/case_studies/hormuz_strait/history/scripts/fetch_and_build.py
```

Use `--refresh-downloads` only when intentionally refreshing the downloadable raw snapshots.
