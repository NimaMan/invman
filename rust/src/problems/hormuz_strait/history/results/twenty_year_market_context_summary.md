# twenty_year_market_context_summary

Generated from the reproducible twenty-year market context within the Hormuz history package.

## Key results

- Brent moved from `$67.58/b` on `2006-04-06` to `$127.61/b` on `2026-04-02`, a `88.8%` move across the `20.00`-year window.
- WTI moved from `$67.22/b` on `2006-04-06` to `$113.23/b` on `2026-04-02`, a `68.4%` move.
- Brent ranged from `$9.12/b` on `2020-04-21` to `$143.95/b` on `2008-07-03`. The latest Brent close sits at the `99.1` percentile of the window.
- WTI ranged from `$-36.98/b` on `2020-04-20` to `$145.31/b` on `2008-07-03`. The latest WTI close sits at the `97.6` percentile of the window.
- Within this horizon, the current crisis window still stands out: Brent averaged `$70.95/b` in the 30 days before `2026-02-28` versus `$103.18/b` during the crisis segment, with a crisis peak of `$127.61/b` on `2026-04-02`.
- Observed AIS commercial transits through Hormuz fell from `148` on `2026-02-28` to a trough of `1` on `2026-03-03`, a `-99.3%` collapse.
- By `2026-03-17`, observed AIS commercial transits were still only `4`, a `-97.3%` gap versus the pre-hostilities snapshot.

## Interpretation

- Twenty-year oil-price context for longer-cycle comparison against the current crisis.
- The history contains a clear regime break after `2026-02-28`: prices rise sharply while observed commercial transits collapse to single digits.
- This is enough evidence for a first-degree FlowNet to treat Hormuz transit capacity as a time-varying shock state rather than a static parameter.
- Shipping counts should be interpreted as lower-bound disruption indicators because JMIC repeatedly warns about AIS suppression, GNSS disruption, and possible dark transits.

## Rebuild

```bash
python rust/src/problems/hormuz_strait/history/scripts/fetch_and_build.py
```

Use `--refresh-downloads` only when intentionally refreshing the downloadable raw snapshots.
