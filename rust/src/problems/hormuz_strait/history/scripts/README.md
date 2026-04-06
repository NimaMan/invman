# scripts

Primary entrypoint:

```bash
python rust/src/problems/hormuz_strait/history/scripts/fetch_and_build.py
```

Refresh downloadable raw snapshots intentionally:

```bash
python rust/src/problems/hormuz_strait/history/scripts/fetch_and_build.py --refresh-downloads
```

What it does:

- reuses the checked-in raw snapshots by default and refreshes them only on request
- downloads the raw Brent and WTI price CSVs from FRED when the local snapshot is missing or when
  `--refresh-downloads` is passed
- falls back to the checked-in local snapshot if a refresh attempt fails at the network layer
- verifies and records all local source files used by the history package
- writes the source manifest and checksums
- builds the processed history tables
- writes a compact backtest-results summary in `../results/`
