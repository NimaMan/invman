# sources

Source tracking for the Hormuz history package.

Files:

- `source_manifest.csv`
  - URLs, publication dates, local paths, and notes
- `checksums.sha256`
  - integrity hashes for the local raw files referenced by the history dataset

The history build regenerates these files from the checked-in raw snapshots. They are the canonical
provenance record for every processed table in `../data/processed/` and every generated summary in
`../results/`.
