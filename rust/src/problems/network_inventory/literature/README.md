# Literature

Current literature anchor for `network_inventory`:

- Pirhooshyaran and Snyder 2021

Current status:

- not literature-verified

Why:

- the Rust package now carries a paper-shaped discrete formulation
- `SINGLE_NODE_BENCHMARK_ROWS` reproduces the paper's analytical newsvendor rows exactly
- `SERIAL_BENCHMARK_ROWS` carries the paper's serial tables and is audited through a continuous
  paper-facing simulator
- the serial audit still does not reproduce the published rows tightly enough for the repo
  implementation to be marked `literature_verified`

Use `literature/references.rs` as the source of truth for:

- `SINGLE_NODE_BENCHMARK_ROWS`
- `SERIAL_BENCHMARK_ROWS`
- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- `WORKED_TRANSITION_REFERENCE`
- carried benchmark-policy names and literature notes
