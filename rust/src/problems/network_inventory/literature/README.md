# Literature

Current literature anchor for `network_inventory`:

- Pirhooshyaran and Snyder 2021

Current status:

- not literature-verified

Why:

- the Rust package now carries a paper-shaped discrete formulation
- `SINGLE_NODE_BENCHMARK_ROWS` reproduces the paper's analytical newsvendor rows exactly
- `SERIAL_BENCHMARK_ROWS` carries the paper's serial tables as literature catalog rows only
- the serial benchmark protocol could not be recovered tightly enough from public sources, so
  those rows are not part of the verification layer

Use `literature/references.rs` as the source of truth for:

- `SINGLE_NODE_BENCHMARK_ROWS`
- `SERIAL_BENCHMARK_ROWS`
- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- `WORKED_TRANSITION_REFERENCE`
- carried benchmark-policy names and literature notes
