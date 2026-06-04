# nonstationary_lot_sizing

Canonical Rust-first home for the nonstationary single-item lot-sizing family
(Dehaybe, Catanzaro & Chevalier 2024, EJOR 314(2):433-445).

## Verification status: literature-verified (published author-repo numbers
## reproduced by the repo solver), with one self-consistent-only fidelity caveat

- Published-number reproduction (LITERATURE-VERIFIED): the eight `references.rs`
  rows are byte-for-byte the author's public testbed CSVs (`HenriDeh/DRL_MMULS`,
  `single-item` branch), and the repo's own solver+simulator reproduces every one of
  them within the stored 35-cost tolerance (±0.17% relative) at 25,000 replications.
  The author CSV values were independently re-confirmed against the GitHub raw files
  in the 2026 literature audit. The anchors are the author's CODE-REPO baseline CSVs
  (their `simple` (s,S) and rolling-DP (s,S) heuristics), not a table printed in the
  EJOR article.
- Model fidelity: the `simple_s_s` heuristic matches the author testbed (s,S) formula
  term-for-term; demand models follow the testbed (CV-Normal for the simple baseline,
  Poisson for the DP baseline).
- Caveat (SELF-CONSISTENT-ONLY): the Section 4.2 worked transition (period cost 130,
  reward -130) is validated only against the repo's own `env.rs::step_state`. The
  attribution of these specific numbers to the paper's printed Section 4.2 could NOT
  be confirmed against the article during the 2026 audit (PDF inaccessible).
- Citation metadata (authors, EJOR 314(2):433-445, 2024, DOI 10.1016/j.ejor.2023.10.007)
  was confirmed exact via IDEAS/RePEc.
- See `literature/README.md` for the full evidence table, audit trail, and source
  pointers, and `verification/README.md` for the executable verifier contract.

## Code

- implementation: `rust/src/problems/nonstationary_lot_sizing/`
- tests: `rust/src/problems/nonstationary_lot_sizing/tests/verification.rs`

## Artifact folders

- `literature/` — paper scope, fidelity argument, reproduced-number table
- `practical/` — checked-in rolling forecast trace, benchmark spec, latest report
- `experiments/` — paper-facing benchmark definition
- `verification/` — human-readable statement of what the verifier asserts

## Canonical anchors

- primary reference instance: `dehaybe2024_lostsales_lt2_b5_k10_constant_10`
- verification instance: `constant_10_rolling_dp_reference`
- practical benchmark dataset: `retail_like_weekly_trace`

## Verification status (HONEST)

`literature_verified = false` for this family.

Per the repo rule, a family is literature-verified only when an in-crate test
re-runs the env/solver and reproduces a number PRINTED IN A PAPER within a
stated tolerance. This family does NOT yet meet that bar:

- The per-instance benchmark rows (mean cost, cost std, shortage rate) are
  reproduced from the AUTHOR'S PUBLIC COMPANION-CODE testbed CSVs
  (`HenriDeh/DRL_MMULS`, branch `single-item`:
  `data/single-item/scarf_testbed_DP_lostsales.csv` and
  `scarf_testbed_simple_lostsales.csv`). That is a reference-implementation
  match, not a value printed in an article table. The testbed grid
  (`product([2,4,8],[5,10],[10,20,30],[true])`) also differs from the article's
  reported experiment grid.
- The Section-4 worked transition (period cost 130 / reward -130) is an INTERNAL
  `step_state` mechanics / self-consistency check. The EJOR full text was not
  accessible to this repo (paywalled; the OA submitted version on the UCLouvain
  DIAL repository was unreachable), so we make NO claim that -130 is printed in
  the article.

Source paper: Dehaybe, Catanzaro & Chevalier (2024), "Deep Reinforcement
Learning for inventory optimization with non-stationary uncertain demand", EJOR
314(2):433-445, DOI 10.1016/j.ejor.2023.10.007.
