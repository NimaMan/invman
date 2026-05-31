# Literature

Current literature anchors for `production_assembly_distribution_network`:

- Pirhooshyaran and Snyder 2021 (paper-shaped network formulation; single-node and serial rows)
- Clark and Scarf 1960 / Snyder and Shen "Fundamentals of Supply Chain Theory" Example 6.1 /
  `stockpyl.ssm_serial` (exact serial multi-echelon optimum; `SERIAL_CLARK_SCARF_REFERENCE`)

Current status (re-investigated 2026-05; references independently re-verified against the
publisher/arXiv sources 2026-05-31, see "Citation verification" below):

ACCURATE verifiability status of THIS env: **faithful-but-no-published-anchor** for the Pirhooshyaran
network env; the carried benchmark tables are **table-only** (stored numbers, verified-correct against
the published tables, but NOT re-derived by this env); the serial optimum is **literature-verified
elsewhere** (in `multi_echelon/serial`, not here). It is NOT "literature-verified" for this env.

- the `env.rs` per-period transition and cost are FAITHFUL to the Pirhooshyaran & Snyder (2021) MDP
  (eq. 1-13 sequence of events; eq. 3 cost), checked equation-by-equation against the paper PDF
- the env is NOT yet anchored to a published benchmark NUMBER reproduced by THIS env, so
  literature_verified = no (recorded as `false` on every reference instance)
- single-node newsvendor rows are reproduced analytically (closed form) and approximately by env
  simulation (L=1 case ~13 vs published 12.71; residual = integer rounding)
- the serial benchmark optima carried here (Pirhooshyaran Tables 2-3) are the TEXTBOOK Clark-Scarf
  optima; their env-faithful, literature-verified home is the `multi_echelon/serial` family. In THAT
  family the repo `exact` solver matches Snyder's public `stockpyl.ssm_serial` (C*=47.6687 for case 3,
  i.e. Example 6.1) within 0.05%; THIS network env does not re-derive it (see root-cause note below)

Why:

- `SINGLE_NODE_BENCHMARK_ROWS` reproduces the paper's Table 1 analytical newsvendor rows exactly
  (closed form)
- `SERIAL_BENCHMARK_ROWS` are the classical periodic-review serial multi-echelon optimum
  (`multi_echelon/serial` problem). The exact solver there reproduces every published serial optimal
  cost within 0.05% relative error. Pirhooshyaran's Table 3 reports that simulating THEIR
  finite-horizon environment with these analytical OULs yields the same cost (case 3: 47.65)
- the Pirhooshyaran network env in THIS family does NOT reproduce those optima when driven with the
  carried OUL levels, but NOT because of a longer effective lead time. CORRECTION: the paper sets
  processing time to zero, and env.rs matches it (effective serial lead time = 2+1+1 = 4, verified);
  holding on in-transit inventory is faithful to eq. 3. The actual cause is a local-vs-echelon
  POLICY/LEVEL-INTERPRETATION mismatch: the carried levels are ECHELON base-stock levels, while
  Pirhooshyaran's pairwise policy (eq. 5) targets the LOCAL raw-material position (excludes finished
  goods), so the levels are the wrong local targets. See `serial_echelon_simulation` and
  `verification/README.md` for the corrected analysis and remaining steps

Use `literature/references.rs` as the source of truth for:

- `SINGLE_NODE_BENCHMARK_ROWS`
- `SERIAL_BENCHMARK_ROWS`
- `SERIAL_CLARK_SCARF_REFERENCE`
- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- carried benchmark-policy names and literature notes

Reference hygiene:

- `references.rs` stores literature rows and problem-instance definitions only
- repo-native worked-transition expected values live in verification fixtures, not in literature references

## References

Serial multi-echelon (Clark-Scarf) exact-theory anchor:

- Clark, A. J., and H. Scarf (1960). "Optimal Policies for a Multi-Echelon Inventory Problem."
  *Management Science* 6(4):475-490. DOI 10.1287/mnsc.6.4.475. Origin of the echelon base-stock
  optimality result and the recursive decomposition this solver implements.
- Federgruen, A., and P. Zipkin (1984). "Computational Issues in an Infinite-Horizon, Multiechelon
  Inventory Model." *Operations Research* 32(4):818-836. DOI 10.1287/opre.32.4.818. Infinite-horizon
  / average-cost stationary echelon base-stock optimality and computation.
- Chen, F., and Y.-S. Zheng (1994). "Lower Bounds for Multi-Echelon Stochastic Inventory Systems."
  *Management Science* 40(11):1426-1443. DOI 10.1287/mnsc.40.11.1426. (Full names: Fangruo Chen,
  Yu-Sheng Zheng.) Echelon cost-function recursion used here; the algorithm implemented by
  `stockpyl.ssm_serial.optimize_base_stock_levels`.
- Snyder, L. V., and Z.-J. M. Shen. *Fundamentals of Supply Chain Theory* (2nd ed., Wiley 2019),
  ISBN 978-1-119-02484-2, DOI 10.1002/9781119584445. Example 6.1 = serial benchmark case 3
  (3 stages, echelon holding [2,2,3], lead times [2,1,1], stockout 37.12, Normal(5,1) demand;
  optimal cost 47.65).
- `stockpyl` (Lawrence V. Snyder), `stockpyl.ssm_serial.optimize_base_stock_levels` — public
  reference implementation (github.com/LarrySnyder/stockpyl, on PyPI). For Example 6.1 it returns
  C*=47.6687 (verified 2026-05). The repo solver `multi_echelon/serial::exact` matches it to machine
  precision on discrete Poisson instances and within 0.05% relative on every serial Normal-demand
  row. https://stockpyl.readthedocs.io/en/latest/api/meio/ssm_serial.html

Paper-shaped network formulation (single-node rows and the carried serial settings):

- Pirhooshyaran, M., and L. V. Snyder (2021). "Simultaneous Decision Making for Stochastic
  Multi-Echelon Inventory Optimization with Deep Neural Networks as Decision Makers."
  arXiv:2006.05608 (v1 Jun 2020, v2 23 Mar 2021), Lehigh University.
  https://arxiv.org/abs/2006.05608. VERIFIED against the paper PDF (2026-05): Table 1 = single-node
  newsvendor rows (h=10, p=30, T=2; `SINGLE_NODE_BENCHMARK_ROWS` = the L=1 analytical column);
  Table 2 = the 10 serial-SCN settings; Table 3 = the OUL/cost comparison
  (`SERIAL_BENCHMARK_ROWS` = Table 2 settings + Table 3 'Analytical' column, case 3 = 47.65). All
  transcribed numbers confirmed exact against the published tables.

## Citation verification (2026-05-31, independent)

Every reference above was re-checked against an authoritative source. All are REAL and the repo
metadata is correct; one URL fix was applied.

- Pirhooshyaran & Snyder 2021 — arXiv:2006.05608 (verified at https://arxiv.org/abs/2006.05608);
  title, authors (Mohammad Pirhooshyaran, Lawrence V. Snyder, Lehigh), and topic match. Table 1 / 2 /
  3 contents and all carried numbers verified against the PDF. CORRECT.
- Clark & Scarf 1960 — *Management Science* 6(4):475-490, DOI 10.1287/mnsc.6.4.475
  (https://doi.org/10.1287/mnsc.6.4.475). CORRECT.
- Federgruen & Zipkin 1984 — *Operations Research* 32(4):818-836, DOI 10.1287/opre.32.4.818
  (https://doi.org/10.1287/opre.32.4.818). CORRECT.
- Chen & Zheng 1994 — *Management Science* 40(11):1426-1443, DOI 10.1287/mnsc.40.11.1426
  (https://doi.org/10.1287/mnsc.40.11.1426). CORRECT.
- Snyder & Shen, *Fundamentals of Supply Chain Theory* 2nd ed. (Wiley 2019), ISBN 978-1-119-02484-2.
  Example 6.1 cost 47.65 corroborated by stockpyl (C*=47.6687). CORRECT.
- stockpyl ssm_serial — FIXED: the deep-link in `references.rs` was
  `.../api/seio/ssm_serial.html` (404); the correct path is `.../api/meio/ssm_serial.html`
  (verified live). Package, author, and `optimize_base_stock_levels` function confirmed.
