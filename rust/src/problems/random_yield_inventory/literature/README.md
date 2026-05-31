# Literature

Current literature anchors for `random_yield_inventory` (see `literature/references.rs` for the
machine-readable source of truth):

All three citations were independently verified against authoritative sources during the 2026-05
audit (Crossref, DBLP, RePEc, and the publisher / open working-paper PDFs). Author lists, year,
venue, volume, pages, and DOIs below are confirmed correct.

- **Yan, Y., Chen, F. (Youhua), Fu, Z. & Bi, W. (2026)**, *Heuristics and deep reinforcement learning
  for the inventory problem with an all-or-nothing yield pattern and non-zero leadtimes*, Computers &
  Operations Research **186**, 107305, https://doi.org/10.1016/j.cor.2025.107305 — the **exact
  structural model match**: single-item, periodic-review, all-or-nothing yield, non-zero lead time,
  backlog, expected total discounted cost. Metadata confirmed (2026-05) via Crossref (full author list
  Yuting Yan, Frank (Youhua) Chen, Zhe Fu, Wenjie Bi; vol. 186, art. 107305, 2026). Full text is
  **paywalled**; **no public per-instance benchmark table** was recoverable, so it **cannot anchor a
  repo assertion**.
- **Chen, F. Y., Hu, F., Yano, C. A. & Yuan, Q. (2018)**, *Heuristics and Bounds for an Inventory
  System with an All-or-Nothing Yield Pattern and Lead-times*, 2018 IEEE International Conference on
  Service Operations and Logistics, and Informatics (SOLI), pp. 180–184,
  https://doi.org/10.1109/SOLI.2018.8476751 — the originating anchor for the **weighted newsvendor
  heuristic (WNH)**: a sample-path rule that weights the order-up-to gap over the yield realizations of
  the pipeline orders. Metadata confirmed (2026-05) via DBLP and IEEE Xplore (doc 8476751). Paywalled;
  **no reusable benchmark numbers recovered**.
- **Inderfurth, K. & Kiesmüller, G. P. (2015)**, *Exact and heuristic linear-inflation policies for an
  inventory model with random yield and arbitrary lead times*, European Journal of Operational
  Research **245**(1), 109–120, https://doi.org/10.1016/j.ejor.2015.03.006 — the canonical
  **linear-inflation rule (LIR)** `q = F * (S - X)^+` with inflation factor `F = 1/E[yield rate]`
  (`F = 1/p` for binomial, `F = 1/μ_Z` for proportional). Confirmed (2026-05) directly from the open
  OVGU working-paper PDF (FEMM No. 7/2013, identical content; metadata cross-checked at the EJOR DOI):
  it is an **infinite-horizon average-cost** model (objective = average holding + backorder cost per
  period) with **per-unit binomial** yield (`Y(Q) ~ Binomial(Q, p)`) or **stochastically proportional**
  yield (`Y(Q) = Z * Q`), NOT this repo's finite-horizon all-or-nothing batch yield. Its published
  numbers (Tables 2–5: μ_D=20, h=1, demand CV {0.1,0.2,0.3,0.5,0.75}, critical ratios
  {0.85,0.90,0.95,0.97,0.99,0.995}, binomial p {0.5,0.7,0.9}, proportional (μ_Z,ρ_Z) pairs, lead times
  {2,5,10}) all describe that **different model**, so they are recorded as related-model context only.

## Why not the same yield model (key root cause)

Inderfurth's **binomial** yield ships `Binomial(Q, p)` units (each unit independently good); the repo's
**all-or-nothing** yield ships the whole batch `Q` with probability `p` or `0` with probability `1-p`.
These coincide only in expected rate (`E = pQ`), so Inderfurth's published average-cost numbers are not
comparable to the repo's discounted all-or-nothing instances. `references.rs` correctly tags the
Inderfurth families `partial_match_general_random_yield` / `related_model_aggregate_only`. The LIR's
inflation factor `1/p` does carry over, because the expected yield rate of an all-or-nothing batch is
also `p`.

## Status: SELF-CONSISTENT-ONLY (not literature-verified)

Per-anchor verdict (audited 2026-05):

| Anchor | Citation metadata | Model fidelity to repo env | Published numbers | Verdict for repo |
| --- | --- | --- | --- | --- |
| Yan et al. (2026) | verified correct (Crossref) | exact structural match (all-or-nothing, +LT, discounted, backlog) | none public (paywalled) | **faithful model, no public anchor** |
| Chen et al. (2018) | verified correct (DBLP/IEEE) | same all-or-nothing setting; WNH source | none public (paywalled) | **faithful model, no public anchor** |
| Inderfurth & Kiesmüller (2015) | verified correct (Crossref/RePEc/PDF) | **different yield model** (binomial / proportional, infinite-horizon avg cost) | published (Tables 2–5), **but for the other model** | **related-model context only / table-not-applicable** |

Net status: **self-consistent-only**.

- The env transition + cost faithfully match the **Yan 2026 model structure** (structural match, no
  reproduced literature number).
- The exact DP is **implementation-correct**: independently reproduced from scratch in a separate
  Python DP of the same MDP — optimal discounted cost `40.0598976099`, first action `4` on
  `VERIFICATION_PROBLEM_INSTANCE`. This is a **repo-native self-consistency** check against the repo's
  own exact solver, NOT a literature number.
- There is **no public benchmark number** for the all-or-nothing finite-horizon discounted model to
  assert against (Yan 2026 / Chen 2018 paywalled; Inderfurth 2015 publishes numbers only for the
  different binomial/proportional infinite-horizon model). The package therefore correctly stays
  `literature_verified = false` everywhere. This is the honest blocker, not a code defect, and the
  repo does **not** overclaim.
- Open formula question: secondary descriptions (Yan 2026 abstract + the Chen 2018 record, the latter
  corroborated in this audit) state the WNH order is the positive target-minus-inventory-position gap
  multiplied by the reciprocal of the mean yield rate, i.e. inflated by `1/p`; the repo WNH does not
  apply that `1/p` inflation. The exact paywalled formula could not be recovered, so the WNH was left
  unchanged (see `verification/README.md` and the root README).
