# Literature

Current literature anchors for `random_yield_inventory` (see `literature/references.rs` for the
machine-readable source of truth):

- **Yan et al. (2026)**, *Heuristics and deep reinforcement learning for the inventory problem with an
  all-or-nothing yield pattern and non-zero leadtimes*, Computers & Operations Research 186, 107305,
  https://doi.org/10.1016/j.cor.2025.107305 — the **exact structural model match**: single-item,
  periodic-review, all-or-nothing yield, non-zero lead time, backlog, expected total discounted cost.
  Verified (2026-05) via the abstract / institutional record. Paywalled full text; **no public
  per-instance benchmark table** was recoverable, so it cannot anchor a repo assertion.
- **Chen et al. (2018)**, *Heuristics and Bounds for an Inventory System with an All-or-Nothing Yield
  Pattern and Lead-times* (IEEE SOLI) — the originating anchor for the **weighted newsvendor heuristic
  (WNH)**: a sample-path rule that weights the order-up-to gap over the yield realizations of the
  pipeline orders. Paywalled; no reusable numbers recovered.
- **Inderfurth & Kiesmüller (2015)**, *Exact and heuristic linear-inflation policies for an inventory
  model with random yield and arbitrary lead times* — the canonical **linear-inflation rule (LIR)**
  `q = F * (S - X)^+` with inflation factor `F = 1/E[yield rate]`. Confirmed from the open
  working-paper PDF: it is an **infinite-horizon average-cost** model with **per-unit binomial** yield
  (`Y(Q) ~ Binomial(Q, p)`) or **stochastically proportional** yield (`Y(Q) = Z * Q`), NOT this repo's
  finite-horizon all-or-nothing batch yield. It publishes numbers, but only for that different model.

## Why not the same yield model (key root cause)

Inderfurth's **binomial** yield ships `Binomial(Q, p)` units (each unit independently good); the repo's
**all-or-nothing** yield ships the whole batch `Q` with probability `p` or `0` with probability `1-p`.
These coincide only in expected rate (`E = pQ`), so Inderfurth's published average-cost numbers are not
comparable to the repo's discounted all-or-nothing instances. `references.rs` correctly tags the
Inderfurth families `partial_match_general_random_yield` / `related_model_aggregate_only`. The LIR's
inflation factor `1/p` does carry over, because the expected yield rate of an all-or-nothing batch is
also `p`.

## Status: not literature-verified

- The env transition + cost faithfully match the **Yan 2026 model structure**.
- The exact DP is **implementation-correct** (independently reproduced: optimal cost
  `40.0598976099`, first action `4` on `VERIFICATION_PROBLEM_INSTANCE`).
- There is **no public benchmark number** to assert against, so the package stays
  `literature_verified = false`. This is the honest blocker, not a code defect.
- Open formula question: the published WNH may inflate the weighted gap by `1/p`; the repo WNH does
  not. The paywalled formula could not be recovered, so the WNH was left unchanged (see
  `verification/README.md` and the root README).
