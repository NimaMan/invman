# Paper subsection draft — Ameliorating inventory (purchase-only price-reactive policy)

Draft text for the paper. Do NOT paste verbatim without an editorial pass; it is written
honestly and conservatively to match the actual full-budget numbers. A BibTeX entry for
Pahr & Grunow (2025) is included inline at the bottom (the orchestrator must not edit
`paper/references.bib` from this run).

---

## X.Y Ameliorating inventory: a price-reactive purchase policy under a perfect-information bound

### Environment and fidelity

We study the ameliorating-inventory control problem of \citet{pahrGrunow2025}, in which a
product *improves* with age (e.g. spirits, port wine): inventory is purchased young, ages
through discrete age classes subject to evaporation and stochastic decay, and is issued to
products defined by target ages. The objective is long-run average profit. We use the
faithful average-profit formulation, including the per-period blending LP that solves
issuance and derives production, exactly as in the companion model.

We anchor the environment to the published perfect-information upper bounds on two reference
instances: the companion default \emph{spirits\_0001} (10 age classes, 3 products, target
ages $\{2,4,6\}$, no blending) and the industry \emph{port\_wine} case study (25 age classes,
2 products, target ages $\{9,19\}$, blending enabled). Our implementation \emph{re-solves}
the steady-state perfect-information LP and reproduces both published upper bounds to within
$10^{-3}$ ($1991.9344$ for spirits\_0001 and $2444.8011$ for port\_wine), an executing
reproduction rather than a stored constant. This fixes the environment's fidelity before any
learning result is reported.

### Policy and action geometry

In the faithful dynamics the only free control is the scalar per-period \emph{purchase}
volume $a^P \in [0, \mathrm{maxInventory}]$; issuance is determined by the environment's
per-period blending LP and production is derived from it. We therefore learn a single
purchase head over the price-augmented state $[\text{price}, \text{inventory}_{0..A}]$, using
a depth-1 oblique soft decision tree with a \emph{linear} leaf so the head can express a
\emph{price-reactive} order-up-to purchase (buy more when the realised, truncated-Normal
purchase price is low). The policy is optimized with CMA-ES warm-started at an order-up-to
purchase $a^P = \mathrm{softplus}(S - \sum \text{inventory})$, so generation 0 reproduces a
simple order-up-to heuristic and the optimizer refines a price-reactive purchase.

### Results

We report long-run average profit under paired common random numbers on a held-out
evaluation block (12{,}000 periods, 24 paired seeds, distinct from training). The natural
in-environment baseline is the best tuned order-up-to purchase (the level $S$ maximizing
profit on the same evaluation block). Because the perfect-information LP value is an
\emph{upper bound} under hindsight and full LP issuance, we report the gap to it and never
treat it as an achievable target.

\begin{table}[t]
\centering
\caption{Ameliorating inventory, full-budget evaluation (12{,}000-period horizon, 24 paired
CRN seeds). The learned price-reactive purchase policy robustly beats the best tuned
order-up-to purchase on both instances; the gap to the perfect-information LP bound is
structural (single purchase action vs full 3-part LP issuance) and is not comparable to the
paper's $\sim$3.5\% DRL gap.}
\label{tab:ameliorating}
\begin{tabular}{lcccc}
\toprule
Instance & Learned profit ($\pm$ SEM) & Best order-up-to & Gain over heuristic & Gap to LP bound \\
\midrule
spirits\_0001 & $115.07 \pm 0.44$ & $20.91$ & $+94.16$ ($+450\%$) & $94.2\%$ \\
port\_wine    & $505.78 \pm 0.59$ & $133.78$ & $+372.00$ ($+278\%$) & $79.3\%$ \\
\bottomrule
\end{tabular}
\end{table}

On both instances the learned price-reactive purchase policy beats the best tuned
order-up-to heuristic by a wide margin: $+94.16$ profit ($+450\%$ over the heuristic) on
spirits\_0001 and $+372.00$ ($+278\%$) on port\_wine. The paired-CRN standard error of the
difference is $0.66$ and $0.61$ respectively, so the advantage is roughly two orders of
magnitude larger than its sampling error and the "beat" is not an artefact of evaluation
noise. The mechanism is the price reactivity: the linear leaf buys more when the realised
purchase price is low, which a fixed order-up-to level cannot exploit.

The gap to the perfect-information LP bound remains large ($94.2\%$ on spirits\_0001,
$79.3\%$ on port\_wine) and we report it honestly. This gap is \emph{structural}, not an
optimization failure: the bound assumes perfect information and the full three-part LP
decision (purchase, production targets, and per-age issuance solved jointly with hindsight),
whereas our policy controls only the scalar purchase volume while the environment charges the
full purchase price ($\sim$200/unit) every period. A feasible single-purchase policy on the
stochastic environment therefore sits well below the hindsight bound by construction. For
this reason our gap is \emph{not} comparable to the $\sim$3.5\% gap reported by
\citet{pahrGrunow2025} for their deep-RL agent, which acts in the full three-part action
space including production targets. The tighter gap on port\_wine ($79.3\%$ vs $94.2\%$) is
explained by blending: issuing across the two target age classes lets each purchased unit
convert to more sold product, so the policy captures more of the bound even with a single
purchase lever.

In summary, on the faithful ameliorating-inventory environment the learned price-reactive
purchase policy is a robust, paired-CRN improvement over the best order-up-to heuristic, the
environment reproduces the published perfect-information bounds exactly, and the residual gap
to those bounds is attributable to the deliberately narrow (purchase-only) action geometry.
Widening to the full three-part action is the natural next step toward the paper's deep-RL
gap and is left to future work.

---

## BibTeX (inline — copy into references.bib outside this run; do not edit references.bib here)

```bibtex
@article{pahrGrunow2025,
  author  = {Pahr, Maximilian and Grunow, Martin},
  title   = {The Value of Blending---Managing Ameliorating Inventory Using Deep Reinforcement Learning},
  journal = {Production and Operations Management},
  year    = {2025},
  volume  = {35},
  number  = {5},
  doi     = {10.1177/10591478251387795},
  url     = {https://journals.sagepub.com/doi/10.1177/10591478251387795},
  note    = {Companion code and per-instance perfect-information upper bounds:
             https://github.com/amelioratinginventory/ameliorating_inventory}
}
```

Citation key used above: `\citet{pahrGrunow2025}`.

### Notes for the editor
- Numbers come from `--budget full` runs recorded in
  `outputs/autoresearch/ameliorating_inventory_average_profit_autoresearch/{spirits_0001,port_wine}_d1_oblique_full.json`
  and tabulated in `scripts/ameliorating_inventory/RESULTS_FULL_BUDGET/README.md`.
- Env fidelity claim is backed by executing tests in
  `src/problems/ameliorating_inventory/tests/verification.rs`.
- The "$+450\%$ / $+278\%$" figures are the gain expressed as a fraction of the heuristic
  baseline; "$+94.16$ / $+372.00$" are the absolute profit gains; the LP-bound gap is the
  fraction of the upper bound left uncaptured. Keep all three framings distinct in the text —
  do not relabel an absolute difference as a percentage.
- Do NOT claim to "beat" or "match" the LP bound anywhere; it is an upper bound only.
```
