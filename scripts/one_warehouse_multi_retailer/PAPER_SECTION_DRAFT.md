# DRAFT — Learned policies on asymmetric / high-variability OWMR instances

> Paper-ready subsection draft for the one-warehouse multi-retailer (OWMR) family.
> Numbers below are filled from the held-out paired-CRN runs in
> `scripts/one_warehouse_multi_retailer/ASYMMETRIC_LEARNED_RESULTS.md`. This file is
> a draft (not committed into `paper/`); it is the source text for the paper editor.

## Problem

We study the one-warehouse multi-retailer system of Kaynov et al. (2024, IJPE 267,
109088): a single warehouse (lead time \(L_w\)) replenishes \(K\) retailers (lead
times \(L_{r,k}\)), with a per-period downstream **allocation** of scarce warehouse
stock and one of three customer regimes (backorder, lost-sales, partial-backorder).
The objective is the 100-period undiscounted expected total cost (linear holding +
backorder/lost-sales penalty), evaluated by simulation under common random numbers.

The symmetric Poisson(3) instances (instances 1/6/7/11) are near-optimally solved by
a tuned echelon base-stock policy: a learned soft-tree only *ties* the tuned
heuristic there (see the companion symmetric result). The interesting question is
the **asymmetric / high-coefficient-of-variation partial-backorder** instances,
where Kaynov's own PPO beats base-stock by ~20%. We test whether a learned policy
with a **per-retailer action geometry** can exploit that structure against the
in-repo tuned gate.

## Instances (Kaynov Table A.3)

All three are partial-backorder, \(L_w=2\), emergency-shipment probability 0.8,
100-period horizon, 1000 replications in the published protocol.

| Instance | \(K\) | \(L_{r}\) | Per-retailer demand | \(h_w\) | \(h_r\) | \(p_r\) | Structure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| instance_12 | 3 | [1,1,1] | N(1,5), N(5,1), Poisson(0.5) | 0.5 | 1 | 9 | heterogeneous (mixed normal+Poisson) |
| instance_13 | 10 | [2]×10 | N(5,14) ×10 | 3 | 3 | 60 | symmetric but very high CV (\(\sigma/\mu=2.8\)) |
| instance_14 | 10 | [2]×10 | N(0,20),N(2,16),N(4,12),N(6,8),N(8,4),N(10,0), Poisson(0.5),Poisson(3),Poisson(9),Poisson(12) | 3 | 3 | 60 | strongly heterogeneous (clipped-normal gradient + Poisson) |

(Demand is rounded and clipped at 0; e.g. N(0,20) realizes a heavily right-skewed
non-negative integer demand with empirical mean ~8.)

## Heuristic gate and published PPO

The **gate** is the strongest in-repo policy: an echelon base-stock policy whose
levels are grid-searched on a search-seed CRN block — a shared warehouse level plus
per-retailer retailer levels (full cartesian for instance_12; symmetric reduction
for instance_13; Kaynov's z0-\(k\) safety-factor candidate set for the heterogeneous
instance_14) — under the better of {proportional, min_shortage} allocation, then
re-scored on a disjoint held-out CRN block. The published **PPO** row (Kaynov Table
A.3, reported as negative reward → cost) is the reference learned comparator (not the
keep/discard gate).

| Instance | Gate (in-repo, held-out) | Published base-stock (min/prop) | Published PPO | PPO vs base-stock |
| --- | ---: | ---: | ---: | ---: |
| instance_12 | 1169.59 (W=39, R=[5,10,1]) | 1406.43 / 1402.38 | 1118.92 | PPO ~20.2% cheaper |
| instance_13 | 91890.25 (symmetric, shared R) | 99882.51 / 101727.47 | 79727.39 | PPO ~21.6% cheaper |
| instance_14 | 50445.20 (W=440, R=[33,30,28,26,27,30,2,10,29,39]) | 52787.41 / 53358.86 | 42835.02 | PPO ~19.7% cheaper |

## Action geometry (the lever)

The symmetric_echelon_targets geometry (control dim 2: one warehouse target + one
**shared** retailer target) cannot express asymmetric per-retailer replenishment and
only ties the gate. We switch to a **per-retailer** geometry:
- **echelon_targets** (control dim \(K+1\)): warehouse target + per-retailer echelon
  base-stock targets — the natural asymmetric generalization of the gate; supports
  both allocation rules. Used for instance_12 and instance_14.
- **direct_orders** (control dim \(K+1\)): raw per-retailer order quantities — most
  expressive, but cannot supply target positions, so it is restricted to proportional
  allocation. Used as the per-retailer ablation on the symmetric-but-high-CV
  instance_13 (whose default geometry is symmetric_echelon_targets).

(`vector_quantity` is the soft-tree's *control mode*, not a policy action mode; the
env binding rejects it as an action mode, so it is not a usable lever.)

## Learned vs gate (held-out, paired CRN)

Soft-tree depth 2, axis-aligned splits, leaf \(\in\) {constant, linear}, CMA-ES
(population 32, 600 generations, train-seed-batch 12 at full budget), 4096 held-out
paths, paired with the gate on identical demand realizations. A win is claimed only
when the paired-difference advantage exceeds its standard error.

| Instance | Geometry | Learned (held-out) ± SEM | Gate ± SEM | Gap % | Paired diff ± SEM | Verdict | Learned vs PPO |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| instance_12 | echelon_targets (linear) | **1154.09 ± 2.12** | 1169.59 ± 2.05 | **+1.33%** | **+15.50 ± 0.97** | **learned wins** | −3.14% |
| instance_13 | symmetric_echelon_targets (linear) | **85974.79 ± 88.29** | 91890.25 ± 99.56 | **+6.44%** | **+5915.47 ± 49.50** | **learned wins** | −7.84% |
| instance_14 | echelon_targets (linear/constant) | 50445.20 ± 61.90 | 50445.20 ± 61.90 | +0.00% | +0.00 ± 0.00 | tie | −17.77% |

## Honest framing

A learned policy with the right action geometry beats the tuned in-repo gate on **two of
the three** instances, and ties on the third:

- **instance_12 (heterogeneous, K=3) — win, +1.33%** (paired +15.50 ± 0.97, ~16 SEM). The
  per-retailer `echelon_targets` soft tree exploits state-dependent deviations from the
  shared base-stock that the gate cannot express.
- **instance_13 (symmetric, K=10, σ/μ=2.8) — win, +6.44%** (paired +5915.47 ± 49.50, ~120
  SEM). Here the lever is *dynamic state-dependence under high demand variance*, not
  per-retailer heterogeneity: the linear-leaf `symmetric_echelon_targets` policy makes the
  order-up-to target a function of the inventory state, closing most of the gap to PPO
  (15.26% → 7.84%). A constant-leaf (static) shared base-stock only ties.
- **instance_14 (strongly heterogeneous, K=10) — tie.** With the generalized per-retailer
  warm-start floor, gen-0 reproduces the gate exactly, but CMA-ES found no improvement in
  600 generations (the trained xbest landed above the gate), so the deployed policy is the
  gate-reproducing anchor. The tie is *search-limited*, not representation-limited.

On all three rows the in-repo gate is already much stronger than Kaynov's published
base-stock and sits between it and the published PPO; PPO remains the strongest learned
policy. The honest claim is therefore: the learned soft tree **beats the strongest in-repo
heuristic gate beyond sampling error on instance_12 and instance_13**, and ties it on the
hardest 10-retailer asymmetric instance — it does **not** beat the published PPO.

### The decisive lever: a per-retailer warm-start floor

The keep/discard floor is the gate-reproducing CMA-ES warm start. Generalizing it from the
2-control `symmetric_echelon_targets` geometry to **any** control dimension — seeding the
leaves so generation 0 emits the gate's full per-retailer target vector [W, r₁,…,r_K] —
makes the floor apply to the richer per-retailer class: the anchor reproduces the gate to
0.0 on identical CRN paths, and CMA-ES then searches outward from the gate. Without this
anchor the richer `echelon_targets` geometry *loses* (instance_12 screening: −12.79%); with
it, instance_12 flips to a +1.33% win. The remaining tie (instance_14) is then a pure search
limit on the 10-dimensional asymmetric target space, not a representation gap — the gate is
inside the learned policy's class, the optimizer just cannot improve on it within budget.
