# Factorized Action Policy Search

This note tracks a reusable action-head investigation rather than a fixed-cost-only benchmark.

The current testbed is the canonical fixed-order-cost lost-sales instance `lit_pois_mu5_l4_p4_k5`,
but the broader question is architectural:

- when does a policy benefit from a factorized action construction?
- what part of that factorization is doing the useful work?
- does the same head transfer back to the plain lost-sales problem?

## Current Question

Can we redesign the scalar dense heads so they no longer bake the environment action cap `Q`
directly into the policy formula, while preserving the good fixed-cost performance?

The original motivation was that some of the strong heads appeared to benefit from factorized
action construction:

- `gate x quantity` for the gated dense head
- `routing weights x leaf quantities` for the soft tree

But the current evidence is sharper than that: the main failure mode is the raw unbounded direct
quantity map, and several `Q`-free heads already recover the good solution class.

## Migration Status

The named/default policies have now been migrated where we had enough evidence:

- `linear_direct_quantity` now resolves to the `Q`-free softplus direct head
- `linear_soft_gated_direct_quantity` is the `Q`-free soft-gated direct head
- `linear_hard_gated_direct_quantity` is the `Q`-free hard-gated direct head
- `soft_tree_depth1_linear_leaf` now resolves to the `Q`-free softplus linear leaf
- `soft_tree_depth2_linear_leaf` now resolves to the `Q`-free softplus linear leaf

The legacy `Q`-scaled variants are still available explicitly:

- `linear_sigmoid_direct_quantity`
- `linear_gated_sigmoid_direct_quantity`
- `soft_tree_depth1_sigmoid_linear_leaf`
- `soft_tree_depth2_sigmoid_linear_leaf`

This migration is intentionally narrow:

- categorical and ordinal heads remain unchanged because they are structurally `Q`-dependent
- generic generated soft-tree `linear_leaf` names were globally repointed to the `Q`-free leaf
- the old sigmoid-to-span leaf map remains available explicitly as `sigmoid_linear_leaf`

## Current Fixed-Cost Evidence

The heads currently tested on the canonical fixed-cost instance are:

- categorical quantity:
  - `a = argmax_k z_k`
- sigmoid direct quantity:
  - `a = round(sigmoid(q) * Q)`
- direct quantity:
  - head output: `\tilde a = round(softplus(q))`
  - env projection: `a = clip(\tilde a, 0, Q)`
- unbounded direct quantity:
  - head output: `\tilde a = round(q)`
  - env projection: `a = clip(\tilde a, 0, Q)`
- gated sigmoid direct quantity:
  - `a = round(sigmoid(g) * sigmoid(q) * Q)`
- gated direct quantity:
  - head output: `\tilde a = round(sigmoid(g) * softplus(q))`
  - env projection: `a = clip(\tilde a, 0, Q)`
- hard-gated direct quantity:
  - head output: `\tilde a = 0` if `sigmoid(g) < 0.5`
  - otherwise `\tilde a = round(softplus(q))`
  - env projection: `a = clip(\tilde a, 0, Q)`
- gated ordinal quantity:
  - `a = round(sigmoid(g) * \sum_{k=1}^{Q} sigmoid(o_k))`
- hard-gated ordinal quantity:
  - `a = 0` if `sigmoid(g) < 0.5`
  - otherwise `a = round(\sum_{k=1}^{Q} sigmoid(o_k))`
- linear soft tree leaf:
  - leaf head output: `\tilde q_\ell(s) = \alpha_\ell^\top s + \beta_\ell`
  - leaf quantity map: `q_\ell(s) = softplus(\tilde q_\ell(s))`
  - tree mixture: `\tilde a = round(\sum_\ell \pi_\ell(s) q_\ell(s))`
  - env projection: `a = clip(\tilde a, 0, Q)`

Here `Q` is the environment action cap `max_order_size`. The important distinction is whether `Q`
appears inside the head parameterization itself, or only in the final projection back into the
environment action range. For the `direct`, `gated_direct`, and `two_stage_direct` heads, `Q` is
not part of the head formula itself. It only appears because the current benchmark problem still
uses the bounded environment action space `0..Q`.

Seed-42 canonical benchmark on `lit_pois_mu5_l4_p4_k5`:

- best heuristic: `9.18235`
- `linear_categorical_quantity`: `9.87591`
- `linear_sigmoid_direct_quantity`: `8.77331`
- `linear_direct_quantity`: `8.77164`
- `linear_unbounded_direct_quantity`: `9.75131`
- `linear_gated_sigmoid_direct_quantity`: `8.77993`
- `linear_soft_gated_direct_quantity`: `8.76964`
- `linear_hard_gated_direct_quantity`: `8.77462`
- `linear_soft_gated_ordinal_quantity`: `8.77502`
- `soft_tree_depth1_sigmoid_linear_leaf`: `8.77345`
- `soft_tree_depth2_sigmoid_linear_leaf`: `8.77725`
- `soft_tree_depth1_linear_leaf`: `8.78111`
- `soft_tree_depth2_linear_leaf`: `8.77689`

Direct `Q`-dependent vs `Q`-free replacements:

- `linear_sigmoid_direct_quantity` (`sigmoid(q) * Q`): `8.77331`
- `linear_direct_quantity` (`softplus(q)` + env projection): `8.77164`
- delta: `-0.00167` in favor of the `Q`-free head

- `linear_gated_sigmoid_direct_quantity` (`sigmoid(g) * sigmoid(q) * Q`): `8.77993`
- `linear_soft_gated_direct_quantity` (`sigmoid(g) * softplus(q)` + env projection): `8.76964`
- delta: `-0.01029` in favor of the `Q`-free head

- `linear_soft_gated_ordinal_quantity` (`Q`-dimensional ordinal head): `8.77502`
- `linear_hard_gated_direct_quantity` (hard zero gate + `softplus(q)` + env projection): `8.77462`
- delta: `-0.00040` in favor of the `Q`-free scalar head

Tree `Q`-dependent vs `Q`-free leaf map:

- `soft_tree_depth1_sigmoid_linear_leaf` (`sigmoid(raw) * span` in each leaf): `8.77345`
- `soft_tree_depth1_linear_leaf` (`softplus(raw)` in each leaf + env projection): `8.78111`
- delta: `+0.00765`, effectively tied

- `soft_tree_depth2_sigmoid_linear_leaf` (`sigmoid(raw) * span` in each leaf): `8.77725`
- `soft_tree_depth2_linear_leaf` (`softplus(raw)` in each leaf + env projection): `8.77689`
- delta: `-0.00035` in favor of the `Q`-free leaf

Observed behavior from short diagnostic rollouts:

- `linear_direct_quantity` produces a strong no-order regime and then large positive orders
- `linear_sigmoid_direct_quantity` is the old Q-scaled baseline
- `linear_unbounded_direct_quantity` mostly orders `4` or `5` and rarely orders `0`
- `linear_soft_gated_direct_quantity` produces a strong no-order regime and then larger positive orders
- `linear_gated_sigmoid_direct_quantity` is the old Q-scaled gated baseline
- `linear_hard_gated_direct_quantity` also produces a strong no-order regime with a separate positive branch
- `linear_soft_gated_ordinal_quantity` produces a strong no-order regime
- soft trees also realize a strong no-order regime, but via conditional leaf routing instead of an
  explicit gate

So the direct head is not the problem by itself. The bad result came specifically from the
unbounded raw-quantity parameterization.

## Current Migration Read

For the scalar dense heads, the current evidence is already strong enough to recommend a `Q`-free
family by default:

- use `direct_quantity` as the default `Q`-free scalar head
- use `soft_gated_direct_quantity` as the default `Q`-free gated scalar head
- keep `hard_gated_direct_quantity` as the clean hard-gated `Q`-free alternative
- keep `unbounded_direct_quantity` only as a negative ablation

What we should **not** claim yet:

- that every policy family can become `Q`-free
- that `categorical_quantity` or `soft_gated_ordinal_quantity` have been fully superseded in general

Those families are structurally `Q`-dependent because `Q` determines either:

- the output dimensionality, or
- the ordinal action construction itself

So the immediate migration target is the scalar head family, not every head family at once.

For the tree family, the first `Q`-free replacement now also looks viable on the canonical fixed-
cost instance:

- replace sigmoid-to-span scaled linear leaves with `softplus` linear leaves
- keep the final bounded env projection unchanged
- treat the old bounded tree leaf as a baseline until we test more than one instance

## Bounded vs Unbounded Clarification

The current successful `soft_gated_direct_quantity` head is not raw-unbounded:

- `gate = sigmoid(g)`
- `quantity = softplus(q)`
- `\tilde a = round(gate * quantity)`
- `action = clip(\tilde a, 0, Q)`

So the current evidence does **not** say that unbounded quantity outputs help. The stronger claim is:

- raw unbounded quantity is harmful here
- bounded direct quantity already works very well
- direct quantity without explicit `Q` in the head also works very well
- multiple factorized bounded constructions also work very well
- a soft-gated direct branch `round(sigmoid(g) * softplus(q))` also works very well

There is also an important implementation detail in the current benchmark pipeline:

- `unbounded_direct_quantity` is only unbounded at the raw-logit level
- the environment still clips final actions into the existing `0..Q` action space

So the real comparison is between:

- bounded quantity parameterization in the head
- raw/unbounded quantity parameterization before projection

It is **not** a comparison between a bounded-action problem and a truly unbounded-action problem.

## Current Interpretation

What we know:

- one-head bounded direct quantity is already enough on this instance
- one-head `Q`-free direct quantity is also enough on this instance
- one-head categorical quantity is not enough
- raw unbounded direct quantity is not enough
- `gate x direct quantity` without explicit `* Q` also recovers the same gain
- a hard gate that only decides `0` vs positive order also recovers the same gain
- `Q`-free linear tree leaves also recover the same gain

What we do not know yet:

- whether factorization adds anything systematic once the scalar head is already `Q`-free
- whether the ordinal quantity representation matters beyond the scalar `Q`-free heads
- how much extra value the richer tree routing adds beyond a simple gate or scalar positive head
- whether the same factorization also helps on the plain lost-sales problem

## Next Ablation

The next clean comparisons are now:

- document the scalar `Q`-free family as the preferred design for fixed-cost dense policies
- optionally verify the same replacement on NN if we need backbone-robustness evidence
- transfer `direct_quantity` and `soft_gated_direct_quantity` back to plain lost sales

The fixed-cost linear result is now clear enough to state:

- explicit `Q` inside the scalar head is not necessary
- raw unbounded direct quantity is the bad case
- multiple `Q`-free scalar heads already land in the good solution class
- the same is now true for the tested linear tree leaves on this canonical instance

## Emerging Structural Hypothesis

The current linear results suggest a more fundamental pattern than "just gating":

- good heads make exact zero actions easy to represent
- good heads also make sizable positive replenishment actions easy to represent
- bad heads fall into a middling-order regime instead

In particular, the following all work well on the canonical fixed-cost instance:

- `linear_direct_quantity`
- `linear_soft_gated_direct_quantity`
- `linear_hard_gated_direct_quantity`
- `linear_soft_gated_ordinal_quantity`
- soft trees with linear leaves

The common property is not one specific decoder formula. The common property is that these heads
make thresholded replenishment structure easy to express, while avoiding the raw unbounded direct
parameterization that tended to settle on moderate positive orders too often.

One especially useful signal is that `linear_hard_gated_direct_quantity` works without baking `Q`
directly into the positive branch:

- if `sigmoid(g) < 0.5`, action `= 0`
- else action `= round(softplus(q))`
- then the bounded environment projects into `0..Q`

So explicit `* Q` scaling inside the head is not necessary for strong performance on this instance.

Interpretation:

- if the NN version shows the same pattern, then the bound effect is robust across backbones
- if `soft_gated_direct_quantity` still helps on plain lost sales, then the factorized-action result
  transfers beyond the fixed-cost benchmark
- if `soft_gated_ordinal_quantity` still beats `soft_gated_direct_quantity`, then ordinal structure adds a
  smaller but real improvement beyond the bound and the factorization
