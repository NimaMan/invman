# Replenishment Geometry Search

This note tracks a reusable replenishment-policy geometry investigation rather than a
fixed-cost-only benchmark.

The current testbed is the canonical fixed-order-cost lost-sales instance `lit_pois_mu5_l4_p4_k5`.
The most important thing we know now is not "gating works" or "Q-free works". The strongest local
fact is behavioral:

- the good learned controllers behave like thresholded replenishment rules;
- the bad raw unbounded direct head does not.

More concretely, on saved good controllers:

- `linear_direct_quantity` orders zero in `50.0%` of post-warmup periods and, conditional on
  ordering, moves the inventory position into a tight band around `27.55` with std `1.24`;
- `linear_soft_gated_direct_quantity` also orders zero in `50.0%` of post-warmup periods and,
  conditional on ordering, moves the inventory position to about `27.15` with std `1.27`;
- on the visited states of that good `linear_soft_gated_direct_quantity` model, the learned gate is
  saturated at `1.0`, so the final controller behaves almost exactly like a direct positive
  quantity map rather than an actively used gate;
- `soft_tree_depth2_linear_leaf` orders zero in `50.0%` of post-warmup periods and, conditional on
  ordering, moves the inventory position to about `27.59` with std `1.45`;
- the same depth-2 tree routes essentially all positive-order states to one leaf and all zero-order
  states to another leaf;
- `linear_unbounded_direct_quantity` has access to the same action space, including zero, but on
  the same probe it does not learn to use that region at all: it mostly places orders `4` or `5`
  and only restores the post-order inventory position to about `24.13`, which is exactly the weak
  `~9.75` basin.

So the best current interpretation is:

- the winning property is not "has a gate";
- it is not "uses `Q` inside the head";
- it is not even "has two heads";
- it is that the parameterization makes it easy to realize a no-order region together with a
  positive replenishment branch that behaves like a smooth order-up-to / deficit-correction map.

For the current `Q`-free direct head,

- `a = clip(round(softplus(w^\top x + b)), 0, Q_env)`,

the `softplus` nonlinearity is a smooth analog of `max(0, \cdot)`. This makes the architecture a
natural way to encode the rule:

- "order nothing when the state is above a threshold-like surface";
- "otherwise order a positive amount that increases as the state falls below that surface."

That is much closer to the classical geometry of setup-cost inventory control than a raw unbounded
scalar `round(w^\top x + b)`, which in our current search setup keeps converging to small positive
orders and pays the fixed cost too often.

## Related Literature

This interpretation is consistent with several distinct strands of prior work:

- Bijvank, Bhulai, and Huh (2015) show that for lost sales with fixed order cost and positive lead
  times there is no simple optimal replenishment policy, but simple parametric policies such as
  `(s, S)`, `(s, nQ)`, and their modified capped `(s, S)` variant can still be near-optimal.
  Their introduction explicitly frames the problem as one where simple replenishment maps remain
  useful even when the exact optimum has no clean closed form.
- The same paper also recalls the classical fixed-cost story: in periodic review models with fixed
  order cost, the natural benchmark geometry is an `(s, S)`-type policy with a no-order region and
  an order-up-to response once the state falls below a reorder threshold.
- In the lost-sales literature without fixed order cost, Zipkin's structural results are summarized
  by Bijvank et al. as implying that optimal order quantities are monotone decreasing in inventory
  position and more sensitive to recent orders. That is exactly the kind of monotone deficit
  correction that the successful softplus direct head can express.
- Huh et al. (2009) show that order-up-to policies are asymptotically optimal when the lost-sales
  penalty becomes large, and Bijvank et al. note that such policies remain popular because they are
  simple and robust.
- In the mixture-of-experts literature, Jordan and Jacobs (1993) describe hierarchical mixtures as
  trees with gating networks at the internal nodes and linear experts at the leaves, i.e. smoothed
  piecewise generalized linear models. Nowlan and Hinton (1991) emphasize that competing expert
  architectures can uncover useful decompositions and often generalize better than a single global
  network.

Taken together, these papers suggest a plausible architectural story:

- fixed-cost inventory wants a thresholded replenishment geometry;
- local linear rules are a natural fit once that geometry is exposed;
- tree routing or expert decomposition can help if the policy really needs multiple local regimes;
- but if a single smooth positive-part map already captures the main thresholded replenishment
  behavior, then the extra gate may help search without being essential in the final policy.

## Current Working Hypothesis

The live question is therefore no longer "does factorization work?" The sharper question is:

- which policy parameterizations make the thresholded replenishment geometry easy enough for CMA-ES
  to find under realistic budgets?

What we should and should not claim at this point:

- We should claim that raw unbounded direct quantity is a bad optimizer-parameterization pair for
  this problem family under our current CMA-ES setup.
- We should claim that the good heads all make exact zero and large positive replenishment easy to
  represent.
- We should claim that the good direct and tree policies are behaving like smooth parametric
  replenishment rules rather than arbitrary neural controllers.
- We should not claim that an explicit gate is necessary, because the best saved soft-gated direct
  controller saturates the gate to `1.0` on the visited states.
- We should not claim that removing `Q` from the head is always free, because search robustness
  still depends on the training budget and CMA population size.

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
  - status: historical negative ablation only; removed from the active experiment set
- soft-gated sigmoid direct quantity:
  - `a = round(sigmoid(g) * sigmoid(q) * Q)`
- soft-gated direct quantity:
  - head output: `\tilde a = round(sigmoid(g) * softplus(q))`
  - env projection: `a = clip(\tilde a, 0, Q)`
- hard-gated direct quantity:
  - head output: `\tilde a = 0` if `sigmoid(g) < 0.5`
  - otherwise `\tilde a = round(softplus(q))`
  - env projection: `a = clip(\tilde a, 0, Q)`
- soft-gated ordinal quantity:
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
environment action range. For the `direct`, `soft_gated_direct`, and `hard_gated_direct` heads,
`Q` is not part of the head formula itself. It only appears because the current benchmark problem
still uses the bounded environment action space `0..Q`.

### Canonical Single-Instance Table

The table below is the clean single-instance comparison for
`lit_pois_mu5_l4_p4_k5` under the stable seed-42 protocol:

- learned policies trained with `5000` CMA iterations and training horizon `2000`
- evaluation over `10` seeds with horizon `10^6`
- absolute costs only, to avoid mixing cross-run heuristic reevaluations
- the later `2000`-iteration dynamic-horizon rename check is intentionally excluded here

| Group | Policy | Mean cost | Std. dev. | Notes |
| --- | --- | ---: | ---: | --- |
| Heuristic | modified `(s,S,q)` | 9.18235 | 0.00591 | best heuristic from canonical heuristic summary |
| Heuristic | `(s,nQ)` | 9.21099 | 0.00743 | canonical heuristic summary |
| Heuristic | `(s,S)` | 9.36945 | 0.01009 | canonical heuristic summary |
| Q-dependent baseline | `linear_categorical_quantity` | 9.87591 | 0.01036 | flat categorical baseline |
| Historical ablation | `linear_unbounded_direct_quantity` | 9.75131 | 0.00753 | raw scalar before projection; excluded from active experiment set |
| Q-dependent baseline | `linear_gated_sigmoid_direct_quantity` | 8.77993 | 0.00681 | soft gate with `sigmoid(q) * Q` |
| Q-dependent baseline | `linear_soft_gated_ordinal_quantity` | 8.77502 | 0.00698 | ordinal head with `Q` outputs |
| Q-dependent baseline | `linear_sigmoid_direct_quantity` | 8.77331 | 0.00818 | direct head with `sigmoid(q) * Q` |
| Q-dependent baseline | `soft_tree_depth1_sigmoid_linear_leaf` | 8.77345 | 0.00772 | tree with sigmoid-to-span leaves |
| Q-dependent baseline | `soft_tree_depth2_sigmoid_linear_leaf` | 8.77725 | 0.00726 | tree with sigmoid-to-span leaves |
| Q-free candidate | `linear_hard_gated_direct_quantity` | 8.77462 | 0.00717 | hard zero gate plus softplus direct quantity |
| Q-free candidate | `linear_direct_quantity` | 8.77164 | 0.00818 | softplus direct head |
| Q-free candidate | `linear_soft_gated_direct_quantity` | 8.76964 | 0.00681 | soft gate plus softplus direct quantity |
| Q-free candidate | `soft_tree_depth1_linear_leaf` | 8.78111 | 0.00810 | tree with softplus leaves |
| Q-free candidate | `soft_tree_depth2_linear_leaf` | 8.77689 | 0.00729 | tree with softplus leaves |

### Summary So Far

- The clearly bad cases are `linear_categorical_quantity` and `linear_unbounded_direct_quantity`.
- Raw unbounded direct quantity is the actual failure mode; it settles into mediocre positive orders and does not recover the good policy class.
- Because it never reaches the best region on this instance, `linear_unbounded_direct_quantity` is
  documented here only as a failed ablation and is not part of the active experiment policy set.
- Explicit `Q` inside the head is not necessary for good performance on this canonical instance.
- The Q-free linear heads all land in essentially the same good regime as the old Q-dependent direct and tree baselines.
- The strongest Q-free linear result so far is `linear_soft_gated_direct_quantity` at `8.76964`.
- The Q-free tree replacement is also viable: depth-2 softplus leaves are effectively tied with the old sigmoid-to-span depth-2 tree.
- So the current takeaway is: remove `Q` from the scalar direct/leaf parameterization where possible, but do not conclude yet that every `Q`-dependent family is obsolete.

### Canonical Single-Instance Table Under The 2k / 1000 -> 3000 Schedule

The table below is the targeted rerun of the direct/tree family on the same canonical instance
under the shorter dynamic-horizon protocol:

- `2000` CMA iterations
- linear training horizon schedule `1000 -> 3000`
- evaluation over `10` seeds with horizon `10^6`
- rows normalized to the current canonical policy names; the underlying completed run predates the
  final naming cleanup

| Group | Policy | Mean cost | Std. dev. | Delta vs stable 5k | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Heuristic | modified `(s,S,q)` | 9.18235 | 0.00591 | 0.00000 | same heuristic baseline |
| Heuristic | `(s,nQ)` | 9.21099 | 0.00743 | 0.00000 | same heuristic baseline |
| Heuristic | `(s,S)` | 9.36945 | 0.01009 | 0.00000 | same heuristic baseline |
| Q-dependent baseline | `linear_sigmoid_direct_quantity` | 8.77835 | 0.00786 | +0.00504 | still strong under the shorter schedule |
| Q-dependent baseline | `linear_gated_sigmoid_direct_quantity` | 8.77415 | 0.00746 | -0.00578 | still strong under the shorter schedule |
| Q-dependent baseline | `soft_tree_depth1_sigmoid_linear_leaf` | 9.74866 | 0.00666 | +0.97521 | collapses under the shorter schedule |
| Q-dependent baseline | `soft_tree_depth2_sigmoid_linear_leaf` | 8.77689 | 0.00759 | -0.00036 | remains robust |
| Q-free candidate | `linear_direct_quantity` | 9.74297 | 0.00709 | +0.97133 | collapses under the shorter schedule |
| Q-free candidate | `linear_soft_gated_direct_quantity` | 8.77804 | 0.00668 | +0.00840 | remains competitive |
| Q-free candidate | `linear_hard_gated_direct_quantity` | 9.74736 | 0.00670 | +0.97274 | collapses under the shorter schedule |
| Q-free candidate | `soft_tree_depth1_linear_leaf` | 9.76153 | 0.00643 | +0.98042 | collapses under the shorter schedule |
| Q-free candidate | `soft_tree_depth2_linear_leaf` | 9.74911 | 0.00687 | +0.97222 | collapses under the shorter schedule |

### What The 2k Schedule Changes

- The earlier `5k` conclusion "the good solution class is not tied to putting `Q` inside the
  head" is still true at the level of representational capacity, but it is no longer enough for a
  default recommendation.
- Under the shorter `2k` dynamic-horizon schedule, most of the `Q`-free heads that looked good
  under `5k` lose the good solution class entirely and fall back to the weak `~9.74` basin.
- The strongest budget-robust policies in this comparison are now:
  - `linear_sigmoid_direct_quantity`
  - `linear_gated_sigmoid_direct_quantity`
  - `linear_soft_gated_direct_quantity`
  - `soft_tree_depth2_sigmoid_linear_leaf`
- So the latest evidence points to search geometry, not just expressivity. Several heads can
  represent the strong policy class, but only some parameterizations let CMA-ES find it reliably
  when the budget is tighter.
- The most interesting survivor in this specific dynamic-horizon comparison is
  `linear_soft_gated_direct_quantity`: it stays strong without putting `Q` directly inside the
  quantity branch. Combined with the later `pop50` recheck, this suggests that the extra gate may
  help the search path even if it is not essential in the final good controller.

### Short-Budget A/B On Horizon Schedule And Population Size

To decide whether the fixed-cost experiments should keep the dynamic horizon schedule, we reran
the canonical instance `lit_pois_mu5_l4_p4_k5` with a single strong policy,
`linear_soft_gated_direct_quantity`, under the short-budget protocol:

- `2000` CMA iterations
- seed `42`
- evaluation over `10` seeds with horizon `10^6`

First, with population `32`, the horizon schedule comparison was:

| Training schedule | Mean cost | Std. dev. | Result |
| --- | ---: | ---: | --- |
| dynamic `1000 -> 3000`, `pop32` | 9.75483 | 0.00687 | no improvement |
| constant `2000`, `pop32` | 9.74971 | 0.00737 | slightly better |

So the `1000 -> 3000` schedule did not help even on a policy we already know can perform well.

Second, with the better constant horizon `2000`, the population-size comparison was:

| Training setup | Mean cost | Std. dev. | Result |
| --- | ---: | ---: | --- |
| constant `2000`, `pop32` | 9.74971 | 0.00737 | weak basin |
| constant `2000`, `pop50` | 8.76964 | 0.00681 | good basin |

So the critical short-budget change was not the horizon schedule. It was the reduction from
population `50` to `32`.

The active fixed-cost experiment protocol therefore uses:

- `training_episodes = 2000`
- `es_population = 50`
- constant training horizon `2000`
- seed `42`
- evaluation over `10` seeds with horizon `10^6`

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

Observed behavior from longer diagnostic rollouts of saved good controllers:

- `linear_direct_quantity` orders zero in `50.0%` of post-warmup periods and, conditional on
  ordering, moves the inventory position to `27.55 ± 1.24`;
- `linear_soft_gated_direct_quantity` also orders zero in `50.0%` of post-warmup periods and,
  conditional on ordering, moves the inventory position to `27.15 ± 1.27`;
- on the probed states of that good `linear_soft_gated_direct_quantity` controller, the learned
  gate is saturated at `1.0`, so the deployed policy is effectively a direct positive-part map;
- `linear_unbounded_direct_quantity` still has zero available in the environment action space, but
  on the saved run it orders zero in `0.0%` of post-warmup periods, mostly places orders `4` or
  `5`, and only restores the inventory position to `24.13 ± 1.47`;
- `soft_tree_depth2_linear_leaf` orders zero in `50.0%` of post-warmup periods, moves the
  inventory position to `27.59 ± 1.45` when it orders, and routes zero-order states and
  positive-order states to different leaves almost perfectly.

So the direct head is not the problem by itself. The bad result is more plausibly a search-space
issue tied to the raw unbounded quantity parameterization.

## Current Migration Read

The migration is no longer settled. The `5k` evidence was strong enough to justify trying a
`Q`-free default, but the `2k` schedule shows that this is not yet robust enough to treat as a
finished conclusion.

What is still safe:

- keep `unbounded_direct_quantity` only as a historical negative ablation in this note, not as an
  active named policy or an experiment candidate;
- keep the Q-free scalar family in the codebase, because it clearly can represent the strong policy
  class;
- keep the old sigmoid/Q-scaled scalar family available explicitly, because it is currently more
  budget-robust.

What is not safe to recommend yet:

- switching the scalar dense defaults entirely to the `Q`-free family;
- replacing all sigmoid/Q-scaled tree leaves with Q-free leaves by default.

What we should **not** claim yet:

- that every policy family can become `Q`-free
- that `categorical_quantity` or `soft_gated_ordinal_quantity` have been fully superseded in general
- that the `Q`-free direct/tree family is as easy for CMA-ES to optimize under shorter budgets

Those families are structurally `Q`-dependent because `Q` determines either:

- the output dimensionality, or
- the ordinal action construction itself

So the immediate task is no longer migration. It is to understand which parameterizations are
search-robust enough to justify becoming defaults.

## Bounded vs Unbounded Clarification

The current successful `soft_gated_direct_quantity` head is not raw-unbounded:

- `gate = sigmoid(g)`
- `quantity = softplus(q)`
- `\tilde a = round(gate * quantity)`
- `action = clip(\tilde a, 0, Q)`

So the current evidence does **not** say that unbounded quantity outputs help. The stronger claim is:

- raw unbounded quantity is harmful here under the current CMA-ES search setup
- bounded direct quantity already works very well
- direct quantity without explicit `Q` in the head also works very well
- multiple threshold-compatible constructions also work very well
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

- under the longer `5k` protocol, one-head bounded direct quantity is enough on this instance
- under the longer `5k` protocol, one-head `Q`-free direct quantity is also enough on this
  instance
- one-head categorical quantity is not enough
- raw unbounded direct quantity is not enough in practice here, even though zero remains in the
  action space
- under the shorter-budget ablations, CMA population size matters materially: `pop50` restores
  `linear_soft_gated_direct_quantity` to the good `~8.77` basin while `pop32` does not
- the good saved linear controllers all behave like smooth thresholded replenishment maps
- the good saved depth-2 tree behaves like an explicit two-regime controller with one leaf for the
  no-order region and one leaf for the positive-order region

What we do not know yet:

- whether factorization adds anything systematic once the scalar head is already `Q`-free under
  the longer protocol, or whether its real benefit is budget-robust searchability
- whether the ordinal quantity representation matters beyond the scalar heads
- how much extra value the richer tree routing adds beyond a simple gate or scalar positive head
- whether the same factorization also helps on the plain lost-sales problem

## Next Ablation

The next clean comparisons are now:

- rerun the surviving heads on more than one fixed-cost instance before changing defaults
- isolate why `soft_gated_direct_quantity` survives the `2k` schedule while
  `direct_quantity` and `hard_gated_direct_quantity` do not
- compare the same survivors on plain lost sales once the fixed-cost picture is stable

The fixed-cost linear result is now clear enough to state more narrowly:

- explicit `Q` inside the scalar head is not necessary for *representing* the good solution class
- raw unbounded direct quantity is the bad search-space case
- under a tighter budget, some parameterizations remain searchable and others do not
- so the next question is optimization robustness, not just expressivity

## Emerging Structural Hypothesis

The current results suggest a more concrete structural pattern than "just gating":

- good heads implement a smooth positive-part affine map, or a piecewise version of one
- that map creates a no-order region and a positive replenishment branch
- when the policy orders, it tends to push the system toward a fairly tight post-order inventory
  position band around `27`
- bad heads fall into a middling-order regime that keeps paying setup cost without restoring the
  system aggressively enough

So the most plausible current hypothesis is:

- explicit `* Q` scaling is not the essential ingredient
- raw unbounded direct quantity is clearly harmful under the current optimizer/parameterization pair
- the real winning structure is a search-friendly parameterization of a thresholded
  deficit-correction / order-up-to-like rule
- tree routing can realize this by separating local regimes, while softplus direct heads can realize
  it with a single smooth positive-part map
- any benefit from soft gating is likely about searchability rather than the final deployed policy,
  because the best saved soft-gated direct controller saturates the gate on the visited states

## Primary References

- Marco Bijvank, Sandjai Bhulai, Woonghee Tim Huh (2015), *Parametric replenishment policies for inventory systems with lost sales and fixed order cost*:
  https://www.math.vu.nl/~sbhulai/publications/ejor2015b.pdf
- Woonghee Tim Huh, Ganesh Janakiraman, John A. Muckstadt, Paat Rusmevichientong (2009), *Asymptotic Optimality of Order-Up-To Policies in Lost Sales Inventory Systems*:
  https://www.columbia.edu/~th2113/files/MS_LostSales_Asymptotic_09.pdf
- Michael I. Jordan, Robert A. Jacobs (1993), *Hierarchical Mixtures of Experts and the EM Algorithm*:
  https://www.cs.toronto.edu/~hinton/absps/hme.pdf
- Steven Nowlan, Geoffrey E. Hinton (1991), *Evaluation of Adaptive Mixtures of Competing Experts*:
  https://www.cs.toronto.edu/~hinton/absps/nh91.html
