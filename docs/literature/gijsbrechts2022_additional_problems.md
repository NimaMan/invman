# Gijsbrechts 2022 Additional Problem Types

This note records the two non-lost-sales benchmark families used by Gijsbrechts et al. (2022),
their related heuristic baselines, and the repo scope chosen for `invman`.

Primary source:

- Gijsbrechts, Boute, and Lambrecht (2022), *Can Deep Reinforcement Learning Improve Inventory Management? Performance on Lost Sales, Dual-Sourcing, and Multi-Echelon Problems*.
  DOI: https://doi.org/10.1287/msom.2021.1064

Supporting sources used for the heuristic definitions:

- Sheopuri, Janakiraman, and Seshadri (2010), *New Policies for the Stochastic Inventory Control Problem with Two Supply Sources*.
  DOI: https://doi.org/10.1287/opre.1090.0782
- Veeraraghavan and Scheller-Wolf (2008), *Now or Later: A Simple Policy for Effective Dual Sourcing in Capacitated Systems*.
  RePEc / DOI landing page: https://ideas.repec.org/a/inm/oropre/v56y2008i4p850-864.html
- Van Roy, Bertsekas, Lee, and Tsitsiklis (1997), *A Neurodynamic Programming Approach to Retailer Inventory Management*.
  DOI: https://doi.org/10.1109/CDC.1997.652501

## Shared A3C Architecture Across the Gijsbrechts Problems

Gijsbrechts et al. keep one shared neural backbone across dual sourcing, lost sales, and
multi-echelon:

- four fully connected layers with widths `[150, 120, 80, 20]`
- ReLU after each layer
- value regularization `0.25`
- four parallel learners
- gradient clipping `40`

The three tuned hyperparameters are:

- learning rate
- entropy regularization
- buffer length

Problem-specific changes are mostly in the input dimension and action-space design:

- lost sales: bounded scalar action `[0, 1, ..., 20]`
- dual sourcing: two order quantities `(q_regular, q_expedited)`
- multi-echelon: discretized base-stock action grid from Van Roy et al. (1997)

## Benchmark-Reference Summary

For the two non-lost-sales problems, the literature gives us the benchmark problem sets and
benchmark policy classes, but not a clean published table of exact per-instance costs for every
setting we use in `invman`.

- Dual sourcing:
  - benchmark problem family: six small-scale settings from Gijsbrechts et al. (2022), Section 6.2,
    inherited from Veeraraghavan and Scheller-Wolf (2008);
  - benchmark comparators: optimal DP, single-index, dual-index, capped dual-index, tailored
    base-surge, and LP-ADP;
  - literature takeaway: Gijsbrechts et al. report that A3C is within 2% of optimal in all six
    settings, while capped dual-index uniformly dominates the other heuristic families in their tests.
- Multi-echelon:
  - benchmark problem family: the two Van Roy et al. (1997) settings reproduced as Table 3 in
    Gijsbrechts et al. (2022), Section 6.3;
  - benchmark comparator: constant base-stock policy, plus the earlier neuro-dynamic programming
    approach of Van Roy et al. (1997);
  - literature takeaway: Gijsbrechts et al. report that A3C improves by about 9% and 12% over
    constant base-stock in the two settings, and note that Van Roy et al. report roughly 10% savings.

Because the published Gijsbrechts paper reports relative performance summaries and figures rather
than exact per-instance cost tables, the exact benchmark costs stored in this repo should be treated
as repo-native baselines anchored to the published problem sets, not as verbatim literature tables.

## Dual Sourcing

Literature setting used in the repo:

- expedited lead time `l_e = 0`
- regular lead time `l_r in {2, 3, 4}`
- demand uniform on `{0, 1, 2, 3, 4}`
- holding cost `h = 5`
- backlog cost `b = 495`
- regular unit cost `c_r = 100`
- expedited unit cost `c_e in {105, 110}`

Implemented heuristic baselines:

- single-index
- dual-index
- capped dual-index
- tailored base-surge

Published benchmark references:

- Gijsbrechts et al. (2022), Section 6.2 and Figure 9:
  - uses exactly these six settings;
  - compares A3C against optimal DP and the four heuristic families above;
  - states that A3C is within 2% of optimal in all six settings;
  - states that capped dual-index is the strongest heuristic benchmark in these tests.
- Veeraraghavan and Scheller-Wolf (2008):
  - source for the small-scale benchmark family used by Gijsbrechts;
  - benchmark takeaway: dual-index style policies are near-optimal and tailored base-surge is a key
    low-dimensional benchmark family.
- Sheopuri et al. (2010):
  - source for the refined two-source benchmark-policy family;
  - benchmark takeaway: improved dual-sourcing heuristics can outperform older index policies on
    broader lead-time structures.

Repo benchmark notes:

- the package includes a bounded dynamic-programming solver over the reduced `l_r`-dimensional state
  representation for the small-scale literature settings;
- benchmark claims should use the DP result only as a bounded finite-state reference, not as a claim
  of a proof-level exact optimum outside the chosen truncation box.

## Multi Echelon

Literature setting used in the repo:

- one warehouse and `K` identical retailers
- warehouse lead time `l_w`
- retailer lead time `l_r`
- demand at retailers sampled from rounded normal distributions
- hybrid same-day-expedite / lost-sales service with probability `P_w`

Reference settings:

- setting 1: `mu=5`, `sigma=14`, `l_w=2`, `l_r=2`, `K=10`
- setting 2: `mu=0`, `sigma=20`, `l_w=5`, `l_r=3`, `K=10`
- shared parameters: `h_w=3`, `h_r=3`, `c_w=0`, `p=60`, `P_w=0.8`, `C_m=100`, `C_w=1000`, `C_r=100`

Published benchmark references:

- Gijsbrechts et al. (2022), Section 6.3 and Table 3:
  - uses these two settings from Van Roy et al. (1997);
  - compares A3C against a constant base-stock benchmark over the same discretized action grid;
  - reports approximate savings of 9% and 12% over constant base-stock.
- Van Roy et al. (1997):
  - original benchmark source for the two-echelon model and the two settings;
  - benchmark takeaway: their neuro-dynamic programming approach is substantially better than
    optimized order-up-to / s-type policies, with roughly 10% cost savings.

Implemented baseline:

- constant base-stock policy over the literature action grids

Learned-policy scope in the repo:

- the learned policy action matches the paper’s reduced action space:
  state-dependent warehouse and retailer base-stock levels `(y_w, y_r)`.

Repo benchmark notes:

- the repo benchmark is faithful to the literature action-grid setup;
- constant base-stock search is exhaustive over the configured warehouse and retailer action grids.

## Additional DRL Inventory Problem Types Outside the Current Repo Scope

Below are adjacent problem families that are plausible next extensions for `invman`.

### Perishable Inventory

Primary source:

- Alvaro Maggiar, Carson Eisenach, Sohrab Andaz, Dean Foster, Akhil Bagaria, Omer Gottesman,
  and Dominique Perrault-Joncas (2025), *Structure-Informed Deep Reinforcement Learning for
  Inventory Management*.
- arXiv: https://arxiv.org/abs/2507.22040

Problem type:

- multi-period lost-sales inventory with fixed shelf life

Architecture used there:

- a WaveNet-style encoder for time-series inputs
- history window `H = 32`
- five stacked causal CNN layers with dilations `1, 2, 4, 8, 16`
- the encoded time-series output is combined with static and endogenous variables in an MLP
- MLP has two hidden layers of `32` neurons
- ELU activations throughout

Why this matters for `invman`:

- this is the cleanest literature-backed next step if we want a fifth problem family beyond the
  current four
- it also provides a non-MLP architecture reference that is still inventory-specific

### Joint Inventory Procurement and Removal

Primary source:

- same Maggiar et al. (2025) paper above

Problem type:

- joint procurement and removal / returns-value inventory control

Architecture used there:

- the same shared WaveNet-plus-MLP policy architecture as above
- no problem-specific architecture change is introduced for this setting

Why this matters for `invman`:

- it is a natural extension if we want to study reverse logistics or inventory liquidation decisions

### Beer Game / Serial Supply-Chain Coordination

Primary source:

- Afshin Oroojlooyjadid, MohammadReza Nazari, Lawrence V. Snyder, and Martin Takac (2020),
  *A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization*.
- arXiv: https://arxiv.org/abs/1708.05924

Problem type:

- decentralized serial supply chain with local observations and cooperative total-cost objective

Architecture and parameter details reported there:

- DQN / SRDQN rather than actor-critic
- state uses the last `m` periods of local observations
- action uses a finite `d + x` adjustment rule instead of an unbounded order quantity
- one explicit network shape reported for transfer-learning experiments is `[50, 180, 130, 61, 5]`
- one explicit parameter setting reported there is `m = 10`, `beta = 20`, and target-network update
  period `C = 10000`

Why this matters for `invman`:

- it is a natural reference if we ever want to move from centralized replenishment problems toward
  decentralized supply-chain learning
