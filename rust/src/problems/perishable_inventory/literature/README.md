# Literature Overview

Primary sources carried by the repo:

- De Moor et al. (2022), perishable inventory with DQN and reward shaping
  - DOI: https://doi.org/10.1016/j.ejor.2021.10.045
- Farrington et al. (2025), published value-iteration and base-stock returns for the same Scenario A
  settings
  - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12350524/

Repo interpretation:

- the repo carries the 32 Scenario A settings in `rust/src/problems/perishable_inventory/references.rs`
- the exact literature-backed verification targets are the `m = 2` experiments 1 and 2
- the exact verifier reproduces:
  - the Figure 3 optimal policy tables
  - the best base-stock levels from Figure 3
  - the rounded value-iteration returns reported in Farrington et al. (2025)

Canonical instance roles:

- primary reference instance:
  - `de_moor2022_m2_exp2_l1_cp7_fifo`
- practical benchmark instance:
  - `de_moor2022_m4_exp6_l2_cp7_fifo`

Benchmark policies carried by the repo:

- `base_stock`
- `bsp_low_ew`
- `dqn`
- `shaped_dqn`
