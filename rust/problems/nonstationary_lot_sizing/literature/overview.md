# Literature Overview

Primary sources carried by the repo:

- Dehaybe et al. (2024), nonstationary single-item lot sizing with rolling forecasts
  - DOI: https://doi.org/10.1016/j.ejor.2023.10.007
- HenriDeh/DRL_MMULS single-item branch
  - URL: https://github.com/HenriDeh/DRL_MMULS/tree/single-item

Repo interpretation:

- the repo carries the eight fixed-forecast lost-sales benchmark instances from the paper family
- the worked transition in Section 4.2 is implemented as an exact mechanics check
- the repo reproduces:
  - the simple `(s,S)` baseline row on the primary constant-10 instance
  - the rolling-DP benchmark row on the same instance

Canonical instance roles:

- primary reference instance:
  - `dehaybe2024_lostsales_lt2_b5_k10_constant_10`
- verification problem instance:
  - `constant_10_rolling_dp_reference`

Benchmark policies carried by the repo:

- `simple_s_s`
- `rolling_dp_s_s`
- `lead_time_base_stock`
- `ppo` as literature comparator only
