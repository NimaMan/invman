# perishable_inventory Practical Benchmark

- dataset: `grocery_like_daily_trace`
- source_kind: `repo_curated_semi_real`
- source_note: Daily grocery-like demand trace with weekday seasonality and two promotion spikes. This is a semi-real practical benchmark, not a literature verification instance.
- practical_goal: Tune simple replenishment heuristics on recent history and evaluate waste-service-cost tradeoffs on a held-out demand block.
- calibration: Tune `base_stock` and `bsp_low_ew` on the train block with deterministic trace search; report both in-sample train and held-out test metrics using the train-demand empirical mean.
- reference_instance_name: `de_moor2022_m4_exp6_l2_cp7_fifo`
- train_mean_demand: `4.6667`
- test_mean_demand: `5.1786`
- train_periods: `42`
- test_periods: `28`

| Policy | Split | Params | Mean Period Cost | Fill Rate | Cycle Service | Waste / Demand | Mean Holding | Mean Order | Positive Order Freq | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base_stock` | `train` | `[10]` | `18.4524` | `0.6480` | `0.4762` | `0.0000` | `0.6667` | `3.1905` | `0.9524` | calibration block |
| `base_stock` | `test` | `[10]` | `21.2143` | `0.5793` | `0.3929` | `0.0000` | `0.5714` | `3.2500` | `0.8929` | held-out evaluation |
| `bsp_low_ew` | `train` | `[10, 10, 0]` | `18.4524` | `0.6480` | `0.4762` | `0.0000` | `0.6667` | `3.1905` | `0.9524` | calibration block |
| `bsp_low_ew` | `test` | `[10, 10, 0]` | `21.2143` | `0.5793` | `0.3929` | `0.0000` | `0.5714` | `3.2500` | `0.8929` | held-out evaluation |
