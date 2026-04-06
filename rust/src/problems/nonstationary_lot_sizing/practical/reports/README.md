# nonstationary_lot_sizing Practical Benchmark

- dataset: `retail_like_weekly_trace`
- source_kind: `repo_curated_semi_real`
- source_note: Weekly retail-like forecast and realized-demand trace with trend, weekly pulses, and forecast error. This is a semi-real practical benchmark, not a literature verification row.
- practical_goal: Evaluate forecast-driven replenishment baselines on a single rolling forecast path using service, cost, and ordering metrics.
- calibration: No train/test split. These heuristics adapt directly from the rolling forecast window; the benchmark evaluates them on one fixed forecast-plus-realization path.
- periods: `32`
- mean_forecast: `11.9688`
- mean_realized_demand: `11.9688`
- forecast_mae: `1.0000`
- forecast_bias: `0.0000`

| Policy | Split | Params | Mean Period Cost | Shortage Rate | Cycle Service | Mean Holding | Mean Order | Positive Order Freq | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lead_time_base_stock` | `eval` | `adaptive` | `15.9345` | `0.0000` | `1.0000` | `5.9345` | `12.2101` | `1.0000` | uses current forecast window directly |
| `simple_s_s` | `eval` | `[32.027, 46.775]` | `20.4885` | `0.0000` | `1.0000` | `15.1760` | `12.7353` | `0.5312` | params column shows first-period levels only |
| `rolling_dp_s_s` | `eval` | `[25.0, 40.0]` | `13.7188` | `0.0000` | `1.0000` | `7.4688` | `12.0938` | `0.6250` | params column shows first-period levels only |
