# one_warehouse_multi_retailer Paper Benchmark

- source: Kaynov et al. (2024), International Journal of Production Economics 267, 109088
- url: https://doi.org/10.1016/j.ijpe.2023.109088
- instances: `14`
- policy family: depth `1` `axis_aligned` soft tree with `linear` leaves
- training allocation: `random_sequential`
- evaluation allocation: `proportional`
- heuristic search: `1000` trajectories of length `100` with common random numbers
- benchmark evaluation: `1000` independent trajectories of length `100`
- instance 14 search note: Kaynov state that instance 14 searches over warehouse level z0 and a shared percentile parameter k. The paper does not publish a discrete k-grid, so the repo enumerates the unique integer retailer-target vectors induced by continuous k in [0, 3].

## Aggregate

- beats best repo heuristic on `3` / `14` instances
- beats best published heuristic on `4` / `14` instances
- beats published PPO on `1` / `14` instances
- mean gap vs best repo heuristic: `-332.675`
- mean gap vs published PPO: `-2330.060`

## Per Instance

| Instance | CB | Learned | Best Repo Heuristic | Best Published Heuristic | Published PPO | Gap vs Repo Best | Gap vs PPO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kaynov2024_instance_1` | `backorder` | `1600.006` | `1562.124` | `1609.470` | `1637.200` | `-37.882` | `37.194` |
| `kaynov2024_instance_2` | `backorder` | `1605.412` | `1381.784` | `1383.880` | `1417.460` | `-223.629` | `-187.952` |
| `kaynov2024_instance_3` | `backorder` | `1870.782` | `1728.020` | `1776.040` | `1731.670` | `-142.762` | `-139.111` |
| `kaynov2024_instance_4` | `backorder` | `1913.986` | `1802.271` | `1857.300` | `1908.950` | `-111.716` | `-5.036` |
| `kaynov2024_instance_5` | `backorder` | `2648.740` | `2454.622` | `2246.840` | `2331.070` | `-194.119` | `-317.670` |
| `kaynov2024_instance_6` | `lost_sales` | `1394.469` | `1346.834` | `1366.510` | `1347.340` | `-47.635` | `-47.130` |
| `kaynov2024_instance_7` | `lost_sales` | `1444.756` | `1390.966` | `1406.270` | `1405.080` | `-53.791` | `-39.677` |
| `kaynov2024_instance_8` | `lost_sales` | `1504.980` | `1473.796` | `1508.120` | `1495.490` | `-31.184` | `-9.490` |
| `kaynov2024_instance_9` | `lost_sales` | `1659.611` | `1521.814` | `1535.960` | `1511.680` | `-137.797` | `-147.931` |
| `kaynov2024_instance_10` | `lost_sales` | `1981.285` | `1777.719` | `1736.550` | `1674.540` | `-203.566` | `-306.745` |
| `kaynov2024_instance_11` | `partial_backorder` | `1141.559` | `1178.381` | `1109.960` | `971.860` | `36.823` | `-169.699` |
| `kaynov2024_instance_12` | `partial_backorder` | `1221.131` | `1240.124` | `1402.380` | `1118.920` | `18.993` | `-102.211` |
| `kaynov2024_instance_13` | `partial_backorder` | `89098.800` | `97166.700` | `99882.510` | `79727.390` | `8067.900` | `-9371.410` |
| `kaynov2024_instance_14` | `partial_backorder` | `64648.989` | `53051.904` | `52787.410` | `42835.020` | `-11597.085` | `-21813.969` |
