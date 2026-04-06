# Fixed-Cost Local Scale Comparison At Population 50

Protocol:

- problem: `lit_pois_mu5_l4_p4_k5`
- backend: Rust
- training: `200` CMA iterations
- population: `50`
- evaluation: horizon `100000`, `3` seeds

| Scale | Ordinal | Tree d1 | Tree d2 |
| ---: | ---: | ---: | ---: |
| 10 | 8.8424 | 8.7678 | 8.7771 |
| 15 | 9.5515 | 8.8061 | 8.7684 |
| 20 | 8.8334 | 8.7719 | 8.7729 |
| 25 | 8.8308 | 9.7522 | 8.7770 |
| 30 | 8.9003 | 8.7736 | 8.7779 |
| 40 | 9.0223 | 8.7738 | 9.7537 |
| 50 | 10.0741 | 9.7787 | 9.7795 |

Quick read:

- the ordinal head is strong around `10` to `30`, then degrades by `40`, and is clearly bad at `50`
- the depth-2 tree is remarkably stable from `10` to `30`, then degrades sharply at `40` and `50`
- the depth-1 tree is also strong in most of the `10` to `40` band, but shows a bad local-basin spike at `25`
- all three policy families are scale-sensitive
- none of them prefer the old fixed-cost `50` scaling at this short proxy budget
