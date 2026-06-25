# Fixed-Cost NN Diagnostics

Canonical instance:

- problem: lost-sales with fixed order cost
- lead time `L = 4`
- shortage cost `p = 4`
- fixed order cost `K = 5`
- holding cost `h = 1`
- demand: Poisson with mean `5`

Reference heuristic benchmark on this instance:

- `s,S`: about `9.43`
- `s,nQ`: about `9.19`
- modified `s,S,q`: about `9.16`

Diagnostic experiment matrix:

| experiment | hidden layers | sigma init | ES iters | learned cost | gap to modified `s,S,q` |
| --- | --- | ---: | ---: | ---: | ---: |
| `diag_fixed_cost_nn_h8x8_sig5_pop10_1k` | `8,8` | `5.0` | `1000` | `10.2501` | `11.95%` |
| `diag_fixed_cost_nn_h16x16_sig5_pop10_1k` | `16,16` | `5.0` | `1000` | `10.2501` | `11.95%` |
| `diag_fixed_cost_nn_h16x16x16_sig5_pop10_1k` | `16,16,16` | `5.0` | `1000` | `10.2501` | `11.95%` |
| `diag_fixed_cost_nn_h16x16x16_sig1_pop10_1k` | `16,16,16` | `1.0` | `1000` | `10.2501` | `11.95%` |

Canonical naming:

- `nn` or `linear`: policy backbone
- `categorical_quantity`: categorical action head over order quantities
- `direct_quantity`: direct scalar action head mapped to an order quantity
- `cma` / `es` / `ga` / `xnes`: parameter optimizers

Main finding:

All tested NN configurations converged to the same effective policy and the same evaluated cost. The trained policy behaves like a constant-order rule, ordering `4` units almost everywhere, while the best heuristic uses a threshold structure that alternates between `0`, `7`, and `8`.

Interpretation:

The failure is not explained by insufficient width or depth alone. The dominant issue is the current policy/optimizer interface: CMA-ES searches a continuous parameter space, but the policy acts through a hard `argmax` over `51` discrete order quantities. That creates large plateaus where many parameter vectors induce the same discrete ordering rule.

Implication for next iteration:

The next useful change is not a larger fully connected network. We should change the policy parameterization to expose the threshold structure directly, for example by learning:

- an order/no-order gate plus a separate positive-order quantity head
- a direct `(s,S)` or `(s,q)` style policy class
- a smoother action-selection surrogate during ES training
