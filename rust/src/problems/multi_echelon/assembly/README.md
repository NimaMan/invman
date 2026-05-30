# multi_echelon / assembly — textbook assembly system (Rosling 1989)

The `assembly` *version* of the multi-echelon problem: several components are procured from
outside suppliers and assembled into one finished product that faces customer demand. Sibling of
`serial` and `general_network` under `multi_echelon/`.

## Problem

- Components `1..M`, each procured from an ample external supplier with a shared lead time `L_c`.
- An assembly/finished stage consumes one of each component per finished unit (lead time `L_a`),
  faces i.i.d. customer demand, installation holding cost, and a backorder penalty.
- Optimal policy: echelon base-stock (the assembly system is equivalent to a serial system).

## Verification — Rosling (1989) equivalence

Rosling (1989) proved an assembly system is equivalent to a serial system. For the equal
component-lead-time case, a balanced echelon base-stock policy stocks every component identically,
so the components collapse into a single "kit" stage and the system is exactly the 2-stage serial
system **kit → finished**. `rosling.rs` performs that reduction, and the optimal echelon base-stock
levels + optimal cost come from the literature-verified `multi_echelon::serial::exact` solver.

`verification.rs` checks that the assembly **`env.rs` simulation** under those echelon base-stock
levels reproduces the exact serial optimum (within Monte-Carlo error), for:

- 2 identical components, `L_c=1`, finished `L_a=1`;
- 3 identical components, `L_c=2` (component/upstream lead time 2 is fully supported), `L_a=1`;
- heterogeneous component holding costs (kit holding = their sum), `L_c=2`, `L_a=1`.

This is the assembly literature anchor: Rosling's equivalence + the serial Clark–Scarf anchor.

## Scope and known limitation

- **Equal component lead time** (the clean Rosling reduction). Distinct component lead times need
  Rosling's lead-time reordering and are out of scope here.
- **Verified for finished (demand-facing) lead time 1.** The shared serial/assembly env has a known
  open discrepancy when the demand-facing stage has lead time ≥ 2 (the multi-stage simulation
  under-counts cost vs the exact solver; single-stage is correct at every lead time, and
  component/upstream lead times ≥ 2 are fine). Must be resolved before training on
  finished-lead-time ≥ 2 instances. See the `env.rs` module docs.

## Package layout

- `env.rs` — clean assembly environment (training-ready; `consume`/`replenish` two-phase API).
- `rosling.rs` — Rosling reduction of the assembly instance to its equivalent serial instance.
- `echelon_base_stock.rs` — optimal echelon base-stock policy + Monte-Carlo evaluator.
- `verification.rs` — env simulation reproduces the Rosling serial optimum.

## References

- Rosling, K. (1989). "Optimal Inventory Policies for Assembly Systems Under Random Demands."
  *Operations Research* 37(4):565–579.
- Clark, A. J., and H. Scarf (1960). "Optimal Policies for a Multi-Echelon Inventory Problem."
  *Management Science* 6(4):475–490 (the serial equivalent and its solver).
