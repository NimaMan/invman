# Literature

Primary anchors:

- Zhou et al. (2022), deep reinforcement learning for joint pricing and inventory under reference-price effects
- Qin, Simchi-Levi, and Wang (2022), data-driven approximation schemes for joint pricing and inventory control

Current status: this package is not literature-verified.

Why:

- Zhou et al. (2022) use an infinite-horizon formulation with a reference-price state and compare
  DRL against other algorithms inside that richer model.
- The repo package keeps a reduced finite-horizon price-sensitive lost-sales formulation and does
  not include the reference-price state.
- Qin et al. (2022) are closer in spirit, but the publicly accessible article page does not expose
  a reusable row-level benchmark table for this package, and the linked supplemental/replication
  material is not openly retrievable from this environment.

So the repo uses these papers as formulation anchors only. The executable verification target for
the current package remains repo-native.

Reference hygiene:

- `references.rs` stores literature anchors and problem-instance definitions only
- repo-native worked-transition expected values live in verification tests, not in literature references
