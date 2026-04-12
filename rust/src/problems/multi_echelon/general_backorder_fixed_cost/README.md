# General Backorder Fixed Cost

This subfolder is reserved for the multi-echelon formulation used by papers such as Geevers et al.
(2023), where the network allows general backorder structure and fixed ordering costs are part of the
benchmark.

It is intentionally separate from `divergent_special_delivery` because that Van Roy family uses a
different event structure and unmet-demand mechanism.

Current status:

- formulation not implemented yet
- no Rust environment or rollout path yet
- no bindings exported yet
