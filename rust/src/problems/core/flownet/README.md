# core/flownet

`flownet` is the short name for the shared inventory-problem language in `problems/core`.

It captures inventory systems as timed networks of:

- stocks
- pipelines
- flows
- random shocks
- control actions
- observations
- scoring rules

The main types in this folder are:

- `FlowNetQuestion`
- `FlowNetFormulation`
- `FlowNetInstance`
- `validate_flownet`

This is not an execution engine. It is the common problem-description layer.
