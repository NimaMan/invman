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
- `PolicyPerformanceTarget`
- `PolicyPerformanceVerificationSummary`
- `validate_flownet`

This is not an execution engine. It is the common problem-description layer.

It also does not require internal estimation or internal optimization.

The intended use is:

- define the principal stocks, flows, shocks, controls, observations, timing rules, and scoring
- supply exogenous event paths or scenario assumptions from outside
- simulate the dynamics under a chosen control regime
- add optimization later on top if needed

One practical lesson from the first three families is that policy verification is not uniform:

- some problems verify lower-is-better discounted or average cost
- others verify higher-is-better discounted return
- some benchmarks are exact-DP checks
- others are literature or rollout anchors

The shared performance types carry the score ordering so problem-level verifiers can stay precise
about what they are checking.

Problem-level FlowNet folders are expected to verify at least four things:

- structure
- reference alignment
- step semantics
- policy performance
