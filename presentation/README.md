# Presentation Story

This deck should tell a focused version of the paper's broader result:

> Compact learned policies can work across inventory-control problems when the learner searches in an action space that matches the structure of the control problem.

The thesis (slide 2) is the portable method itself:

> One gradient-free recipe (CMA-ES) learns compact, interpretable policies that are competitive across ten structurally different inventory problems.

The action-space idea — that the action parameterization is a design choice, not a fixed property of the environment — is a *supporting insight*, a piece of the solution that explains why such small policies suffice. It belongs on the mechanism slides (policy parameterization, woven through the case studies), never as the headline thesis. The contribution is the portable CMA-ES recipe over compact policy-owned decoders, evaluated with careful match / beat / context accounting.

## Core Intuition

The same operational decision can be represented in several ways:

- raw order quantities
- order-up-to levels
- inventory positions
- caps and thresholds
- residuals around a heuristic
- problem-specific target coordinates

These choices are not neutral. Some action spaces make good policies easy to express and easy for search to find. Others force the learner to rediscover both the control law and the useful coordinates.

Classical inventory heuristics often reveal useful coordinates. We can use that heuristic intuition to design the policy's decoder, then let CMA-ES tune a compact policy inside or around that action space.

## What / Why / How / Now What

### What

We learn compact, interpretable inventory-control policies with CMA-ES.

The policy maps state to a valid action, but the action is not always a raw order. The policy owns the decoder that turns latent outputs into an inventory action.

### Why

Inventory control is hard because lead times, pipelines, shortage rules, and network structure make the state and action spaces grow quickly.

Classical heuristics are strong and interpretable, but they are problem-specific. Deep RL is flexible, but can be large, tuning-heavy, and hard to compare fairly across papers.

The opening tension should be:

> Can we keep the portability of learning while using the structure that makes heuristics strong?

### How

Use one simple training loop:

1. Choose a compact policy class.
2. Choose an action parameterization informed by the inventory structure.
3. Roll out policies in simulation.
4. Let CMA-ES minimize realized operating cost directly.
5. Report results with the right verdict: match optima, beat same-protocol heuristics, and keep cross-protocol DRL as context.

The key design move is not simply "bigger model" or "better optimizer." It is choosing an action space where a good policy is simple.

### Evidence

Use three case studies as teaching examples.

**Lost Sales**

The action is a scalar order, so this is the cleanest setting for seeing decoder effects. Direct, ordinal, gated, and tree decoders make different policy classes easy or hard to optimize.

Claim hygiene: the `22 / 24` vanilla result is a broad benchmark sweep using single optimizer runs per cell, not the same five-optimizer-seed claim used later.

**Dual Sourcing**

The useful action coordinates are close to capped-dual-index controls. The learned policy should be framed as recovering a near-optimal structured policy, not beating it.

Main message: when the heuristic is already near-optimal, the honest goal is match / recovery.

**Divergent Multi-Echelon**

This is the strongest visual example. The same learner behaves very differently when the reachable action set changes. Direct order-up-to levels can express the operating region; the reduced grid cannot.

Main message: the action space can determine whether the learner can even reach the useful region.

## Subtle Value Proposition

Avoid making a sales pitch. Let the values appear through the story:

- compact policies are easier to inspect and deploy
- heuristic-informed action spaces make learning more reliable
- one optimizer loop can transfer across problem families
- match / beat / context accounting makes the evidence more defensible
- the benchmark suite can become a reusable baseline ledger for future learners

## Suggested Slide Flow

1. Title: paper title and subtitle, authors, and affiliation (match the .tex \title / \author). No hero figure.
2. Thesis: one gradient-free recipe learns compact, interpretable policies competitive across ten problems.
3. Problem: lead times and networks make inventory control hard.
4. Tension: heuristics are structured but bespoke; generic learners are flexible but heavy.
5. Recipe: CMA-ES optimizes compact policies on rollout cost.
6. Policy interface: state -> normalization -> backbone -> decoder -> valid action.
7. Evidence frame: match / beat / context.
8. Optional overview: ten problem families and verdict badges.
9. Lost sales: scalar order and decoder comparison.
10. Lost sales result: 22 / 24 broad-grid evidence, with caveat.
11. Dual sourcing: CDI-shaped coordinates.
12. Dual sourcing result: 6 / 6 match, not a beat.
13. Multi-echelon: direct levels vs reduced grid.
14. Multi-echelon result: robust same-protocol base-stock beat.
15. Synthesis: same recipe, decoder changes shape.
16. Takeaways: structure in the action space beats raw capacity.
17. Now what: search decoder classes systematically; pair structured decoders with more sample-efficient learners.

## Tone and Register

Keep the deck official and academic. Present results, not self-assessment.

- Do not label the work's own reporting as "honest", "rigorous", or "never overclaimed" on the slides. State results neutrally as match / beat / context and let the discipline speak for itself.
- The title and metadata carry formal identifiers (author, affiliation, venue), not casual taglines.
- Prefer neutral, descriptive headings over conversational ones (avoid "the story", "read out with ...").
- Eyebrows are formal section labels: Thesis, Motivation, Method, Policy parameterization, Evaluation protocol, Scope and results, Case study N — <problem>, Results, Synthesis, Conclusions, Limitations and future work.
- Headlines state a single finding declaratively (assertion-evidence). Avoid rhetorical questions, punchy fragments (e.g. "Strong but bespoke ..."), and promotional verbs (e.g. "robustly beat").
- No first person or self-reference on slides ("we already do", "reproduce ours"); phrase directions impersonally.

## Phrases To Use

- The action space is a design choice, not just an environment detail.
- Some action spaces make good policies easier to express and easier to find.
- Heuristics give useful coordinates; learning tunes within or around them.
- The decoder is where inventory structure enters the learned policy.
- We match optima, beat same-protocol heuristics, and keep cross-protocol DRL as context.

## Phrases To Avoid

- Raw orders are the wrong language.
- The learner discovers the structure from scratch.
- We beat DRL broadly.
- One leaderboard proves the method.
- Bigger networks are unnecessary in general.
- Honest verdicts / we report honestly / never overclaimed.
- The story is not a leaderboard.

These are too absolute. The more defensible claim is that action parameterization often matters more than model capacity in these inventory benchmarks.
