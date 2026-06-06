// agentic_policy_search :: agent :: brain.rs
// =================================================================================================
// ALGORITHMIC DESCRIPTION
// -------------------------------------------------------------------------------------------------
// This module builds the *proposer* brain for the OWMR policy-structure search and provides an
// explicit, opt-in deterministic stub so the whole evolution loop is testable without spending
// Codex calls. There are two paths and choosing between them is never silent:
//
//   build_brain(cfg) -> Arc<dyn ConversationBrain>
//     * DEFAULT (real): a `beyin-codex` CodexBrain in a ReadOnly sandbox. Codex may only reason and
//       emit ONE structured tool request; it cannot touch the filesystem or run the evaluation. Its
//       JSON output schema is fixed by beyin-codex (message + tool_requests[] + meta). The OWMR
//       system prompt (below) instructs it to recombine the archive's best specs into exactly one
//       new policy-spec and to call the `invman.evaluate_policy_spec` tool with that spec as the
//       tool arguments (a compact JSON string of the DSL object).
//     * STUB (opt-in): when the env var APS_STUB_BRAIN is set to a truthy value ("1"/"true"/"yes"),
//       build_brain returns `StubProposerBrain` instead and PRINTS a one-line notice to stderr so
//       the choice is visible in logs. The stub is a hand-written ConversationBrain that:
//         - on the first brain step of a turn (no tool result yet in the transcript): emits a single
//           `invman.evaluate_policy_spec` tool request carrying a deterministic OWMR spec drawn from
//           a fixed rotation of structural variants (so successive generations propose *different*
//           structures, exercising the archive/ranking path), status = NeedsToolResults.
//         - on the second brain step (the evaluation tool result is now in the transcript): reads
//           the result back, emits a short assistant message summarizing deployed_cost /
//           robust_gate_beat, and completes the turn (status = Completed).
//       The rotation index is derived from the generation number injected into the run_turn context
//       (context.generation), so the stub is fully deterministic given the loop's generation count.
//
// WHY A SINGLE TOOL CALL PER TURN: the runtime loops the brain after each action; emitting exactly
// one evaluate request per turn keeps one spec == one generation == one archive row, matching the
// README's outer loop. The brain's *second* step (post-result) just narrates and completes, so each
// generation costs at most two brain steps (well under the default max_brain_steps = 8).
//
// The OWMR system prompt encodes the README DSL contract verbatim enough that Codex emits evaluable
// specs (problem fixed to one_warehouse_multi_retailer, instance 14), and the honest-reporting rule
// (robust gate-beat = all seeds below gate AND mean+std < gate) so the proposer optimizes the right
// objective rather than a lucky single seed.
//
// NOVELTY PRESSURE (the fix for the verified re-proposal plateau): the per-generation context built
// in main.rs no longer feeds a flat top_k-by-cost (which collapsed to ~5 near-duplicates of the
// single archived best, so Codex deterministically re-emitted it). Instead it carries
//   * diverse_elites    — the BEST spec per OCCUPIED structural niche (action_head|leaf_type|
//                         split_type), best-first: structurally-DISTINCT parents to recombine across,
//   * tried_signatures  — every DISTINCT structure already evaluated (a key over action_head,
//                         leaf_type, split_type, depth, per_retailer_targets, features, backbone,
//                         warm_start) with its best deployed_cost / robust_gate_beat / times_tried,
//   * best_signature    — the signature of the current archive best (the referent to deviate from),
//   * untried_niches    — concrete unexplored DSL cells, plus n_occupied/n_total coverage.
// The owmr_system_prompt below consumes these and applies a HARD rule: NEVER re-propose a signature
// in tried_signatures, and differ from best_signature on >=1 structural axis (prefer an untried
// niche). The signature deliberately ignores temperature/continuous params (CMA-ES owns those), so
// changing only temperature does NOT count as a new structure. The StubProposerBrain is left
// UNCHANGED and is intentionally novelty-agnostic: it already rotates 4 distinct structures by
// generation index, so it still exercises the archive/ranking/new-context path end-to-end.
// =================================================================================================

use std::path::PathBuf;
use std::sync::Arc;

use beyin::{
    BrainError, ConversationBrain, ConversationBrainRequest, ConversationBrainResponse,
    ConversationTurnStatus,
};
use beden_core::ActionRequest;
use beyin_codex::{CodexBrain, CodexBrainConfig, CodexSandbox, CodexSessionStrategy};
use omurga_agent::TranscriptItemKind;
use serde_json::{Value, json};

use crate::actions::EvaluatePolicySpecAction;

/// Env var that, when truthy, selects the deterministic stub proposer instead of CodexBrain.
pub const STUB_BRAIN_ENV: &str = "APS_STUB_BRAIN";

/// Is the value of an env var "truthy" (1/true/yes, case-insensitive)?
fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// The OWMR system prompt: instructs Codex to recombine the archive into ONE evaluable policy-spec
/// and call the evaluate tool with it, under the NOVELTY-PRESSURE rule (never repeat a tried
/// signature; vary >=1 structural axis vs best_signature; prefer untried niches). Kept as a
/// function so the instance/problem can be templated.
pub fn owmr_system_prompt(problem: &str, instance: u32) -> String {
    format!(
        "You are the policy-STRUCTURE proposer for an automated inventory-control search.\n\
\n\
OBJECTIVE (the only success metric): discover a policy-spec that ROBUSTLY beats the in-repo \
echelon-base-stock GATE on {problem} instance {instance} (OWMR, K=10 strongly heterogeneous \
retailers, partial backorder). Robust gate-beat means: EVERY evaluation seed is strictly below \
the gate cost AND mean+std < gate cost. Anything weaker is parity, not a win. The published PPO \
figure is cross-protocol context only and must NEVER be treated as the comparator.\n\
\n\
WHAT YOU DO each turn: study the archive evidence in the context, then propose EXACTLY ONE new \
policy-spec that recombines or mutates the best structures to try to beat the gate. Change the \
STRUCTURE (action head, per-dimension geometry, state-dependent leaf, split type, feature basis), \
not the continuous parameters — an inner CMA-ES owns those and is warm-started at the gate anchor. \
The context gives you (a) `diverse_elites`: the BEST spec found so far IN EACH DISTINCT STRUCTURAL \
NICHE (a niche = action_head x leaf_type x split_type), ranked best-first by deployed_cost — these \
are structurally-different parents to recombine ACROSS, not one winner to copy; (b) \
`tried_signatures`: every structure already evaluated (a key over action_head, leaf_type, \
split_type, depth, per_retailer_targets, features, backbone, warm_start) with its best \
deployed_cost and whether it robustly beat the gate; (c) `best_signature`: the signature of the \
current archive best; (d) `untried_niches`: structural niches not yet tried at all, with your \
coverage `n_occupied_niches`/`n_total_niches`.\n\
\n\
NOVELTY REQUIREMENT (hard): you MUST NOT propose any spec whose signature equals an entry in \
`tried_signatures` — re-proposing a tried structure is a wasted generation and is forbidden. Your \
proposal MUST differ from `best_signature` on AT LEAST ONE structural axis (a different \
action_head, leaf_type, split_type, depth, per_retailer_targets, features subset, backbone, or \
warm_start). Prefer to either (i) FILL one of the `untried_niches` (set its action_head, \
leaf_type, split_type to that cell, warm-started at the gate so it ties at worst), or (ii) \
RECOMBINE two STRUCTURALLY DIFFERENT `diverse_elites` (take the action_head/leaf geometry of one \
and the split_type/feature basis of another) into a niche different from every elite shown. The \
signature IGNORES temperature and continuous parameters (CMA-ES owns those), so changing only \
temperature does NOT count as a new structure. In your one-sentence `rationale`, name which \
structural axis you changed versus `best_signature` (or which untried niche you filled / which two \
elites you crossed) and the hypothesis.\n\
\n\
You make this happen by calling the body tool `invman.evaluate_policy_spec` EXACTLY ONCE, with \
`arguments` set to a COMPACT JSON STRING of the policy-spec DSL object. Do not call any other tool \
and do not call it more than once. The DSL object MUST have this shape (problem and instance are \
fixed):\n\
{{\n\
  \"problem\": \"{problem}\",\n\
  \"instance\": {instance},\n\
  \"backbone\": \"soft_tree\" | \"linear\",\n\
  \"depth\": <int, soft_tree only>,\n\
  \"split_type\": \"oblique\" | \"axis_aligned\",\n\
  \"leaf_type\": \"constant\" | \"linear\",\n\
  \"temperature\": <float>,\n\
  \"action_head\": \"echelon_targets\" | \"symmetric_echelon_targets\" | \
\"echelon_targets_with_alloc_targets\" | \"direct_orders\",\n\
  \"per_retailer_targets\": <bool>,\n\
  \"features\": [subset of \"on_hand\",\"backlog\",\"pipeline\"],\n\
  \"warm_start\": \"gate_invertible\" | \"none\",\n\
  \"rationale\": \"<one sentence: which archived structure you recombined and the hypothesis>\"\n\
}}\n\
\n\
Constraints (every emitted spec must be evaluable): unknown enum values are rejected by the \
compiler; `symmetric_echelon_targets` is valid ONLY on a homogeneous-retailer instance with \
per_retailer_targets=false, so on this strongly-heterogeneous instance it is INVALID -- do NOT \
propose it and SKIP any untried niche whose action_head is symmetric_echelon_targets (the valid \
heads here are echelon_targets, echelon_targets_with_alloc_targets, and direct_orders, all with \
per_retailer_targets=true); echelon_targets with per_retailer_targets=true grows control_dim to K+1; prefer \
warm_start=\"gate_invertible\" so generation-0 ties the gate exactly and any improvement is real. \
Proven exploitable levers from sibling instances: changing the leaf/target head already flipped \
the sign (use leaf_type=\"linear\" and per_retailer_targets to add state-dependent, per-retailer \
control). After the tool returns, briefly state deployed_cost and whether it was a robust gate-beat, \
then stop.\n"
    )
}

/// Build the proposer brain. Default = real CodexBrain (ReadOnly); stub iff APS_STUB_BRAIN is truthy.
///
/// `working_dir` is the directory Codex runs `codex exec` in (ReadOnly): point it at the
/// agentic_policy_search dir so the brain can read the README/DSL if it chooses. `model` is the
/// optional Codex model override; None lets Codex use its default.
pub fn build_brain(
    problem: &str,
    instance: u32,
    working_dir: PathBuf,
    model: Option<String>,
) -> Arc<dyn ConversationBrain> {
    if env_truthy(STUB_BRAIN_ENV) {
        eprintln!(
            "[brain] {STUB_BRAIN_ENV} is set -> using deterministic StubProposerBrain (NO Codex calls)"
        );
        return Arc::new(StubProposerBrain::new(problem.to_string(), instance));
    }
    eprintln!("[brain] using real CodexBrain (ReadOnly sandbox)");
    let config = CodexBrainConfig::new()
        .with_name("brain.codex.owmr")
        .with_sandbox(CodexSandbox::ReadOnly)
        .with_session_strategy(CodexSessionStrategy::Ephemeral)
        .with_working_dir(Some(working_dir))
        .with_model(model)
        .with_system_prompt(owmr_system_prompt(problem, instance));
    Arc::new(CodexBrain::new(config))
}

// =================================================================================================
// StubProposerBrain — deterministic, explicit, no-Codex proposer for loop testing.
// =================================================================================================

/// A deterministic ConversationBrain that emits one OWMR spec per turn from a fixed rotation, then
/// narrates the evaluation result and completes. Used only when APS_STUB_BRAIN is truthy.
pub struct StubProposerBrain {
    problem: String,
    instance: u32,
}

impl StubProposerBrain {
    pub fn new(problem: String, instance: u32) -> Self {
        Self { problem, instance }
    }

    /// A small rotation of structurally distinct OWMR specs. Index modulo length selects one, so the
    /// loop proposes a different structure each generation.
    fn variant(&self, idx: usize) -> Value {
        // (backbone, depth, split, leaf, temp, head, per_retailer, features)
        let variants: &[(&str, u64, &str, &str, f64, &str, bool, &[&str])] = &[
            ("linear", 0, "axis_aligned", "linear", 0.0, "echelon_targets", false, &["on_hand", "backlog", "pipeline"]),
            ("soft_tree", 2, "oblique", "linear", 0.25, "echelon_targets", true, &["on_hand", "backlog", "pipeline"]),
            ("soft_tree", 2, "oblique", "constant", 0.25, "symmetric_echelon_targets", true, &["on_hand", "backlog"]),
            ("soft_tree", 3, "oblique", "linear", 0.15, "echelon_targets_with_alloc_targets", true, &["on_hand", "backlog", "pipeline"]),
        ];
        let (backbone, depth, split, leaf, temp, head, per_retailer, features) =
            variants[idx % variants.len()];
        json!({
            "problem": self.problem,
            "instance": self.instance,
            "backbone": backbone,
            "depth": depth,
            "split_type": split,
            "leaf_type": leaf,
            "temperature": temp,
            "action_head": head,
            "per_retailer_targets": per_retailer,
            "features": features,
            "warm_start": "gate_invertible",
            "rationale": format!(
                "stub deterministic variant {} ({} backbone, {} leaf, {} head)",
                idx % variants.len(), backbone, leaf, head
            ),
        })
    }
}

impl ConversationBrain for StubProposerBrain {
    fn name(&self) -> &str {
        "brain.stub.owmr"
    }

    fn think_conversation(
        &self,
        request: ConversationBrainRequest,
    ) -> Result<ConversationBrainResponse, BrainError> {
        let action = EvaluatePolicySpecAction::name();

        // Second step: the evaluation tool result is already in the transcript -> narrate + finish.
        if let Some(tool_result) = request.transcript.iter().rev().find(|item| {
            item.kind == TranscriptItemKind::ToolResult && item.action.as_ref() == Some(&action)
        }) {
            let deployed = tool_result
                .payload
                .get("deployed_cost")
                .and_then(Value::as_f64);
            let robust = tool_result
                .payload
                .get("robust_gate_beat")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let summary = match deployed {
                Some(d) => format!(
                    "evaluated spec: deployed_cost={d:.4}, robust_gate_beat={robust}"
                ),
                None => format!(
                    "evaluation returned no deployed_cost; raw result={}",
                    tool_result.payload
                ),
            };
            return Ok(ConversationBrainResponse {
                assistant_message: Some(summary),
                action_requests: Vec::new(),
                status: ConversationTurnStatus::Completed,
                meta: json!({ "brain": "stub", "phase": "narrate" }),
            });
        }

        // First step: choose the variant from the injected generation index and request evaluation.
        if !request.resolved_available_actions().contains(&action) {
            return Err(BrainError::Rejected(format!(
                "required action `{}` is not available",
                action.as_str()
            )));
        }
        let generation = request
            .context
            .get("generation")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let spec = self.variant(generation);

        Ok(ConversationBrainResponse {
            assistant_message: Some(format!(
                "proposing OWMR spec for generation {generation}"
            )),
            action_requests: vec![ActionRequest {
                action,
                input: spec,
            }],
            status: ConversationTurnStatus::NeedsToolResults,
            meta: json!({ "brain": "stub", "phase": "propose", "generation": generation }),
        })
    }
}
