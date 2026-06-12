// policy_search/agentic :: agent :: main.rs
// =================================================================================================
// ALGORITHMIC DESCRIPTION  (the outer evolution loop, on beden)
// -------------------------------------------------------------------------------------------------
// This binary is the AlphaEvolve/ShinkaEvolve-style outer loop from the README, implemented as a
// `beden` agent. Codex (the brain) proposes ONE policy-spec per generation; the spec is scored by
// the Python rollout oracle through the default-deny omurga gate; the result is appended to an
// archive ranked by the honest deploy floor; the best-so-far is logged. The keep/gate decision is
// deterministic and lives in the oracle (robust_gate_beat + deployed_cost), never in the brain.
//
// CONFIG (CLI flags, all optional; sensible OWMR instance_14 defaults):
//   --problem <id>          default one_warehouse_multi_retailer
//   --instance <n>          default 14
//   --seeds <n>             default 5        (>=5 seed-robust mandate; passed to the oracle)
//   --budget <tier>         default small    (inner CMA-ES budget tier)
//   --generations <G>       default 3        (number of Codex propose->evaluate rounds)
//   --python <bin>          default "python"
//   --oracle <path>         default <base>/evaluate_policy_spec.py
//   --gate-spec <path>      default <base>/specs/gate_anchor_echelon_targets.json
//   --archive <path>        default <base>/agent/archive.jsonl   (git-ignored runtime population)
//   --working-dir <path>    default <base>   (ReadOnly dir Codex runs in)
//   --model <name>          default none     (Codex default model)
//   where <base> = /home/nima/code/ml/invman/policy_search/agentic
//
// PROBLEM-AWARE DEFAULTS (additive; OWMR untouched): when --problem ==
// production_assembly_distribution_network (PADN), any of {--oracle, --gate-spec, --archive,
// --instance, --budget} left at the OWMR default is re-pointed to the PADN counterpart:
//   --oracle    -> <base>/evaluate_policy_spec_padn.py
//   --gate-spec -> <base>/specs/gate_anchor_padn_residual_base_stock.json
//   --archive   -> <base>/agent/archive_padn.jsonl   (separate file: PADN/OWMR niches never mix)
//   --instance  -> 0   (PADN is a single fixed instance; the oracle parses-and-ignores it)
//   --budget    -> the OWMR 'small'/'tiny' tier is MAPPED to PADN 'screening'/'smoke' (the agent
//                  default 'small' is not a PADN budget); a valid PADN budget passes through.
// An explicit flag ALWAYS wins over the problem default. The brain prompt + the niche DSL are also
// routed by problem (see brain.rs::build_brain and actions.rs::dsl_for).
// Env: APS_STUB_BRAIN truthy -> deterministic stub proposer (no Codex), see brain.rs.
//
// WIRING (copied from beden examples/minimal-agents/minimal_tool_agent.rs):
//   gate    := ExecutionGate allowing the CLI source for BOTH custom actions (default-deny otherwise)
//   omurga  := OmurgaPlug::new(gate); register EvaluatePolicySpecAction + ArchiveAction
//   beyin   := BeyinPlug::direct(build_brain(..))     (CodexBrain ReadOnly, or stub)
//   plug    := AgentPlug::new(AgentIdentity(tools_enabled).with_system_prompt(owmr_prompt), omurga, beyin)
//
// ALGORITHM:
//   0. Fail fast: assert the oracle CLI and gate-spec files exist (raw error, no fallback).
//   1. SEED gen-0: read the gate spec, run EvaluatePolicySpecAction directly on it (deterministic,
//      no brain), and ArchiveAction.append_row it as generation 0. This is the heuristic-gate anchor
//      the README requires; doing it directly keeps gen-0 reproducible and saves a Codex call.
//   2. For g in 1..=G:
//        a. ctx := NOVELTY-PRESSURE context built from the FULL archive (the fix for the verified
//           re-proposal plateau where gens 1-4 re-emitted the identical archived-best spec because
//           ctx + prompt were constant each generation). ctx carries:
//             { generation, problem, instance, gate_cost,
//               diverse_elites: archive.diverse_elites(K)  // MAP-Elites: best row per OCCUPIED
//                                                           // structural niche (action_head|leaf|
//                                                           // split), best-first; replaces the ~5
//                                                           // near-duplicate top_k parents with
//                                                           // structurally-DISTINCT ones,
//               best_signature: structural_signature(archive best),  // referent for "vary >=1 axis",
//               tried_signatures: archive.tried_signatures(), // every DISTINCT structure already
//                                                             // tried + its best deployed_cost +
//                                                             // robust_gate_beat + times_tried,
//               untried_niches: archive.untried_niches(cap),  // concrete unexplored DSL cells,
//               n_occupied_niches / n_total_niches }           // coverage signal.
//           The prompt (brain.rs) turns this into a HARD rule: never re-propose a tried_signature,
//           and differ from best_signature on >=1 structural axis (prefer an untried_niche).
//        b. run_turn(brain, session, source, user_msg, ctx)  // user_msg restates the constraint
//             -> brain emits ONE invman.evaluate_policy_spec request
//             -> runtime gates+runs it; tool result lands in the session transcript
//             -> brain second step narrates + completes
//        c. Pull the latest evaluate tool result + the spec that produced it from the transcript,
//           and append {spec, result} to the archive as generation g.
//        d. Update best_deployed := min(best_deployed, result.deployed_cost); log the line.
//   3. Print the best spec, its result, and a robust-gate-beat summary over the whole archive.
//
// NOVELTY-PRESSURE NOTE: diverse_elites / tried_signatures / best_signature / untried_niches are
// PURE READS over the canonical archive (no new persisted state, archive.jsonl schema unchanged).
// The honest-metric path (robust_gate_beat, deployed_cost=min(trained,gate)) and the PPO-is-cross-
// protocol rule are untouched; top_k still backs the gen-0 anchor and the summary. The change is
// context+prompt only: it APPLIES novelty pressure, it does not hard-reject a duplicate at eval
// time (the one-Codex-call/one-eval-per-generation contract is preserved).
//
// NO SILENT FALLBACKS: missing files, a turn that produced no evaluate result, or a result without
// deployed_cost all abort the generation with an explicit message; the loop continues to the next
// generation (recording the failure) rather than fabricating a score.
// =================================================================================================

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use beden_core::{SessionId, SourceId};
use beden_plug::{AgentPlug, BeyinPlug, OmurgaPlug};
use omurga_action::{Action, ActionExecutionRequest};
use omurga_agent::TranscriptItemKind;
use omurga_execution_gate::{ExecutionGate, ExecutionRule};
use omurga_runtime::ConversationRuntimeConfig;
use oz::AgentIdentity;
use serde_json::{Value, json};

mod actions;
mod brain;

use actions::{ArchiveAction, EvaluatePolicySpecAction};
use brain::{build_brain, system_prompt_for};

const BASE_DIR: &str = "/home/nima/code/ml/invman/policy_search/agentic";

/// Max number of untried structural niches to list in the per-generation context (bounds context
/// bloat; the niche space is only 16 cells, so this is a soft cap that rarely binds).
const UNTRIED_NICHE_CAP: usize = 8;

/// Minimal config resolved from CLI args + defaults.
struct Config {
    problem: String,
    instance: u32,
    seeds: u32,
    budget: String,
    generations: u32,
    python_bin: String,
    oracle_cli: PathBuf,
    gate_spec: PathBuf,
    archive: PathBuf,
    working_dir: PathBuf,
    model: Option<String>,
    top_k: usize,
}

/// The PADN problem id. When --problem is this, the oracle / gate-spec / archive / budget defaults
/// are routed to the PADN counterparts (deliverable 3). OWMR defaults are kept for every other
/// problem (byte-for-byte unchanged).
const PADN_PROBLEM: &str = "production_assembly_distribution_network";
/// Valid PADN budget tiers (must match evaluate_policy_spec_padn.py's BUDGETS).
const PADN_BUDGETS: &[&str] = &["smoke", "screening", "full"];

impl Config {
    fn from_args() -> Result<Self, String> {
        let base = PathBuf::from(BASE_DIR);
        let mut cfg = Config {
            problem: "one_warehouse_multi_retailer".to_string(),
            instance: 14,
            seeds: 5,
            budget: "small".to_string(),
            generations: 3,
            python_bin: "python".to_string(),
            oracle_cli: base.join("evaluate_policy_spec.py"),
            gate_spec: base.join("specs/gate_anchor_echelon_targets.json"),
            archive: base.join("agent/archive.jsonl"),
            working_dir: base.clone(),
            model: None,
            top_k: 5,
        };

        // Track which path/budget/instance flags the user EXPLICITLY set, so problem-aware defaults
        // only fill in the ones left at the OWMR default (an explicit flag always wins).
        let mut set_oracle = false;
        let mut set_gate_spec = false;
        let mut set_archive = false;
        let mut set_budget = false;
        let mut set_instance = false;

        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let mut take = || {
                args.next()
                    .ok_or_else(|| format!("flag `{flag}` requires a value"))
            };
            match flag.as_str() {
                "--problem" => cfg.problem = take()?,
                "--instance" => {
                    cfg.instance = take()?
                        .parse()
                        .map_err(|e| format!("--instance must be an integer: {e}"))?;
                    set_instance = true;
                }
                "--seeds" => {
                    cfg.seeds = take()?
                        .parse()
                        .map_err(|e| format!("--seeds must be an integer: {e}"))?
                }
                "--budget" => {
                    cfg.budget = take()?;
                    set_budget = true;
                }
                "--generations" => {
                    cfg.generations = take()?
                        .parse()
                        .map_err(|e| format!("--generations must be an integer: {e}"))?
                }
                "--python" => cfg.python_bin = take()?,
                "--oracle" => {
                    cfg.oracle_cli = PathBuf::from(take()?);
                    set_oracle = true;
                }
                "--gate-spec" => {
                    cfg.gate_spec = PathBuf::from(take()?);
                    set_gate_spec = true;
                }
                "--archive" => {
                    cfg.archive = PathBuf::from(take()?);
                    set_archive = true;
                }
                "--working-dir" => cfg.working_dir = PathBuf::from(take()?),
                "--model" => cfg.model = Some(take()?),
                "--top-k" => {
                    cfg.top_k = take()?
                        .parse()
                        .map_err(|e| format!("--top-k must be an integer: {e}"))?
                }
                "-h" | "--help" => return Err("help".to_string()),
                other => return Err(format!("unknown flag `{other}`")),
            }
        }

        // PROBLEM-AWARE DEFAULTS (deliverable 3, additive): for PADN, route the oracle / gate-spec /
        // archive / budget / instance to the PADN counterparts UNLESS the user set them explicitly.
        // This keeps the PADN archive separate from the OWMR archive (so niches never mix) and
        // ensures a valid PADN --budget is passed (the agent default 'small' is OWMR's tier and is
        // not a PADN budget). OWMR is untouched.
        if cfg.problem == PADN_PROBLEM {
            if !set_oracle {
                cfg.oracle_cli = base.join("evaluate_policy_spec_padn.py");
            }
            if !set_gate_spec {
                cfg.gate_spec = base.join("specs/gate_anchor_padn_residual_base_stock.json");
            }
            if !set_archive {
                cfg.archive = base.join("agent/archive_padn.jsonl");
            }
            if !set_instance {
                // PADN is a single fixed instance; the oracle parses-and-ignores --instance, but
                // default it to 0 (the mixed SCN) rather than OWMR's 14 for honest provenance.
                cfg.instance = 0;
            }
            if !set_budget {
                // Map the OWMR default 'small' to PADN's 'screening' (the comparable mid tier).
                cfg.budget = "screening".to_string();
            } else {
                // Map any OWMR tier the caller passed onto the nearest PADN tier; pass a valid PADN
                // budget through unchanged. Reject an unmappable tier loudly (no silent fallback).
                cfg.budget = match cfg.budget.as_str() {
                    "tiny" => "smoke".to_string(),
                    "small" => "screening".to_string(),
                    b if PADN_BUDGETS.contains(&b) => b.to_string(),
                    other => {
                        return Err(format!(
                            "--budget={other:?} is not a valid PADN budget (expected one of {} or an \
                             OWMR tier tiny/small to map)",
                            PADN_BUDGETS.join("/")
                        ));
                    }
                };
            }
        }

        if cfg.seeds < 5 {
            return Err(format!(
                "--seeds={} violates the >=5-seed seed-robust mandate",
                cfg.seeds
            ));
        }
        Ok(cfg)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[agentic-policy-search-agent] FATAL: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let cfg = Config::from_args().map_err(|e| {
        if e == "help" {
            print_help();
        }
        e
    })?;

    // 0. Fail fast on missing inputs (no fallback).
    if !actions::is_file(&cfg.oracle_cli) {
        return Err(format!("oracle CLI not found at {:?}", cfg.oracle_cli));
    }
    if !actions::is_file(&cfg.gate_spec) {
        return Err(format!("gate spec not found at {:?}", cfg.gate_spec));
    }

    eprintln!(
        "[agentic-policy-search-agent] problem={} instance={} seeds={} budget={} generations={}",
        cfg.problem, cfg.instance, cfg.seeds, cfg.budget, cfg.generations
    );

    // Build the two actions (the evaluate action is also used directly to seed gen-0).
    let evaluate = EvaluatePolicySpecAction {
        python_bin: cfg.python_bin.clone(),
        oracle_cli: cfg.oracle_cli.clone(),
        problem: cfg.problem.clone(),
        instance: cfg.instance,
        seeds: cfg.seeds,
        budget: cfg.budget.clone(),
    };
    let archive = ArchiveAction {
        archive_path: cfg.archive.clone(),
        problem: cfg.problem.clone(),
    };

    // The CLI source that drives the loop; the gate allows ONLY this source for our two actions.
    let source = SourceId::new("surface.cli").map_err(|e| e.to_string())?;
    let gate = ExecutionGate::new()
        .with_rule(ExecutionRule::allow_source_for(
            source.clone(),
            EvaluatePolicySpecAction::name(),
        ))
        .with_rule(ExecutionRule::allow_source_for(
            source.clone(),
            ArchiveAction::name(),
        ));

    // One brain step per generation: Codex proposes exactly ONE spec, the evaluate action runs
    // once, then the turn stops. The `codex exec` brain is itself autonomous and will keep
    // re-proposing+re-evaluating if looped (double-agency), so we cap at a single
    // proposal+evaluation per generation and salvage the result from the transcript (run_turn then
    // returns the step-limit error BY DESIGN). Cross-generation memory is carried by the archive
    // injected into the brain context, not by the conversation transcript.
    let mut omurga = OmurgaPlug::new(gate).with_runtime_config(ConversationRuntimeConfig {
        max_brain_steps: 1,
        ..Default::default()
    });
    omurga
        .register_action(EvaluatePolicySpecAction {
            python_bin: cfg.python_bin.clone(),
            oracle_cli: cfg.oracle_cli.clone(),
            problem: cfg.problem.clone(),
            instance: cfg.instance,
            seeds: cfg.seeds,
            budget: cfg.budget.clone(),
        })
        .map_err(|e| format!("failed to register evaluate action: {e}"))?;
    omurga
        .register_action(ArchiveAction {
            archive_path: cfg.archive.clone(),
            problem: cfg.problem.clone(),
        })
        .map_err(|e| format!("failed to register archive action: {e}"))?;

    let brain = build_brain(
        &cfg.problem,
        cfg.instance,
        cfg.working_dir.clone(),
        cfg.model.clone(),
    );

    let identity = AgentIdentity::new("policy_search/agentic", "Agentic Policy Search")
        .with_description("OWMR policy-structure proposer (Codex brain) scored by the invman oracle")
        .with_tools_enabled(true)
        .with_system_prompt(system_prompt_for(&cfg.problem, cfg.instance));

    let plug = AgentPlug::new(identity, omurga, BeyinPlug::direct(Arc::clone(&brain)));

    // 1. SEED gen-0 with the gate spec, evaluated directly (deterministic, no brain).
    let gate_spec: Value = {
        let text = std::fs::read_to_string(&cfg.gate_spec)
            .map_err(|e| format!("failed to read gate spec {:?}: {e}", cfg.gate_spec))?;
        serde_json::from_str(&text)
            .map_err(|e| format!("gate spec {:?} is not valid JSON: {e}", cfg.gate_spec))?
    };

    eprintln!("[gen 0] seeding archive with the GATE anchor spec...");
    let gen0_session = SessionId::new("policy_search/agentic:gen0").map_err(|e| e.to_string())?;
    let gate_result = evaluate
        .execute(ActionExecutionRequest {
            session_id: gen0_session,
            source: source.clone(),
            input: gate_spec.clone(),
        })
        .map_err(|e| format!("gate anchor evaluation failed: {e}"))?;
    archive
        .append_row(gate_spec.clone(), gate_result.clone(), 0)
        .map_err(|e| format!("failed to archive gen-0: {e}"))?;

    let mut best_deployed = gate_result
        .get("deployed_cost")
        .and_then(Value::as_f64)
        .ok_or_else(|| "gate anchor result missing deployed_cost".to_string())?;
    let gate_cost = gate_result.get("gate_cost").and_then(Value::as_f64);
    let mut best_spec = gate_spec.clone();
    let mut best_result = gate_result.clone();
    eprintln!("[gen 0] gate anchor deployed_cost={best_deployed:.4} gate_cost={gate_cost:?}");

    // 2. Generations 1..=G: brain proposes -> evaluate runs -> archive -> log best.
    // Each generation uses a FRESH session: cross-generation memory is carried by the archive
    // injected into the brain context (the novelty-pressure payload: diverse_elites +
    // tried_signatures + best_signature + untried_niches), so a fresh session avoids stale-result
    // contamination and makes the post-turn salvage unambiguous.
    let mut runtime = plug.conversation_runtime();
    let brain_ref = plug.conversation_brain();

    for g in 1..=cfg.generations {
        // NOVELTY-PRESSURE context, all derived from the FULL canonical archive (pure reads):
        //   diverse_elites  = best row per OCCUPIED structural niche (structurally-distinct parents),
        //   tried_signatures = every DISTINCT structure already tried + its best cost / robustness,
        //   best_signature   = the referent the proposal must differ from on >=1 axis,
        //   untried_niches   = concrete unexplored DSL cells (+ coverage counts).
        let diverse_elites = archive
            .diverse_elites(cfg.top_k)
            .map_err(|e| format!("failed to compute diverse elites: {e}"))?;
        let tried = archive
            .tried_signatures()
            .map_err(|e| format!("failed to compute tried signatures: {e}"))?;
        let best_signature = archive
            .best_signature()
            .map_err(|e| format!("failed to compute best signature: {e}"))?
            .unwrap_or_else(|| "none".to_string());
        let (untried_niches, n_occupied, n_total) = archive
            .untried_niches(UNTRIED_NICHE_CAP)
            .map_err(|e| format!("failed to compute untried niches: {e}"))?;

        let ctx = json!({
            "generation": g,
            "problem": cfg.problem,
            "instance": cfg.instance,
            "gate_cost": gate_cost,
            "diverse_elites": diverse_elites,
            "best_signature": best_signature,
            "tried_signatures": tried,
            "untried_niches": untried_niches,
            "n_occupied_niches": n_occupied,
            "n_total_niches": n_total,
        });

        eprintln!(
            "[gen {g}] proposing ({} diverse elites, {} tried signatures, {}/{} niches filled)...",
            ctx["diverse_elites"].as_array().map(|a| a.len()).unwrap_or(0),
            ctx["tried_signatures"].as_array().map(|a| a.len()).unwrap_or(0),
            n_occupied,
            n_total,
        );

        let sid = format!("policy_search/agentic:gen{g}");
        let mut session = plug
            .new_session(SessionId::new(&sid).map_err(|e| e.to_string())?)
            .map_err(|e| format!("failed to open session for gen {g}: {e}"))?;

        // With max_brain_steps=1 the run_turn returns the step-limit error BY DESIGN after the
        // single proposal+evaluation; that is expected here, not a failure. Either way we salvage
        // the evaluate result the action produced this turn from the session transcript.
        match runtime.run_turn(
            brain_ref.as_ref(),
            &mut session,
            source.clone(),
            format!(
                "Propose policy-spec for generation {g} and evaluate it. Do NOT repeat any signature \
                 in tried_signatures ({} already tried); your spec MUST differ from best_signature \
                 ({}) on at least one structural axis. Coverage: {}/{} structural niches filled — \
                 prefer an unfilled cell from untried_niches.",
                tried.len(),
                best_signature,
                n_occupied,
                n_total,
            ),
            ctx,
        ) {
            Ok(t) => eprintln!(
                "[gen {g}] turn completed cleanly: stop_reason={:?} steps={}",
                t.stop_reason, t.steps
            ),
            Err(e) => eprintln!("[gen {g}] turn ended at the single-proposal cap (expected): {e}"),
        }

        // Pull the latest evaluate tool result + the spec (tool call input) from the transcript.
        let (spec, result) = match latest_eval_from_transcript(&session) {
            Some(pair) => pair,
            None => {
                eprintln!("[gen {g}] no evaluate result produced this turn (recorded as miss)");
                continue;
            }
        };

        archive
            .append_row(spec.clone(), result.clone(), g as u64)
            .map_err(|e| format!("failed to archive gen {g}: {e}"))?;

        match result.get("deployed_cost").and_then(Value::as_f64) {
            Some(deployed) => {
                let robust = result
                    .get("robust_gate_beat")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if deployed < best_deployed {
                    best_deployed = deployed;
                    best_spec = spec.clone();
                    best_result = result.clone();
                }
                eprintln!(
                    "[gen {g}] deployed_cost={deployed:.4} robust_gate_beat={robust} best_so_far={best_deployed:.4}"
                );
            }
            None => {
                eprintln!(
                    "[gen {g}] result missing deployed_cost (compiled_ok={:?}, error={:?})",
                    result.get("compiled_ok"),
                    result.get("error")
                );
            }
        }
    }

    // 3. Summary.
    print_summary(&cfg, &best_spec, &best_result, best_deployed, gate_cost)?;
    Ok(())
}

/// Find, in the most recent transcript items, the latest `invman.evaluate_policy_spec` ToolResult
/// and the ToolCall input (the spec) that produced it. Returns (spec, result) or None.
fn latest_eval_from_transcript(session: &omurga_agent::AgentSession) -> Option<(Value, Value)> {
    let action = EvaluatePolicySpecAction::name();
    let items = session.transcript().items();
    // Latest ToolResult for our action.
    let result_idx = items.iter().rposition(|it| {
        it.kind == TranscriptItemKind::ToolResult && it.action.as_ref() == Some(&action)
    })?;
    let result = items[result_idx].payload.clone();
    let call_id = items[result_idx].call_id.clone();

    // The matching ToolCall: same call_id if available, else the nearest preceding ToolCall.
    let spec = items[..result_idx]
        .iter()
        .rev()
        .find(|it| {
            it.kind == TranscriptItemKind::ToolCall
                && it.action.as_ref() == Some(&action)
                && (call_id.is_none() || it.call_id == call_id)
        })
        .map(|it| it.payload.clone())
        // Normalize: the evaluate action accepts a bare DSL or {spec:..}; store the inner spec.
        .map(|input| {
            input
                .get("spec")
                .filter(|s| s.is_object())
                .cloned()
                .unwrap_or(input)
        })
        .unwrap_or(Value::Null);

    Some((spec, result))
}

fn print_summary(
    cfg: &Config,
    best_spec: &Value,
    best_result: &Value,
    best_deployed: f64,
    gate_cost: Option<f64>,
) -> Result<(), String> {
    let archive = ArchiveAction {
        archive_path: cfg.archive.clone(),
        problem: cfg.problem.clone(),
    };
    let all = archive
        .top_k(usize::MAX)
        .map_err(|e| format!("failed to read archive for summary: {e}"))?;
    let n_rows = all.len();
    let robust_beats = all
        .iter()
        .filter(|row| {
            row.get("result")
                .and_then(|r| r.get("robust_gate_beat"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .count();

    println!("================ policy_search/agentic :: summary ================");
    println!(
        "problem={} instance={} seeds={} budget={} generations={}",
        cfg.problem, cfg.instance, cfg.seeds, cfg.budget, cfg.generations
    );
    println!("archive: {:?} ({n_rows} rows)", cfg.archive);
    println!("gate_cost: {gate_cost:?}");
    println!("best deployed_cost: {best_deployed:.4}");
    println!("robust gate-beats in archive: {robust_beats}/{n_rows}");
    println!(
        "best result: {}",
        serde_json::to_string(best_result).unwrap_or_else(|_| "<unserializable>".to_string())
    );
    println!(
        "best spec:\n{}",
        serde_json::to_string_pretty(best_spec).unwrap_or_else(|_| "<unserializable>".to_string())
    );
    println!("=================================================================");
    Ok(())
}

fn print_help() {
    eprintln!(
        "agentic-policy-search-agent — OWMR policy-structure search (beden + Codex)\n\
\n\
Usage: agentic-policy-search-agent [flags]\n\
  --problem <id>        default one_warehouse_multi_retailer\n\
  --instance <n>        default 14\n\
  --seeds <n>           default 5  (>=5 mandate)\n\
  --budget <tier>       default small\n\
  --generations <G>     default 3\n\
  --python <bin>        default python\n\
  --oracle <path>       default <base>/evaluate_policy_spec.py\n\
  --gate-spec <path>    default <base>/specs/gate_anchor_echelon_targets.json\n\
  --archive <path>      default <base>/agent/archive.jsonl\n\
  --working-dir <path>  default <base>\n\
  --model <name>        default (Codex default)\n\
  --top-k <k>           default 5\n\
\n\
Env: APS_STUB_BRAIN=1 uses the deterministic stub proposer (no Codex calls).\n"
    );
}
