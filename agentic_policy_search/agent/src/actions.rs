// agentic_policy_search :: agent :: actions.rs
// =================================================================================================
// ALGORITHMIC DESCRIPTION
// -------------------------------------------------------------------------------------------------
// This module defines the two `omurga` Actions that are the agent's ONLY side-effect capabilities.
// Everything Codex wants to make happen in the world must travel through these, behind the
// default-deny execution gate (wired in main.rs), so the whole search is auditable in the
// transcript. There are no silent fallbacks: any failure becomes an explicit, raw-error tool
// result that the loop records and the brain can read back as evidence.
//
// ACTION 1 — EvaluatePolicySpecAction  (name: "invman.evaluate_policy_spec")
//   Purpose: score ONE policy-spec (the README DSL JSON) under the seed-robust gate, by delegating
//   to the Python rollout oracle. It never evaluates anything itself.
//   Algorithm:
//     1. Read the policy-spec object from the action input. The brain may pass it either as
//        input.spec (an object) or as the input object itself (when it emits the DSL directly).
//        Unknown/missing spec -> ActionError::Execution with the raw reason (no coercion).
//     2. Serialize the spec to a uniquely-named temp file under the OS temp dir
//        (aps_spec_<nanos>_<counter>.json). We write a file rather than piping so the oracle CLI's
//        --spec <path> contract is honoured exactly and the artifact is inspectable on failure.
//     3. Shell out:  python <oracle_cli> --spec <tmp> --problem <problem> --instance <instance>
//                                         --seeds <seeds> --budget <budget>
//        The oracle path, problem, instance, seeds, budget are all fields of the action (set once
//        at construction from CLI/env in main.rs) so the Codex-supplied JSON cannot redirect the
//        subprocess — the brain proposes structure, the harness fixes the evaluation protocol.
//     4. Require exit status success. On non-zero exit, return ActionError::Execution carrying
//        BOTH stdout and stderr verbatim (the oracle is contractually required to print a JSON
//        result with compiled_ok=false + error even on a spec that fails to compile; a non-zero
//        exit therefore means an *infrastructure* failure, which we surface raw, not swallow).
//     5. Parse stdout as JSON (the evaluate I/O contract object). Parse failure -> Execution error
//        with the raw stdout/stderr so the brain/operator can see exactly what the oracle emitted.
//     6. Return the parsed result Value unchanged as the tool result. The temp file is removed on
//        the success path; on error it is intentionally left for post-mortem inspection.
//
// ACTION 2 — ArchiveAction  (name: "invman.archive")
//   Purpose: the population store for the AlphaEvolve-style archive. Two operations, selected by
//   input.op:
//     * op = "append": append one JSONL row {spec, result, generation, ts_ms} to archive.jsonl
//       (create parent dir + file if absent). Returns {appended:true, path, rows_after}.
//     * op = "top_k": read archive.jsonl, keep rows whose result is present, sort ascending by the
//       honest ranking key result.deployed_cost (rows missing the key sort last), and return the
//       best K as {top_k:[...], n_rows}. K defaults to 5, overridable via input.k.
//   Ranking key rationale: deployed_cost = min(trained, gate) is the README's honest deploy floor,
//   so ranking by it never rewards a lucky-seed trained policy that the gate floor would override.
//   I/O errors (bad path, malformed JSONL line) are returned as ActionError::Execution with the
//   raw error + offending line; nothing is silently skipped except blank lines.
//
//   NOVELTY-PRESSURE READ HELPERS (added to break the re-proposal plateau; pure reads, no new
//   persisted state, archive.jsonl schema unchanged):
//     * structural_signature(spec) -> canonical key string over the DSL's STRUCTURAL axes only
//       (action_head | leaf_type | split_type | depth | per_retailer_targets | features-sorted |
//       backbone | warm_start). Temperature and continuous params are EXCLUDED on purpose — the
//       inner CMA-ES owns those, so "same structure" ignores them. Features are lowercased + SORTED
//       so feature ORDER never fabricates a false-new signature. depth is "na" for linear backbones.
//     * tried_signatures() -> for EVERY distinct signature seen anywhere in the FULL archive, the
//       best (min) deployed_cost, an OR-reduce of robust_gate_beat, and times_tried. This is the
//       explicit "what has already been tried" memory the loop was missing.
//     * diverse_elites(k) -> the single BEST (min deployed_cost) row per OCCUPIED structural niche
//       (niche = action_head|leaf_type|split_type), best-first, capped at k. This is the MAP-Elites
//       quality-diversity parent set: it turns the ~5 near-duplicate top_k rows (all collapsing to
//       one niche) into structurally-DISTINCT parents to recombine across.
//     * untried_niches(cap) -> niches in the finite (action_head x leaf_type x split_type) DSL
//       cross-product that have NO archived row yet (concrete exploration targets), with the total
//       niche count for a coverage signal.
//   top_k, append_row, deployed_cost_of, read_rows, and the row schema are UNCHANGED; the helpers
//   are pure read-side derivations recomputed from the canonical archive each generation, so they
//   never diverge from what append_row wrote.
//
// Both actions are declared read_only=false / Exclusive so the runtime never parallelizes them
// (evaluation is a heavyweight subprocess; archive append must be serialized).
// =================================================================================================

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use beden_core::ActionName;
use omurga_action::{Action, ActionDescriptor, ActionError, ActionExecutionRequest};
use serde_json::{Map, Value, json};

// =================================================================================================
// DSL STRUCTURAL ENUMERATIONS  (single source of truth = README.md "Policy-spec DSL" section,
// lines ~95-104). These mirror the README contract so untried_niches can enumerate the finite
// (action_head x leaf_type x split_type) cross-product. If the README DSL gains an axis value,
// these MUST be updated in lockstep (documented coupling; the README is the contract).
// =================================================================================================

/// All `action_head` enum values from the README DSL.
pub const ACTION_HEAD_DOMAIN: &[&str] = &[
    "echelon_targets",
    "symmetric_echelon_targets",
    "echelon_targets_with_alloc_targets",
    "direct_orders",
];
/// All `leaf_type` enum values from the README DSL.
pub const LEAF_TYPE_DOMAIN: &[&str] = &["constant", "linear"];
/// All `split_type` enum values from the README DSL.
pub const SPLIT_TYPE_DOMAIN: &[&str] = &["oblique", "axis_aligned"];

// The MAP-Elites NICHE key is the 3 axes action_head x leaf_type x split_type (a small,
// low-cardinality descriptor chosen because the live archive shows these collapse to one cell).
// Depth, features, per_retailer, backbone, warm_start are part of the finer SIGNATURE but not the
// niche key, to keep the cell count small (4 x 2 x 2 = 16) without every spec becoming its own
// niche. The niche key is built inline by `niche_key` / `untried_niches`.

/// Read a spec string field, returning a "?" placeholder (never a panic) when absent/non-string.
fn spec_str(spec: &Value, key: &str) -> String {
    spec.get(key)
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_ascii_lowercase()
}

/// The canonical STRUCTURAL signature of a policy-spec: a pipe-joined key over the discrete
/// structural axes ONLY (temperature / continuous params excluded — CMA-ES owns those). Features
/// are lowercased + sorted so order never fabricates a false-new signature; depth is "na" for the
/// linear backbone (no tree depth). Pure, no I/O; reads with explicit placeholders, never panics.
pub fn structural_signature(spec: &Value) -> String {
    let backbone = spec_str(spec, "backbone");
    let depth = if backbone == "linear" {
        "na".to_string()
    } else {
        spec.get("depth")
            .and_then(Value::as_u64)
            .map(|d| d.to_string())
            .unwrap_or_else(|| "?".to_string())
    };
    let action_head = spec_str(spec, "action_head");
    let leaf_type = spec_str(spec, "leaf_type");
    let split_type = spec_str(spec, "split_type");
    let per_retailer = spec
        .get("per_retailer_targets")
        .and_then(Value::as_bool)
        .map(|b| b.to_string())
        .unwrap_or_else(|| "?".to_string());
    let warm_start = spec_str(spec, "warm_start");
    let mut features: Vec<String> = spec
        .get("features")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(|s| s.to_ascii_lowercase())
                .collect()
        })
        .unwrap_or_default();
    features.sort();
    let features = if features.is_empty() {
        "none".to_string()
    } else {
        features.join(",")
    };
    format!(
        "head={action_head}|leaf={leaf_type}|split={split_type}|backbone={backbone}|depth={depth}|per_retailer={per_retailer}|features={features}|warm_start={warm_start}"
    )
}

/// The NICHE key (a 3-axis subset of the full signature) used for MAP-Elites bucketing: just
/// action_head|leaf_type|split_type. Returns the pipe-joined cell key.
fn niche_key(spec: &Value) -> String {
    format!(
        "head={}|leaf={}|split={}",
        spec_str(spec, "action_head"),
        spec_str(spec, "leaf_type"),
        spec_str(spec, "split_type"),
    )
}

/// A niche descriptor object {action_head, leaf_type, split_type} for the prompt/context.
fn niche_descriptor(action_head: &str, leaf_type: &str, split_type: &str) -> Value {
    json!({
        "action_head": action_head,
        "leaf_type": leaf_type,
        "split_type": split_type,
    })
}

/// Monotonic counter to disambiguate temp spec filenames written within the same nanosecond.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Current wall-clock time in milliseconds since the UNIX epoch (best-effort; 0 if the clock is
/// before the epoch, which we surface as a benign timestamp rather than an error).
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// =================================================================================================
// EvaluatePolicySpecAction
// =================================================================================================

/// Shells out to the Python rollout oracle CLI to score one policy-spec under the seed-robust gate.
///
/// All evaluation-protocol parameters are fixed at construction time so a Codex-supplied spec can
/// only influence the *structure* being scored, never the subprocess invocation.
pub struct EvaluatePolicySpecAction {
    /// Path to the python interpreter (e.g. "python" or an absolute venv python).
    pub python_bin: String,
    /// Absolute path to evaluate_policy_spec.py (the oracle CLI).
    pub oracle_cli: PathBuf,
    /// invman problem id, e.g. "one_warehouse_multi_retailer".
    pub problem: String,
    /// Instance index, e.g. 14.
    pub instance: u32,
    /// Number of paired-CRN seeds (>=5 per the honest gate-beat metric).
    pub seeds: u32,
    /// Inner CMA-ES budget tier, e.g. "small".
    pub budget: String,
}

impl EvaluatePolicySpecAction {
    pub fn name() -> ActionName {
        ActionName::new("invman.evaluate_policy_spec").expect("hardcoded action name is valid")
    }

    /// Extract the policy-spec object the brain wants scored. Accepts either `{ "spec": {..} }`
    /// (the explicit wrapper) or a bare DSL object (the spec inlined as the whole input). Anything
    /// that is not an object is an explicit error — no silent coercion.
    fn extract_spec(input: &Value) -> Result<Value, ActionError> {
        if let Some(spec) = input.get("spec") {
            if spec.is_object() {
                return Ok(spec.clone());
            }
            return Err(ActionError::Execution(format!(
                "`spec` field must be a JSON object, got: {spec}"
            )));
        }
        if input.is_object() {
            // Treat the whole input as the DSL spec (the common Codex path).
            return Ok(input.clone());
        }
        Err(ActionError::Execution(format!(
            "evaluate input must be a JSON object (the policy-spec DSL) or contain a `spec` object; got: {input}"
        )))
    }

    /// Write the spec to a unique temp file and return its path.
    fn write_spec_tempfile(spec: &Value) -> Result<PathBuf, ActionError> {
        let pretty = serde_json::to_string_pretty(spec)
            .map_err(|e| ActionError::Execution(format!("failed to serialize spec to JSON: {e}")))?;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!("aps_spec_{nanos}_{counter}.json"));
        let mut file = fs::File::create(&path).map_err(|e| {
            ActionError::Execution(format!("failed to create temp spec file {path:?}: {e}"))
        })?;
        file.write_all(pretty.as_bytes()).map_err(|e| {
            ActionError::Execution(format!("failed to write temp spec file {path:?}: {e}"))
        })?;
        Ok(path)
    }
}

impl Action for EvaluatePolicySpecAction {
    fn descriptor(&self) -> ActionDescriptor {
        ActionDescriptor::new(
            Self::name(),
            "1.0.0",
            "Score one policy-spec (DSL JSON) under the seed-robust gate via the Python rollout \
             oracle CLI. Pass the policy-spec DSL object directly as the input, or wrap it as \
             {\"spec\": {..}}. Returns the evaluate I/O contract object (compiled_ok, mean_cost, \
             std_cost, per_seed, gate_cost, gate_gap_pct, n_seeds_below_gate, deployed_cost, \
             robust_gate_beat, error).",
        )
        .with_input_schema(json!({
            "type": "object",
            "description": "The policy-spec DSL object to evaluate (see agentic_policy_search README DSL).",
            "properties": {
                "spec": {
                    "type": "object",
                    "description": "Optional explicit wrapper; if omitted the whole input is treated as the spec."
                }
            }
        }))
        .with_output_schema(json!({
            "type": "object",
            "properties": {
                "compiled_ok": { "type": "boolean" },
                "mean_cost": { "type": "number" },
                "std_cost": { "type": "number" },
                "per_seed": { "type": "array" },
                "n_seeds": { "type": "integer" },
                "gate_cost": { "type": "number" },
                "gate_gap_pct": { "type": "number" },
                "n_seeds_below_gate": { "type": "integer" },
                "deployed_cost": { "type": "number" },
                "robust_gate_beat": { "type": "boolean" },
                "error": { "type": ["string", "null"] }
            }
        }))
        .with_read_only(false)
    }

    fn execute(&self, request: ActionExecutionRequest) -> Result<Value, ActionError> {
        let spec = Self::extract_spec(&request.input)?;
        let spec_path = Self::write_spec_tempfile(&spec)?;

        let instance = self.instance.to_string();
        let seeds = self.seeds.to_string();

        let output = Command::new(&self.python_bin)
            .arg(&self.oracle_cli)
            .arg("--spec")
            .arg(&spec_path)
            .arg("--problem")
            .arg(&self.problem)
            .arg("--instance")
            .arg(&instance)
            .arg("--seeds")
            .arg(&seeds)
            .arg("--budget")
            .arg(&self.budget)
            .output()
            .map_err(|e| {
                ActionError::Execution(format!(
                    "failed to spawn oracle `{} {:?}` (spec left at {:?}): {e}",
                    self.python_bin, self.oracle_cli, spec_path
                ))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            // Non-zero exit == infrastructure failure (the oracle is contractually required to
            // print a JSON result even for a spec that fails to compile). Surface raw, no swallow.
            return Err(ActionError::Execution(format!(
                "oracle exited with status {:?} (spec left at {:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
                output.status.code(),
                spec_path,
                stdout.trim(),
                stderr.trim()
            )));
        }

        let result: Value = serde_json::from_str(stdout.trim()).map_err(|e| {
            ActionError::Execution(format!(
                "oracle stdout was not valid JSON: {e} (spec left at {:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
                spec_path,
                stdout.trim(),
                stderr.trim()
            ))
        })?;

        // Success path: remove the temp artifact (ignore removal errors — they are non-fatal and
        // the result is already in hand).
        let _ = fs::remove_file(&spec_path);

        Ok(result)
    }
}

// =================================================================================================
// ArchiveAction
// =================================================================================================

/// The population store for the evolution archive: append rows, or read the best-K by deployed_cost.
pub struct ArchiveAction {
    /// Absolute path to archive.jsonl.
    pub archive_path: PathBuf,
}

impl ArchiveAction {
    pub fn name() -> ActionName {
        ActionName::new("invman.archive").expect("hardcoded action name is valid")
    }

    /// The honest ranking key: result.deployed_cost (min of trained, gate). Rows missing the key
    /// sort last (treated as +inf) so malformed/failed evaluations never rank above real ones.
    fn deployed_cost_of(row: &Value) -> f64 {
        row.get("result")
            .and_then(|r| r.get("deployed_cost"))
            .and_then(Value::as_f64)
            .unwrap_or(f64::INFINITY)
    }

    /// Read and parse every non-blank line of archive.jsonl into Values. Missing file -> empty.
    fn read_rows(&self) -> Result<Vec<Value>, ActionError> {
        if !self.archive_path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&self.archive_path).map_err(|e| {
            ActionError::Execution(format!(
                "failed to read archive {:?}: {e}",
                self.archive_path
            ))
        })?;
        let mut rows = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let row: Value = serde_json::from_str(trimmed).map_err(|e| {
                ActionError::Execution(format!(
                    "archive {:?} line {} is not valid JSON: {e}; line={}",
                    self.archive_path,
                    i + 1,
                    trimmed
                ))
            })?;
            rows.push(row);
        }
        Ok(rows)
    }

    /// Append one {spec, result, generation, ts_ms} row and return the total row count afterward.
    /// Public so main.rs can seed the archive directly (the gate gen-0 anchor) without round-tripping
    /// through the brain.
    pub fn append_row(
        &self,
        spec: Value,
        result: Value,
        generation: u64,
    ) -> Result<usize, ActionError> {
        if let Some(parent) = self.archive_path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    ActionError::Execution(format!(
                        "failed to create archive parent dir {parent:?}: {e}"
                    ))
                })?;
            }
        }
        let row = json!({
            "spec": spec,
            "result": result,
            "generation": generation,
            "ts_ms": now_ms(),
        });
        let line = serde_json::to_string(&row).map_err(|e| {
            ActionError::Execution(format!("failed to serialize archive row: {e}"))
        })?;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.archive_path)
            .map_err(|e| {
                ActionError::Execution(format!(
                    "failed to open archive {:?} for append: {e}",
                    self.archive_path
                ))
            })?;
        writeln!(file, "{line}").map_err(|e| {
            ActionError::Execution(format!(
                "failed to append to archive {:?}: {e}",
                self.archive_path
            ))
        })?;
        Ok(self.read_rows()?.len())
    }

    /// Return the best K rows by ascending deployed_cost (best first).
    pub fn top_k(&self, k: usize) -> Result<Vec<Value>, ActionError> {
        let mut rows = self.read_rows()?;
        rows.sort_by(|a, b| {
            Self::deployed_cost_of(a)
                .partial_cmp(&Self::deployed_cost_of(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows.truncate(k);
        Ok(rows)
    }

    // =============================================================================================
    // NOVELTY-PRESSURE READ HELPERS (anti-repeat + quality-diversity). All pure reads over the
    // FULL archive; no writes, no schema change. See the file-top header ACTION 2 block.
    // =============================================================================================

    /// The signature of one archive ROW's spec (or "?" if the row has no spec object).
    fn row_signature(row: &Value) -> String {
        row.get("spec")
            .map(structural_signature)
            .unwrap_or_else(|| "?".to_string())
    }

    /// Every DISTINCT structural signature seen anywhere in the FULL archive, each with its best
    /// (min) deployed_cost, an OR-reduce of robust_gate_beat, and times_tried. Deterministic order
    /// (BTreeMap-sorted by signature). This is the explicit "already tried" memory: the prompt
    /// forbids re-proposing any signature in this list.
    pub fn tried_signatures(&self) -> Result<Vec<Value>, ActionError> {
        let rows = self.read_rows()?;
        // signature -> (best_deployed_cost, robust_or_reduce, times_tried)
        let mut agg: BTreeMap<String, (f64, bool, u64)> = BTreeMap::new();
        for row in &rows {
            let sig = Self::row_signature(row);
            let deployed = Self::deployed_cost_of(row);
            let robust = row
                .get("result")
                .and_then(|r| r.get("robust_gate_beat"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let entry = agg.entry(sig).or_insert((f64::INFINITY, false, 0));
            if deployed < entry.0 {
                entry.0 = deployed;
            }
            entry.1 = entry.1 || robust;
            entry.2 += 1;
        }
        let out = agg
            .into_iter()
            .map(|(sig, (best, robust, count))| {
                let best_json = if best.is_finite() {
                    json!(best)
                } else {
                    Value::Null
                };
                json!({
                    "signature": sig,
                    "best_deployed_cost": best_json,
                    "robust_gate_beat": robust,
                    "times_tried": count,
                })
            })
            .collect();
        Ok(out)
    }

    /// The structural_signature of the current archive BEST (lowest deployed_cost) spec, the
    /// referent for the prompt's "vary >=1 axis vs the best" rule. None if the archive is empty.
    pub fn best_signature(&self) -> Result<Option<String>, ActionError> {
        let top = self.top_k(1)?;
        Ok(top
            .first()
            .and_then(|r| r.get("spec"))
            .map(structural_signature))
    }

    /// MAP-Elites quality-diversity parents: the single BEST (min deployed_cost) row per OCCUPIED
    /// structural niche (action_head|leaf_type|split_type), best-first by deployed_cost, capped at
    /// k. Each element is a compact object {niche, spec, deployed_cost, robust_gate_beat,
    /// gate_gap_pct} (NOT the full row) to bound context size. This replaces the ~5 near-duplicate
    /// top_k parents with one structurally-distinct representative per niche.
    pub fn diverse_elites(&self, k: usize) -> Result<Vec<Value>, ActionError> {
        let rows = self.read_rows()?;
        // niche_key -> index of the best row in that niche
        let mut best_per_niche: BTreeMap<String, usize> = BTreeMap::new();
        for (i, row) in rows.iter().enumerate() {
            let spec = match row.get("spec") {
                Some(s) if s.is_object() => s,
                _ => continue, // malformed row -> skip, never panic
            };
            let key = niche_key(spec);
            match best_per_niche.get(&key) {
                Some(&cur) if Self::deployed_cost_of(&rows[cur]) <= Self::deployed_cost_of(row) => {}
                _ => {
                    best_per_niche.insert(key, i);
                }
            }
        }
        let mut elites: Vec<Value> = best_per_niche
            .values()
            .map(|&i| {
                let row = &rows[i];
                let spec = row.get("spec").cloned().unwrap_or(Value::Null);
                let deployed = Self::deployed_cost_of(row);
                let robust = row
                    .get("result")
                    .and_then(|r| r.get("robust_gate_beat"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let gate_gap = row
                    .get("result")
                    .and_then(|r| r.get("gate_gap_pct"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let niche = niche_descriptor(
                    spec.get("action_head").and_then(Value::as_str).unwrap_or("?"),
                    spec.get("leaf_type").and_then(Value::as_str).unwrap_or("?"),
                    spec.get("split_type").and_then(Value::as_str).unwrap_or("?"),
                );
                json!({
                    "niche": niche,
                    "spec": spec,
                    "deployed_cost": if deployed.is_finite() { json!(deployed) } else { Value::Null },
                    "robust_gate_beat": robust,
                    "gate_gap_pct": gate_gap,
                })
            })
            .collect();
        elites.sort_by(|a, b| {
            let da = a.get("deployed_cost").and_then(Value::as_f64).unwrap_or(f64::INFINITY);
            let db = b.get("deployed_cost").and_then(Value::as_f64).unwrap_or(f64::INFINITY);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        elites.truncate(k);
        Ok(elites)
    }

    /// Untried structural niches: cells of the finite (action_head x leaf_type x split_type) DSL
    /// cross-product that have NO archived row yet, capped at `cap`, plus (n_occupied, n_total) for
    /// a coverage signal. These are concrete exploration targets the prompt prefers on novelty.
    pub fn untried_niches(&self, cap: usize) -> Result<(Vec<Value>, usize, usize), ActionError> {
        let rows = self.read_rows()?;
        let mut occupied: BTreeSet<String> = BTreeSet::new();
        for row in &rows {
            if let Some(spec) = row.get("spec").filter(|s| s.is_object()) {
                occupied.insert(niche_key(spec));
            }
        }
        let n_total = ACTION_HEAD_DOMAIN.len() * LEAF_TYPE_DOMAIN.len() * SPLIT_TYPE_DOMAIN.len();
        let mut untried = Vec::new();
        for &head in ACTION_HEAD_DOMAIN {
            for &leaf in LEAF_TYPE_DOMAIN {
                for &split in SPLIT_TYPE_DOMAIN {
                    let key = format!("head={head}|leaf={leaf}|split={split}");
                    if !occupied.contains(&key) {
                        untried.push(niche_descriptor(head, leaf, split));
                    }
                }
            }
        }
        let n_occupied = n_total.saturating_sub(untried.len());
        untried.truncate(cap);
        Ok((untried, n_occupied, n_total))
    }
}

impl Action for ArchiveAction {
    fn descriptor(&self) -> ActionDescriptor {
        ActionDescriptor::new(
            Self::name(),
            "1.0.0",
            "Population archive store. op=\"append\" appends {spec,result} as a JSONL row; \
             op=\"top_k\" returns the best K evaluated rows ranked ascending by \
             result.deployed_cost (the honest deploy floor).",
        )
        .with_input_schema(json!({
            "type": "object",
            "properties": {
                "op": { "type": "string", "enum": ["append", "top_k"] },
                "spec": { "type": "object", "description": "Required for op=append." },
                "result": { "type": "object", "description": "Required for op=append." },
                "generation": { "type": "integer", "description": "Optional for op=append." },
                "k": { "type": "integer", "description": "Top-K size for op=top_k (default 5)." }
            },
            "required": ["op"]
        }))
        .with_output_schema(json!({ "type": "object" }))
        .with_read_only(false)
    }

    fn execute(&self, request: ActionExecutionRequest) -> Result<Value, ActionError> {
        let input: &Map<String, Value> = request.input.as_object().ok_or_else(|| {
            ActionError::Execution(format!(
                "archive input must be a JSON object; got: {}",
                request.input
            ))
        })?;
        let op = input
            .get("op")
            .and_then(Value::as_str)
            .ok_or_else(|| ActionError::Execution("archive input missing string `op`".to_string()))?;

        match op {
            "append" => {
                let spec = input.get("spec").cloned().ok_or_else(|| {
                    ActionError::Execution("archive op=append requires `spec`".to_string())
                })?;
                let result = input.get("result").cloned().ok_or_else(|| {
                    ActionError::Execution("archive op=append requires `result`".to_string())
                })?;
                let generation = input.get("generation").and_then(Value::as_u64).unwrap_or(0);
                let rows_after = self.append_row(spec, result, generation)?;
                Ok(json!({
                    "appended": true,
                    "path": self.archive_path.to_string_lossy(),
                    "rows_after": rows_after,
                }))
            }
            "top_k" => {
                let k = input.get("k").and_then(Value::as_u64).unwrap_or(5) as usize;
                let top = self.top_k(k)?;
                let n_rows = self.read_rows()?.len();
                Ok(json!({
                    "top_k": top,
                    "n_rows": n_rows,
                }))
            }
            other => Err(ActionError::Execution(format!(
                "unknown archive op `{other}` (expected `append` or `top_k`)"
            ))),
        }
    }
}

/// Convenience: does this path point at a regular file? Used by main.rs for fail-fast oracle checks.
pub fn is_file(path: &Path) -> bool {
    path.is_file()
}
