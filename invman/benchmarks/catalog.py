"""Baseline catalog API over BENCHMARK_MANIFEST.json (the single source of truth).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Give a benchmark CONSUMER a minimal-effort, dependency-light way to, for any one
of the 14 inventory-control problem families in this repo:
  (a) identify the reference instances,
  (b) read the baseline heuristics + exact-solver + published rows,
  (c) read the reference RESULTS (so they can compare their own approach),
  (d) get the exact reproduce command(s) + expected value + tolerance,
  (e) read the difficulty + verification tier (honestly labeled).

Design invariant
----------------
`docs/benchmarks/BENCHMARK_MANIFEST.json` is the SINGLE SOURCE OF TRUTH. This
module never duplicates manifest data; it LOADS the JSON and STRUCTURES it into
dataclasses + render functions. Nothing here hardcodes a problem's numbers; if
the manifest changes, this API reflects the change with no code edit.

Data flow / algorithm
---------------------
1. `_manifest_path()` resolves the manifest relative to this file
   (repo_root = parents[2] of this module), with an env-var override
   `INVMAN_BENCHMARK_MANIFEST` for out-of-tree use.
2. `_load_manifest()` reads + json.loads the file once and memoizes the parsed
   list-of-dicts (module-level cache; `reload()` clears it). Pure stdlib.
3. `_build_card(entry)` wraps one manifest entry into a `ProblemCard`:
   - nested `Instance` objects (name, dimensions, literature_verified_flag),
   - a `Baselines` object (heuristics / exact_solver / published_rows),
   - a `Verification` object (status + published_value + reproduced_value +
     rerun_method), whose `tier` is READ VERBATIM from the manifest's explicit
     `verification_tier` field (NOT string-derived from the status). The full
     verbatim `verification.status` is always surfaced so the nuance is never
     lost. For `mixed`/split entries a per-subfamily tier map
     (`verification_tier_by_subfamily`) and a one-line `verification_tier_note`
     are carried and rendered on the card,
   - a list of `Result` objects (claim / seed_reporting / at_risk),
   - difficulty + difficulty_rationale (+ optional per-subfamily maps for the
     multi_echelon umbrella entry),
   - the raw reproduce_commands list.
4. Lookups:
   - `list_problems(difficulty=?, verified=?)` -> problem short-names in manifest
     order, optionally filtered by difficulty bucket or verification tier.
   - `get(name)` -> ProblemCard; accepts the short problem name ('lost_sales') or
     the full manifest `problem` string (they are identical in this manifest, but
     `get` also tolerates case/whitespace and fails LOUDLY (KeyError) on unknown.
   - `render_card(name)` -> a Markdown BENCHMARK_CARD string (title, difficulty,
     verification tier, instances, baselines, results, "How to reproduce &
     compare" block with command + expected value + tolerance pulled from the
     verification strings).
   - `render_all_cards(out_dir)` -> writes one card per problem to
     `<out_dir>/<problem>.md` plus a `README.md` index, returns the dict of
     written paths.

Difficulty rubric (folds three axes; recorded per entry in the manifest)
-----------------------------------------------------------------------
  (a) state/action dimensionality,
  (b) exact-solver availability (an exact VI/DP true optimum makes a problem
      easier to BENCHMARK against, even if the env itself is rich),
  (c) comparator type: true_optimum_match_only (easiest to score) <
      heuristic_to_beat ~ bound_gap < self_consistent (hardest to score
      honestly).
The manifest carries the final `difficulty` + one-line `difficulty_rationale`
per entry; the prose rubric lives in docs/benchmarks/README.md.

Verification tiers (honest labels — AUTHORED in the manifest, reconciled against
docs/benchmarks/VERIFICATION_LEDGER.md + FUNDAMENTAL_QUESTIONS.md; the ledger wins)
----------------------------------------------------------
  strict   = the benchmarked env re-runs a number PRINTED IN A PEER-REVIEWED
             paper within tolerance,
  reference= the env re-runs a companion-code / closed-form / reduced-module
             number, OR reproduces an ACTION (not a printed cost),
  faithful = no public anchor; validated only against the repo's own exact DP,
  mixed    = an umbrella family whose sub-families have DIFFERENT tiers (see the
             per-subfamily map). Used only by multi_echelon.
The tier is read VERBATIM from each manifest entry's `verification_tier` field;
it is NOT string-derived. Split families (spare_parts_inventory,
ameliorating_inventory, joint_replenishment, decentralized_inventory_control)
keep the honest HEADLINE tier on the family and record the split in a one-line
`verification_tier_note`.
`verified='strict'|'reference'|'faithful'|'mixed'|'any'` filters on this field.

Dependencies: Python stdlib only (json, os, pathlib, dataclasses, typing).
================================================================================
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Manifest location + loading (single source of truth; memoized)
# ---------------------------------------------------------------------------

_MANIFEST_ENV_VAR = "INVMAN_BENCHMARK_MANIFEST"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST = _REPO_ROOT / "docs" / "benchmarks" / "BENCHMARK_MANIFEST.json"

_VALID_DIFFICULTIES = ("easy", "medium", "hard")
_VALID_TIERS = ("strict", "reference", "faithful", "mixed")

_manifest_cache: Optional[list[dict[str, Any]]] = None


def _manifest_path() -> Path:
    """Resolve the manifest path (env override > repo-relative default)."""
    override = os.environ.get(_MANIFEST_ENV_VAR)
    if override:
        return Path(override)
    return _DEFAULT_MANIFEST


def _load_manifest() -> list[dict[str, Any]]:
    """Read + parse the manifest once; memoize the list-of-entries."""
    global _manifest_cache
    if _manifest_cache is None:
        path = _manifest_path()
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"benchmark manifest must be a JSON list: {path}")
        _manifest_cache = data
    return _manifest_cache


def reload() -> None:
    """Clear the memoized manifest (use after editing the JSON in-process)."""
    global _manifest_cache, _cards_cache
    _manifest_cache = None
    _cards_cache = None


# ---------------------------------------------------------------------------
# Dataclasses — the structured view a consumer reads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One named reference instance of a problem family."""

    name: str
    dimensions: list[str]
    literature_verified_flag: str

    @property
    def is_literature_verified(self) -> bool:
        """True only when the flag clearly asserts 'true' (provenance-level)."""
        return self.literature_verified_flag.strip().lower().startswith("true")


@dataclass(frozen=True)
class Baselines:
    """The comparators shipped with the problem."""

    heuristics: list[str]
    exact_solver: str
    published_rows: list[str]

    @property
    def has_exact_solver(self) -> bool:
        """Heuristic check: is there an in-repo exact/bound solver (vs 'none')?"""
        text = self.exact_solver.strip().lower()
        return bool(text) and not text.startswith(("none", "no "))


@dataclass(frozen=True)
class Verification:
    """Honest verification record + the manifest-authored tier.

    `tier` is read VERBATIM from the manifest `verification_tier` field (never
    string-derived). `tier_by_subfamily` (present for the multi_echelon umbrella
    'mixed' entry) and `tier_note` (present for split families) capture the
    nuance so it is never collapsed into the single headline label.
    """

    status: str
    published_value: str
    reproduced_value: str
    rerun_method: str
    tier: str  # 'strict' | 'reference' | 'faithful' | 'mixed'
    tier_by_subfamily: dict[str, str] = field(default_factory=dict)
    tier_note: str = ""

    @property
    def tier_label(self) -> str:
        return {
            "strict": "re-runs a PEER-REVIEWED printed number",
            "reference": "re-runs a companion / closed-form / reduced-module number, or a published action",
            "faithful": "faithful_unverified (validated only vs the repo's own exact DP)",
            "mixed": "umbrella — tiers differ per sub-family (see map)",
        }.get(self.tier, self.tier)


@dataclass(frozen=True)
class Result:
    """One reported learned-policy result + its honesty flags."""

    claim: str
    seed_reporting: str
    at_risk: bool

    @property
    def is_seed_robust(self) -> bool:
        return self.seed_reporting == "multi_seed_mean_std" and not self.at_risk


@dataclass(frozen=True)
class ProblemCard:
    """The full structured benchmark card for one problem family."""

    problem: str
    subfamily: str
    difficulty: str
    difficulty_rationale: str
    instances: list[Instance]
    baselines: Baselines
    verification: Verification
    results: list[Result]
    reproduce_commands: list[str]
    # Optional per-subfamily difficulty maps (present only for umbrella entries
    # such as multi_echelon); empty dict otherwise.
    difficulty_by_subfamily: dict[str, str] = field(default_factory=dict)
    difficulty_by_subfamily_rationale: dict[str, str] = field(default_factory=dict)

    @property
    def verification_tier(self) -> str:
        return self.verification.tier

    @property
    def has_seed_robust_result(self) -> bool:
        return any(r.is_seed_robust for r in self.results)

    # -- executable layer (lazy bridge to invman.benchmarks.runners) -------
    # These turn the metadata card into a RUNNABLE handle. They import the
    # runners package (and through it `invman_rust`) lazily, so reading a card
    # never requires the compiled extension; only running the env does.
    @property
    def has_runner(self) -> bool:
        """True if this family has an executable baseline runner."""
        from invman.benchmarks import runners

        return runners.has_runner(self.problem)

    def runner(self):
        """Return the `ProblemRunner` for this family (raises if none exists)."""
        from invman.benchmarks import runners

        return runners.get_runner(self.problem)

    def list_instances(self) -> list[str]:
        """Reference-instance names a user can `load_instance(name)`."""
        return self.runner().list_instances()

    def load_instance(self, name: Optional[str] = None):
        """Load a runnable `ReferenceInstance` (env params + baselines + evaluate).

        With `name=None` the family's primary/canonical instance is returned.
        This is the one-call "get the baseline problem" entry point: the result
        carries the env params, the published baselines, `run_baselines()` to
        re-run them on the live env, and `evaluate()` to score your own policy.
        """
        return self.runner().load_instance(name)


# ---------------------------------------------------------------------------
# Verification-tier read (manifest is the authority — NO string derivation)
# ---------------------------------------------------------------------------


def _read_tier(entry: dict[str, Any]) -> str:
    """Read the honest verification tier VERBATIM from the manifest entry.

    The tier is AUTHORED in `BENCHMARK_MANIFEST.json` (`verification_tier`),
    reconciled against docs/benchmarks/VERIFICATION_LEDGER.md +
    FUNDAMENTAL_QUESTIONS.md. It is NOT derived from the prose status string
    (that over-labelled approx-only / action-reproducing / adjacent-module
    families as `strict`). Fails LOUDLY if absent or not one of the honest tiers.

    Tiers:
      strict   -> the benchmarked env re-runs a PEER-REVIEWED printed number,
      reference-> re-runs companion-code / closed-form / reduced-module number,
                  OR reproduces an action (not a printed cost),
      faithful -> no public anchor; validated only vs the repo's own exact DP,
      mixed    -> umbrella family whose sub-families have different tiers.
    """
    raw = entry.get("verification_tier")
    if raw is None:
        raise ValueError(
            f"problem {entry.get('problem')!r} is missing the explicit "
            f"'verification_tier' manifest field (expected one of {_VALID_TIERS})"
        )
    tier = str(raw).strip().lower()
    if tier not in _VALID_TIERS:
        raise ValueError(
            f"problem {entry.get('problem')!r} has invalid verification_tier "
            f"{raw!r}; expected one of {_VALID_TIERS}"
        )
    return tier


# ---------------------------------------------------------------------------
# Card construction
# ---------------------------------------------------------------------------


def _build_card(entry: dict[str, Any]) -> ProblemCard:
    instances = [
        Instance(
            name=str(item.get("name", "")),
            dimensions=list(item.get("dimensions", [])),
            literature_verified_flag=str(item.get("literature_verified_flag", "")),
        )
        for item in entry.get("instances", [])
    ]

    raw_baselines = entry.get("baselines", {}) or {}
    baselines = Baselines(
        heuristics=list(raw_baselines.get("heuristics", [])),
        exact_solver=str(raw_baselines.get("exact_solver", "")),
        published_rows=list(raw_baselines.get("published_rows", [])),
    )

    raw_verification = entry.get("verification", {}) or {}
    status = str(raw_verification.get("status", ""))
    verification = Verification(
        status=status,
        published_value=str(raw_verification.get("published_value", "")),
        reproduced_value=str(raw_verification.get("reproduced_value", "")),
        rerun_method=str(raw_verification.get("rerun_method", "")),
        tier=_read_tier(entry),
        tier_by_subfamily=dict(entry.get("verification_tier_by_subfamily", {})),
        tier_note=str(entry.get("verification_tier_note", "")),
    )

    results = [
        Result(
            claim=str(item.get("claim", "")),
            seed_reporting=str(item.get("seed_reporting", "")),
            at_risk=bool(item.get("at_risk", False)),
        )
        for item in entry.get("results", [])
    ]

    difficulty = str(entry.get("difficulty", "")).strip().lower()
    if difficulty not in _VALID_DIFFICULTIES:
        raise ValueError(
            f"problem {entry.get('problem')!r} has invalid difficulty {difficulty!r}; "
            f"expected one of {_VALID_DIFFICULTIES}"
        )

    return ProblemCard(
        problem=str(entry.get("problem", "")),
        subfamily=str(entry.get("subfamily", "")),
        difficulty=difficulty,
        difficulty_rationale=str(entry.get("difficulty_rationale", "")),
        instances=instances,
        baselines=baselines,
        verification=verification,
        results=results,
        reproduce_commands=list(entry.get("reproduce_commands", [])),
        difficulty_by_subfamily=dict(entry.get("difficulty_by_subfamily", {})),
        difficulty_by_subfamily_rationale=dict(
            entry.get("difficulty_by_subfamily_rationale", {})
        ),
    )


_cards_cache: Optional[dict[str, ProblemCard]] = None


def _cards() -> dict[str, ProblemCard]:
    """Build (once) and memoize the ordered {problem: ProblemCard} map."""
    global _cards_cache
    if _cards_cache is None:
        _cards_cache = {}
        for entry in _load_manifest():
            card = _build_card(entry)
            _cards_cache[card.problem] = card
    return _cards_cache


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------


def list_problems(
    difficulty: Optional[str] = None,
    verified: Optional[str] = None,
    literature_verified: Optional[bool] = None,
) -> list[str]:
    """Return problem short-names in manifest order, optionally filtered.

    Args:
        difficulty: keep only 'easy' | 'medium' | 'hard' (case-insensitive).
        verified:   keep only the verification tier 'strict' | 'reference' |
                    'faithful' | 'mixed' (or 'any'/None for no filter). The tier
                    is the explicit, ledger-reconciled manifest field — so e.g.
                    verified='strict' returns ONLY families whose benchmarked env
                    re-runs a peer-reviewed printed number (NOT OWMR /
                    joint_replenishment / spare_parts, which are approx-only /
                    action-only / adjacent-module).
        literature_verified: coarse keep/exclude on the literature anchor. True
                    keeps the 9 families with a real anchor (tier != 'faithful');
                    False keeps only the 5 repo-native 'faithful' families. This
                    is the keep/exclude line the adversarial audit drew (see
                    docs/benchmarks/LITERATURE_VERIFICATION_AUDIT_2026_06_12.md).
    """
    cards = _cards()
    names: Iterable[str] = cards.keys()

    if literature_verified is not None:
        names = [
            n for n in names
            if (cards[n].verification.tier != "faithful") == bool(literature_verified)
        ]

    if difficulty is not None:
        wanted = difficulty.strip().lower()
        if wanted not in _VALID_DIFFICULTIES:
            raise ValueError(
                f"unknown difficulty {difficulty!r}; expected one of {_VALID_DIFFICULTIES}"
            )
        names = [n for n in names if cards[n].difficulty == wanted]

    if verified is not None and verified.strip().lower() != "any":
        wanted_tier = verified.strip().lower()
        if wanted_tier not in _VALID_TIERS:
            raise ValueError(
                f"unknown verification tier {verified!r}; expected one of "
                f"{_VALID_TIERS} (or 'any')"
            )
        names = [n for n in names if cards[n].verification.tier == wanted_tier]

    return list(names)


def get(name: str) -> ProblemCard:
    """Return the ProblemCard for `name`.

    Accepts the short problem name ('lost_sales') or the full manifest `problem`
    string (identical in this manifest). Tolerant of surrounding whitespace and
    case. Fails LOUDLY (KeyError) on an unknown problem — no silent fallback.
    """
    if not isinstance(name, str) or not name.strip():
        raise KeyError(f"unknown benchmark problem: {name!r}")
    cards = _cards()
    key = name.strip()
    if key in cards:
        return cards[key]
    lowered = key.lower()
    for problem, card in cards.items():
        if problem.lower() == lowered:
            return card
    raise KeyError(
        f"unknown benchmark problem: {name!r}. Known problems: {sorted(cards)}"
    )


def all_cards() -> list[ProblemCard]:
    """Return every ProblemCard in manifest order."""
    return list(_cards().values())


# ---------------------------------------------------------------------------
# Markdown rendering — the BENCHMARK_CARD
# ---------------------------------------------------------------------------


def _bullets(items: list[str]) -> list[str]:
    if not items:
        return ["- _(none recorded)_"]
    return [f"- {item}" for item in items]


def render_card(name: str) -> str:
    """Render the Markdown BENCHMARK_CARD for one problem (round-trippable)."""
    card = get(name)
    lines: list[str] = []

    lines.append(f"# Benchmark card — `{card.problem}`")
    lines.append("")
    lines.append(f"**Subfamily:** {card.subfamily}")
    lines.append("")
    lines.append(f"**Difficulty:** `{card.difficulty}` — {card.difficulty_rationale}")
    if card.difficulty_by_subfamily:
        lines.append("")
        lines.append("**Difficulty by subfamily:**")
        for sub, diff in card.difficulty_by_subfamily.items():
            rationale = card.difficulty_by_subfamily_rationale.get(sub, "")
            lines.append(f"- `{sub}` → `{diff}` — {rationale}")
    lines.append("")
    lines.append(
        f"**Verification tier:** `{card.verification.tier}` "
        f"({card.verification.tier_label})"
    )
    if card.verification.tier_by_subfamily:
        lines.append("")
        lines.append("**Verification tier by subfamily:**")
        for sub, sub_tier in card.verification.tier_by_subfamily.items():
            lines.append(f"- `{sub}` → `{sub_tier}`")
    if card.verification.tier_note:
        lines.append("")
        lines.append(f"**Tier note:** {card.verification.tier_note}")
    lines.append("")
    lines.append(f"> Status (manifest, verbatim): {card.verification.status}")
    lines.append("")

    # Instances
    lines.append("## Reference instances")
    lines.append("")
    lines.append("| Instance | literature_verified | Dimensions |")
    lines.append("| --- | --- | --- |")
    for inst in card.instances:
        dims = ", ".join(inst.dimensions) if inst.dimensions else ""
        flag = inst.literature_verified_flag.replace("|", "\\|")
        dims = dims.replace("|", "\\|")
        name_cell = inst.name.replace("|", "\\|")
        lines.append(f"| {name_cell} | {flag} | {dims} |")
    lines.append("")

    # Baselines
    lines.append("## Baselines")
    lines.append("")
    lines.append("**Heuristics**")
    lines.extend(_bullets(card.baselines.heuristics))
    lines.append("")
    lines.append("**Exact solver / bound**")
    lines.append("")
    lines.append(card.baselines.exact_solver or "_(none)_")
    lines.append("")
    lines.append("**Published rows**")
    lines.extend(_bullets(card.baselines.published_rows))
    lines.append("")

    # Reference results
    lines.append("## Reference results (compare your approach against these)")
    lines.append("")
    if card.results:
        lines.append("| seed_reporting | at_risk | seed-robust | Claim |")
        lines.append("| --- | --- | --- | --- |")
        for res in card.results:
            claim = res.claim.replace("|", "\\|")
            robust = "yes" if res.is_seed_robust else "no"
            lines.append(
                f"| `{res.seed_reporting}` | {res.at_risk} | {robust} | {claim} |"
            )
    else:
        lines.append("_(no learned-policy results recorded for this problem yet.)_")
    lines.append("")

    # Reproduce + compare
    lines.append("## How to reproduce & compare")
    lines.append("")
    lines.append(f"**Expected (published) value:** {card.verification.published_value}")
    lines.append("")
    lines.append(f"**Reproduced value (this audit):** {card.verification.reproduced_value}")
    lines.append("")
    lines.append(f"**Rerun method / tolerance:** {card.verification.rerun_method}")
    lines.append("")
    lines.append("**Reproduce command(s):**")
    lines.append("")
    if card.reproduce_commands:
        lines.append("```bash")
        for cmd in card.reproduce_commands:
            lines.append(cmd)
        lines.append("```")
    else:
        lines.append("_(no reproduce commands recorded.)_")
    lines.append("")
    lines.append(
        "To compare your own policy: run the command(s) above to regenerate the "
        "baseline on the named instance(s), evaluate your policy under the SAME "
        "instance + eval protocol (seeds / horizon / tolerance shown above), and "
        "report mean±std over ≥5 optimizer seeds vs the strongest baseline."
    )
    lines.append("")
    lines.append(
        "_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via "
        "`invman.benchmarks.catalog.render_card`. Do not edit by hand._"
    )
    lines.append("")
    return "\n".join(lines)


def render_all_cards(out_dir: str | Path) -> dict[str, Path]:
    """Write one card per problem to `<out_dir>/<problem>.md` + a README index.

    Returns a dict {problem: written_path} plus the index under key '__index__'.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    cards = all_cards()
    for card in cards:
        card_path = out_path / f"{card.problem}.md"
        card_path.write_text(render_card(card.problem) + "\n", encoding="utf-8")
        written[card.problem] = card_path

    index_path = out_path / "README.md"
    index_path.write_text(_render_index(cards) + "\n", encoding="utf-8")
    written["__index__"] = index_path
    return written


def _render_index(cards: list[ProblemCard]) -> str:
    """Render the per-folder cards/README.md index (functionality doc)."""
    lines: list[str] = []
    lines.append("# Benchmark cards — index")
    lines.append("")
    lines.append(
        "One **BENCHMARK_CARD** per problem family, auto-generated from "
        "`../BENCHMARK_MANIFEST.json` (the single source of truth) by "
        "`invman.benchmarks.catalog.render_all_cards`. Each card carries the "
        "difficulty, the honest verification tier, the reference instances, the "
        "baselines (heuristics / exact solver / published rows), the reference "
        "results, and a **How to reproduce & compare** block (command + expected "
        "value + tolerance) so a user can regenerate the baseline and compare "
        "their own approach with minimal effort."
    )
    lines.append("")
    lines.append("Regenerate with:")
    lines.append("")
    lines.append("```bash")
    lines.append(
        'python -c "from invman.benchmarks import catalog; '
        "catalog.render_all_cards('docs/benchmarks/cards')\""
    )
    lines.append("```")
    lines.append("")
    lines.append("| Problem | Difficulty | Verification tier | Card |")
    lines.append("| --- | --- | --- | --- |")
    for card in cards:
        lines.append(
            f"| `{card.problem}` | `{card.difficulty}` "
            f"| `{card.verification.tier}` | [{card.problem}](./{card.problem}.md) |"
        )
    lines.append("")
    lines.append("## Difficulty rubric")
    lines.append("")
    lines.append(
        "Difficulty folds three axes: (a) state/action dimensionality, (b) "
        "exact-solver availability (an exact VI/DP true optimum makes a problem "
        "easier to benchmark), and (c) comparator type "
        "(`true_optimum_match_only` easiest → `heuristic_to_beat` ~ `bound_gap` "
        "→ `self_consistent` hardest). The per-problem `difficulty` + one-line "
        "`difficulty_rationale` live in the manifest; the full rubric + the three "
        "verification tiers are documented in `../README.md`."
    )
    lines.append("")
    lines.append(
        "_Generated by `invman.benchmarks.catalog`. Do not edit by hand._"
    )
    lines.append("")
    return "\n".join(lines)
