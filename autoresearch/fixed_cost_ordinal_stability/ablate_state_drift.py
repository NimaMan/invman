#!/usr/bin/env python3
"""Archived fixed-cost ordinal state-drift ablation.

This probe depended on the deleted Python lost-sales environments and the old
torch-backed ``LinearPolicyNet`` checkpoint format. The fixed-cost/lost-sales
runtime is now Rust-first, and current trained policies are saved as
``policy_artifact.json`` + ``model_params.npy``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEGACY_MESSAGE = (
    "ablate_state_drift.py is archived. It requires deleted Python env/model APIs "
    "(invman.problems.* and invman.policies.linear.LinearPolicyNet) plus an old "
    "model_params.torch checkpoint. Use scripts/evaluate_saved_policy.py for "
    "current Rust-backed policy_artifact.json evaluations."
)


def main() -> None:
    raise SystemExit(LEGACY_MESSAGE)


if __name__ == "__main__":
    main()
