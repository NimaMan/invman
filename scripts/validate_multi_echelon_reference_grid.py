import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.problems.multi_echelon.benchmark import evaluate_default_heuristics
from invman.problems.multi_echelon.reference_instances import build_reference_args, list_reference_instances


def main():
    results = {}
    for name in list_reference_instances():
        args = build_reference_args(name)
        args.rollout_backend = "rust"
        args.eval_seeds = 2
        results[name] = evaluate_default_heuristics(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
