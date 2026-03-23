import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.problems.dual_sourcing.benchmark import benchmark_reference_instance
from invman.problems.dual_sourcing.reference_instances import list_reference_instances, build_reference_args


def main():
    results = {}
    for name in list_reference_instances():
        args = build_reference_args(name)
        args.rollout_backend = "rust"
        args.eval_seeds = 2
        results[name] = benchmark_reference_instance(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
