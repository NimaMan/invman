import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits_from_argv

configure_process_cpu_limits_from_argv(sys.argv[1:])

from invman.config import get_config
from invman.experiment_runner import run_experiment


def main():
    args = get_config()
    result_payload, results_path = run_experiment(args)

    import json

    print(json.dumps(result_payload["evaluation"], indent=2))
    print(f"saved results to {results_path}")


if __name__ == "__main__":
    main()
