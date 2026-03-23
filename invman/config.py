import argparse
from pathlib import Path

from invman.policies.common import (
    normalize_policy_head,
    normalize_tree_action_adapter,
    normalize_tree_leaf_type,
    normalize_tree_split_type,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return False


def _default_output_dirs() -> tuple[Path, Path, Path]:
    package_root = Path(__file__).resolve().parents[1]
    outputs_root = package_root / "outputs"
    return outputs_root / "results", outputs_root / "logs", outputs_root / "models"


def _normalize_state_features(state_features: str) -> str:
    aliases = {
        "pipeline": "pipeline",
        "pipeline_plus_summary": "pipeline_plus_summary",
        "augmented": "pipeline_plus_summary",
        "feature_augmented": "pipeline_plus_summary",
    }
    normalized = aliases.get(state_features)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown state feature mode '{state_features}'. Expected one of: {valid}")
    return normalized


def get_config(argv=None):
    load_dotenv()

    default_results_dir, default_log_dir, default_models_dir = _default_output_dirs()

    parser = argparse.ArgumentParser(description="Lost-sales inventory management experiments")

    parser.add_argument(
        "--problem",
        default="lost_sales",
        choices=["lost_sales", "lost_sales_fixed_order_cost", "dual_sourcing", "multi_echelon"],
        help="Problem variant to simulate.",
    )

    parser.add_argument(
        "--training_method",
        default="cma",
        choices=["cma"],
        help="Parameter optimizer.",
    )
    parser.add_argument("--training_episodes", default=500, type=int, help="Number of ES iterations.")
    parser.add_argument("--mp_num_processors", default=4, type=int, help="Worker processes for parallel rollouts.")
    parser.add_argument("--sigma_init", default=5.0, type=float, help="Initial search variance.")
    parser.add_argument("--es_population", default=50, type=int, help="Population size per ES iteration.")
    parser.add_argument("--same_seed", action="store_true", help="Use common random numbers within each ES batch.")
    parser.add_argument("--dynamic_horizon", action="store_true", help="Increase the rollout horizon over training.")
    parser.add_argument("--min_dynamic_horizon", default=100, type=int, help="Lower bound for dynamic horizon.")
    parser.add_argument("--max_dynamic_horizon", default=5000, type=int, help="Upper bound for dynamic horizon.")

    parser.add_argument(
        "--demand_dist_name",
        default="Poisson",
        choices=["Poisson", "Geometric"],
        help="Demand distribution.",
    )
    parser.add_argument("--demand_rate", default=5.0, type=float, help="Mean demand per period.")
    parser.add_argument("--max_order_size", default=20, type=int, help="Maximum feasible order quantity.")
    parser.add_argument("--lead_time", default=2, type=int, help="Lead time in review periods.")
    parser.add_argument("--shortage_cost", default=4.0, type=float, help="Lost-sales penalty.")
    parser.add_argument("--holding_cost", default=1.0, type=float, help="Holding cost.")
    parser.add_argument("--procurement_cost", default=0.0, type=float, help="Per-unit procurement cost.")
    parser.add_argument(
        "--fixed_order_cost",
        default=0.0,
        type=float,
        help="Fixed setup cost charged whenever a positive order is placed.",
    )
    parser.add_argument("--regular_lead_time", default=2, type=int, help="Regular supplier lead time.")
    parser.add_argument("--expedited_lead_time", default=0, type=int, help="Expedited supplier lead time.")
    parser.add_argument("--regular_order_cost", default=100.0, type=float, help="Regular supplier unit cost.")
    parser.add_argument("--expedited_order_cost", default=105.0, type=float, help="Expedited supplier unit cost.")
    parser.add_argument("--regular_max_order_size", default=20, type=int, help="Max regular-source order quantity.")
    parser.add_argument("--expedited_max_order_size", default=20, type=int, help="Max expedited-source order quantity.")
    parser.add_argument("--dual_demand_low", default=0, type=int, help="Lower support bound for dual-sourcing demand.")
    parser.add_argument("--dual_demand_high", default=4, type=int, help="Upper support bound for dual-sourcing demand.")
    parser.add_argument("--warehouse_lead_time", default=2, type=int, help="Manufacturer to warehouse lead time.")
    parser.add_argument("--retailer_lead_time", default=2, type=int, help="Warehouse to retailer lead time.")
    parser.add_argument("--num_retailers", default=10, type=int, help="Number of identical retailers.")
    parser.add_argument("--warehouse_holding_cost", default=3.0, type=float, help="Warehouse holding cost.")
    parser.add_argument("--retailer_holding_cost", default=3.0, type=float, help="Retailer holding cost.")
    parser.add_argument("--warehouse_expedited_cost", default=0.0, type=float, help="Unit cost of same-day warehouse expediting.")
    parser.add_argument("--warehouse_lost_sale_cost", default=60.0, type=float, help="Lost-sales penalty in the multi-echelon model.")
    parser.add_argument("--expedited_service_prob", default=0.8, type=float, help="Probability a retailer stockout customer accepts same-day warehouse service.")
    parser.add_argument("--warehouse_capacity", default=100, type=int, help="Warehouse production/replenishment capacity.")
    parser.add_argument("--warehouse_inventory_cap", default=1000, type=int, help="Warehouse inventory-position cap.")
    parser.add_argument("--retailer_inventory_cap", default=100, type=int, help="Retail inventory-position cap.")
    parser.add_argument("--multi_demand_mean", default=5.0, type=float, help="Mean retailer demand in the multi-echelon model.")
    parser.add_argument("--multi_demand_std", default=14.0, type=float, help="Std. dev. of retailer demand in the multi-echelon model.")
    parser.add_argument("--horizon", default=2000, type=int, help="Training rollout horizon.")
    parser.add_argument("--eval_horizon", default=10000, type=int, help="Evaluation rollout horizon.")
    parser.add_argument("--eval_seeds", default=10, type=int, help="Number of evaluation seeds.")
    parser.add_argument("--track_demand", action="store_true", help="Pre-sample demand paths for reproducible evaluations.")
    parser.add_argument("--warm_up_periods_ratio", default=0.2, type=float, help="Warm-up fraction discarded from the mean cost.")
    parser.add_argument("--inventory_upper_bound", default=200, type=int, help="One-hot helper upper bound retained for legacy utilities.")
    parser.add_argument(
        "--state_features",
        default="pipeline",
        help="State representation fed to the policy approximator.",
    )
    parser.add_argument(
        "--rollout_backend",
        default="python",
        choices=["python", "rust"],
        help="Simulator backend used for policy rollouts when supported.",
    )

    parser.add_argument("--policy_type", default="nn", choices=["nn", "linear", "soft_tree"], help="Policy backbone.")
    parser.add_argument(
        "--policy_head",
        "--action_output_mode",
        dest="policy_head",
        default="categorical_quantity",
        help="Action head used by the policy approximator.",
    )
    parser.add_argument(
        "--hidden_dim",
        nargs="+",
        type=int,
        default=[32, 32],
        help="Hidden-layer widths for the neural policy.",
    )
    parser.add_argument(
        "--activation",
        default="selu",
        choices=["selu", "gelu", "relu"],
        help="Activation used by the neural policy.",
    )
    parser.add_argument("--tree_depth", default=2, type=int, help="Depth of the soft tree policy.")
    parser.add_argument(
        "--tree_temperature",
        default=0.25,
        type=float,
        help="Temperature used by soft tree split sigmoids.",
    )
    parser.add_argument(
        "--tree_split_type",
        default="oblique",
        help="Tree split structure used by the soft tree policy.",
    )
    parser.add_argument(
        "--tree_leaf_type",
        default="constant",
        help="Leaf output type used by the soft tree policy.",
    )
    parser.add_argument(
        "--tree_action_adapter",
        default="identity",
        help="Structured action adapter used by the soft tree policy.",
    )

    parser.add_argument("--experiment_name", default="lost_sales", help="Name used for saved artifacts.")
    parser.add_argument("--results_dir", default=str(default_results_dir), help="Directory for JSON summaries.")
    parser.add_argument("--log_dir", default=str(default_log_dir), help="Directory for training logs.")
    parser.add_argument("--trained_models_dir", default=str(default_models_dir), help="Directory for saved models.")
    parser.add_argument("--save_every", default=100, type=int, help="Checkpoint frequency in ES iterations.")
    parser.add_argument("--save_solutions", action="store_true", help="Persist ES solution populations.")
    parser.add_argument("--seed", default=1234, type=int, help="Base random seed used for evaluation helpers.")

    args = parser.parse_args(argv)
    args.results_dir = str(Path(args.results_dir).expanduser())
    args.log_dir = str(Path(args.log_dir).expanduser())
    args.trained_models_dir = str(Path(args.trained_models_dir).expanduser())
    try:
        args.policy_head = normalize_policy_head(args.policy_head)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.state_features = _normalize_state_features(args.state_features)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.tree_split_type = normalize_tree_split_type(args.tree_split_type)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.tree_leaf_type = normalize_tree_leaf_type(args.tree_leaf_type)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.tree_action_adapter = normalize_tree_action_adapter(args.tree_action_adapter)
    except ValueError as exc:
        parser.error(str(exc))
    # Backward-compatible alias retained for older scripts.
    args.action_output_mode = args.policy_head

    if args.problem == "lost_sales_fixed_order_cost" and args.fixed_order_cost <= 0:
        parser.error("--fixed_order_cost must be positive when --problem=lost_sales_fixed_order_cost")
    if args.problem == "dual_sourcing" and args.expedited_lead_time > args.regular_lead_time:
        parser.error("--expedited_lead_time must be <= --regular_lead_time for dual_sourcing")
    if args.tree_depth < 1:
        parser.error("--tree_depth must be at least 1")
    if args.tree_temperature <= 0:
        parser.error("--tree_temperature must be positive")

    return args
