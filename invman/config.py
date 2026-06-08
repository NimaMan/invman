import argparse
from pathlib import Path

from invman.cpu_limits import normalize_args_cpu_limits
from invman.policy_registry import apply_policy_name, resolve_policy_name

SUPPORTED_DEMAND_NAMES = ("Poisson", "Geometric", "MarkovModulatedPoisson2")
DEFAULT_MMPP2_LAMBDA_LOW = 3.0
DEFAULT_MMPP2_LAMBDA_HIGH = 7.0
DEFAULT_MMPP2_POSITIVE_P00 = 0.9
DEFAULT_MMPP2_POSITIVE_P11 = 0.9

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


def _normalize_state_normalizer(state_normalizer: str) -> str:
    aliases = {
        "identity": "identity",
        "none": "identity",
        "raw": "identity",
        "quantity_scale": "quantity_scale",
        "qscale": "quantity_scale",
        "scale": "quantity_scale",
        "divide_by_scale": "quantity_scale",
        "scalar_divide": "quantity_scale",
    }
    normalized = aliases.get(state_normalizer)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown state normalizer '{state_normalizer}'. Expected one of: {valid}"
        )
    return normalized


def _parse_csv_values(raw_value, *, cast, argument_name: str):
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    else:
        parts = list(raw_value)
    if not parts:
        raise ValueError(f"{argument_name} must contain at least one value")
    try:
        return [cast(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{argument_name} must be a comma-separated list of {cast.__name__} values") from exc


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
        choices=["cma", "ppo"],
        help=(
            "Policy trainer. 'cma' = gradient-free CMA-ES over the policy "
            "parameter vector via es_mp.train (the default param-vector path). "
            "'ppo' = the reusable Rust PPO trainer (a neural actor-critic, "
            "gradient-based); it does NOT use the es_mp ask/tell loop -- invoke it "
            "via invman.ppo_trainer.train_ppo(problem, ...)."
        ),
    )
    parser.add_argument("--training_episodes", default=500, type=int, help="Number of ES iterations.")
    parser.add_argument("--mp_num_processors", default=4, type=int, help="Worker processes for parallel rollouts.")
    parser.add_argument("--sigma_init", default=5.0, type=float, help="Initial search variance.")
    parser.add_argument("--es_population", default=50, type=int, help="Population size per ES iteration.")
    parser.add_argument(
        "--es_population_sampling",
        default="fixed",
        choices=["fixed", "categorical"],
        help="Whether to keep the ES population fixed or sample it per iteration from a categorical distribution.",
    )
    parser.add_argument(
        "--es_population_candidates",
        default=None,
        help="Comma-separated candidate ES population sizes used when sampling populations per iteration.",
    )
    parser.add_argument(
        "--es_population_probabilities",
        default=None,
        help="Comma-separated nonnegative weights aligned with --es_population_candidates.",
    )
    parser.add_argument("--same_seed", action="store_true", help="Use common random numbers within each ES batch.")
    parser.add_argument("--dynamic_horizon", action="store_true", help="Increase the rollout horizon over training.")
    parser.add_argument("--min_dynamic_horizon", default=100, type=int, help="Lower bound for dynamic horizon.")
    parser.add_argument("--max_dynamic_horizon", default=5000, type=int, help="Upper bound for dynamic horizon.")

    parser.add_argument(
        "--demand_dist_name",
        default="Poisson",
        choices=list(SUPPORTED_DEMAND_NAMES),
        help="Demand distribution.",
    )
    parser.add_argument("--demand_rate", default=5.0, type=float, help="Mean demand per period.")
    parser.add_argument(
        "--demand_lambda_low",
        default=DEFAULT_MMPP2_LAMBDA_LOW,
        type=float,
        help="Low-state Poisson mean for MarkovModulatedPoisson2 demand.",
    )
    parser.add_argument(
        "--demand_lambda_high",
        default=DEFAULT_MMPP2_LAMBDA_HIGH,
        type=float,
        help="High-state Poisson mean for MarkovModulatedPoisson2 demand.",
    )
    parser.add_argument(
        "--demand_p00",
        default=DEFAULT_MMPP2_POSITIVE_P00,
        type=float,
        help="Stay-in-low transition probability for MarkovModulatedPoisson2 demand.",
    )
    parser.add_argument(
        "--demand_p11",
        default=DEFAULT_MMPP2_POSITIVE_P11,
        type=float,
        help="Stay-in-high transition probability for MarkovModulatedPoisson2 demand.",
    )
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
    parser.add_argument("--inventory_dynamics_mode", default="gijs_2022", choices=["gijs_2022", "van_roy_1997"], help="Multi-echelon timing convention: gijs_2022 (paper-faithful Eq.(2), pre-shipment warehouse order, end-of-period holding) or van_roy_1997 (reproduction).")
    parser.add_argument("--multi_action_design", default="direct_level", choices=["grid", "direct_level"], help="Multi-echelon learned-policy action parameterization (autoresearch search dimension): 'grid' (pick order-up-to levels from the Gijs reduced grid) or 'direct_level' (directly estimate order-up-to levels bounded by the physical caps).")
    parser.add_argument("--demand_distribution", default="normal_rounded_clipped", choices=["normal_rounded_clipped", "poisson"], help="Multi-echelon retailer demand distribution.")
    parser.add_argument("--horizon", default=2000, type=int, help="Training rollout horizon.")
    parser.add_argument("--eval_horizon", default=10000, type=int, help="Evaluation rollout horizon.")
    parser.add_argument("--eval_seeds", default=10, type=int, help="Number of evaluation seeds.")
    parser.add_argument("--track_demand", action="store_true", help="Pre-sample demand paths for reproducible evaluations.")
    parser.add_argument("--warm_up_periods_ratio", default=0.2, type=float, help="Warm-up fraction discarded from the mean cost.")
    parser.add_argument(
        "--one_hot_inventory_upper_bound",
        default=200,
        type=int,
        help="Inventory cap used by the lost-sales one-hot state encoding helper.",
    )
    parser.add_argument(
        "--state_features",
        default="pipeline",
        help="State representation fed to the policy approximator.",
    )
    parser.add_argument(
        "--state_normalizer",
        default="identity",
        help="Policy-side input normalization mode.",
    )
    parser.add_argument(
        "--state_scale",
        default=None,
        type=float,
        help="Optional scalar divisor used by the policy-side input normalizer.",
    )
    parser.add_argument(
        "--rollout_backend",
        default="python",
        choices=["python", "rust"],
        help="Simulator backend used for policy rollouts when supported.",
    )
    parser.add_argument(
        "--policy_name",
        default=None,
        help="Unique identifier of the learned policy architecture to run.",
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
        args.state_features = _normalize_state_features(args.state_features)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.state_normalizer = _normalize_state_normalizer(args.state_normalizer)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.es_population_candidates = _parse_csv_values(
            args.es_population_candidates,
            cast=int,
            argument_name="--es_population_candidates",
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.es_population_probabilities = _parse_csv_values(
            args.es_population_probabilities,
            cast=float,
            argument_name="--es_population_probabilities",
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.policy_name is not None:
        try:
            resolve_policy_name(args.policy_name)
        except ValueError as exc:
            parser.error(str(exc))

    if args.problem == "lost_sales_fixed_order_cost" and args.fixed_order_cost <= 0:
        parser.error("--fixed_order_cost must be positive when --problem=lost_sales_fixed_order_cost")
    if args.problem == "dual_sourcing" and args.expedited_lead_time > args.regular_lead_time:
        parser.error("--expedited_lead_time must be <= --regular_lead_time for dual_sourcing")
    if args.es_population_candidates is not None and any(value <= 0 for value in args.es_population_candidates):
        parser.error("--es_population_candidates must all be positive")
    if args.es_population_probabilities is not None:
        if args.es_population_candidates is None:
            parser.error("--es_population_probabilities requires --es_population_candidates")
        if len(args.es_population_probabilities) != len(args.es_population_candidates):
            parser.error("--es_population_probabilities must match --es_population_candidates in length")
        if any(value < 0 for value in args.es_population_probabilities):
            parser.error("--es_population_probabilities must be nonnegative")
        if sum(args.es_population_probabilities) <= 0:
            parser.error("--es_population_probabilities must sum to a positive value")
    if args.es_population_sampling == "categorical" and args.es_population_candidates is None:
        parser.error("--es_population_sampling=categorical requires --es_population_candidates")
    if argv != [] and args.policy_name is None:
        parser.error("--policy_name is required")
    if args.policy_name is not None:
        apply_policy_name(args)

    normalize_args_cpu_limits(args)
    return args
