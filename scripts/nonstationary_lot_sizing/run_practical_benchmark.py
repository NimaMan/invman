import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.benchmarks.practical import dumps_json, load_dataset, write_report

import invman_rust


DEFAULT_DATASET = (
    PACKAGE_ROOT
    / "rust"
    / "problems"
    / "nonstationary_lot_sizing"
    / "practical"
    / "datasets"
    / "retail_like_weekly_trace.json"
)

DEFAULT_OUTPUT_JSON = (
    PACKAGE_ROOT
    / "rust"
    / "problems"
    / "nonstationary_lot_sizing"
    / "practical"
    / "reports"
    / "latest_report.json"
)

DEFAULT_OUTPUT_MARKDOWN = (
    PACKAGE_ROOT
    / "rust"
    / "problems"
    / "nonstationary_lot_sizing"
    / "practical"
    / "reports"
    / "latest_report.md"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the practical nonstationary-lot-sizing benchmark on a checked-in trace."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output_markdown", default=str(DEFAULT_OUTPUT_MARKDOWN))
    return parser.parse_args()


def mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_policy(dataset, policy_name, params):
    return dict(
        invman_rust.nonstationary_lot_sizing_policy_trace_summary_from_demands(
            policy_name=policy_name,
            params=list(params),
            forecast_means=list(dataset["forecast_means"]),
            forecast_horizon=int(dataset["forecast_horizon"]),
            initial_net_inventory=float(dataset["initial_net_inventory"]),
            pipeline_orders=list(dataset["pipeline_orders"]),
            demands=list(dataset["demands"]),
            holding_cost=float(dataset["holding_cost"]),
            shortage_cost=float(dataset["shortage_cost"]),
            fixed_order_cost=float(dataset["fixed_order_cost"]),
            demand_distribution=str(dataset["demand_distribution"]),
            demand_cv=float(dataset["demand_cv"]),
            procurement_cost=float(dataset["procurement_cost"]),
            lost_sales=bool(dataset["lost_sales"]),
        )
    )


def rolling_dp_summary(dataset):
    return dict(
        invman_rust.nonstationary_lot_sizing_rolling_dp_trace_summary_from_demands(
            forecast_means=list(dataset["forecast_means"]),
            forecast_horizon=int(dataset["forecast_horizon"]),
            initial_net_inventory=float(dataset["initial_net_inventory"]),
            pipeline_orders=list(dataset["pipeline_orders"]),
            demands=list(dataset["demands"]),
            holding_cost=float(dataset["holding_cost"]),
            shortage_cost=float(dataset["shortage_cost"]),
            fixed_order_cost=float(dataset["fixed_order_cost"]),
            procurement_cost=float(dataset["procurement_cost"]),
            lost_sales=bool(dataset["lost_sales"]),
            discount_factor=0.99,
            stationary_tail_periods=32,
        )
    )


def main():
    parsed = parse_args()
    dataset = load_dataset(parsed.dataset)

    periods = len(dataset["demands"])
    initial_window = dataset["forecast_means"][: dataset["forecast_horizon"]]
    simple_levels = invman_rust.nonstationary_lot_sizing_simple_s_s_levels(
        forecast_window=list(initial_window),
        lead_time=int(dataset["lead_time"]),
        holding_cost=float(dataset["holding_cost"]),
        shortage_cost=float(dataset["shortage_cost"]),
        fixed_order_cost=float(dataset["fixed_order_cost"]),
        demand_distribution=str(dataset["demand_distribution"]),
        demand_cv=float(dataset["demand_cv"]),
    )
    rolling_levels = invman_rust.nonstationary_lot_sizing_rolling_dp_s_s_levels(
        forecast_window=list(initial_window),
        lead_time=int(dataset["lead_time"]),
        holding_cost=float(dataset["holding_cost"]),
        shortage_cost=float(dataset["shortage_cost"]),
        fixed_order_cost=float(dataset["fixed_order_cost"]),
        demand_distribution="poisson",
        discount_factor=0.99,
        stationary_tail_periods=32,
    )

    observed = dataset["demands"]
    forecast = dataset["forecast_means"][:periods]
    forecast_errors = [forecast[i] - observed[i] for i in range(periods)]
    forecast_mae = mean([abs(value) for value in forecast_errors])
    forecast_bias = mean(forecast_errors)

    payload = {
        "family": "nonstationary_lot_sizing",
        "dataset": dataset,
        "calibration_protocol": (
            "No train/test split. These heuristics adapt directly from the rolling forecast window; "
            "the benchmark evaluates them on one fixed forecast-plus-realization path."
        ),
        "dataset_diagnostics": {
            "periods": periods,
            "mean_forecast": mean(forecast),
            "mean_realized_demand": mean(observed),
            "forecast_mae": forecast_mae,
            "forecast_bias": forecast_bias,
        },
        "metric_order": [
            "mean_period_cost",
            "shortage_rate",
            "cycle_service_level",
            "mean_holding_inventory",
            "mean_order_quantity",
            "positive_order_frequency",
        ],
        "metric_labels": {
            "mean_period_cost": "Mean Period Cost",
            "shortage_rate": "Shortage Rate",
            "cycle_service_level": "Cycle Service",
            "mean_holding_inventory": "Mean Holding",
            "mean_order_quantity": "Mean Order",
            "positive_order_frequency": "Positive Order Freq",
        },
        "policy_rows": [
            {
                "policy": "lead_time_base_stock",
                "split": "eval",
                "params": "adaptive",
                "metrics": evaluate_policy(dataset, "lead_time_base_stock", []),
                "notes": "uses current forecast window directly",
            },
            {
                "policy": "simple_s_s",
                "split": "eval",
                "params": [round(simple_levels[0], 3), round(simple_levels[1], 3)],
                "metrics": evaluate_policy(dataset, "simple_s_s", []),
                "notes": "params column shows first-period levels only",
            },
            {
                "policy": "rolling_dp_s_s",
                "split": "eval",
                "params": [round(rolling_levels[0], 3), round(rolling_levels[1], 3)],
                "metrics": rolling_dp_summary(dataset),
                "notes": "params column shows first-period levels only",
            },
        ],
    }

    payload = write_report(
        payload,
        output_json=parsed.output_json,
        output_markdown=parsed.output_markdown,
    )
    print(dumps_json(payload))
    print()
    print(payload["markdown"])


if __name__ == "__main__":
    main()
