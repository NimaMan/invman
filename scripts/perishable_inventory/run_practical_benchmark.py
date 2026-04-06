import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import get_reference, zero_state

from invman.benchmarks.practical import dumps_json, load_dataset, write_report

import invman_rust


DEFAULT_DATASET = (
    PACKAGE_ROOT
    / "rust"
    / "src"
    / "problems"
    / "perishable_inventory"
    / "practical"
    / "datasets"
    / "grocery_like_daily_trace.json"
)

DEFAULT_OUTPUT_JSON = (
    PACKAGE_ROOT
    / "rust"
    / "src"
    / "problems"
    / "perishable_inventory"
    / "practical"
    / "reports"
    / "latest_report.json"
)

DEFAULT_OUTPUT_MARKDOWN = (
    PACKAGE_ROOT
    / "rust"
    / "src"
    / "problems"
    / "perishable_inventory"
    / "practical"
    / "reports"
    / "README.md"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the practical perishable-inventory benchmark on a checked-in demand trace."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output_markdown", default=str(DEFAULT_OUTPUT_MARKDOWN))
    return parser.parse_args()


def empirical_mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def base_stock_params(reference, train_demands, estimated_mean):
    on_hand, pipeline_orders = zero_state(reference)
    best, _ = invman_rust.perishable_inventory_base_stock_search_from_demands(
        on_hand=on_hand,
        pipeline_orders=pipeline_orders,
        demands=list(train_demands),
        lead_time=int(reference["lead_time"]),
        max_order_size=int(reference["max_order_size"]),
        demand_mean=float(estimated_mean),
        holding_cost=float(reference["holding_cost"]),
        shortage_cost=float(reference["shortage_cost"]),
        waste_cost=float(reference["waste_cost"]),
        position_upper_bound=int(reference["max_order_size"]),
        procurement_cost=float(reference["procurement_cost"]),
        warm_up_periods_ratio=0.0,
        issuing_policy=str(reference["issuing_policy"]),
        top_k=12,
    )
    return [int(best[0])]


def bsp_low_ew_params(reference, train_demands, estimated_mean):
    on_hand, pipeline_orders = zero_state(reference)
    best, _ = invman_rust.perishable_inventory_bsp_low_ew_search_from_demands(
        on_hand=on_hand,
        pipeline_orders=pipeline_orders,
        demands=list(train_demands),
        lead_time=int(reference["lead_time"]),
        max_order_size=int(reference["max_order_size"]),
        demand_mean=float(estimated_mean),
        holding_cost=float(reference["holding_cost"]),
        shortage_cost=float(reference["shortage_cost"]),
        waste_cost=float(reference["waste_cost"]),
        position_upper_bound=int(reference["max_order_size"]),
        procurement_cost=float(reference["procurement_cost"]),
        warm_up_periods_ratio=0.0,
        issuing_policy=str(reference["issuing_policy"]),
        top_k=12,
    )
    return [int(best[0]), int(best[1]), int(best[2])]


def evaluate_trace(reference, policy_name, params, demands, estimated_mean):
    on_hand, pipeline_orders = zero_state(reference)
    return dict(
        invman_rust.perishable_inventory_policy_trace_summary_from_demands(
            policy_name=policy_name,
            params=list(params),
            on_hand=on_hand,
            pipeline_orders=pipeline_orders,
            demands=list(demands),
            lead_time=int(reference["lead_time"]),
            max_order_size=int(reference["max_order_size"]),
            demand_mean=float(estimated_mean),
            holding_cost=float(reference["holding_cost"]),
            shortage_cost=float(reference["shortage_cost"]),
            waste_cost=float(reference["waste_cost"]),
            procurement_cost=float(reference["procurement_cost"]),
            issuing_policy=str(reference["issuing_policy"]),
        )
    )


def main():
    parsed = parse_args()
    dataset = load_dataset(parsed.dataset)
    reference = get_reference(dataset["reference_instance_name"])

    train_demands = [int(value) for value in dataset["train_demands"]]
    test_demands = [int(value) for value in dataset["test_demands"]]
    estimated_mean = empirical_mean(train_demands)

    base_stock = base_stock_params(reference, train_demands, estimated_mean)
    bsp_low_ew = bsp_low_ew_params(reference, train_demands, estimated_mean)

    payload = {
        "family": "perishable_inventory",
        "dataset": dataset,
        "calibration_protocol": (
            "Tune `base_stock` and `bsp_low_ew` on the train block with deterministic trace search; "
            "report both in-sample train and held-out test metrics using the train-demand empirical mean."
        ),
        "dataset_diagnostics": {
            "reference_instance_name": dataset["reference_instance_name"],
            "train_mean_demand": estimated_mean,
            "test_mean_demand": empirical_mean(test_demands),
            "train_periods": len(train_demands),
            "test_periods": len(test_demands),
        },
        "metric_order": [
            "mean_period_cost",
            "fill_rate",
            "cycle_service_level",
            "waste_rate",
            "mean_holding_inventory",
            "mean_order_quantity",
            "positive_order_frequency",
        ],
        "metric_labels": {
            "mean_period_cost": "Mean Period Cost",
            "fill_rate": "Fill Rate",
            "cycle_service_level": "Cycle Service",
            "waste_rate": "Waste / Demand",
            "mean_holding_inventory": "Mean Holding",
            "mean_order_quantity": "Mean Order",
            "positive_order_frequency": "Positive Order Freq",
        },
        "policy_rows": [
            {
                "policy": "base_stock",
                "split": "train",
                "params": base_stock,
                "metrics": evaluate_trace(reference, "base_stock", base_stock, train_demands, estimated_mean),
                "notes": "calibration block",
            },
            {
                "policy": "base_stock",
                "split": "test",
                "params": base_stock,
                "metrics": evaluate_trace(reference, "base_stock", base_stock, test_demands, estimated_mean),
                "notes": "held-out evaluation",
            },
            {
                "policy": "bsp_low_ew",
                "split": "train",
                "params": bsp_low_ew,
                "metrics": evaluate_trace(reference, "bsp_low_ew", bsp_low_ew, train_demands, estimated_mean),
                "notes": "calibration block",
            },
            {
                "policy": "bsp_low_ew",
                "split": "test",
                "params": bsp_low_ew,
                "metrics": evaluate_trace(reference, "bsp_low_ew", bsp_low_ew, test_demands, estimated_mean),
                "notes": "held-out evaluation",
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
