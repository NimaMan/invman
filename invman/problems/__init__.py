"""Problem packages and registry for experiment dispatch."""

from importlib import import_module

PROBLEM_MODULES = {
    "lost_sales": "invman.problems.lost_sales",
    "lost_sales_fixed_order_cost": "invman.problems.lost_sales_fixed_order_cost",
}


def get_problem_module(problem_name: str):
    try:
        module_path = PROBLEM_MODULES[problem_name]
    except KeyError as exc:
        valid = ", ".join(sorted(PROBLEM_MODULES))
        raise ValueError(f"Unknown problem '{problem_name}'. Expected one of: {valid}") from exc
    return import_module(module_path)


__all__ = ["PROBLEM_MODULES", "get_problem_module"]
