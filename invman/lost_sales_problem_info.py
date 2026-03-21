"""
Benchmark problem information adopted from the lost-sales literature:
- Zipkin, "Old and New Methods for Lost-Sales Inventory Systems" (1990)
- "Capped Base-Stock Policies in Lost-Sales Inventory Models"

The values below are kept as compact reference data for the classic lost-sales
problem used in this repository.
"""

import numpy as np


Poisson_demand_shortage_cost_4 = {
    2: {"optimal": 4.40, "M2": 4.41, "M1": 4.56, "SVBS": 4.80, "CappedBS": 4.41291875, "CAP_levels": (17, 5)},
    4: {"optimal": 4.73, "M2": 4.82, "M1": 5.06, "SVBS": 5.83, "CappedBS": 4.80, "CAP_levels": (26, 5)},
    6: {"optimal": None, "M2": 5.04476375, "M1": 5.4140775, "SVBS": 6.77253875, "CappedBS": 5.03894625, "CAP_levels": (34, 5)},
    8: {"optimal": None, "M2": 5.1991675, "M1": 5.6982775, "SVBS": 7.6943475, "CappedBS": 5.19198, "CAP_levels": (43, 5)},
    10: {"optimal": None, "M2": 5.31, "M1": None, "SVBS": None, "CappedBS": 5.27, "CAP_levels": None},
    16: {"optimal": None, "M2": 5.467825, "M1": 6.3501375, "SVBS": 10.144, "CappedBS": 5.26011625, "CAP_levels": (112, 4)},
}

Poisson_demand_shortage_cost_19 = {
    2: {"optimal": 7.66, "M2": 7.67173875, "M1": 7.7767725, "SVBS": 8.08596, "CappedBS": 7.71932375, "CAP_levels": (21, 7)},
    4: {"optimal": 8.89, "M2": 8.95289875, "M1": 9.17321125, "SVBS": 9.56453125, "CappedBS": 8.95},
    6: {"optimal": None, "M2": 9.8306675, "M1": 10.26865, "SVBS": 11.23402, "CappedBS": 9.80, "CAP_levels": (41, 6)},
    8: {"optimal": None, "M2": 10.510595, "M1": 11.0916, "SVBS": 12.3333375, "CappedBS": 10.33341375, "CAP_levels": (53, 5)},
    10: {"optimal": None, "M2": 11.09, "M1": np.nan, "SVBS": None, "CappedBS": 10.66, "CAP_levels": (91, 5)},
    16: {"optimal": None, "M2": np.nan, "M1": np.nan, "SVBS": None, "CappedBS": 11.35735875, "CAP_levels": (91, 5)},
}

Geometric_demand_shortage_cost_4 = {
    2: {"optimal": 10.24, "M2": 10.29, "M1": None, "SVBS": None, "CappedBS": 10.32},
    4: {"optimal": 10.61, "M2": 10.80, "M1": None, "SVBS": None, "CappedBS": 10.70},
    6: {"optimal": None, "M2": 11.08, "M1": None, "SVBS": None, "CappedBS": 10.91},
    8: {"optimal": None, "M2": 11.27, "M1": None, "SVBS": None, "CappedBS": 10.96},
    10: {"optimal": None, "M2": 11.40, "M1": None, "SVBS": None, "CappedBS": 10.98},
    16: {"optimal": None, "M2": None, "M1": None, "SVBS": None, "CappedBS": None},
}

Geometric_demand_shortage_cost_19 = {
    2: {"optimal": 20.89, "M2": 20.95, "M1": None, "SVBS": None, "CappedBS": 21.06},
    4: {"optimal": 22.95, "M2": 23.28, "M1": None, "SVBS": None, "CappedBS": 23.29},
    6: {"optimal": None, "M2": 24.93, "M1": None, "SVBS": None, "CappedBS": 24.49},
    8: {"optimal": None, "M2": 26.21, "M1": None, "SVBS": None, "CappedBS": 25.38},
    10: {"optimal": None, "M2": 27.27, "M1": None, "SVBS": None, "CappedBS": 25.98},
    16: {"optimal": None, "M2": None, "M1": None, "SVBS": None},
}


problem_info = {
    "Poisson_demand_shortage_cost_4": Poisson_demand_shortage_cost_4,
    "Poisson_demand_shortage_cost_19": Poisson_demand_shortage_cost_19,
    "Geometric_demand_shortage_cost_4": Geometric_demand_shortage_cost_4,
    "Geometric_demand_shortage_cost_19": Geometric_demand_shortage_cost_19,
}
