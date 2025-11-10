import os
import logging
from pyBEEP import (
    setup_logging,
    connect_to_potentiostat,
    #plot_cdl_points
)

setup_logging(level=logging.INFO)

controller = connect_to_potentiostat()

folder = os.path.join("results", "example_Cdl")
os.makedirs(folder, exist_ok=True)

# --- 3. Cyclic Voltammetry (CV) ---
cdl_params = {
    "vertex_a": -2.0,
    "vertex_b": 2.0,
    "scan_rates": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],  # [V/s]
    "cycles_per_rate": 2,
    "rest_time": 5.0,
    "start": None,  # optional
    "end": None,  # optional
}

cdl_filename = "test_CDL.csv"

controller.apply_measurement(
    mode="CDL", params=cdl_params, tia_gain=0, filename=cdl_filename, folder=folder
)

csv_path = os.path.join(folder, cdl_filename)
