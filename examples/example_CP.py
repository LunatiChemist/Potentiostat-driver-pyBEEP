import os
import logging
from pyBEEP import (
    plot_time_series,
    setup_logging,
    connect_to_potentiostat,
)

setup_logging(level=logging.INFO)

controller = connect_to_potentiostat()

folder = os.path.join("results", "example_CP")
os.makedirs(folder, exist_ok=True)

# --- 1. Constant Amperometry (CA) ---
ca_file = os.path.join(folder, "test_CP_busy.csv")
for time in [
    120,
]:
    ca_params = {"current": 0.001, "duration": time}
    controller.apply_measurement(
        mode="CP",
        params=ca_params,
        tia_gain=0,
        filename="test_CP_busy.csv",
        folder=folder,
        charge_cutoff_c=0.05
    )
plot_time_series(ca_file, figpath=ca_file.replace(".csv", ".png"), show=True)
