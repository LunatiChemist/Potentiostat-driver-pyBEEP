import os
import logging
from pyBEEP import (
    plot_time_series,
    setup_logging,
    connect_to_potentiostat,
)

setup_logging(level=logging.INFO)

controller = connect_to_potentiostat()

folder = os.path.join("results", "example_OCP")
os.makedirs(folder, exist_ok=True)

# --- 1. Constant Amperometry (CA) ---
ca_file = os.path.join(folder, "test_OCP.csv")
for time in [
    5,
]:
    ocp_params = {"duration": time}
    controller.apply_measurement(
        mode="OCP",
        params=ocp_params,
        tia_gain=0,
        filename="test_OCP.csv",
        sampling_interval=0.01,
        folder=folder,
    )
plot_time_series(ca_file, figpath=ca_file.replace(".csv", ".png"), show=True)
