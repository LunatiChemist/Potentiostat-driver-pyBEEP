import os
import logging
from pyBEEP import (
    plot_time_series,
    setup_logging,
    connect_to_potentiostat,
)

setup_logging(level=logging.DEBUG)

controller = connect_to_potentiostat()

folder = os.path.join("results", "example_CA")
os.makedirs(folder, exist_ok=True)

# --- 1. Constant Amperometry (CA) ---
ca_file = os.path.join(folder, "test_CA_busy.csv")
for time in [
    120.0,
]:
    ca_params = {"potential": 0.17, "duration": time}
    controller.apply_measurement(
        mode="CA",
        params=ca_params,
        tia_gain=0,
        filename="test_CA_busy.csv",
        folder=folder,
        charge_cutoff_c=0.05,
    )
plot_time_series(ca_file, figpath=ca_file.replace(".csv", ".png"), show=True)
