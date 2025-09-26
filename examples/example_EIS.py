import os
import logging
from pyBEEP import (
    plot_eis_impedance,
    setup_logging,
    connect_to_potentiostat,
)

setup_logging(level=logging.INFO)

controller = connect_to_potentiostat()

folder = os.path.join("results", "example_EIS")
os.makedirs(folder, exist_ok=True)

# --- Electrochemical Impedance Spectroscopy (EIS) ---
eis_file = os.path.join(folder, "test_EIS.csv")

eis_params = {
    "start_freq": 1000,
    "end_freq": 10,
    "dc_potential": 5,
    "perturbation_potential": 0.01,
    "point_per_decade": 10,
}
controller.apply_measurement(
    mode="EIS",
    params=eis_params,
    tia_gain=0,
    filename="test_EIS.csv",
    folder=folder,
)

plot_eis_impedance(eis_file, figpath=eis_file.replace(".csv", ".png"), show=True)
