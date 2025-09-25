import os
import logging
from pyBEEP import (
    plot_time_series,
    setup_logging,
    connect_to_potentiostat,
)

setup_logging(level=logging.INFO)

controller = connect_to_potentiostat()

folder = os.path.join("results", "example_EIS")
os.makedirs(folder, exist_ok=True)

# --- Electrochemical Impedance Spectroscopy (EIS) ---
eis_file = os.path.join(folder, "test_EIS.csv")

start_freq: int
end_freq: int
duration: float

eis_params = {"start_freq": 1000, "end_freq": 100, "duration": 5}
controller.apply_measurement(
    mode="EIS",
    params=eis_params,
    tia_gain=0,
    filename="test_EIS.csv",
    folder=folder,
)

plot_time_series(eis_file, figpath=eis_file.replace(".csv", ".png"), show=True)
