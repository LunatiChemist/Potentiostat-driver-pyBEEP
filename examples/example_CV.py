import os
import logging
from pyBEEP import (
    plot_cv_cycles,
    setup_logging,
    connect_to_potentiostat,
)

setup_logging(level=logging.INFO)

controller = connect_to_potentiostat()
for v in [0.5]:
    folder = os.path.join("results", "example_CV")
    os.makedirs(folder, exist_ok=True)

    # --- 3. Cyclic Voltammetry (CV) ---
    cv_file = os.path.join(folder, f"test_CV_{v}_2.csv")
    cv_params = {
        "start": 0.5,
        "vertex1": 0.5,
        "vertex2": -0.5,
        "end": 0.5,
        "scan_rate": v,
        "cycles": 2,
    }
    controller.apply_measurement(
        mode="CV",
        params=cv_params,
        tia_gain=0,
        filename=f"test_CV_{v}_2.csv",
        folder=folder,
    )
    # If you know scan_points per cycle, set it below:
    plot_cv_cycles(cv_file, figpath=cv_file.replace(".csv", ".png"), show=True, cycles=2)
