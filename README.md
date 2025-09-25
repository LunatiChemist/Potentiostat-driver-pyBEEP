# Potentiostat Driver - pyBEEP

Welcome to **pyBEEP** – a Python library for controlling BEEP (Basic Electrochemical Experimentation Potentiostat).  
Run common electrochemical experiments (chronoamperometry, chronopotentiometry, cyclic voltammetry, and more) with 
robust data logging and plotting.

This driver is designed to be used alongside the following:
- Potentiostat [**firmware**](https://github.com/aurelienblanc2/Potentiostat-firmware)
- Potentiostat python package datapipeline [**potentiopipe**](https://github.com/aurelienblanc2/Potentiostat-datapipeline)

If you’d like to share or explore all three related repositories together, here is a [**link**](https://github.com/stars/aurelienblanc2/lists/potentiostat)

Below, an image of the Potentiostat device:
![Potentiostat](docs/Potentiostat.png)

---

# Overview

pyBEEP provides:
- An easy interface for running standard electrochemical experiments over serial (Modbus).
- Flexible parameterization for experiment setup.
- Threaded data acquisition and automatic CSV logging.
- Optional plotting and data reduction.

---

# Table of Contents

- [Main Functionalities](#main-functionalities)
- [Installation](#installation)
- [How to use](#how-to-use)
  - [pyBEEP GUI](#pybeep-gui)
  - [Examples](#examples)
- [File Structure](#file-structure)
- [Notes](#notes)
- [Bugs & Support](#bugs--support)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

# Main Functionalities

- **Easy experiment setup and execution:**  
  Use `PotentiostatController.apply_measurement()` to run a wide range of electrochemical experiments.
  - **Supported modes:**  
    - Discover all available experiment types with `get_available_modes()`.
    - For each mode, list all required and optional parameters with `get_mode_params(mode)`.
  - **Configurable parameters include:**  
    - `mode` (`str`): Type of experiment (e.g., `'CA'`, `'CP'`, `'CV'`, `'STEPSEQ'`, etc.)
    - *Mode-specific parameters*: (e.g., `start`, `vertex`, `scan_rate`, `currents`, etc.)
    - `tia_gain` (`int`): Sets current range by selecting the transimpedance amplifier gain.
    - `filename`, `folder` (`str`): Output file name and results directory.
    - `plot` (`bool`): Automatically show a plot after data acquisition.
- **Default results folder management:**  
  - Set a default output folder using `set_default_folder()`.
  - Retrieve or prompt for the default folder with `get_default_folder()`.

- **Low-level device control:**  
  - Access and manage the underlying potentiostat hardware via the `PotentiostatDevice` class (Modbus serial communication).

- **Automated data logging and visualization:**  
  - All experiment data is saved automatically as CSV files.
  - Optional plotting after measurement using matplotlib.

---

# Installation

Clone this repository and install using pip:

```bash
git clone https://github.com/aurelienblanc2/Potentiostat-driver-pyBEEP
cd pyBEEP
pip install .
```

---

# How to use

## pyBEEP GUI

After installing the package, a terminal command is available to open the pyBEEP GUI
```bash
pyBEEP_GUI
```
This function can also be called directly in Python once the package is imported
```python
import pyBEEP

pyBEEP.launch_GUI()
```

The GUI provides a simple way to configure and run an experiment.

## Examples

A very simple example to start using the Potentiostat driver is provided below. Keep in mind that the [examples](https://github.com/aurelienblanc2/Potentiostat-driver-pyBEEP/tree/main/examples) 
folder in the repository provides use cases for the different modes of the potentiostat.

```python
from pyBEEP import PotentiostatDevice, PotentiostatController

device = PotentiostatDevice(port='COM3', address=1)  # Adjust port and address as needed
controller = PotentiostatController(device=device)

# Run a cyclic voltametry experiment
controller.apply_measurement(
     mode= "CV",
     params= {
         'start': -0.25,
         'vertex': 1.8,
         'end': 0,
         'scan_rate': 0.1,
         'cycles': 3
     },
     tia_gain=0,
     filename= "result_ca.csv",
     folder= 'C:/experiments',
)
        

# Run a stepped chronopotentiometry experiment
controller.apply_measurement(
     mode = "STEPSEQ",
     params= {
       'currents': [0.002, 0.01, 0.05],
       'duration': 5
     },
     tia_gain=0,
     filename= "result_ca.csv",
     folder= 'C:/experiments',
)
```

---

# File Structure

```
pyBEEP/
├── pyproject.toml               # Project configuration
├── README.md                    # This file
├── docs                         # Folder for the ressources used by the README.md
├── LICENSE                      # MIT
├── requirements.txt             # Python dependencies
├── uv.lock                      # Lockfile used by UV for reproducible builds
├── .pre-commit-config.yaml      # Pre-commit configuration for developing this package
│
├── examples/                    # Example scripts for running experiments
│   ├── example_CA.py
│   ├── example_CA_TIA.py
│   ├── example_CV.py
│   ├── example_CV_TIA.py
│   ├── example_EIS.py
│   ├── example_GCV.py
│   ├── example_LSV.py
│   ├── example_OCP.py
│   ├── example_PSTEP.py
│   └── examples_methods.py
│
├── src/
│   └── pyBEEP/
│       ├── __init__.py 
│       ├── controller.py        # High-level control and experiment logic
│       ├── device.py            # Low-level Modbus device communication
│       ├── logger.py            # Threaded data logging to file
│       ├── plotter.py           # Data plotting utilities
│       ├── utils
│       │   ├── __init__.py 
│       │   ├── utils.py            # Utility functions (e.g. file/folder selection)
│       │   └──constants.py         # Hardware and experiment constants
│       │
│       ├── measurement_modes
│       │   ├── __init__.py 
│       │   ├── measurement_modes.py
│       │   ├── waveform_outputs.py
│       │   ├── waveform_params.py
│       │   ├── waveforms_gal.py
│       │   ├── waveforms_ocp.py
│       │   └── waveforms_pot.py
│       │
│       └── gui
│           ├── __init__.py
│           └── main_window.py
│
└── tests/
    └── test_init.py             # Test files for the proper package import check

```

- **examples/** contains working scripts that demonstrate how to use pyBEEP for different types of experiments.  
- **src/pyBEEP/** is the main package source code.

---

# Notes

- Select the TIA gain (`tia_gain`) carefully to match your experimental current/potential range.
- The library creates CSV files with timestamp, experiment type, and parameters in the filename.
- Data is logged as a numpy array with potential (V) and current (A) pairs.

---

# Bugs & Support

If you encounter a bug, have a feature request, or need help:
- contact: aurelien.blanc@utoronto.ca
- Or contact: adpisa@gmail.com

---

# Contributing

Contributions are very welcome!  
If you’d like to add features, fix bugs, or improve documentation, please submit a merge request or open an issue to discuss your ideas.

---

# License

MIT License

---

# Author

Adrián Pinilla-Sánchez - adpisa@gmail.com  
Aurelien Blanc - aurelien.blanc@utoronto.ca

---
