import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_time_series(
    filepaths: str | list[str],
    figpath: str | None = None,
    show: bool = False,
):
    """
    Plot current and potential vs time for CA, CP, GS, etc.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle("Current & Potential vs Time")

    for fp in filepaths:
        data = pd.read_csv(fp)
        label = os.path.basename(fp)
        axs[0].plot(data["Time (s)"], data["Current (A)"], label=label)  # Current (A)
        axs[1].plot(
            data["Time (s)"], data["Potential (V)"], label=label
        )  # Potential (V)

    axs[0].set_ylabel("Current (A)", color="tab:red")
    axs[1].set_ylabel("Potential (V)", color="tab:blue")
    axs[1].set_xlabel("Time (s)")

    if len(filepaths) > 1:
        axs[0].legend()
        axs[1].legend()

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if show:
        plt.show()
    if figpath:
        fig.savefig(figpath)
    plt.close(fig)


def plot_iv_curve(
    filepaths: str | list[str], figpath: str | None = None, show: bool = False
):
    """
    Plot current vs potential for LSV, CV, GCV, etc.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Current vs Potential")

    for fp in filepaths:
        data = pd.read_csv(fp)
        label = os.path.basename(fp)
        ax.plot(data["Potential (V)"], data["Current (A)"], label=label)

    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current (A)")
    if len(filepaths) > 1:
        ax.legend()
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if show:
        plt.show()
    if figpath:
        fig.savefig(figpath)
    plt.close(fig)


def plot_cv_cycles(
    filepaths: str | list[str],
    figpath: str | None = None,
    show: bool = False,
    cycles: int | None = None,
):
    """
    Plot CV data with each cycle shown in a different color.
    Accepts list of filepaths; cycles in each file are plotted as separate groups.
    Assumes the data in each file is ordered as [current, potential] rows, scans concatenated.
    Provide scan_points (points per scan, optional) and cycles (optional) if known.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Cyclic Voltammetry (CV) - Individual Cycles")

    color_map = plt.get_cmap("tab10")
    color_idx = 0

    for fp in filepaths:
        data = pd.read_csv(fp)

        if cycles is not None:
            for n, cycles in zip(data["Cycle"].unique(), range(1, cycles + 1)):
                label = (
                    f"{os.path.basename(fp)} - Cycle {n}"
                    if len(filepaths) > 1 or len(data["Cycle"].unique()) > 1
                    else os.path.basename(fp)
                )
                data_cycle = data[data["Cycle"] == n]
                ax.plot(
                    data_cycle["Potential (V)"],
                    data_cycle["Current (A)"],
                    label=label,
                    color=color_map(color_idx % 10),
                )
                color_idx += 1

    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current (A)")
    ax.legend()
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if show:
        plt.show()
    if figpath:
        fig.savefig(figpath)
    plt.close(fig)
