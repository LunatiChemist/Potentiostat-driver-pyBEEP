import numpy as np

from pyBEEP.measurement_modes.waveform_outputs import (
    PotenOutput,
    SteppedPotenOutput,
    CyclicPotenOutput,
    EisPotenOutput,
)
from pyBEEP.utils.constants import POINT_INTERVAL, POINT_INTERVAL_EIS


def constant_waveform(potential: float, duration: float) -> PotenOutput:
    """
    Generates a constant waveform for a specified duration.

    Args:
        potential (float): The constant value (e.g., voltage or current) to apply.
        duration (float): Total time (in seconds) for which the value is held.

    Returns:
        PotenOutput: Pydantic model with:
            - applied_potential (np.ndarray): Constant potential, shape (N,)
            - time (np.ndarray): Time vector (s), shape (N,)
    """
    length = int(duration / POINT_INTERVAL)
    applied_potential = np.full(length, potential, dtype=np.float32)
    time = np.arange(length) * POINT_INTERVAL
    return PotenOutput(
        applied_potential=applied_potential,
        time=time,
    )


def potential_steps(
    potentials: list[float], step_duration: float
) -> SteppedPotenOutput:
    """
    Generates a potentiostatic waveform consisting of consecutive potential steps,
    each held for the specified duration.

    Args:
        potentials (list[float]): List of potential values (e.g., in Volts) to apply sequentially.
        step_duration (float): Duration (in seconds) for which each potential is held.

    Returns:
        SteppedPotenOutput: Pydantic model with:
            - applied_potential (np.ndarray): Concatenated potentials, shape (N,)
            - time (np.ndarray): Time vector (s), shape (N,)
            - step (np.ndarray): Step indices (0-based), shape (N,)
    """
    length_step = int(step_duration / POINT_INTERVAL)
    applied_potential = np.concatenate(
        [np.full(length_step, potential, dtype=np.float32) for potential in potentials]
    )
    total_length = len(applied_potential)
    time = np.arange(total_length) * POINT_INTERVAL
    step = np.concatenate(
        [np.full(length_step, i, dtype=np.int32) for i in range(len(potentials))]
    )
    return SteppedPotenOutput(
        applied_potential=applied_potential,
        time=time,
        step=step,
    )


def linear_sweep(start: float, end: float, scan_rate: float) -> PotenOutput:
    """
    Generates a linear sweep waveform from a start to an end value at a fixed scan rate.

    Args:
        start (float): Starting value (e.g., voltage or current).
        end (float): Ending value.
        scan_rate (float): Rate of change per second (units per second).

    Returns:
        PotenOutput: Pydantic model with:
            - applied_potential (np.ndarray): Linearly ramped potential, shape (N,)
            - time (np.ndarray): Time vector (s), shape (N,)
    """
    duration = abs(end - start) / scan_rate
    length = int(duration / POINT_INTERVAL)
    applied_potential = np.linspace(start, end, length, dtype=np.float32)
    time = np.arange(length) * POINT_INTERVAL
    return PotenOutput(
        applied_potential=applied_potential,
        time=time,
    )


def cyclic_voltammetry(
    start: float,
    vertex1: float,
    vertex2: float,
    end: float,
    scan_rate: float,
    cycles: int,
) -> CyclicPotenOutput:
    """
    Generates a cyclic voltammetry waveform with asymmetric start and cycles.

    First cycle: start → vertex1 → vertex2
    Then cycles 2 to N: vertex2 → vertex1 → vertex2
    Final segment (if end ≠ vertex2): vertex2 → vertex1 → vertex2 → end

    Args:
        start (float): Initial potential.
        vertex1 (float): First vertex potential.
        vertex2 (float): Second vertex potential.
        end (float): Final potential after the last cycle.
        scan_rate (float): Scan rate (V/s).
        cycles (int): Number of full cycles (excluding first initial sweep).

    Returns:
        CyclicPotenOutput: Pydantic model with:
            - applied_potential (np.ndarray): Cyclic potential waveform, shape (N,)
            - time (np.ndarray): Time vector (s), shape (N,)
            - cycle (np.ndarray): Cycle index (1-based), shape (N,)
    """
    segments = []
    cycle = []

    # First cycle: start → vertex1 → vertex2
    seg1 = linear_sweep(start, vertex1, scan_rate).applied_potential
    seg2 = linear_sweep(vertex1, vertex2, scan_rate).applied_potential
    segments.extend([seg1, seg2])
    cycle.extend([1] * (len(seg1) + len(seg2)))

    # Middle cycles: vertex2 → vertex1 → vertex2
    for n in range(2, cycles + 1):
        seg_up = linear_sweep(vertex2, vertex1, scan_rate).applied_potential
        seg_down = linear_sweep(vertex1, vertex2, scan_rate).applied_potential
        segments.extend([seg_up, seg_down])
        cycle.extend([n] * (len(seg_up) + len(seg_down)))

    # Final segment (optional): vertex2 → vertex1 → vertex2 → end
    if end != vertex2:
        seg_extra = linear_sweep(vertex2, end, scan_rate).applied_potential
        segments.extend(
            [
                seg_extra,
            ]
        )
        cycle.extend([cycles] * len(seg_extra))

    applied_potential = np.concatenate(segments)
    total_length = len(applied_potential)
    time = np.arange(total_length) * POINT_INTERVAL  # Must be defined globally
    cycle = np.array(cycle, dtype=np.int32)

    return CyclicPotenOutput(
        applied_potential=applied_potential,
        time=time,
        cycle=cycle,
    )


def eis_sweep(
    start_freq: int,
    end_freq: int,
    duration: float,
) -> EisPotenOutput:
    """
    Generate a linear voltage sweep waveform.

    Args:
        start_freq (int): Initial frequency in Hz
        end_freq (int): End frequency in Hz
        duration (float): Duration of the seep in seconds

    Returns:
        CyclicPotenOutput: Pydantic model with:
            - start_freq (int): Initial frequency in Hz
            - end_freq (int): End frequency in Hz
            - duration (float): Duration of the seep in seconds
    """

    time = np.arange(0, duration + POINT_INTERVAL_EIS, POINT_INTERVAL_EIS)

    # Sweep rate
    k = (end_freq - start_freq) / duration

    # Command signal
    applied_potential = np.sin(2 * np.pi * (start_freq * time + 0.5 * k * time**2))

    return EisPotenOutput(
        applied_potential=applied_potential,
        time=time,
        start_freq=start_freq,
        end_freq=end_freq,
        duration=duration,
    )
