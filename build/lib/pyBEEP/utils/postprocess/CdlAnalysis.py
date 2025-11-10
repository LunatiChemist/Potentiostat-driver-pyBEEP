# utils/postprocess/CdlAnalysis.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def _middle_window_mask(E: np.ndarray, edge_exclude: float):
    lo = np.nanquantile(E, edge_exclude)
    hi = np.nanquantile(E, 1.0 - edge_exclude)
    return (E >= lo) & (E <= hi), float(lo), float(hi)


def _compute_scanrate_from_cycle(df: pd.DataFrame, edge_exclude: float = 0.15) -> float:
    """Robust scan-rate (V/s) from median(|dE/dt|) in the middle potential window."""
    E = df["_E"].to_numpy()
    t = df["Time (s)"].to_numpy()
    dE = np.gradient(E)
    dt = np.gradient(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        dEdt = np.where(dt != 0, dE / dt, np.nan)
    mask, _, _ = _middle_window_mask(E, edge_exclude)
    return float(np.nanmedian(np.abs(dEdt[mask])))


def _label_branches(df: pd.DataFrame) -> pd.DataFrame:
    """Anodic/cathodic by sign of dE/dt on E."""
    E = df["_E"].to_numpy()
    dE = np.gradient(E)
    out = df.copy()
    out["_branch"] = np.where(dE >= 0, "an", "kat")
    return out


def _choose_cycles_present(
    df: pd.DataFrame, prefer_cycles: Optional[list]
) -> Optional[list]:
    if "Cycle" not in df.columns or df["Cycle"].isna().all():
        return None
    all_cycles = sorted(pd.unique(df["Cycle"]))
    if prefer_cycles:
        chosen = [c for c in all_cycles if c in set(prefer_cycles)]
        return chosen or all_cycles
    # default: drop first cycle if more than one (conditioning)
    if len(all_cycles) > 1:
        return [c for c in all_cycles if c != min(all_cycles)]
    return all_cycles


def _through_origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom > 0 else np.nan


def _through_origin_r2(x: np.ndarray, y: np.ndarray, slope: float) -> float:
    y_pred = slope * x
    sse = float(np.sum((y - y_pred) ** 2))
    sst0 = float(np.sum(y**2))
    return float(1.0 - sse / sst0) if sst0 > 0 else np.nan


def estimate_cdl_from_csv(
    filepath: str,
    vertex_a: Optional[float] = None,
    vertex_b: Optional[float] = None,
    window: float = 0.01,  # half-width around E* for medians
    *,
    cycles_to_use: Optional[list] = None,
    edge_exclude: float = 0.15,
    points_round: int = 4,
    save_points_csv: bool = True,
) -> Dict[str, Any]:
    """
    Estimate double-layer capacitance (C_dl) from a single CV CSV with multiple cycles.

    Method (per cycle):
      - Choose a single target potential E*:
          * if vertex_a & vertex_b given: E* = (A+B)/2
          * else: E* = median(E)
      - I_cap(E*) = (median(I_an in [E*-w, E*+w]) - median(I_cat in [E*-w, E*+w])) / 2
      - Scan-rate v: median(|dE/dt|) in the middle potential window
      - Fit through origin: I_cap = C_dl * v  (intercept fixed to 0)

    Returns final C_dl (F/mF), R², cycle points and an optional *_cdl.csv with points.
    """
    df = pd.read_csv(filepath)

    # Columns
    time_col = "Time (s)" if "Time (s)" in df.columns else None
    pot_col = (
        "Applied potential (V)"
        if "Applied potential (V)" in df.columns
        else ("Potential (V)" if "Potential (V)" in df.columns else None)
    )
    curr_col = "Current (A)"
    if time_col is None or pot_col is None or curr_col not in df.columns:
        raise ValueError(
            "CSV must contain ['Time (s)', '...potential...', 'Current (A)']."
        )

    df = df.sort_values(time_col).reset_index(drop=True)
    df["_E"] = df[pot_col].astype(float)  # no IR correction
    df[curr_col] = df[curr_col].astype(float)

    # Cycle selection
    use_cycles = _choose_cycles_present(df, cycles_to_use)
    if use_cycles is not None and "Cycle" in df.columns:
        df = df[df["Cycle"].isin(use_cycles)].copy()

    # Branch tagging
    df = _label_branches(df)

    # Define single target E*
    if (vertex_a is not None) and (vertex_b is not None):
        E_star = float(0.5 * (vertex_a + vertex_b))
    else:
        E_star = float(np.nanmedian(df["_E"]))

    # Walk cycles
    if "Cycle" in df.columns:
        cycles = sorted(pd.unique(df["Cycle"]))
    else:
        df["_synthetic_cycle"] = 1
        df["Cycle"] = df.get("Cycle", df["_synthetic_cycle"])
        cycles = [1]

    per_cycle = []
    for cyc in cycles:
        dcy = df[df["Cycle"] == cyc].copy()
        if dcy.empty:
            continue

        # scan-rate
        v = _compute_scanrate_from_cycle(dcy, edge_exclude=edge_exclude)

        # window around E*
        win = dcy[(dcy["_E"] >= E_star - window) & (dcy["_E"] <= E_star + window)]
        if win.empty:
            continue

        Ian = np.nanmedian(win.loc[win["_branch"] == "an", curr_col])
        Ikat = np.nanmedian(win.loc[win["_branch"] == "kat", curr_col])
        if not (np.isfinite(Ian) and np.isfinite(Ikat)):
            continue

        Icap = float((Ian - Ikat) / 2.0)
        per_cycle.append({"cycle": int(cyc), "scan_rate": float(v), "Icap": Icap})

    # Aggregate by scan-rate (median over cycles with ~same v)
    pts_df = pd.DataFrame(per_cycle)
    if pts_df.empty:
        scan_rates = np.array([], dtype=float)
        Icaps = np.array([], dtype=float)
        n_cycles_pt = np.array([], dtype=int)
    else:
        v_key = pts_df["scan_rate"].round(points_round)
        grouped = pts_df.groupby(v_key, dropna=True)
        pts = grouped.agg(
            scan_rate=("scan_rate", "median"),
            Icap=("Icap", "median"),
            n_cycles=("Icap", "size"),
        ).reset_index(drop=True)
        scan_rates = pts["scan_rate"].to_numpy(dtype=float)
        Icaps = pts["Icap"].to_numpy(dtype=float)
        n_cycles_pt = pts["n_cycles"].to_numpy(dtype=int)

    # Fit through origin
    slope = _through_origin_slope(scan_rates, Icaps) if len(scan_rates) >= 1 else np.nan
    r2 = (
        _through_origin_r2(scan_rates, Icaps, slope)
        if np.isfinite(slope) and len(scan_rates) >= 2
        else np.nan
    )
    Cdl_F = slope
    Cdl_mF = Cdl_F * 1e3 if np.isfinite(Cdl_F) else np.nan

    # Points CSV for plotter (same suffix as before)
    out_csv = None
    if save_points_csv and len(scan_rates) > 0:
        out = pd.DataFrame(
            {"scan_rate_V_per_s": scan_rates, "Icap_A": Icaps, "n_cycles": n_cycles_pt}
        )
        out_csv = filepath.replace(".csv", "_cdl.csv")
        out.to_csv(out_csv, index=False)

    return {
        "Cdl_F": float(Cdl_F) if np.isfinite(Cdl_F) else np.nan,
        "Cdl_mF": float(Cdl_mF) if np.isfinite(Cdl_mF) else np.nan,
        "slope": float(Cdl_F) if np.isfinite(Cdl_F) else np.nan,
        "intercept": 0.0,
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "points": [
            {"scan_rate": float(v), "Icap": float(i), "n_cycles": int(n)}
            for v, i, n in zip(scan_rates, Icaps, n_cycles_pt)
        ],
        "per_cycle": per_cycle,
        "csv": out_csv,
        "n_points": int(len(scan_rates)),
        "n_cycles_total": int(len(per_cycle)),
        "E_star": E_star,
        "window": float(window),
        "edge_exclude": float(edge_exclude),
    }
