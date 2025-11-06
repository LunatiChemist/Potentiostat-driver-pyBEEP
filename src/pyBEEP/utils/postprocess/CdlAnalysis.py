# src/pyBEEP/post/CdlAnalysis.py
import numpy as np
import pandas as pd
from typing import Dict, Any


def estimate_cdl_from_csv(
    filepath: str,
    vertex_a: float,
    vertex_b: float,
    window: float = 0.01,  # ±10 mV um Vmid mitteln
) -> Dict[str, Any]:
    """Schätze Cdl aus einer CV-Sequenz-CSV (Logger-Format).
    Methode: mittlere i_an/i_cat in kleinem Potentialfenster um Vmid,
    v = dV/dt lokal aus der Spalte 'Applied potential (V)' und 'Time (s)'.
    """
    df = pd.read_csv(filepath)

    # Fallbacks für Spaltennamen aus dem Logger
    time = df["Time (s)"].to_numpy() if "Time (s)" in df else np.arange(len(df))
    pot_applied = (
        df["Applied potential (V)"].to_numpy()
        if "Applied potential (V)" in df
        else df["Potential (V)"].to_numpy()
    )
    curr = df["Current (A)"].to_numpy()
    cycles = df["Cycle"].to_numpy() if "Cycle" in df else np.ones(len(df), dtype=int)

    v_mid = 0.5 * (vertex_a + vertex_b)
    dvdt = np.gradient(pot_applied, time, edge_order=2)
    dir_sign = np.sign(dvdt)

    results = []
    for cyc in np.unique(cycles):
        m = (cycles == cyc) & (np.abs(pot_applied - v_mid) <= window)
        pos = m & (dir_sign > 0)  # anodisch
        neg = m & (dir_sign < 0)  # kathodisch
        if not np.any(pos) or not np.any(neg):
            continue

        i_an = float(np.median(curr[pos]))
        i_cat = float(np.median(curr[neg]))  # < 0
        v_pos = float(np.median(dvdt[pos]))
        v_neg = float(-np.median(dvdt[neg]))
        v_local = np.mean([abs(v_pos), abs(v_neg)])
        cdl = (i_an - i_cat) / (2.0 * v_local)  # i_cat ist negativ

        results.append(
            {
                "cycle": int(cyc),
                "scan_rate": v_local,
                "i_an": i_an,
                "i_cat": i_cat,
                "c_dl": cdl,
            }
        )

    out = {
        "per_cycle": results,
        "cdl_mean": float("nan"),
        "cdl_std": float("nan"),
        "csv": None,
    }
    if results:
        df_out = pd.DataFrame(results)
        out_csv = filepath.replace(".csv", "_cdl.csv")
        df_out.to_csv(out_csv, index=False)
        out.update(
            {
                "cdl_mean": float(df_out["c_dl"].mean()),
                "cdl_std": float(
                    df_out["c_dl"].std(ddof=1) if len(df_out) > 1 else 0.0
                ),
                "csv": out_csv,
            }
        )
    return out
