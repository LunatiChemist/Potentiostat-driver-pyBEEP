import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Dict


# -----------------------------
# Core functions
# -----------------------------
def _dir_masks(potential: pd.Series, time: pd.Series) -> Tuple[pd.Series, pd.Series]:
    dEdt = np.gradient(potential.values, time.values)
    return pd.Series(dEdt > 0, index=potential.index), pd.Series(
        dEdt < 0, index=potential.index
    )


def _central_window(
    potential: pd.Series,
    base_mask: pd.Series,
    lower_q: float = 0.2,
    upper_q: float = 0.8,
) -> pd.Series:
    E = potential[base_mask]
    if E.empty:
        return potential.index == -1  # empty mask
    lo = E.quantile(lower_q)
    hi = E.quantile(upper_q)
    return base_mask & (potential >= lo) & (potential <= hi)


def _estimate_scanrate(time: pd.Series, potential: pd.Series, mask: pd.Series) -> float:
    t = time[mask].values
    E = potential[mask].values
    if len(t) < 3:
        return np.nan
    dE = np.diff(E)
    dt = np.diff(t)
    return float(np.median(np.abs(dE / dt)))


def _trapz(time: pd.Series, current: pd.Series, mask: pd.Series) -> float:
    t = time[mask].values
    i = current[mask].values
    if len(t) < 2:
        return np.nan
    return float(np.trapz(i, t))


def _infer_columns(df: pd.DataFrame) -> Dict[str, str]:
    # Try common column names
    cols = {
        "time": None,
        "potential": None,
        "current": None,
        "cycle": None,
    }
    cand_time = ["Time (s)", "time_s", "time", "t"]
    cand_E = ["Potential (V)", "E (V)", "potential", "E"]
    cand_I = ["Current (A)", "I (A)", "current", "I"]
    cand_cycle = ["Cycle", "cycle", "segment"]
    for c in cand_time:
        if c in df.columns:
            cols["time"] = c
            break
    for c in cand_E:
        if c in df.columns:
            cols["potential"] = c
            break
    for c in cand_I:
        if c in df.columns:
            cols["current"] = c
            break
    for c in cand_cycle:
        if c in df.columns:
            cols["cycle"] = c
            break
    missing = [k for k, v in cols.items() if v is None and k != "cycle"]
    if missing:
        raise ValueError(
            f"Spalten nicht gefunden (mind. Zeit/Spannung/Strom nötig). Gefunden: {df.columns.tolist()}"
        )
    return cols


def load_ocp_offset(
    ocp_csv_path: Optional[str], current_col_candidates: List[str] = None
) -> Optional[float]:
    """Liest eine OCP-CSV und gibt den mittleren Strom als Offset zurück."""
    if ocp_csv_path is None:
        return None
    if current_col_candidates is None:
        current_col_candidates = ["Current (A)", "I (A)", "current", "I"]
    ocp = pd.read_csv(ocp_csv_path)
    Icol = next((c for c in current_col_candidates if c in ocp.columns), None)
    if Icol is None:
        raise ValueError(
            f"In OCP-Datei keine Stromspalte gefunden. Spalten: {ocp.columns.tolist()}"
        )
    return float(ocp[Icol].mean())


def compute_capacitance_from_cv(
    cv_csv_path: str,
    ocp_csv_path: Optional[str] = None,
    ocp_offset: Optional[float] = None,
    lower_q: float = 0.2,
    upper_q: float = 0.8,
    save_prefix: Optional[str] = None,
    make_plots: bool = True,
) -> pd.DataFrame:
    """
    Berechnet die Kapazität aus einem CV mit zwei Methoden:
      (1) Plateau (strombasiert):  C = |i_fwd_avg - i_rev_avg| / (2*v)
      (2) Ladung (integriert):     C = |Q_fwd - Q_rev| / (2*ΔV)
    Optional: OCP-Offset abziehen (aus Datei oder als Zahl).
    """
    df = pd.read_csv(cv_csv_path)
    cols = _infer_columns(df)
    t = df[cols["time"]]
    E = df[cols["potential"]]
    I = df[cols["current"]].copy()

    # Offset bestimmen
    if ocp_offset is None and ocp_csv_path is not None:
        ocp_offset = load_ocp_offset(ocp_csv_path)
    if ocp_offset is not None:
        I = I - ocp_offset  # Offsetkorrektur

    # Gruppen (Cycles) vorbereiten
    if cols["cycle"] and cols["cycle"] in df.columns:
        groups = [
            (
                f"Cycle {int(c)}" if float(c).is_integer() else f"Cycle {c}",
                df[df[cols["cycle"]] == c],
            )
            for c in sorted(df[cols["cycle"]].unique())
        ]
        if len(groups) > 1:
            groups.append(("All", df))
    else:
        groups = [("All", df)]

    rows = []
    plot_paths = []

    for label, d in groups:
        time = d[cols["time"]]
        pot = d[cols["potential"]]
        cur = (
            (d[cols["current"]] - ocp_offset)
            if ocp_offset is not None
            else d[cols["current"]]
        )

        fwd_mask, rev_mask = _dir_masks(pot, time)
        fwd_central = _central_window(pot, fwd_mask, lower_q, upper_q)
        rev_central = _central_window(pot, rev_mask, lower_q, upper_q)

        v_fwd = _estimate_scanrate(time, pot, fwd_central)
        v_rev = _estimate_scanrate(time, pot, rev_central)
        v_mean = np.nanmean([v_fwd, v_rev])

        i_fwd = float(cur[fwd_central].mean())
        i_rev = float(cur[rev_central].mean())
        C_plateau = (
            abs(i_fwd - i_rev) / (2.0 * v_mean)
            if np.isfinite(v_mean) and v_mean != 0
            else np.nan
        )

        Q_fwd = _trapz(time, cur, fwd_central)
        Q_rev = _trapz(time, cur, rev_central)
        dV_fwd = (
            float(
                pot[fwd_central].quantile(upper_q) - pot[fwd_central].quantile(lower_q)
            )
            if fwd_central.any()
            else np.nan
        )
        dV_rev = (
            float(
                pot[rev_central].quantile(upper_q) - pot[rev_central].quantile(lower_q)
            )
            if rev_central.any()
            else np.nan
        )
        dV_mean = np.nanmean([dV_fwd, dV_rev])
        C_charge = (
            abs(Q_fwd - Q_rev) / (2.0 * dV_mean)
            if np.isfinite(dV_mean) and dV_mean != 0
            else np.nan
        )

        rows.append(
            {
                "cycle": label,
                "v_forward_est (V/s)": v_fwd,
                "v_reverse_est (V/s)": v_rev,
                "v_mean_est (V/s)": v_mean,
                "i_forward_avg (A)": i_fwd,
                "i_reverse_avg (A)": i_rev,
                "C_plateau (F)": C_plateau,
                "Q_forward (C)": Q_fwd,
                "Q_reverse (C)": Q_rev,
                "ΔV_mean (V)": dV_mean,
                "C_charge (F)": C_charge,
            }
        )

        if make_plots:
            plt.figure(figsize=(7, 5))
            plt.plot(pot, cur, linewidth=0.8, label="CV")
            plt.scatter(pot[fwd_central], cur[fwd_central], s=6, label="forward window")
            plt.scatter(pot[rev_central], cur[rev_central], s=6, label="reverse window")
            ttl = f"CV selection – {label}"
            if ocp_offset is not None:
                ttl += f" (offset {ocp_offset:.3e} A)"
            plt.title(ttl)
            plt.xlabel("Potential (V)")
            plt.ylabel("Current (A)")
            plt.legend()
            plt.tight_layout()
            if save_prefix:
                outp = Path(f"{save_prefix}_selection_{label.replace(' ','_')}.png")
                plt.savefig(outp, dpi=150)
                plot_paths.append(str(outp))
            plt.close()

    res = pd.DataFrame(rows)
    if save_prefix:
        out_csv = Path(f"{save_prefix}_capacitance_summary.csv")
        res.to_csv(out_csv, index=False)
        print(f"Summary saved to: {out_csv}")
        if plot_paths:
            print("Plots:", *plot_paths, sep="\n  ")
    return res


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Pfade anpassen:
    cv_path = r"tests\test_CV_0.8.csv"  # dein CV
    ocp_path = None  # optional: "your_OCP.csv"
    ocp_I_offset = -6.2e-5  # optional: z.B. -6.2e-5 (A)

    # Rechne und speichere zusätzlich Dateien mit Prefix:
    df_out = compute_capacitance_from_cv(
        cv_csv_path=cv_path,
        ocp_csv_path=ocp_path,  # oder None
        ocp_offset=ocp_I_offset,  # oder None (dann wird ocp_path genommen)
        lower_q=0.2,
        upper_q=0.8,
        save_prefix="cv",  # z.B. "cv" => cv_capacitance_summary.csv & Plots
        make_plots=True,
    )
    print(df_out)
