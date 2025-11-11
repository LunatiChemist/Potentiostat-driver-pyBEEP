import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_Cedl_from_folder(folder_path, well_number=1, target_voltage=0.1):
    """
    Minimal-invasive Erweiterung von compute_Cedl, die alle test_CV_*.csv Dateien
    im angegebenen Ordner verarbeitet und daraus C_dl berechnet.

    Erwartet:
        - Dateien heißen z.B. test_CV_0.1.csv, test_CV_0.3.csv, ...
        - Jede Datei enthält Spalten wie ['Time (s)', 'Potential (V)', 'Current (A)', 'Cycle', 'Exp']
    """

    # --- Alle CSV-Dateien holen ---
    csv_files = [
        f for f in os.listdir(folder_path) if f.startswith("test_CV_") and f.endswith(".csv")
    ]
    if not csv_files:
        raise FileNotFoundError("Keine test_CV_*.csv Dateien im angegebenen Ordner gefunden.")

    # --- Feste Reihenfolge von 1.0 bis 0.1 ---
    scanrate_order = ["1.0", "0.9", "0.8", "0.7", "0.5"]

    # sortiere csv_files nach Reihenfolge oben, falls enthalten
    csv_files = [f for rate in scanrate_order for f in csv_files if f"_{rate}" in f]
    scan_rates = []
    Cedl_currents = []

    for file in csv_files:
        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        df = df.sort_values(["Time (s)"]).reset_index(drop=True)
        df["Current_mA"] = df["Current (A)"] * 1e3

        # Scanrate aus dE/dt schätzen
        dE = np.gradient(df["Potential (V)"].to_numpy())
        dt = np.gradient(df["Time (s)"].to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            dEdt = np.where(dt != 0, dE / dt, np.nan)
        sr = np.nanmedian(np.abs(dEdt))
        scan_rates.append(sr)

        # Strom bei target_voltage extrahieren (nächster Punkt)
        idx = np.argmin(np.abs(df["Potential (V)"] - target_voltage))
        Cedl_currents.append(df["Current_mA"].iloc[idx])

    scan_rates = np.array(scan_rates)
    Cedl_currents = np.array(Cedl_currents)

    # --- Lineare Regression (I vs. Scanrate) ---
    coefficients = np.polyfit(scan_rates, Cedl_currents, 1)
    slope, intercept = coefficients
    line_fit = slope * scan_rates + intercept
    Cedl = slope / 2  # mF

    print(f"{len(csv_files)} CV-Dateien gefunden:")
    for name, sr in zip(csv_files, scan_rates):
        print(f"  {name:<20s}  ≈ {sr:.3f} V/s")

    print(f"\nBerechnete Doppelschichtkapazität (C_dl): {Cedl:.5f} mF")

    # --- Plots ---
    volt_dir = os.path.join(folder_path, "Voltammetry Data")
    os.makedirs(volt_dir, exist_ok=True)

    fig, ax = plt.subplots()
    ax.scatter(scan_rates, Cedl_currents, label=f"I bei {target_voltage:.2f} V")
    ax.plot(scan_rates, line_fit, linewidth=1.7, label="lineare Regression")
    ax.set_title("Double Layer Capacitance", fontsize=20, style="italic")
    ax.set_xlabel("Scan Rate (V/s)", fontsize=16)
    ax.set_ylabel("Current (mA)", fontsize=16)
    ax.axhline(0, color="#DEDEDE", linestyle="--")
    ax.legend(loc="lower right")
    ax.text(
        0.05,
        0.95,
        f"Slope = {slope:.5f} mA/(V/s)",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    fig.tight_layout()
    fig.savefig(os.path.join(volt_dir, "Cedl_from_folder.png"), dpi=180)
    plt.close(fig)

    return Cedl, slope, intercept, scan_rates, Cedl_currents

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_Cdl_from_folder_pro(
    folder_path,
    target_potentials=None,  # Liste von E* (V); None => automatisch 5 Werte im mittleren Fenster
    band=0.01,  # Halbbreite um E* (V) für Medianbildung
    cycles_to_use=None,  # z.B. (2,3,4); None => alle vorhandenen, außer ggf. Cycle 1 wenn >1 vorhanden
    Ru=None,  # Ohm; None => keine iR-Korrektur
    save_plot=True,
    show_plot=True,
    file_prefix="test_CV_",
    file_suffix=".csv",
    scanrate_order=(
        "1.0",
        "0.9",
        "0.8",
        "0.7",
        "0.6",
        "0.5",
        "0.4",
        "0.3",
        "0.2",
        "0.1",
    ),
    edge_exclude=0.15,  # Anteil des Potenzialfensters, der an den Rändern ignoriert wird
    verbose=True,
):
    """
    Liest test_CV_*.csv Dateien, bestimmt C_dl über I_cap vs. Scanrate und erstellt einen Plot.

    WICHTIGSTE PRINZIPIEN:
      - I_cap(E*) = (I_an(E*) - I_kat(E*))/2  (Median in einem ±band Fenster)
      - Mehrere E* im nicht-faradaischen Mittelbereich -> Median über E*
      - Scanrate v aus linearem Fit E(t) im mittleren Potenzialbereich
      - Optionale iR-Korrektur: E_corr = E - I*Ru
      - Robuster gegen Rauschen und Faradaik an den Umkehrpunkten

    Erwartete Spalten in jedem CSV:
        ['Time (s)', 'Potential (V)', 'Current (A)', 'Cycle']  (weitere Spalten werden ignoriert)

    Rückgabe:
        dict mit
        - 'Cdl_F' (F), 'Cdl_mF' (mF), 'slope' (F), 'intercept' (A), 'R2',
        - 'scan_rates' (V/s), 'Icap' (A) pro Datei,
        - 'by_file' (Liste mit pro-Datei-Infos)
    """

    # --- Dateien sammeln & sortieren in gewünschter Reihenfolge ---
    csv_files = [
        f
        for f in os.listdir(folder_path)
        if f.startswith(file_prefix) and f.endswith(file_suffix)
    ]
    if not csv_files:
        raise FileNotFoundError("Keine passenden CSV-Dateien gefunden.")

    # Ordnung gemäß scanrate_order falls Rate im Dateinamen steckt
    ordered = []
    for rate in scanrate_order:
        for f in csv_files:
            if f"_{rate}" in f:
                ordered.append(f)
    # Ergänze ggf. übriggebliebene Dateien
    leftovers = [f for f in csv_files if f not in ordered]
    csv_files = ordered + leftovers

    # --- Hilfsfunktionen ---
    def choose_cycles_present(df):
        if "Cycle" not in df.columns or df["Cycle"].isna().all():
            return None
        all_cycles = sorted(pd.unique(df["Cycle"]))
        if cycles_to_use is not None:
            return [c for c in all_cycles if c in cycles_to_use] or all_cycles
        # Standard: wenn >1 Zyklus existiert, ersten Zyklus verwerfen (Konditionierung)
        if len(all_cycles) > 1:
            return [c for c in all_cycles if c != min(all_cycles)]
        return all_cycles

    def add_ir_corrected_potential(df):
        if Ru is None:
            df["_E"] = df["Potential (V)"].astype(float)
        else:
            df["_E"] = (
                df["Potential (V)"].astype(float) - df["Current (A)"].astype(float) * Ru
            )
        return df

    def middle_window_mask(E):
        # Mittlerer Bereich des Potenzialfensters (Ränder ausschließen)
        lo = np.quantile(E, edge_exclude)
        hi = np.quantile(E, 1 - edge_exclude)
        return (E >= lo) & (E <= hi), lo, hi

    def compute_scanrate_from_cv(df, cycle=None, edge_exclude=0.1):
        """
        Robuste Schätzung der Scanrate (V/s) aus dE/dt im mittleren Fenster.
        """
        df = df.sort_values("Time (s)").reset_index(drop=True)
        if cycle is not None and "Cycle" in df.columns:
            df = df[df["Cycle"] == cycle].copy()

        E = df["_E"].to_numpy() if "_E" in df.columns else df["Potential (V)"].to_numpy()
        t = df["Time (s)"].to_numpy()
        dE = np.gradient(E)
        dt = np.gradient(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            dEdt = np.where(dt != 0, dE / dt, np.nan)

        lo = np.quantile(E, edge_exclude)
        hi = np.quantile(E, 1 - edge_exclude)
        mask = (E >= lo) & (E <= hi)
        dEdt_mid = dEdt[mask]

        return float(np.nanmedian(np.abs(dEdt_mid)))

    def label_branches(df):
        # Anodisch/kathodisch anhand Vorzeichen von dE/dt
        dE = np.gradient(df["_E"].values)
        df = df.copy()
        df["_branch"] = np.where(dE >= 0, "an", "kat")
        return df

    def choose_target_potentials(df):
        # Wenn target_potentials nicht vorgegeben: nimm 5 gleichverteilte Werte im mittleren Fenster
        msk, lo, hi = middle_window_mask(df["_E"].values)
        return np.linspace(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), 5)

    # --- Sammellisten ---
    scan_rates = []
    Icaps = []
    per_file_info = []

    # --- Hauptschleife über Dateien ---
    for fname in csv_files:
        path = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            if verbose:
                print(f"Überspringe {fname} (Lesefehler): {e}")
            continue

        needed = {"Time (s)", "Potential (V)", "Current (A)"}
        if not needed.issubset(df.columns):
            if verbose:
                print(
                    f"Überspringe {fname}: fehlende Spalten {needed - set(df.columns)}"
                )
            continue

        df = df.sort_values("Time (s)").reset_index(drop=True)
        df = add_ir_corrected_potential(df)

        # Zyklen filtern
        use_cyc = choose_cycles_present(df)
        if use_cyc is not None and "Cycle" in df.columns:
            df = df[df["Cycle"].isin(use_cyc)].copy()

        # Scanrate
        v = v = compute_scanrate_from_cv(
            df, cycle=(cycles_to_use[0] if cycles_to_use else None)
        )

        # Branch-Zuordnung
        df = label_branches(df)

        # Zielpotenziale
        Es = np.array(
            target_potentials
            if target_potentials is not None
            else choose_target_potentials(df)
        )

        # I_cap pro E*
        Icaps_E = []
        for E_star in Es:
            win = df[(df["_E"] >= E_star - band) & (df["_E"] <= E_star + band)]
            Ian = win[win["_branch"] == "an"]["Current (A)"].median()
            Ikat = win[win["_branch"] == "kat"]["Current (A)"].median()
            if np.isfinite(Ian) and np.isfinite(Ikat):
                Icap = (Ian - Ikat) / 2.0  # A
                Icaps_E.append(Icap)

        if len(Icaps_E) == 0:
            if verbose:
                print(f"Warnung: keine gültigen Fenster in {fname}")
            continue

        Icap_file = np.median(Icaps_E)  # robuste Zusammenfassung über mehrere E*
        scan_rates.append(v)
        Icaps.append(Icap_file)

        per_file_info.append(
            {
                "file": fname,
                "scan_rate_V_per_s": v,
                "Icap_A": Icap_file,
                "n_targets_used": len(Icaps_E),
                "cycles_used": use_cyc,
            }
        )

    if len(scan_rates) < 2:
        raise RuntimeError("Zu wenige gültige Dateien/Scanraten für eine Regression.")

    scan_rates = np.array(scan_rates, dtype=float)
    Icaps = np.array(Icaps, dtype=float)

    # --- Lineare Regression: I_cap = C_dl * v + b ---
    coeff = np.polyfit(scan_rates, Icaps, 1)
    slope, intercept = coeff[0], coeff[1]
    y_fit = slope * scan_rates + intercept

    # R²
    ss_res = np.sum((Icaps - y_fit) ** 2)
    ss_tot = np.sum((Icaps - np.mean(Icaps)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    Cdl_F = slope  # F (weil I = C*v)
    Cdl_mF = Cdl_F * 1e3

    # --- Plot ---
    if save_plot or show_plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(scan_rates, Icaps, label=r"$I_{\mathrm{cap}}$ aus CV", zorder=3)
        # Linie über einen glatten Bereich
        xs = np.linspace(0, max(scan_rates) * 1.05, 100)
        ax.plot(
            xs,
            slope * xs + intercept,
            linewidth=1.8,
            label="lineare Regression",
            zorder=2,
        )
        ax.axhline(0, linestyle="--", alpha=0.4)
        ax.set_xlabel("Scan rate v (V/s)")
        ax.set_ylabel(r"$I_{\mathrm{cap}}$ (A)")
        ax.set_title("Double-Layer Capacitance aus CV (robuster Ansatz)")
        ax.legend(loc="best")
        txt = (
            f"C_dl = {Cdl_mF:.3f} mF  (R² = {R2:.3f})\n"
            f"Intercept = {intercept:.3e} A\n"
            f"n = {len(scan_rates)} Dateien"
        )
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        fig.tight_layout()

        out_dir = os.path.join(folder_path, "Voltammetry Data")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "Cdl_from_folder_pro.png")
        if save_plot:
            fig.savefig(out_path, dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)

        if verbose and save_plot:
            print(f"Plot gespeichert: {out_path}")

    if verbose:
        print(f"\nErgebnis: C_dl = {Cdl_mF:.3f} mF (slope={Cdl_F:.6e} F, R²={R2:.3f})")
        for info in per_file_info:
            print(
                f"  {info['file']:<24s}  v≈{info['scan_rate_V_per_s']:.4f} V/s, "
                f"Icap≈{info['Icap_A']:.3e} A, cycles={info['cycles_used']}"
            )

    return {
        "Cdl_F": Cdl_F,
        "Cdl_mF": Cdl_mF,
        "slope": Cdl_F,
        "intercept": intercept,
        "R2": R2,
        "scan_rates": scan_rates,
        "Icap": Icaps,
        "by_file": per_file_info,
    }


if __name__ == "__main__":
    folder = os.path.join("results", "example_CV", "known_10")
    compute_Cedl_from_folder(folder, well_number=1, target_voltage=0.0)

    res = compute_Cdl_from_folder_pro(
        folder,
        scanrate_order=(
            "1.0",
            "0.9",
            "0.8",
            "0.7",
            "0.6",
            "0.5",
        ),
    )
    print(res["Cdl_mF"], "mF")
