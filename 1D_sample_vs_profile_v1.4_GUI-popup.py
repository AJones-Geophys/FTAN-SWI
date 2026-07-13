"""
===============================================================================
1D Sampled VS Profile Builder for Surfer Export (Enhanced / Versioned)
File: 1D_sample_vs_profile_v1.4_GUI-popup.py
===============================================================================

PURPOSE
-------
This script reads layered 1D shear-wave velocity model files ("*.txt") from a
user-selected input folder and converts each model to a uniformly sampled depth
profile, then appends all sampled profiles into a single Surfer-style XYZ-like
DAT file.

This version adds GUI pop-up dialogs so the user can select:
1) the input folder,
2) the output folder,
3) the ShotData Excel file.

This avoids failures caused by hard-coded paths and makes the script easier to
run interactively.
===============================================================================
"""

import os
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================
# USER INPUTS
# ============================================================
output_filename = "Surfer_xyz.dat"

# Vertical sampling interval for generated profile grid.
sampling_interval_m = 2.0

# Width of linear transition zone centered on each interface.
# Set <= 0 to disable smoothing.
interface_transition_m = 1.0

# Optional maximum sampled depth (m). If model is shallower, model depth wins.
max_depth_m = 40.0

# Round layer thickness values to reduce tiny decimal artefacts.
round_thickness = True
thickness_rounding_m = 1.0

# If True, write depth as negative values in output (common for Surfer sections).
negative_depth_for_surfer = True

# Set True for more verbose diagnostics.
print_diagnostics = True


# ============================================================
# GUI HELPERS
# ============================================================
def select_paths_via_gui():
    """Open GUI dialogs for selecting input folder, output folder, and ShotData file."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    input_folder = filedialog.askdirectory(title="Select input folder containing model .txt files")
    if not input_folder:
        messagebox.showerror("Selection Error", "No input folder selected. Script will exit.")
        raise SystemExit("No input folder selected.")

    output_folder = filedialog.askdirectory(title="Select output folder for DAT file")
    if not output_folder:
        messagebox.showerror("Selection Error", "No output folder selected. Script will exit.")
        raise SystemExit("No output folder selected.")

    shotdata_path = filedialog.askopenfilename(
        title="Select ShotData Excel file",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
    )
    if not shotdata_path:
        messagebox.showerror("Selection Error", "No ShotData file selected. Script will exit.")
        raise SystemExit("No ShotData file selected.")

    root.destroy()
    return input_folder, output_folder, shotdata_path


# ============================================================
# VALIDATION HELPERS
# ============================================================
def _require_positive_number(value, name):
    """Raise ValueError if value is not a positive finite number."""
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number. Got: {value}")


def _require_nonnegative_number(value, name):
    """Raise ValueError if value is not a non-negative finite number."""
    if value is None or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number. Got: {value}")


# ============================================================
# LOAD ELEVATION DATA
# ============================================================
def load_positioning_data(xlsx_path):
    """
    Load and validate positioning sheet with Distance and Elevation columns.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe sorted by Distance ascending.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"ShotData Excel file not found: {xlsx_path}")

    try:
        df_pos = pd.read_excel(xlsx_path, sheet_name="Positioning")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read sheet 'Positioning' from {xlsx_path}: {exc}"
        ) from exc

    required_cols = {"Distance", "Elevation"}
    missing = required_cols - set(df_pos.columns)
    if missing:
        raise ValueError(
            f"Positioning sheet missing required columns: {sorted(missing)}"
        )

    df_pos = df_pos[["Distance", "Elevation"]].copy()
    df_pos["Distance"] = pd.to_numeric(df_pos["Distance"], errors="coerce")
    df_pos["Elevation"] = pd.to_numeric(df_pos["Elevation"], errors="coerce")
    df_pos = df_pos.dropna(subset=["Distance", "Elevation"]).copy()

    if df_pos.empty:
        raise ValueError("Positioning sheet has no valid Distance/Elevation rows.")

    df_pos = (
        df_pos.sort_values("Distance")
        .groupby("Distance", as_index=False)["Elevation"]
        .mean()
        .reset_index(drop=True)
    )

    if len(df_pos) < 2:
        raise ValueError(
            "At least two unique Distance points are required for interpolation."
        )

    return df_pos


# ============================================================
# FUNCTION: PARSE MODEL FILE
# ============================================================
def parse_model_file(filepath):
    """
    Parse model file rows into thickness/depth increment and Vs.

    Expected minimum row format (whitespace separated):
      col[0] = d_km
      col[2] = vs_kms

    Filters out non-numeric, non-positive values.
    """
    data = []
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                d_km = float(parts[0])
                vs_kms = float(parts[2])
            except Exception:
                continue

            if d_km > 0 and vs_kms > 0:
                data.append([d_km, vs_kms])

    return pd.DataFrame(data, columns=["d_km", "vs_kms"])


# ============================================================
# FUNCTION: BUILD PROFILE (STEP5-STYLE INTERVAL HANDLING)
# ============================================================
def build_step_profile(df, dz=1.0, max_depth=None):
    _require_positive_number(dz, "dz")

    if df is None or df.empty:
        raise ValueError("Input model dataframe is empty.")

    layer_df = df.copy()
    layer_df["thickness_m"] = pd.to_numeric(layer_df["d_km"], errors="coerce") * 1000.0
    layer_df["vs_ms"] = pd.to_numeric(layer_df["vs_kms"], errors="coerce") * 1000.0

    layer_df = layer_df.dropna(subset=["thickness_m", "vs_ms"]).copy()
    layer_df = layer_df[(layer_df["thickness_m"] > 0) & (layer_df["vs_ms"] > 0)].copy()

    if layer_df.empty:
        raise ValueError("No valid positive thickness/Vs rows found after numeric cleaning.")

    if round_thickness and thickness_rounding_m > 0:
        layer_df["thickness_m"] = (
            np.round(layer_df["thickness_m"] / thickness_rounding_m) * thickness_rounding_m
        )
        layer_df = layer_df[layer_df["thickness_m"] > 0].reset_index(drop=True)

    if layer_df.empty:
        raise ValueError("All layers were removed after thickness rounding/filtering.")

    layer_df["top_depth_m"] = layer_df["thickness_m"].cumsum() - layer_df["thickness_m"]
    layer_df["bottom_depth_m"] = layer_df["thickness_m"].cumsum()

    model_max_depth = float(layer_df["bottom_depth_m"].iloc[-1])

    if max_depth is None:
        sample_max_depth = model_max_depth
    else:
        _require_nonnegative_number(max_depth, "max_depth")
        sample_max_depth = min(float(max_depth), model_max_depth)

    if sample_max_depth <= 0:
        raise ValueError("Computed sample_max_depth is non-positive.")

    depths = np.arange(0.0, sample_max_depth + dz * 0.5, dz)
    depths = depths[depths <= sample_max_depth + 1e-9]

    if len(depths) == 0:
        raise ValueError("No depth samples generated; check dz and model depth.")

    vs_profile = np.full_like(depths, np.nan, dtype=float)

    for _, layer in layer_df.iterrows():
        top = float(layer["top_depth_m"])
        bot = float(layer["bottom_depth_m"])
        val = float(layer["vs_ms"])
        mask = (depths >= top) & (depths < bot)
        vs_profile[mask] = val

    if np.isclose(depths[-1], sample_max_depth, atol=1e-9):
        assigned = False
        for _, layer in layer_df.iterrows():
            if float(layer["top_depth_m"]) <= sample_max_depth <= float(layer["bottom_depth_m"]):
                vs_profile[-1] = float(layer["vs_ms"])
                assigned = True
                break
        if not assigned:
            vs_profile[-1] = float(layer_df["vs_ms"].iloc[-1])

    if np.any(~np.isfinite(vs_profile)):
        valid = np.where(np.isfinite(vs_profile))[0]
        if len(valid) == 0:
            raise ValueError("All sampled Vs values are NaN after interval assignment.")
        for i in range(1, len(vs_profile)):
            if not np.isfinite(vs_profile[i]):
                vs_profile[i] = vs_profile[i - 1]
        if not np.isfinite(vs_profile[0]):
            first_valid = np.where(np.isfinite(vs_profile))[0][0]
            vs_profile[:first_valid] = vs_profile[first_valid]

    sampled_df = pd.DataFrame({"depth_m": depths, "vs_ms": vs_profile})
    return sampled_df, layer_df


# ============================================================
# FUNCTION: APPLY SMOOTHING
# ============================================================
def apply_interface_smoothing(sampled_df, layer_df, transition_m=1.0):
    if transition_m is None or transition_m <= 0:
        return sampled_df.copy()

    out_df = sampled_df.copy()
    depths = out_df["depth_m"].to_numpy(dtype=float)
    vs = out_df["vs_ms"].to_numpy(dtype=float).copy()

    half_width = float(transition_m) / 2.0

    for i in range(len(layer_df) - 1):
        interface_depth = float(layer_df["bottom_depth_m"].iloc[i])
        vs_upper = float(layer_df["vs_ms"].iloc[i])
        vs_lower = float(layer_df["vs_ms"].iloc[i + 1])

        z0 = interface_depth - half_width
        z1 = interface_depth + half_width
        mask = (depths >= z0) & (depths <= z1)

        if np.any(mask):
            t = (depths[mask] - z0) / float(transition_m)
            vs[mask] = vs_upper + t * (vs_lower - vs_upper)

    out_df["vs_ms"] = vs
    return out_df


# ============================================================
# FUNCTION: EXTRACT CHAINAGE FROM FILENAME
# ============================================================
def get_chainage_from_filename(file_name):
    basename = os.path.basename(file_name)
    stem = basename.replace(".sac.txt", "")
    numbers = re.findall(r"-?\d+(?:\.\d*)?", stem)

    if len(numbers) < 2:
        raise ValueError(f"Not enough numeric values in filename: {basename}")

    return float(numbers[-2])


# ============================================================
# FUNCTION: WRITE TO MASTER SURFER FILE
# ============================================================
def write_surfer_xyz(df, master_file, chainage, negative_depth=True):
    for _, row in df.iterrows():
        depth_val = -float(row["depth_m"]) if negative_depth else float(row["depth_m"])
        master_file.write(
            f"{chainage:.2f} "
            f"{depth_val:.2f} "
            f"{float(row['vs_ms']):.2f} "
            f"{float(row['elevation_m']):.2f}\n"
        )


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    input_folder, output_folder, shotdata_path = select_paths_via_gui()

    _require_positive_number(sampling_interval_m, "sampling_interval_m")
    _require_nonnegative_number(interface_transition_m, "interface_transition_m")
    if max_depth_m is not None:
        _require_nonnegative_number(max_depth_m, "max_depth_m")

    os.makedirs(output_folder, exist_ok=True)

    df_pos = load_positioning_data(shotdata_path)
    distance = df_pos["Distance"].to_numpy(dtype=float)
    elevation = df_pos["Elevation"].to_numpy(dtype=float)

    elev_interp_func = interp1d(distance, elevation, fill_value="extrapolate")

    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"Input folder does not exist or is not a directory: {input_path}")

    txt_files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    if len(txt_files) == 0:
        raise FileNotFoundError(f"No .txt model files found in input folder: {input_path}")

    master_dat_path = os.path.join(output_folder, output_filename)
    print(f"Output file: {master_dat_path}")

    processed = 0
    skipped = 0

    with open(master_dat_path, "w", encoding="utf-8") as master_file:
        for filepath in txt_files:
            file_name = filepath.name
            print(f"Processing: {file_name}")

            try:
                df_model = parse_model_file(filepath)
                if df_model.empty:
                    raise ValueError("No valid model rows parsed.")

                chainage = get_chainage_from_filename(file_name)

                step_df, layer_df = build_step_profile(
                    df_model,
                    dz=sampling_interval_m,
                    max_depth=max_depth_m,
                )

                smooth_df = apply_interface_smoothing(
                    step_df,
                    layer_df,
                    transition_m=interface_transition_m,
                )

                surface_elevation = float(elev_interp_func(chainage))
                smooth_df["elevation_m"] = surface_elevation - smooth_df["depth_m"]

                write_surfer_xyz(
                    smooth_df,
                    master_file,
                    chainage,
                    negative_depth=negative_depth_for_surfer,
                )

                processed += 1
                print(
                    f"Appended {file_name} | chainage={chainage:.2f} "
                    f"| samples={len(smooth_df)} | surf_elev={surface_elevation:.2f}"
                )

            except Exception as exc:
                skipped += 1
                print(f"Skipping {file_name}: {exc}")
                continue

    print("Done.")
    print(f"Output file: {master_dat_path}")
    print(f"Profiles processed: {processed}")
    print(f"Profiles skipped:   {skipped}")

    if print_diagnostics:
        print("\nDiagnostics:")
        print(f"  Positioning points used: {len(df_pos)}")
        print(f"  Input text files found:  {len(txt_files)}")
        print(f"  Sampling interval (m):   {sampling_interval_m}")
        print(f"  Interface transition (m): {interface_transition_m}")
        print(f"  Max depth (m):           {max_depth_m}")
        print(f"  ShotData file used:      {shotdata_path}")

    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Processing Complete",
            f"Done.\n\nOutput file:\n{master_dat_path}\n\nProfiles processed: {processed}\nProfiles skipped: {skipped}",
        )
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
