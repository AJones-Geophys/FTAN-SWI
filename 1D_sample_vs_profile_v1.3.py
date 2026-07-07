"""
===============================================================================
1D Sampled VS Profile Builder for Surfer Export (Enhanced / Versioned)
File: 1D_sample_vs_profile_v1.3.py
===============================================================================

PURPOSE
-------
This script reads layered 1D shear-wave velocity model files ("*.txt") from an
input folder and converts each model to a uniformly sampled depth profile,
then appends all sampled profiles into a single Surfer-style XYZ-like DAT file.

Compared with earlier versions, this script aligns depth/thickness handling more
closely with the STEP5 regularization philosophy:

- Treat layer intervals explicitly using top/bottom depths.
- Build sampling grid with deterministic bounds and tolerance-aware endpoint
  handling.
- Assign properties using clear interval rules and explicit bottom-boundary
  behavior.
- Enforce robust input checks so malformed files fail safely with diagnostics.

This script is intended as a preprocessing step for downstream gridding and
visualization (e.g., Surfer workflows).

-------------------------------------------------------------------------------
WORKFLOW OVERVIEW
-----------------
1) Read station/shot positioning data from Excel sheet "Positioning".
2) Build interpolation function for surface elevation vs chainage.
3) For each model text file in the input folder:
   a) Parse model rows (thickness in km, Vs in km/s).
   b) Convert to meters and validate.
   c) Reconstruct layered depth intervals.
   d) Generate a regular depth grid at dz (sampling_interval_m).
   e) Assign piecewise-constant Vs to each sampled depth.
   f) Optionally smooth across interfaces within a transition window.
   g) Compute RL/elevation at each sampled depth.
   h) Append chainage, depth, Vs, and elevation to a master DAT file.
4) Save outputs and print a processing summary.

-------------------------------------------------------------------------------
INPUTS
------
- Model files in input_folder (plain text with numeric columns where:
  * column 1 = layer thickness/depth increment in km
  * column 3 = Vs in km/s)
- Excel file with sheet "Positioning" and columns:
  * Distance
  * Elevation

OUTPUT
------
A single DAT text file with rows:
    chainage depth vs elevation

Depth can be written as negative (Surfer-friendly depth-positive-down display)
or positive based on settings.

-------------------------------------------------------------------------------
NOTES ON VERSIONING / ROLLBACK
------------------------------
This file is intentionally saved as a new versioned filename:
    1D_sample_vs_profile_v1.3.py

Your original script remains unchanged. In GitHub, all commits are also tracked,
so you can restore or compare any version later.
===============================================================================
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================
# USER INPUTS
# ============================================================
input_folder = r"./SAC/"
output_folder = os.path.join(input_folder, "Line1_sampled_profiles")
shotdata_path = r"./SAC/ShotData.xlsx"

output_filename = "Line1_surfer_xyz.dat"

# Vertical sampling interval for generated profile grid.
sampling_interval_m = 5.0

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

# Ensure output folder exists.
os.makedirs(output_folder, exist_ok=True)


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

    # Sort and collapse duplicate distances by averaging elevation.
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

            # Keep only physically meaningful positive values.
            if d_km > 0 and vs_kms > 0:
                data.append([d_km, vs_kms])

    return pd.DataFrame(data, columns=["d_km", "vs_kms"])


# ============================================================
# FUNCTION: BUILD PROFILE (STEP5-STYLE INTERVAL HANDLING)
# ============================================================
def build_step_profile(df, dz=1.0, max_depth=None):
    """
    Reconstruct layer intervals and sample onto a regular depth grid.

    Depth/thickness handling is aligned with STEP5-style logic:
      - Build explicit top/bottom interval bounds for each layer.
      - Sample regular grid deterministically from z=0 to z_max.
      - Use interval inclusion rule [top, bottom) per layer.
      - Explicitly assign last sample if it lands on bottom boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain d_km, vs_kms.
    dz : float
        Sampling interval in meters.
    max_depth : float or None
        Optional depth cap in meters.

    Returns
    -------
    sampled_df : pd.DataFrame
        Columns: depth_m, vs_ms
    layer_df : pd.DataFrame
        Layer table with interval bounds and Vs.
    """
    _require_positive_number(dz, "dz")

    if df is None or df.empty:
        raise ValueError("Input model dataframe is empty.")

    layer_df = df.copy()
    layer_df["thickness_m"] = pd.to_numeric(layer_df["d_km"], errors="coerce") * 1000.0
    layer_df["vs_ms"] = pd.to_numeric(layer_df["vs_kms"], errors="coerce") * 1000.0

    # Drop invalid numeric rows early.
    layer_df = layer_df.dropna(subset=["thickness_m", "vs_ms"]).copy()
    layer_df = layer_df[(layer_df["thickness_m"] > 0) & (layer_df["vs_ms"] > 0)].copy()

    if layer_df.empty:
        raise ValueError("No valid positive thickness/Vs rows found after numeric cleaning.")

    # Optional rounding of thickness to practical increments.
    if round_thickness and thickness_rounding_m > 0:
        layer_df["thickness_m"] = (
            np.round(layer_df["thickness_m"] / thickness_rounding_m) * thickness_rounding_m
        )
        layer_df = layer_df[layer_df["thickness_m"] > 0].reset_index(drop=True)

    if layer_df.empty:
        raise ValueError("All layers were removed after thickness rounding/filtering.")

    # Explicit interval bounds.
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

    # STEP5-like deterministic grid generation with tolerance-aware clip.
    depths = np.arange(0.0, sample_max_depth + dz * 0.5, dz)
    depths = depths[depths <= sample_max_depth + 1e-9]

    if len(depths) == 0:
        raise ValueError("No depth samples generated; check dz and model depth.")

    vs_profile = np.full_like(depths, np.nan, dtype=float)

    # Piecewise-constant assignment using [top, bottom) intervals.
    for _, layer in layer_df.iterrows():
        top = float(layer["top_depth_m"])
        bot = float(layer["bottom_depth_m"])
        val = float(layer["vs_ms"])
        mask = (depths >= top) & (depths < bot)
        vs_profile[mask] = val

    # Explicit bottom boundary behavior.
    if np.isclose(depths[-1], sample_max_depth, atol=1e-9):
        # Assign from the interval containing sample_max_depth.
        assigned = False
        for _, layer in layer_df.iterrows():
            if float(layer["top_depth_m"]) <= sample_max_depth <= float(layer["bottom_depth_m"]):
                vs_profile[-1] = float(layer["vs_ms"])
                assigned = True
                break
        if not assigned:
            vs_profile[-1] = float(layer_df["vs_ms"].iloc[-1])

    # Guard against any unassigned samples due to floating edge cases.
    if np.any(~np.isfinite(vs_profile)):
        valid = np.where(np.isfinite(vs_profile))[0]
        if len(valid) == 0:
            raise ValueError("All sampled Vs values are NaN after interval assignment.")
        # Forward-fill then back-fill nearest valid value.
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
    """
    Apply linear transition around each interface across a finite window.

    If transition_m <= 0, returns unchanged sampled dataframe.
    """
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
            # 0 at upper bound, 1 at lower bound in transition window.
            t = (depths[mask] - z0) / float(transition_m)
            vs[mask] = vs_upper + t * (vs_lower - vs_upper)

    out_df["vs_ms"] = vs
    return out_df


# ============================================================
# FUNCTION: EXTRACT CHAINAGE FROM FILENAME
# ============================================================
def get_chainage_from_filename(file_name):
    """
    Extract chainage from filename by taking the second-last numeric token.

    Example tokenization approach mirrors prior script behavior.
    """
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
    """Append sampled profile rows to the master output file."""
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
    # Validate global numeric settings early.
    _require_positive_number(sampling_interval_m, "sampling_interval_m")
    _require_nonnegative_number(interface_transition_m, "interface_transition_m")
    if max_depth_m is not None:
        _require_nonnegative_number(max_depth_m, "max_depth_m")

    # Load and validate positioning data.
    df_pos = load_positioning_data(shotdata_path)
    distance = df_pos["Distance"].to_numpy(dtype=float)
    elevation = df_pos["Elevation"].to_numpy(dtype=float)

    # Build interpolation function for surface RL/elevation.
    elev_interp_func = interp1d(distance, elevation, fill_value="extrapolate")

    # Gather candidate text files.
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


if __name__ == "__main__":
    main()
