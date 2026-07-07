import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================
# USER INPUTS
# ============================================================
input_folder = r"\\ghdnet\ghd\AU\hobart\General\GeoPhysics\Projects\12658085 - Rentails E Dam Embankment SRT FTAN\06_Processing\Line 1\Data\1\SAC"
output_folder = os.path.join(input_folder, "Line1_sampled_profiles")
shotdata_path = r"\\ghdnet\ghd\AU\hobart\General\GeoPhysics\Projects\12658085 - Rentails E Dam Embankment SRT FTAN\06_Processing\Line 1\Data\1\SAC\ShotData.xlsx"

output_filename = "Line1_surfer_xyz.dat"

sampling_interval_m = 2.0
interface_transition_m = 1.0
max_depth_m = 30

round_thickness = True
thickness_rounding_m = 1.0

negative_depth_for_surfer = True

os.makedirs(output_folder, exist_ok=True)

# ============================================================
# LOAD ELEVATION DATA (FROM STEP 4 WORKFLOW)
# ============================================================

df_pos = pd.read_excel(shotdata_path, sheet_name="Positioning")

distance = df_pos["Distance"].to_numpy()
Z = df_pos["Elevation"].to_numpy()

elev_interp_func = interp1d(distance, Z, fill_value="extrapolate")

# ============================================================
# FUNCTION: PARSE MODEL FILE
# ============================================================
def parse_model_file(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    d_km = float(parts[0])
                    vs_kms = float(parts[2])
                    if d_km > 0 and vs_kms > 0:
                        data.append([d_km, vs_kms])
                except Exception:
                    continue
    return pd.DataFrame(data, columns=["d_km", "vs_kms"])

# ============================================================
# FUNCTION: BUILD PROFILE
# ============================================================
def build_step_profile(df, dz=1.0, max_depth_m=None):
    layer_df = df.copy()

    layer_df["thickness_m"] = layer_df["d_km"] * 1000.0
    layer_df["vs_ms"] = layer_df["vs_kms"] * 1000.0

    if round_thickness and thickness_rounding_m > 0:
        layer_df["thickness_m"] = (
            np.round(layer_df["thickness_m"] / thickness_rounding_m)
            * thickness_rounding_m
        )
        layer_df = layer_df[layer_df["thickness_m"] > 0].reset_index(drop=True)

    layer_df["top_depth_m"] = layer_df["thickness_m"].cumsum() - layer_df["thickness_m"]
    layer_df["bottom_depth_m"] = layer_df["thickness_m"].cumsum()

    model_max_depth = layer_df["bottom_depth_m"].iloc[-1]
    sample_max_depth = model_max_depth if max_depth_m is None else min(max_depth_m, model_max_depth)

    depths = np.arange(0.0, sample_max_depth + dz, dz)
    vs_profile = np.zeros_like(depths, dtype=float)

    for _, layer in layer_df.iterrows():
        mask = (depths >= layer["top_depth_m"]) & (depths < layer["bottom_depth_m"])
        vs_profile[mask] = layer["vs_ms"]

    if len(depths) and np.isclose(depths[-1], sample_max_depth):
        vs_profile[-1] = layer_df["vs_ms"].iloc[-1]

    return pd.DataFrame({"depth_m": depths, "vs_ms": vs_profile}), layer_df

# ============================================================
# FUNCTION: APPLY SMOOTHING
# ============================================================
def apply_interface_smoothing(sampled_df, layer_df, transition_m=1.0):
    if transition_m is None or transition_m <= 0:
        return sampled_df.copy()

    out_df = sampled_df.copy()
    depths = out_df["depth_m"].values
    vs = out_df["vs_ms"].values.copy()

    half_width = transition_m / 2.0

    for i in range(len(layer_df) - 1):
        interface_depth = layer_df["bottom_depth_m"].iloc[i]
        vs_upper = layer_df["vs_ms"].iloc[i]
        vs_lower = layer_df["vs_ms"].iloc[i + 1]

        z0 = interface_depth - half_width
        z1 = interface_depth + half_width

        mask = (depths >= z0) & (depths <= z1)

        if np.any(mask):
            t = (depths[mask] - z0) / transition_m
            vs[mask] = vs_upper + t * (vs_lower - vs_upper)

    out_df["vs_ms"] = vs
    return out_df

# ============================================================
# FUNCTION: EXTRACT CHAINAGE FROM FILENAME
# ============================================================
def get_chainage_from_filename(file):

    basename = os.path.basename(file)
    stem = basename.replace(".sac.txt", "")

    numbers = re.findall(r'-?\d+(?:\.\d*)?', stem)

    if len(numbers) < 2:
        raise ValueError(f"Not enough numeric values in: {basename}")

    return float(numbers[-2])

# ============================================================
# WRITE TO MASTER SURFER FILE
# ============================================================
def write_surfer_xyz(df, master_file, chainage, negative_depth=True):

    for _, row in df.iterrows():

        depth_val = -row["depth_m"] if negative_depth else row["depth_m"]

        master_file.write(
            f"{chainage:.2f} "
            f"{depth_val:.2f} "
            f"{row['vs_ms']:.2f} "
            f"{row['elevation_m']:.2f}\n"
        )

# ============================================================
# MAIN LOOP
# ============================================================
master_dat_path = os.path.join(output_folder, output_filename)

print(f"Output file: {master_dat_path}")

with open(master_dat_path, "w") as master_file:

    for file in os.listdir(input_folder):

        if not file.lower().endswith(".txt"):
            continue

        filepath = os.path.join(input_folder, file)

        print(f"Processing: {file}")

        df_model = parse_model_file(filepath)

        if df_model is None or df_model.empty:
            print(f"Skipping {file}")
            continue

        try:
            chainage = get_chainage_from_filename(file)

        except ValueError as e:
            print(f"Skipping {file}: {e}")
            continue

        step_df, layer_df = build_step_profile(
            df_model,
            sampling_interval_m,
            max_depth_m
        )

        smooth_df = apply_interface_smoothing(
            step_df,
            layer_df,
            interface_transition_m
        )

        surface_elevation = float(elev_interp_func(chainage))

        smooth_df["elevation_m"] = (
            surface_elevation - smooth_df["depth_m"]
        )

        write_surfer_xyz(
            smooth_df,
            master_file,
            chainage,
            negative_depth=negative_depth_for_surfer
        )

        print(f"Appended {file} (chainage={chainage})")

print("Done.")
print(f"Output file: {master_dat_path}")
