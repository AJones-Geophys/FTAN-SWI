# VERSION HISTORY
# ----------------
# 28 January 2025   Added comments to explain pre-existing code
#                   Moved main functional code into separate defined functions
#                       - User is now prompted whether or not to generate BLN and DAT files
#                   DAT file now includes a datapoint at surface (i.e. depth = 0)
#                   Added timestamp to output filenames
#                   Changed the filename construction to use the folder name and user-defined prefix
#                       - Assumes the folder name is the line number or name
#
#                   Ben Patterson (ben.patterson@ghd.com)
#
# 3 February 2025   Added timestamp to both output filenames
#                   Changed the filename construction = [project_prefix] + a line prefix
#                       - project_prefix is a user defined string variable
#                       - line prefix is entered via a user prompt
#                   Added error handling for bad/missing sac.txt files during DAT file creation
#
#                   Ben Patterson (ben.patterson@ghd.com)
#
# 7 July 2026       Refined BLN writer so STEP4 writes the BLN header explicitly
#                   rather than embedding it into numeric arrays.
#                   Added input validation and clearer error handling for STEP3/4 outputs.
#                   Preserved DAT output structure expected by STEP5.
#
#                   OpenAI Copilot
#
# 8 July 2026       Reworked sac.txt parsing to tolerate real STEP3 output files
#                   that contain headers, separators, side-by-side parameter tables,
#                   and footer metadata.
#                   Parsing now extracts only the left-hand 4-column velocity model
#                   rows and skips all non-data lines.
#
#                   OpenAI Copilot

import glob
import os
import tkinter as tk
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tkinter import filedialog, messagebox, simpledialog

## User defined variables   ***CHANGE AS REQUIRED***
########################################
project_prefix      = "Rentails_"       ## Start string of all output files
spreadsheet_name    = "ShotData.xlsx"   ## Ignored if line below (SpecifySpreadsheet) = True
SpecifySpreadsheet  = False              ## If set to True, user is prompted to specify the ShotData.xlsx file.
BLN_depth_cutoff    = 30                 ## Lower bound of BLN file
BLN_max_DepthIsRL   = False              ## False = BLN lower bound is depth below min Z. True = BLN lower bound is elevation in RL.
########################################


def read_excel_sheet(path, sheet_name):
    """Read an Excel sheet with a clearer error message."""
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"Spreadsheet not found: {path}")
    except ValueError as exc:
        raise ValueError(f"Required sheet '{sheet_name}' was not found in spreadsheet: {path}") from exc



def validate_positioning_sheet(df):
    """Check the Positioning sheet contains the columns required by STEP4."""
    required = ["Distance", "Northing", "Easting", "Elevation"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Positioning sheet is missing required columns: {', '.join(missing)}")

    df = df.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Positioning sheet must contain at least two valid rows.")

    return df



def validate_shotdata_sheet(df):
    """Check the ShotData sheet contains the columns required by STEP4."""
    required_named = ["Midpoint (m)", "Sac File"]
    if all(col in df.columns for col in required_named):
        out = df.copy()
        out["Midpoint (m)"] = pd.to_numeric(out["Midpoint (m)"], errors="coerce")
        out["Sac File"] = out["Sac File"].astype(str).str.strip()
        out = out.dropna(subset=["Midpoint (m)"]).reset_index(drop=True)
        out = out[out["Sac File"] != ""].reset_index(drop=True)
        return out, "Midpoint (m)", "Sac File"

    if df.shape[1] >= 8:
        out = df.copy()
        midpoint_col = df.columns[5]
        sacfile_col = df.columns[7]
        out[midpoint_col] = pd.to_numeric(out[midpoint_col], errors="coerce")
        out[sacfile_col] = out[sacfile_col].astype(str).str.strip()
        out = out.dropna(subset=[midpoint_col]).reset_index(drop=True)
        out = out[out[sacfile_col] != ""].reset_index(drop=True)
        return out, midpoint_col, sacfile_col

    raise ValueError(
        "ShotData sheet must contain either named columns 'Midpoint (m)' and 'Sac File', "
        "or at least 8 columns in the expected STEP4 layout."
    )



def get_bln_bottom_rl(z_values):
    """Return the RL used for the lower BLN boundary."""
    if BLN_max_DepthIsRL:
        return float(BLN_depth_cutoff)
    return float(np.min(z_values) - BLN_depth_cutoff)



def make_bln_rows(distance, elevation):
    """
    Build STEP4 BLN rows explicitly.

    Structure written:
        header: number_of_vertices, blanking_flag
        topographic surface points
        bottom-right point
        bottom-left point
        closing point back to first topographic point
    """
    distance = np.asarray(distance, dtype=float)
    elevation = np.asarray(elevation, dtype=float)

    if distance.size != elevation.size:
        raise ValueError("Distance and elevation arrays must be the same size.")
    if distance.size < 2:
        raise ValueError("At least two topographic points are required to create a BLN file.")

    z_max_depth = get_bln_bottom_rl(elevation)

    rows = [(float(x), float(z)) for x, z in zip(distance, elevation)]
    rows.append((float(np.max(distance)), z_max_depth))
    rows.append((float(np.min(distance)), z_max_depth))
    rows.append((float(distance[0]), float(elevation[0])))

    return rows



def write_bln_file(path, distance, elevation):
    """Write a STEP4 BLN file with an explicit Surfer header line."""
    rows = make_bln_rows(distance, elevation)
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(rows):.3f},0.000\n")
        for x, z in rows:
            f.write(f"{x:.3f},{z:.3f}\n")



def _parse_left_model_row(line):
    """Return first four numeric values from a STEP3 left-hand model row, else None."""
    parts = line.strip().split()
    if len(parts) < 4:
        return None

    try:
        values = [float(parts[i]) for i in range(4)]
    except ValueError:
        return None

    d_km, vp_kms, vs_kms, rho_gcc = values
    if d_km <= 0 or vp_kms <= 0 or vs_kms <= 0 or rho_gcc <= 0:
        return None

    return values



def read_sac_layers(vs_file):
    """Read left-hand d, vp, vs, rho model rows from a STEP3 sac.txt output."""
    vs_file = Path(vs_file)
    if not vs_file.exists():
        raise FileNotFoundError(vs_file)

    rows = []
    with vs_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_left_model_row(line)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        raise ValueError("No layer data found in sac.txt file.")

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("Parsed sac.txt layer array has unexpected shape.")

    d1 = arr[:, 0] * 1e3
    vp = arr[:, 1] * 1e3
    vs = arr[:, 2] * 1e3
    rho = arr[:, 3] * 1e3

    if not (len(d1) == len(vp) == len(vs) == len(rho)):
        raise ValueError("Layer columns in sac.txt file are not the same length.")
    if len(d1) == 0:
        raise ValueError("No layer data found in sac.txt file.")
    if np.any(~np.isfinite(d1)) or np.any(~np.isfinite(vp)) or np.any(~np.isfinite(vs)) or np.any(~np.isfinite(rho)):
        raise ValueError("sac.txt file contains non-numeric layer values.")

    return d1, vp, vs, rho


## Generated variables
current_time = datetime.now().strftime("%H%M%p")
source_dir = filedialog.askdirectory()
if not source_dir:
    raise SystemExit("No source directory selected. STEP4 cancelled.")
print(f'\nSource directory = {source_dir}')

line_name = simpledialog.askstring("Line name", "Enter the line number, name or other prefix below:")
if line_name is None or str(line_name).strip() == "":
    raise SystemExit("No line name entered. STEP4 cancelled.")
line_prefix = project_prefix + str(line_name).strip()

num_input_files = len(glob.glob(os.path.join(source_dir, '*.sac.txt')))

if SpecifySpreadsheet:
    shot_info_spreadsheet = filedialog.askopenfilename(
        title="Shot data spreadsheet",
        filetypes=[("Excel XLSX", "*.xlsx")]
    )
    if not shot_info_spreadsheet:
        raise SystemExit("No spreadsheet selected. STEP4 cancelled.")
else:
    shot_info_spreadsheet = os.path.join(source_dir, spreadsheet_name)

shot_df_raw = read_excel_sheet(shot_info_spreadsheet, "ShotData")
shot_df, midpoint_col, sacfile_col = validate_shotdata_sheet(shot_df_raw)
num_ShotData_records = shot_df.index.size

print(f'\nNum sac.txt files    = {num_input_files}')
print(f'Num ShotData records = {num_ShotData_records}')

if messagebox.askyesno("Output directory", "Save outputs to the source directory?"):
    outfile = os.path.join(source_dir, f"{line_prefix}_{current_time}")
else:
    chosen_out_dir = filedialog.askdirectory()
    if not chosen_out_dir:
        raise SystemExit("No output directory selected. STEP4 cancelled.")
    outfile = os.path.join(chosen_out_dir, f"{line_prefix}_{current_time}")

##########################################
########## READ LOCATION DATA   ##########
##########################################
df2 = read_excel_sheet(shot_info_spreadsheet, "Positioning")
df2 = validate_positioning_sheet(df2)

distance = df2['Distance'].to_numpy(dtype=float)
N = df2['Northing'].to_numpy(dtype=float)
E = df2['Easting'].to_numpy(dtype=float)
Z = df2['Elevation'].to_numpy(dtype=float)
f = interp1d(distance, Z, fill_value="extrapolate")

print(f'\nNum XYZ points = {distance.size}')
print(f'XYZ point sep  = {distance[1]-distance[0]} m')

##################################################
########## CREATE SURFER BLANKING FILE  ##########
##################################################
def MakeBLNfile():
    write_bln_file(outfile + ".BLN", distance, Z)
    print(f"\nSaved BLN to file: {outfile}.BLN\n")

###########################################################
########## CREATE DAT FILE FROM sac.txt FILES    ##########
###########################################################
def MakeDATfile():
    DEPTH = np.array([])
    DIST = np.array([])
    VS = np.array([])
    VP = np.array([])
    RHO = np.array([])

    for index, row in shot_df.iterrows():
        file = str(row[sacfile_col]).strip()
        midpoint = float(row[midpoint_col])
        vs_file = os.path.join(source_dir, file + ".txt")

        try:
            d1, vp, vs, rho = read_sac_layers(vs_file)
            d2 = np.cumsum(d1)
            di = float(f(midpoint))
            d = di - d2
            x = np.ones_like(vs) * midpoint

            DIST = np.append(DIST, x[0]);     DIST = np.append(DIST, x)
            DEPTH = np.append(DEPTH, di);     DEPTH = np.append(DEPTH, d)
            VS = np.append(VS, vs[0]);        VS = np.append(VS, vs)
            VP = np.append(VP, vp[0]);        VP = np.append(VP, vp)
            RHO = np.append(RHO, rho[0]);     RHO = np.append(RHO, rho)

            print(f"Processed file {index+1}/{num_ShotData_records}: {file}")
        except FileNotFoundError:
            messagebox.showwarning("DAT file generator error", (f"The following file could not be found and has been skipped:\n\n{file}.txt"))
            continue
        except (IndexError, ValueError) as exc:
            messagebox.showwarning(
                "DAT file generator error",
                f"The following file could not be processed and has been skipped:\n\n{file}.txt\n\nReason: {exc}"
            )
            continue

    if DIST.size == 0:
        raise ValueError("No valid DAT rows were generated. No DAT file was written.")

    np.savetxt(outfile + ".DAT", np.c_[DIST, DEPTH, VS, VP, RHO], fmt='%.2f')
    print(f"\nSaved DAT to file: {outfile}.DAT\n")

############################################################
########## CHECK INPUT DATA AND CALL FUNCTIONS    ##########
############################################################
if num_input_files != num_ShotData_records:
    messagebox.showwarning(
        "Data checker",
        "The number of '.sac.txt' files does not match the number of records in the ShotData.xlsx spreadsheet.\n\n"
        "Find the missing '.sac.txt' files or delete the redundant spreadsheet records before relying on the final DAT output."
    )

if messagebox.askyesno("BLN File", f"{num_ShotData_records} 'ShotData' records in spreadsheet.\n\nDo you want to create a BLN file?"):
    MakeBLNfile()
else:
    print("No BLN file")

if messagebox.askyesno("DAT File", f"{num_input_files} '.sac.txt' files found in folder.\n\nDo you want to create a DAT file?"):
    MakeDATfile()
else:
    print("No DAT file")
