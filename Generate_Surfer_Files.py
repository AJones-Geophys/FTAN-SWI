import os
import glob
import numpy as np
import pandas as pd

# -----------------------------
# SETTINGS YOU MAY CHANGE
# -----------------------------
# New workflow:
#   Geospatial/positioning data are read from an Excel workbook sheet named
#   "Positioning" instead of from *_geospatial.csv.
#
# The Positioning sheet is expected to contain these columns:
#   Distance, Elevation, Easting, Northing
# These are normalised internally to:
#   chainage, surface_rl, easting, northing

INPUT_POSITIONING_EXCEL_SUFFIXES = [".xlsx", ".xlsm", ".xls"]
POSITIONING_SHEET_NAME = "Positioning"

# Optional. If left as None, the script will use the first Excel workbook in the
# current folder that contains a sheet named POSITIONING_SHEET_NAME.
# Example: POSITIONING_EXCEL_FILE = "ShotData.xlsx"
POSITIONING_EXCEL_FILE = None

# Optional. If your FTAN files have a line prefix such as L1_output.dat or
# L1_some_section_output.dat, set LINE_PREFIX = "L1" to restrict processing.
# If left as None, all *_output.dat and *_output.txt files in the folder are processed.
LINE_PREFIX = None

INPUT_FTAN_SUFFIXES = ["_output.dat", "_output.txt"]

OUT_SUFFIX = "_XYZVCD.csv"
WRITE_SPACE_DELIM_DAT = True

CHAINAGE_STEP_CHECK = 0.2

# Used to check no Surfer NoData values propagate to output
SURFER_NODATA = 1.70141e38

APPLY_Z_BOUNDS = True
Z_MIN = 485
Z_MAX = 520


# -----------------------------
# HELPERS
# -----------------------------
def find_positioning_excel(cwd):
    """
    Locate the Excel workbook containing the Positioning sheet.

    If POSITIONING_EXCEL_FILE is set, that file is used directly.
    Otherwise, the script scans Excel workbooks in the current directory and
    selects the first workbook that contains POSITIONING_SHEET_NAME.
    """
    if POSITIONING_EXCEL_FILE:
        path = os.path.join(cwd, POSITIONING_EXCEL_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"POSITIONING_EXCEL_FILE not found: {path}")
        return path

    candidates = []
    for suffix in INPUT_POSITIONING_EXCEL_SUFFIXES:
        candidates.extend(glob.glob(os.path.join(cwd, f"*{suffix}")))

    candidates = sorted(candidates)

    if not candidates:
        raise FileNotFoundError(
            f"No Excel files found in {cwd}. Expected one of: {INPUT_POSITIONING_EXCEL_SUFFIXES}"
        )

    valid = []
    for path in candidates:
        try:
            xl = pd.ExcelFile(path)
            sheet_names_stripped = [str(s).strip() for s in xl.sheet_names]
            if POSITIONING_SHEET_NAME in sheet_names_stripped:
                valid.append(path)
        except Exception as exc:
            print(f"Warning: could not inspect Excel file {os.path.basename(path)}: {exc}")

    if not valid:
        raise FileNotFoundError(
            f"No Excel workbook in {cwd} contains a sheet named '{POSITIONING_SHEET_NAME}'"
        )

    if len(valid) > 1:
        print("Warning: multiple Excel workbooks contain a Positioning sheet. Using the first one:")
        for p in valid:
            print(f"  - {os.path.basename(p)}")
        print(f"Selected: {os.path.basename(valid[0])}")

    return valid[0]


def read_positioning_excel(path):
    """
    Read geospatial information from the Positioning tab of an Excel workbook.

    Required input columns in the Positioning sheet:
        Distance, Elevation, Easting, Northing

    Returned columns:
        chainage, surface_rl, easting, northing
    """
    xl = pd.ExcelFile(path)

    # Match sheet name after stripping whitespace, but preserve actual workbook sheet name.
    sheet_lookup = {str(s).strip(): s for s in xl.sheet_names}
    if POSITIONING_SHEET_NAME not in sheet_lookup:
        raise ValueError(
            f"Excel workbook {os.path.basename(path)} does not contain sheet '{POSITIONING_SHEET_NAME}'. "
            f"Available sheets: {xl.sheet_names}"
        )

    df = pd.read_excel(path, sheet_name=sheet_lookup[POSITIONING_SHEET_NAME])
    df.columns = df.columns.astype(str).str.strip()

    rename = {
        "Distance": "chainage",
        "distance": "chainage",
        "Chainage": "chainage",
        "chainage": "chainage",
        "Elevation": "surface_rl",
        "Elevation1": "surface_rl",
        "Surface_RL": "surface_rl",
        "surface_rl": "surface_rl",
        "Easting": "easting",
        "easting": "easting",
        "x": "easting",
        "X": "easting",
        "Northing": "northing",
        "northing": "northing",
        "y": "northing",
        "Y": "northing",
    }
    df = df.rename(columns=rename)

    needed = ["chainage", "easting", "northing"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Positioning sheet in {os.path.basename(path)} missing columns {missing}. "
            "Expected Distance, Elevation, Easting, Northing."
        )

    for c in ["chainage", "easting", "northing"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "surface_rl" in df.columns:
        df["surface_rl"] = pd.to_numeric(df["surface_rl"], errors="coerce")
    else:
        df["surface_rl"] = np.nan

    df = df.dropna(subset=["chainage", "easting", "northing"]).copy()

    if df.empty:
        raise ValueError(
            f"No valid positioning rows found in {os.path.basename(path)} after numeric conversion."
        )

    # If duplicated chainages exist, average numeric values at the same chainage.
    # This avoids np.interp problems caused by duplicate x-values.
    df = (
        df.groupby("chainage", as_index=False)
          .agg({"easting": "mean", "northing": "mean", "surface_rl": "mean"})
          .sort_values("chainage")
          .reset_index(drop=True)
    )

    if df["chainage"].nunique() < 2:
        raise ValueError(
            f"Positioning sheet in {os.path.basename(path)} requires at least two unique chainage values."
        )

    check_chainage_spacing(df, source_name=os.path.basename(path))

    return df


def check_chainage_spacing(df, source_name="positioning data"):
    """
    Non-fatal QA check on chainage spacing.

    CHAINAGE_STEP_CHECK is retained from the original script, but is used only as
    a warning threshold rather than forcing failure. The attached Positioning
    data are not spaced at 0.2 m, so a hard check would be inappropriate here.
    """
    ch = df["chainage"].to_numpy(dtype=float)
    diffs = np.diff(ch)

    if len(diffs) == 0:
        return

    if np.any(diffs <= 0):
        raise ValueError(f"Non-increasing chainage values detected in {source_name}")

    median_step = float(np.nanmedian(diffs))
    if CHAINAGE_STEP_CHECK is not None and median_step > CHAINAGE_STEP_CHECK:
        print(
            f"Warning: median positioning chainage spacing is {median_step:.6g} m in {source_name}, "
            f"which is greater than CHAINAGE_STEP_CHECK={CHAINAGE_STEP_CHECK}. "
            "This is a warning only; interpolation will still run."
        )


def read_ftan_dat(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)

    if df.shape[1] < 3:
        raise ValueError(f"FTAN file has <3 columns: {os.path.basename(path)}")

    df = df.iloc[:, :3]
    df.columns = ["chainage", "elev_rl", "velocity"]

    df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    df["elev_rl"] = pd.to_numeric(df["elev_rl"], errors="coerce")
    df["velocity"] = pd.to_numeric(df["velocity"], errors="coerce")

    # Remove Surfer NoData values
    df.loc[df["velocity"] >= SURFER_NODATA * 0.99, "velocity"] = np.nan

    # Apply elevation bounds
    if APPLY_Z_BOUNDS:
        df.loc[(df["elev_rl"] < Z_MIN) | (df["elev_rl"] > Z_MAX), "velocity"] = np.nan

    df = df.dropna(subset=["chainage", "elev_rl"]).sort_values("chainage")
    return df


def interp_no_extrap(x_src, y_src, x_tgt):
    """
    1D linear interpolation with NO extrapolation.
    Values outside [min(x_src), max(x_src)] become NaN.
    """
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_tgt = np.asarray(x_tgt, dtype=float)

    valid = np.isfinite(x_src) & np.isfinite(y_src)
    x_src = x_src[valid]
    y_src = y_src[valid]

    if len(x_src) < 2:
        return np.full_like(x_tgt, np.nan, dtype=float)

    order = np.argsort(x_src)
    x_src = x_src[order]
    y_src = y_src[order]

    y_tgt = np.interp(x_tgt, x_src, y_src)
    mask = (x_tgt >= x_src.min()) & (x_tgt <= x_src.max())
    y_tgt = np.where(mask, y_tgt, np.nan)

    return y_tgt


def find_ftan_files(cwd):
    """
    Find FTAN files to process.

    If LINE_PREFIX is set, only files starting with LINE_PREFIX and ending with
    one of INPUT_FTAN_SUFFIXES are processed.

    If LINE_PREFIX is None, all files ending with one of INPUT_FTAN_SUFFIXES are
    processed. This is intentional because the Excel workbook name, for example
    ShotData.xlsx, may not share the same prefix as files such as L1_output.dat.
    """
    ftan_paths = []

    prefix = "*" if LINE_PREFIX is None else f"{LINE_PREFIX}*"
    for suf in INPUT_FTAN_SUFFIXES:
        ftan_paths.extend(glob.glob(os.path.join(cwd, f"{prefix}{suf}")))

    return sorted(set(ftan_paths))


def get_section_name(ftan_path):
    section_name = os.path.basename(ftan_path)
    for suf in INPUT_FTAN_SUFFIXES:
        if section_name.endswith(suf):
            section_name = section_name[: -len(suf)]
            break
    return section_name


# -----------------------------
# MAIN BATCH PROCESS
# -----------------------------
def main():
    cwd = os.getcwd()

    positioning_excel = find_positioning_excel(cwd)
    print(f"Using positioning data from: {positioning_excel}")

    geo = read_positioning_excel(positioning_excel)
    print(
        f"Loaded {len(geo)} positioning rows from sheet '{POSITIONING_SHEET_NAME}'. "
        f"Chainage range: {geo['chainage'].min():.6f} to {geo['chainage'].max():.6f} m"
    )

    ftan_paths = find_ftan_files(cwd)

    if not ftan_paths:
        if LINE_PREFIX is None:
            raise FileNotFoundError(
                f"No FTAN files ending with {INPUT_FTAN_SUFFIXES} found in {cwd}"
            )
        raise FileNotFoundError(
            f"No FTAN files matching prefix '{LINE_PREFIX}' and suffixes {INPUT_FTAN_SUFFIXES} found in {cwd}"
        )

    print(f"Found {len(ftan_paths)} FTAN file(s) to process.")

    # Interpolation source arrays from Positioning sheet
    ch_geo = geo["chainage"].values
    e_geo = geo["easting"].values
    n_geo = geo["northing"].values
    s_geo = geo["surface_rl"].values if "surface_rl" in geo.columns else np.full(len(geo), np.nan)

    for ftan_path in ftan_paths:
        section_name = get_section_name(ftan_path)
        print(f"\nProcessing section: {section_name}")

        ftan = read_ftan_dat(ftan_path)
        ch_ftan = ftan["chainage"].values

        # Interpolate spatial coordinates onto FTAN chainage
        e_ftan = interp_no_extrap(ch_geo, e_geo, ch_ftan)
        n_ftan = interp_no_extrap(ch_geo, n_geo, ch_ftan)
        s_ftan = interp_no_extrap(ch_geo, s_geo, ch_ftan)

        # --------------------------------------------------
        # Build output DataFrame
        # --------------------------------------------------
        out = pd.DataFrame({
            "Easting": e_ftan,
            "Northing": n_ftan,
            "Elevation": ftan["elev_rl"].values,
            "Velocity": ftan["velocity"].values,
            "Chainage": ftan["chainage"].values,
            "Surface_RL": s_ftan,
            "Section": section_name
        })

        # --------------------------------------------------
        # Calculate Depth(RL)
        # --------------------------------------------------
        # Depth(RL) is defined as:
        #     Depth(RL) = Surface_RL - Elevation
        #
        # Interpretation:
        #   > 0  = below ground surface
        #   = 0  = at surface
        #   < 0  = above surface, which may indicate data issues or structures
        #
        # If either input is NaN, result will be NaN.
        # --------------------------------------------------
        out["Depth(RL)"] = out["Surface_RL"] - out["Elevation"]

        # Remove points outside spatial range or without elevation.
        # Velocity is not included here so that blanked/NoData cells can remain
        # as NaN in the CSV output, while the point geometry remains available.
        out = out.dropna(subset=["Easting", "Northing", "Elevation"])

        if out.empty:
            print(
                f"  Skipping {section_name}: no FTAN points fall inside the Positioning chainage range "
                f"{ch_geo.min():.6f} to {ch_geo.max():.6f} m"
            )
            continue

        # Save CSV
        out_csv = os.path.join(cwd, f"{section_name}{OUT_SUFFIX}")
        out.to_csv(out_csv, index=False, float_format="%.6f")
        print(f"  Wrote {out_csv}")

        # Save DAT (Surfer-friendly)
        if WRITE_SPACE_DELIM_DAT:
            # For Surfer point import, do not export NaN velocity rows to the DAT.
            out_dat_df = out.dropna(subset=["Velocity"])
            out_dat = os.path.join(cwd, f"{section_name}_surfer3d_points.dat")
            out_dat_df[["Easting", "Northing", "Elevation", "Velocity"]].to_csv(
                out_dat,
                index=False,
                header=False,
                sep=" ",
                float_format="%.6f"
            )
            print(f"  Wrote {out_dat}")

    print("\nDone.")


if __name__ == "__main__":
    main()
