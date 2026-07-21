import os
import re
import glob
import numpy as np
import pandas as pd

# -----------------------------
# SETTINGS YOU MAY CHANGE
# -----------------------------
# Root directory to process. If None, uses current working directory.
ROOT_DIR = None

# Recurse into subfolders while searching for input files.
RECURSIVE = True

# Positioning workbook/sheet settings
INPUT_POSITIONING_EXCEL_SUFFIXES = [".xlsx", ".xlsm", ".xls"]
POSITIONING_SHEET_NAME = "Positioning"

# FTAN/SRT input DAT file pattern(s)
INPUT_DAT_SUFFIXES = ["_surfer_output.dat", "_surfer_output.txt", "_output.dat", "_output.txt"]

OUT_SUFFIX = "_XYZVCD.csv"
WRITE_SPACE_DELIM_DAT = True

# Optional: warning threshold only (non-fatal)
CHAINAGE_STEP_CHECK = 0.2

# Used to check no Surfer NoData values propagate to output
SURFER_NODATA = 1.70141e38

# Optional elevation bounds applied to velocity
APPLY_Z_BOUNDS = False
Z_MIN = 485
Z_MAX = 520

# If True, only report actions without writing outputs
DRY_RUN = False


# -----------------------------
# HELPERS
# -----------------------------
def normalize_line_token(token: str) -> str:
    """Normalize line token to canonical format, e.g. Line1 / line_01 -> LINE1."""
    if token is None:
        return ""
    t = str(token).strip()
    t = t.replace("(", "").replace(")", "")
    t = t.replace("_", "").replace("-", "")
    t = t.upper()

    m = re.search(r"LINE\s*(\d+)", t)
    if m:
        return f"LINE{int(m.group(1))}"

    m = re.search(r"L\s*(\d+)", t)
    if m:
        return f"LINE{int(m.group(1))}"

    return t


def extract_alignment_key(name: str):
    """
    Extract an alignment key from file names such as:
      - FTAN_H-01_(Line1)_1x0.5_surfer_output.dat
      - SRT_V-01_(Line3)_surfer_output.dat
      - H-01_Line1_ShotData.xlsx
      - V-06_Line10_ShotData.xlsx

    Returns tuple (profile, line), e.g. ("H-01", "LINE1")
    or None if key cannot be extracted.
    """
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]

    # Remove known prefixes for DAT names
    stem = re.sub(r"^(FTAN|SRT)[_\-]", "", stem, flags=re.IGNORECASE)

    # Profile token like H-01, V-06, H01, V6
    profile_match = re.search(r"\b([HV])\s*[-_]?\s*(\d{1,2})\b", stem, flags=re.IGNORECASE)
    if not profile_match:
        return None

    profile = f"{profile_match.group(1).upper()}-{int(profile_match.group(2)):02d}"

    # Line token like (Line1), Line_10, line6
    line_match = re.search(r"\bLINE\s*[_\-]?\s*(\d{1,3})\b", stem, flags=re.IGNORECASE)
    if not line_match:
        return None

    line = f"LINE{int(line_match.group(1))}"
    return profile, line


def find_files(root, suffixes, recursive=True):
    paths = []
    if recursive:
        for suf in suffixes:
            paths.extend(glob.glob(os.path.join(root, "**", f"*{suf}"), recursive=True))
    else:
        for suf in suffixes:
            paths.extend(glob.glob(os.path.join(root, f"*{suf}")))
    return sorted(set(paths))


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


def read_dat_xyzv(path):
    """
    Read input DAT/TXT as three columns:
      chainage, elev_rl, velocity
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)

    if df.shape[1] < 3:
        raise ValueError(f"Input DAT/TXT has <3 columns: {os.path.basename(path)}")

    df = df.iloc[:, :3].copy()
    df.columns = ["chainage", "elev_rl", "velocity"]

    df["chainage"] = pd.to_numeric(df["chainage"], errors="coerce")
    df["elev_rl"] = pd.to_numeric(df["elev_rl"], errors="coerce")
    df["velocity"] = pd.to_numeric(df["velocity"], errors="coerce")

    # Remove Surfer NoData values
    df.loc[df["velocity"] >= SURFER_NODATA * 0.99, "velocity"] = np.nan

    # Apply elevation bounds (optional)
    if APPLY_Z_BOUNDS:
        df.loc[(df["elev_rl"] < Z_MIN) | (df["elev_rl"] > Z_MAX), "velocity"] = np.nan

    df = df.dropna(subset=["chainage", "elev_rl"]).sort_values("chainage").reset_index(drop=True)
    return df


def interp_no_extrap(x_src, y_src, x_tgt):
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


def build_excel_index(excel_paths):
    """Map alignment key -> excel path; warn on duplicates."""
    idx = {}
    for p in excel_paths:
        key = extract_alignment_key(p)
        if key is None:
            continue
        if key in idx:
            print(
                f"Warning: multiple Excel files map to key {key}: "
                f"{os.path.basename(idx[key])} and {os.path.basename(p)}. "
                f"Keeping first."
            )
            continue
        idx[key] = p
    return idx


def get_section_name(dat_path):
    name = os.path.basename(dat_path)
    for suf in INPUT_DAT_SUFFIXES:
        if name.endswith(suf):
            return name[:-len(suf)]
    return os.path.splitext(name)[0]


def process_one(dat_path, positioning_excel):
    section_name = get_section_name(dat_path)

    geo = read_positioning_excel(positioning_excel)
    dat = read_dat_xyzv(dat_path)

    ch_geo = geo["chainage"].values
    e_geo = geo["easting"].values
    n_geo = geo["northing"].values
    s_geo = geo["surface_rl"].values

    ch_dat = dat["chainage"].values
    e_dat = interp_no_extrap(ch_geo, e_geo, ch_dat)
    n_dat = interp_no_extrap(ch_geo, n_geo, ch_dat)
    s_dat = interp_no_extrap(ch_geo, s_geo, ch_dat)

    out = pd.DataFrame({
        "Easting": e_dat,
        "Northing": n_dat,
        "Elevation": dat["elev_rl"].values,
        "Velocity": dat["velocity"].values,
        "Chainage": dat["chainage"].values,
        "Surface_RL": s_dat,
        "Section": section_name,
    })

    out["Depth(RL)"] = out["Surface_RL"] - out["Elevation"]

    out = out.dropna(subset=["Easting", "Northing", "Elevation"]).copy()
    if out.empty:
        print(
            f"  Skipping {section_name}: no DAT points fall inside positioning chainage range "
            f"{ch_geo.min():.6f} to {ch_geo.max():.6f} m"
        )
        return

    out_dir = os.path.dirname(dat_path)
    out_csv = os.path.join(out_dir, f"{section_name}{OUT_SUFFIX}")
    out_dat = os.path.join(out_dir, f"{section_name}_surfer3d_points.dat")

    if DRY_RUN:
        print(f"  DRY_RUN: would write {out_csv} ({len(out)} rows)")
        if WRITE_SPACE_DELIM_DAT:
            print(f"  DRY_RUN: would write {out_dat}")
        return

    out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"  Wrote {out_csv}")

    if WRITE_SPACE_DELIM_DAT:
        out_dat_df = out.dropna(subset=["Velocity"]) 
        out_dat_df[["Easting", "Northing", "Elevation", "Velocity"]].to_csv(
            out_dat,
            index=False,
            header=False,
            sep=" ",
            float_format="%.6f",
        )
        print(f"  Wrote {out_dat}")


def main():
    root = os.path.abspath(ROOT_DIR if ROOT_DIR else os.getcwd())
    print(f"Root directory: {root}")
    print(f"Recursive search: {RECURSIVE}")

    excel_paths = find_files(root, INPUT_POSITIONING_EXCEL_SUFFIXES, recursive=RECURSIVE)
    dat_paths = find_files(root, INPUT_DAT_SUFFIXES, recursive=RECURSIVE)

    if not excel_paths:
        raise FileNotFoundError(
            f"No Excel files found under {root}. Expected one of: {INPUT_POSITIONING_EXCEL_SUFFIXES}"
        )
    if not dat_paths:
        raise FileNotFoundError(
            f"No DAT/TXT files found under {root} matching suffixes: {INPUT_DAT_SUFFIXES}"
        )

    excel_index = build_excel_index(excel_paths)
    if not excel_index:
        raise RuntimeError(
            "No Excel file names could be mapped to alignment keys. "
            "Expected names like H-01_Line1_ShotData.xlsx"
        )

    print(f"Found {len(dat_paths)} DAT/TXT file(s).")
    print(f"Indexed {len(excel_index)} positioning workbook(s) by key.")

    processed = 0
    skipped = 0

    for dat_path in dat_paths:
        key = extract_alignment_key(dat_path)
        section_name = get_section_name(dat_path)

        print(f"\nProcessing: {section_name}")

        if key is None:
            print(f"  Skip: could not parse alignment key from DAT name: {os.path.basename(dat_path)}")
            skipped += 1
            continue

        excel_path = excel_index.get(key)
        if excel_path is None:
            print(f"  Skip: no matching Excel for key {key}")
            skipped += 1
            continue

        print(f"  Match key: {key} -> {os.path.basename(excel_path)}")

        try:
            process_one(dat_path, excel_path)
            processed += 1
        except Exception as exc:
            print(f"  Error processing {os.path.basename(dat_path)}: {exc}")
            skipped += 1

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Skipped/Error: {skipped}")


if __name__ == "__main__":
    main()
