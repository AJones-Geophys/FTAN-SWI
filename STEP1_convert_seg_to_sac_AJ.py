from obspy import read
from obspy.io.sac.sactrace import SACTrace
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# User settings
# -----------------------
EXCEL_FILE = "ShotData.xlsx"
SHEET_NAME = "ShotData"
SAVE_PLOTS = True   # True  → saves a QC PNG per trace (default)
                    # False → skips plot generation entirely

excel_path = Path(EXCEL_FILE).resolve()
base_dir = excel_path.parent

df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)


def read_waveform_auto(path: Path):
    """
    Read a seismic waveform file using a layered strategy:

      1. Try ObsPy auto-detection (no format specified) — handles the widest
         range of SEG-Y dialects, SEG-2, and other supported formats.
      2. If that fails, explicitly retry SEG-Y with both endians.
      3. If that fails, explicitly retry SEG-2.
      4. If all attempts fail, raise a RuntimeError with a clear summary of
         what was tried.

    All intermediate errors are printed so you can see exactly what ObsPy
    reported for each attempt.
    """
    errors = []

    # --- Strategy 1: ObsPy auto-detect ---
    try:
        st = read(str(path))
        print(f"  [ok] {path.name}: read via ObsPy auto-detect")
        return st
    except Exception as e:
        errors.append(f"  auto-detect : {e}")
        print(f"  [warn] {path.name} auto-detect failed: {e}")

    # --- Strategy 2: Explicit SEG-Y (big then little endian) ---
    for endian, label in ((">", "big-endian"), ("<", "little-endian")):
        try:
            st = read(str(path), format="SEGY", endian=endian)
            print(f"  [ok] {path.name}: read as SEG-Y ({label})")
            return st
        except Exception as e:
            errors.append(f"  SEGY {label}: {e}")
            print(f"  [warn] {path.name} SEGY {label} failed: {e}")

    # --- Strategy 3: Explicit SEG-2 ---
    try:
        st = read(str(path), format="SEG2")
        print(f"  [ok] {path.name}: read as SEG-2")
        return st
    except Exception as e:
        errors.append(f"  SEG2        : {e}")
        print(f"  [warn] {path.name} SEG-2 failed: {e}")

    # --- All strategies exhausted ---
    raise RuntimeError(
        f"\n{path.name} could not be read. Attempts made:\n" +
        "\n".join(errors)
    )


for index, row in df.iterrows():
    # ---- Excel-controlled parameters ----
    filename  = str(row.iloc[0])
    geophone  = int(row.iloc[3]) - 1
    dist      = float(row.iloc[6])
    file_out  = str(row.iloc[7])

    # ---- Resolve paths relative to the Excel file location ----
    in_path = Path(filename)
    if not in_path.is_absolute():
        in_path = base_dir / in_path

    out_path = Path(file_out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRow {index}: {in_path.name} | trace {geophone + 1} | output {out_path.name}")

    # ---- Read waveform ----
    st = read_waveform_auto(in_path)

    if geophone < 0 or geophone >= len(st):
        raise IndexError(
            f"Requested trace {geophone + 1} out of range "
            f"(file has {len(st)} traces)"
        )

    tr = st[geophone]

    # ---- Save QC plot ----
    if SAVE_PLOTS:
        fig_name = out_path.parent / f"{in_path.stem}_geo{geophone + 1}.png"
        tr.plot(outfile=str(fig_name))
        plt.close("all")  # prevent memory accumulation over many traces

    # ---- Convert to SAC ----
    t = tr.stats.starttime
    sac = SACTrace.from_obspy_trace(tr)

    sac.a      = t      # preserve actual trace start time
    sac.b      = 0.0
    sac.dist   = dist
    sac.az     = 0
    sac.baz    = 0
    sac.evla   = 0
    sac.evlo   = 0
    sac.stla   = 0
    sac.stlo   = 0
    sac.gcarc  = 0.001
    sac.kstnm  = "50m"
    sac.kcmpnm = "nada"

    sac.write(str(out_path))

print("\ndone")
