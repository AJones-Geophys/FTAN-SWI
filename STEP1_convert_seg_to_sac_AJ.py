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
    Read SEG-Y (big- or little-endian), SEG-2, or data stored as .dat.
    Extension-based routing with silent endian retry and SEG-2 fallback.
    """

    ext = path.suffix.lower()

    def try_segy(p: Path):
        for endian in (">", "<"):
            try:
                return read(str(p), format="SEGY", endian=endian)
            except Exception:
                pass
        return None

    def try_seg2(p: Path):
        try:
            return read(str(p), format="SEG2")
        except Exception:
            return None

    # Explicit SEG-Y extensions
    if ext in (".segy", ".sgy"):
        st = try_segy(path)
        if st is not None:
            return st
        raise RuntimeError(f"{path.name} could not be read as SEG-Y")

    # Explicit SEG-2 extensions
    if ext in (".seg2", ".sg2"):
        st = try_seg2(path)
        if st is not None:
            return st
        raise RuntimeError(f"{path.name} could not be read as SEG-2")

    # .dat or unknown extension: try SEG-Y first, then SEG-2
    st = try_segy(path)
    if st is not None:
        return st

    st = try_seg2(path)
    if st is not None:
        return st

    raise RuntimeError(f"{path.name} could not be read as SEG-Y or SEG-2")


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

    # ---- Read waveform (SEG-Y / SEG-2 / .dat) ----
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
