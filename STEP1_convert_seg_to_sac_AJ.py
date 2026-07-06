from obspy import read
from obspy.io.sac.sactrace import SACTrace
import pandas as pd
import os

df = pd.read_excel("ShotData.xlsx", sheet_name="ShotData")

for index, row in df.iterrows():
    filename = str(row[0])
    print("Row:::", filename, type(row[0]))

    st = read(filename, format="SEGY", endian='>')
    
    geophone = int(row[3]) - 1
    dist = row[6]
    file_out = row[7]

    tr = st[geophone]

    # --- build output figure name ---
    base = os.path.splitext(os.path.basename(filename))[0]
    fig_name = f"{base}_geo{geophone+1}.png"

    # --- save plot (no display) ---
    tr.plot(outfile=fig_name)

    # --- SAC handling ---
    t = tr.stats.starttime
    sac = SACTrace.from_obspy_trace(tr)
    sac.a = t
    sac.az = 0
    sac.b = 0
    sac.baz = 0
    sac.dist = dist
    sac.degree = 0
    sac.evla = 0
    sac.evlo = 0
    sac.stla = 0
    sac.stlo = 0
    sac.kstnm = "50m"
    sac.kcmpnm = "nada"
    sac.gcarc = 0.001

    sac.write(file_out)

print("done")