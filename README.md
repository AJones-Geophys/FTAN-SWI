# FTAN-SWI
Scripts associated with GHD's internal Frequency-Time Analysis Shear Wave inversion (FTAN-SWI) workflow

## undulating_blanking_file_creation.py

Creates Surfer BLN outputs from existing STEP4-style BLN files using an undulating mirrored base.

### What it does
For each input `.BLN` file in a selected folder, it writes three new files:

1. `*_smoothed_<Xm>_data_window.BLN`  
   - Polygon defining the in-model data window between:
     - the topographic surface, and
     - an undulating base calculated as `topography - thickness_m`.

2. `*_smoothed_<Xm>_blank_above.BLN`  
   - Polygon above topography up to a user-defined upper RL.

3. `*_smoothed_<Xm>_blank_below.BLN`  
   - Polygon below the mirrored base down to a user-defined lower RL.

### BLN header format
The script preserves Surfer BLN header style and writes the first row as:

`<vertex_count>,<flag>`

This is consistent with existing STEP4/Surfer expectations (e.g., `227.000,0.000`).

### Inputs expected
- Existing STEP4-style BLN polygon files where the body follows:
  - topographic points,
  - bottom-right point,
  - bottom-left point,
  - closing point (same as first topographic point).

### How to run
From a Python environment with `tkinter` available:

```bash
python undulating_blanking_file_creation.py
```

The script will prompt for:
1. Folder containing `.BLN` files
2. Mirrored thickness in metres (default `30`)
3. Upper blanking RL
4. Lower blanking RL

### Notes
- Outputs are written to the same folder as the input BLN files.
- If a file cannot be parsed, it is skipped and reported in the console.
