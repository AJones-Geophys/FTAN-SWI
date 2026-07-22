#!/usr/bin/env python3
"""
undulating_blanking_file_creation.py

Create Surfer BLN polygons from existing BLN line files:
1) Data window polygon using topography and an undulating mirrored base (top - thickness)
2) Blank-above polygon from topography up to a user upper RL
3) Blank-below polygon from mirrored base down to a user lower RL

Notes on BLN format used here:
- First row is written as: <vertex_count>,<flag>
- This script preserves the incoming BLN flag when present.
- Output first row always follows the same Surfer-style format expected by existing scripts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

Point = Tuple[float, float]


def _is_close(p1: Point, p2: Point, tol: float = 1e-9) -> bool:
    return math.isclose(p1[0], p2[0], abs_tol=tol) and math.isclose(p1[1], p2[1], abs_tol=tol)


def parse_bln(path: Path) -> Tuple[List[Point], float]:
    """
    Parse a Surfer BLN polygon.

    Expected structure for the files used in this workflow:
    - Row 1: "count,flag"
    - Then `count` coordinate rows.

    Returns
    -------
    (points, flag)
      points: list of all coordinate pairs from the BLN body
      flag: header flag value (defaults to 0.0 if not parseable)
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 4:
        raise ValueError(f"{path.name}: file is too short to be a valid BLN polygon.")

    # Parse header
    first = lines[0].split(",")
    if len(first) < 2:
        raise ValueError(f"{path.name}: first row is not in 'count,flag' format.")

    try:
        count = int(round(float(first[0])))
    except ValueError as exc:
        raise ValueError(f"{path.name}: cannot parse BLN point count from first row.") from exc

    try:
        flag = float(first[1])
    except ValueError:
        flag = 0.0

    body = lines[1:]
    if len(body) < count:
        raise ValueError(
            f"{path.name}: header count={count} but only {len(body)} coordinate rows found."
        )

    points: List[Point] = []
    for i, ln in enumerate(body[:count], start=2):
        parts = ln.split(",")
        if len(parts) < 2:
            raise ValueError(f"{path.name}: invalid coordinate at line {i}: '{ln}'")
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"{path.name}: non-numeric coordinate at line {i}: '{ln}'") from exc
        points.append((x, y))

    return points, flag


def extract_top_surface(points: List[Point]) -> List[Point]:
    """
    Extract topographic polyline from a STEP4-style BLN polygon.

    STEP4 BLN body pattern is assumed:
      top points ...,
      bottom-right,
      bottom-left,
      closing point (same as first top point)

    Therefore top polyline = points[:-3].
    """
    if len(points) < 5:
        raise ValueError("BLN body does not contain enough points for STEP4-style structure.")

    top = points[:-3]

    if len(top) < 2:
        raise ValueError("Topographic surface has fewer than two points.")

    return top


def ensure_x_monotonic(top: List[Point]) -> List[Point]:
    """Ensure x increases left-to-right for predictable polygon construction."""
    if top[0][0] <= top[-1][0]:
        return top
    return list(reversed(top))


def build_data_window_polygon(top: List[Point], thickness_m: float) -> List[Point]:
    """
    Build polygon between topography and mirrored undulating base (top - thickness).

    Polygon order:
      top left->right,
      base right->left,
      close to first top point
    """
    base = [(x, y - thickness_m) for x, y in top]

    poly = []
    poly.extend(top)
    poly.extend(reversed(base))
    poly.append(top[0])

    return poly


def build_blank_above_polygon(top: List[Point], upper_rl: float) -> List[Point]:
    """
    Build polygon above topography to a constant upper RL.

    Polygon order:
      top left->right,
      upper-right,
      upper-left,
      close to first top point
    """
    x_left = top[0][0]
    x_right = top[-1][0]

    poly = []
    poly.extend(top)
    poly.append((x_right, upper_rl))
    poly.append((x_left, upper_rl))
    poly.append(top[0])
    return poly


def build_blank_below_polygon(top: List[Point], thickness_m: float, lower_rl: float) -> List[Point]:
    """
    Build polygon below mirrored base down to a constant lower RL.

    Polygon order:
      base left->right,
      lower-right,
      lower-left,
      close to first base point
    """
    base = [(x, y - thickness_m) for x, y in top]
    x_left = base[0][0]
    x_right = base[-1][0]

    poly = []
    poly.extend(base)
    poly.append((x_right, lower_rl))
    poly.append((x_left, lower_rl))
    poly.append(base[0])
    return poly


def write_bln(path: Path, polygon: List[Point], flag: float) -> None:
    """Write polygon as BLN with first row strictly '<count>,<flag>' format."""
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(polygon):.3f},{flag:.3f}\n")
        for x, y in polygon:
            f.write(f"{x:.3f},{y:.3f}\n")


def proposed_upper_rl(top: List[Point], margin: float = 10.0) -> float:
    return max(y for _, y in top) + margin


def proposed_lower_rl(top: List[Point], thickness_m: float, margin: float = 10.0) -> float:
    return min(y - thickness_m for _, y in top) - margin


def process_folder(
    folder: Path,
    thickness_m: float,
    upper_rl: float,
    lower_rl: float,
) -> Tuple[int, int]:
    files = sorted(folder.glob("*.BLN")) + sorted(folder.glob("*.bln"))
    if not files:
        return 0, 0

    processed = 0
    failed = 0

    for file in files:
        try:
            points, flag = parse_bln(file)
            top = extract_top_surface(points)
            top = ensure_x_monotonic(top)

            poly_data = build_data_window_polygon(top, thickness_m)
            poly_above = build_blank_above_polygon(top, upper_rl)
            poly_below = build_blank_below_polygon(top, thickness_m, lower_rl)

            stem = file.stem

            out_data = file.with_name(f"{stem}_smoothed_{thickness_m:g}m_data_window.BLN")
            out_above = file.with_name(f"{stem}_smoothed_{thickness_m:g}m_blank_above.BLN")
            out_below = file.with_name(f"{stem}_smoothed_{thickness_m:g}m_blank_below.BLN")

            write_bln(out_data, poly_data, flag)
            write_bln(out_above, poly_above, flag)
            write_bln(out_below, poly_below, flag)

            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {file.name}: {exc}")

    return processed, failed


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    folder_selected = filedialog.askdirectory(title="Select folder containing BLN files")
    if not folder_selected:
        print("No folder selected. Cancelled.")
        return

    folder = Path(folder_selected)

    thickness_m = simpledialog.askfloat(
        "Thickness",
        "Distance below topography for mirrored base (m):",
        initialvalue=30.0,
        minvalue=0.001,
    )
    if thickness_m is None:
        print("No thickness entered. Cancelled.")
        return

    # Build a quick preview top surface from first BLN for sensible defaults
    blns = sorted(folder.glob("*.BLN")) + sorted(folder.glob("*.bln"))
    if not blns:
        messagebox.showerror("No BLN files", "No .BLN files were found in the selected folder.")
        return

    try:
        p0, _ = parse_bln(blns[0])
        top0 = ensure_x_monotonic(extract_top_surface(p0))
        upper_default = proposed_upper_rl(top0, margin=10.0)
        lower_default = proposed_lower_rl(top0, thickness_m=thickness_m, margin=10.0)
    except Exception:
        upper_default = 300.0
        lower_default = 0.0

    upper_rl = simpledialog.askfloat(
        "Upper blanking RL",
        "Upper elevation (RL) for blank-above polygon:",
        initialvalue=float(upper_default),
    )
    if upper_rl is None:
        print("No upper RL entered. Cancelled.")
        return

    lower_rl = simpledialog.askfloat(
        "Lower blanking RL",
        "Lower elevation (RL) for blank-below polygon:",
        initialvalue=float(lower_default),
    )
    if lower_rl is None:
        print("No lower RL entered. Cancelled.")
        return

    processed, failed = process_folder(
        folder=folder,
        thickness_m=float(thickness_m),
        upper_rl=float(upper_rl),
        lower_rl=float(lower_rl),
    )

    message = (
        f"Finished.\n\n"
        f"Processed BLN files: {processed}\n"
        f"Failed BLN files: {failed}\n\n"
        f"Each processed file produced:\n"
        f"  - *_smoothed_{thickness_m:g}m_data_window.BLN\n"
        f"  - *_smoothed_{thickness_m:g}m_blank_above.BLN\n"
        f"  - *_smoothed_{thickness_m:g}m_blank_below.BLN"
    )
    print(message)
    messagebox.showinfo("undulating_blanking_file_creation", message)


if __name__ == "__main__":
    main()
