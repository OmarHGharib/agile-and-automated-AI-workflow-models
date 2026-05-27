#!/usr/bin/env python3
"""Diagnose a TotalSegmentator output folder.

Tells you EXACTLY why a thorax model might be missing the sternum / cartilage /
rib-sternum connection: which structures exist, their voxel counts, and the gap
from each true rib's anterior end to the sternum.

Run on your Mac after segmentation:
    python diagnose_segmentation.py runs/RibFrac501/segmentations
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ribpipe import RIB_LABELS, VERTEBRA_LABELS, geometry as geo  # noqa: E402
from ribpipe.io_utils import load_mask                            # noqa: E402
from ribpipe.segment import find_mask                             # noqa: E402
from ribpipe.config import DEFAULT                                # noqa: E402
from ribpipe import stages                                        # noqa: E402


def voxels(seg_dir, label):
    p = find_mask(seg_dir, label)
    if p is None:
        return None
    m, _ = load_mask(p)
    return int(m.sum())


def main():
    if len(sys.argv) < 2:
        print("usage: python diagnose_segmentation.py <segmentations_dir>")
        sys.exit(1)
    seg = sys.argv[1]
    print(f"\n=== Diagnosing {seg} ===\n")

    # key connectors
    conn = {}
    for key in ("sternum", "costal_cartilages"):
        v = voxels(seg, key)
        conn[key] = v
        state = "MISSING (no file)" if v is None else (
            "EMPTY (0 voxels)" if v == 0 else f"{v} voxels")
        flag = "  <-- needed to connect ribs to sternum" if (v is None or v == 0) else "  OK"
        print(f"  {key:20s}: {state}{flag}")

    # ribs / vertebrae presence
    ribs = {l: voxels(seg, l) for l in RIB_LABELS}
    verts = {l: voxels(seg, l) for l in VERTEBRA_LABELS}
    n_rib = sum(1 for v in ribs.values() if v)
    n_vert = sum(1 for v in verts.values() if v)
    print(f"\n  ribs present : {n_rib}/24")
    print(f"  vertebrae T* : {n_vert}/12")
    missing_ribs = [l for l, v in ribs.items() if not v]
    if missing_ribs:
        print("  missing/empty ribs:", ", ".join(missing_ribs))

    # per-true-rib anterior-end -> sternum gap
    print("\n  rib anterior-end -> sternum gap (true ribs 1-7):")
    try:
        frame, vc = stages.stage_03_spine(seg, DEFAULT)
        cls = stages.stage_04_centerlines(seg, frame, vc, DEFAULT)
        sp = find_mask(seg, "sternum")
        if sp is not None:
            sm, sa = load_mask(sp)
            sv = geo.mask_to_world_points(sm, sa)
            from scipy.spatial import cKDTree
            tree = cKDTree(sv) if len(sv) else None
            for (side, n), cl in sorted(cls.items()):
                if n > 7 or tree is None:
                    continue
                d, _ = tree.query(np.asarray(cl)[-1])
                print(f"    rib_{side}_{n:<2d}: {d:6.1f} mm")
        else:
            print("    (no sternum -> pipeline will bridge to an estimated midline)")
    except Exception as e:
        print("    could not compute gaps:", e)

    print("\nInterpretation:")
    st, ca = conn.get("sternum"), conn.get("costal_cartilages")
    if ca and ca > 0:
        print("  * costal_cartilages PRESENT -> it is meshed directly as the")
        print("    rib-sternum connector (green). Ensure meshing uses")
        print("    keep_largest=False so ALL ~24 cartilage pieces are kept.")
    else:
        print("  * costal_cartilages EMPTY/MISSING -> normal if radiolucent;")
        print("    the pipeline synthesises curved cartilage bridges instead.")
    if st and st > 0:
        print("  * sternum PRESENT -> used as the anterior target. OK.")
    else:
        print("  * sternum MISSING/EMPTY -> pipeline bridges ribs to an")
        print("    estimated anterior midline bar instead.")
    print("  * rib->sternum gaps grow for lower true ribs (1->7); the 7th")
    print("    costal cartilage is the longest, so 80-110 mm is expected.\n")


if __name__ == "__main__":
    main()
