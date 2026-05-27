#!/usr/bin/env python3
"""Render a folder of per-structure STL files into a high-quality colour PNG.

Uses PyVista (OpenGL) when available -- run this on your Mac (M1 GPU) for the
clean, solid, lit look. Falls back to the headless Matplotlib renderer if
OpenGL is unavailable.

Example:
  python render_stls.py --in-dir runs/RibFrac501/meshes \
                        --out runs/RibFrac501/thorax_hq.png --backend vtk
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ribpipe import meshing                              # noqa: E402


def load_stls(in_dir):
    from stl import mesh as stlmesh
    meshes = {}
    for path in sorted(glob.glob(os.path.join(in_dir, "*.stl"))):
        label = os.path.splitext(os.path.basename(path))[0]
        if label.startswith("bony_thorax"):             # skip the combined file
            continue
        m = stlmesh.Mesh.from_file(path)
        v = m.vectors.reshape(-1, 3)
        f = np.arange(len(v)).reshape(-1, 3)
        color = meshing.GROUP_COLORS[meshing.group_of(label)]
        meshes[label] = (v, f, color)
    return meshes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="folder of per-structure STLs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vtk", "mpl"], default="vtk")
    ap.add_argument("--title", default="Bony thorax")
    a = ap.parse_args()

    meshes = load_stls(a.in_dir)
    if not meshes:
        print("No per-structure STLs found in", a.in_dir); return
    print(f"Loaded {len(meshes)} structures")

    if a.backend == "vtk":
        try:
            meshing.render_meshes_vtk(meshes, a.out, title=a.title)
            print("VTK render ->", a.out); return
        except Exception as e:
            print("VTK unavailable (%s); using matplotlib fallback" % e)
    meshing.render_meshes(meshes, a.out, title=a.title)
    print("Matplotlib render ->", a.out)


if __name__ == "__main__":
    main()
