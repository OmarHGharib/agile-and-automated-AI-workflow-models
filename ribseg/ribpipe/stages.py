"""High-level stage functions (03-07) that operate on a TotalSegmentator output
folder.  They glue together io_utils / geometry / meshing / mechanics.

Each function is small and side-effect-light so the orchestrator in
``pipeline.py`` reads as a linear recipe and so individual stages can be re-run
in isolation.
"""
from __future__ import annotations

import logging
import numpy as np

from . import RIB_SIDES, RIB_NUMBERS, VERTEBRA_LABELS, ALL_THORAX_LABELS
from . import geometry as geo
from . import meshing
from . import mechanics
from .config import Config, DEFAULT
from .io_utils import load_mask
from .segment import find_mask

log = logging.getLogger("ribpipe.stages")


# --------------------------------------------------------------------------- #
def stage_03_spine(seg_dir: str, cfg: Config = DEFAULT):
    """Build the anatomical frame from T1..T12 centroids.

    Returns (frame, vert_centroids dict {"T1":(3,),...}).
    """
    centroids, names = [], []
    for n, label in zip(range(1, 13), VERTEBRA_LABELS):
        p = find_mask(seg_dir, label)
        if p is None:
            continue
        mask, aff = load_mask(p)
        if mask.sum() == 0:
            continue
        centroids.append(geo.center_of_mass_world(mask, aff))
        names.append(f"T{n}")
    if len(centroids) < 2:
        raise RuntimeError("fewer than 2 thoracic vertebrae segmented -- "
                           "cannot define a spinal axis")
    centroids = np.array(centroids)
    frame = geo.build_anatomical_frame(centroids)
    vert_centroids = dict(zip(names, centroids))
    log.info("Spinal axis from %d vertebrae; Z=%s", len(names),
             np.round(frame["z_axis"], 3))
    return frame, vert_centroids


def stage_04_centerlines(seg_dir: str, frame, vert_centroids,
                         cfg: Config = DEFAULT):
    """Extract an ordered world-coordinate centerline for every present rib."""
    centerlines = {}
    for side in RIB_SIDES:
        for n in RIB_NUMBERS:
            label = f"rib_{side}_{n}"
            p = find_mask(seg_dir, label)
            if p is None:
                continue
            mask, aff = load_mask(p)
            if mask.sum() < 10:
                continue
            root = vert_centroids.get(f"T{n}")
            cl = geo.extract_centerline(
                mask, aff, n_points=cfg.centerline_n_points,
                smooth_sigma=cfg.centerline_smooth_sigma, root_world=root,
                teasar_params=cfg.teasar_params)
            centerlines[(side, n)] = cl
    log.info("Extracted %d rib centerlines", len(centerlines))
    return centerlines


def stage_05_morphometry(seg_dir: str, frame, centerlines,
                         cfg: Config = DEFAULT):
    """Compute morphometric measurements for every rib with a centerline."""
    rows = []
    for (side, n), cl in sorted(centerlines.items()):
        p = find_mask(seg_dir, f"rib_{side}_{n}")
        mask, aff = load_mask(p)
        m = geo.rib_measurements(mask, aff, frame, cl)
        m.update({"rib": n, "side": side, "label": f"rib_{side}_{n}"})
        rows.append(m)
    return rows


def stage_06_mesh(seg_dir: str, out_dir, cfg: Config = DEFAULT,
                  labels=ALL_THORAX_LABELS):
    """Marching-cubes meshes for every present structure, with export.

    Writes per-structure STLs into ``out_dir`` and a single combined
    ``bony_thorax.stl`` (+ colour-coded GLB/OBJ) into its parent.
    Returns {label: (verts, faces)}.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for label in labels:
        p = find_mask(seg_dir, label)
        if p is None:
            continue
        mask, aff = load_mask(p)
        # keep_largest=False: structures like costal_cartilages are ONE mask
        # holding ~24 separate pieces, and a fractured rib is >1 piece -- we
        # must mesh ALL components, not just the largest.
        verts, faces = meshing.mask_to_mesh(mask, aff, level=cfg.marching_level,
                                            keep_largest=False)
        if len(verts) == 0:
            continue
        meshes[label] = (verts, faces)
        meshing.save_stl(verts, faces, out_dir / f"{label}.stl")
    if meshes:
        paths = meshing.export_combined(meshes, out_dir.parent, "bony_thorax")
        log.info("Built %d structure meshes; combined model: %s",
                 len(meshes), ", ".join(paths) or "none")
    else:
        log.warning("No meshes built -- check segmentation coverage")
    return meshes


def coverage_report(seg_dir: str, labels=ALL_THORAX_LABELS) -> dict:
    """Report which expected thorax structures are present / missing / empty.

    Helps explain a model that is missing the sternum, cartilages, etc. --
    those structures are only in the 3D model if TotalSegmentator produced a
    non-empty mask for them.
    """
    present, empty, missing = [], [], []
    for label in labels:
        p = find_mask(seg_dir, label)
        if p is None:
            missing.append(label)
            continue
        mask, _ = load_mask(p)
        (present if mask.sum() > 0 else empty).append(label)
    rep = {"n_present": len(present), "n_empty": len(empty),
           "n_missing": len(missing), "present": present,
           "empty": empty, "missing": missing}
    log.info("Coverage: %d present, %d empty, %d missing. Missing=%s",
             len(present), len(empty), len(missing), missing or "none")
    return rep


def stage_07_mechanics(centerlines, vert_centroids, meshes,
                       cfg: Config = DEFAULT):
    """Run the biomechanical verification checks."""
    surfaces = {label: verts for label, (verts, _f) in meshes.items()}
    return mechanics.verify_thorax(centerlines, vert_centroids, surfaces, cfg)
