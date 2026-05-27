"""Stage 07 -- biomechanical / anatomical consistency verification.

Implements the four check classes from the project approach:
  A. costo-vertebral connectivity  (every rib head near its vertebra)
  B. costo-sternal connectivity     (true ribs 1-7 reach the sternum/cartilage)
  C. inter-vertebral connectivity   (adjacent T-vertebrae in near contact)
  D. floating-rib check             (ribs 11-12 are FAR from the sternum)

Distances are computed with KD-trees over mesh vertices (closest-surface-point
distance), so the module needs only numpy + scipy -- no PyVista -- and is
therefore unit-testable against the synthetic phantom.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import Config, DEFAULT


def _closest_surface_dist(point: np.ndarray, verts: np.ndarray) -> float:
    if verts is None or len(verts) == 0:
        return float("inf")
    tree = cKDTree(verts)
    d, _ = tree.query(np.atleast_2d(point), k=1)
    return float(d[0])


def check_costo_vertebral(rib_n, side, centerline_world, vert_centroid,
                          cfg: Config = DEFAULT) -> dict:
    """Posterior centerline endpoint must be near the corresponding vertebra."""
    post = np.asarray(centerline_world)[0]
    dist = float(np.linalg.norm(post - np.asarray(vert_centroid)))
    return {"check": "costo_vertebral", "rib": rib_n, "side": side,
            "dist_mm": round(dist, 2), "threshold_mm": cfg.costo_vertebral_mm,
            "status": "PASS" if dist <= cfg.costo_vertebral_mm else "FAIL"}


def check_costo_sternal(rib_n, side, centerline_world, sternum_verts,
                        cartilage_verts=None, cfg: Config = DEFAULT) -> dict:
    """True ribs (1-7): anterior endpoint must reach sternum/costal cartilage."""
    ant = np.asarray(centerline_world)[-1]
    d = min(_closest_surface_dist(ant, sternum_verts),
            _closest_surface_dist(ant, cartilage_verts))
    return {"check": "costo_sternal", "rib": rib_n, "side": side,
            "dist_mm": round(d, 2), "threshold_mm": cfg.costo_sternal_mm,
            "status": "PASS" if d <= cfg.costo_sternal_mm else "FAIL"}


def check_floating_rib(rib_n, side, centerline_world, sternum_verts,
                       cfg: Config = DEFAULT) -> dict:
    """Floating ribs (11-12): anterior end must be FAR from the sternum."""
    ant = np.asarray(centerline_world)[-1]
    d = _closest_surface_dist(ant, sternum_verts)
    return {"check": "floating_rib", "rib": rib_n, "side": side,
            "dist_mm": round(d, 2), "threshold_mm": cfg.floating_min_mm,
            "status": "PASS" if d >= cfg.floating_min_mm else "WARN"}


def check_intervertebral(n, verts_a, verts_b, cfg: Config = DEFAULT) -> dict:
    """Adjacent thoracic vertebrae Tn / Tn+1 must be in near contact."""
    if verts_a is None or verts_b is None or len(verts_a) == 0 or len(verts_b) == 0:
        return {"check": "intervertebral", "pair": f"T{n}-T{n+1}",
                "min_dist_mm": float("nan"), "threshold_mm": cfg.intervertebral_mm,
                "status": "SKIP"}
    tree = cKDTree(verts_a)
    d, _ = tree.query(verts_b, k=1)
    mind = float(d.min())
    return {"check": "intervertebral", "pair": f"T{n}-T{n+1}",
            "min_dist_mm": round(mind, 2), "threshold_mm": cfg.intervertebral_mm,
            "status": "PASS" if mind <= cfg.intervertebral_mm else "FAIL"}


def verify_thorax(centerlines: dict, vert_centroids: dict,
                  surfaces: dict, cfg: Config = DEFAULT) -> list:
    """Run all checks and return a flat list of result dicts.

    Parameters
    ----------
    centerlines   : {(side, n): (Npts,3) world centerline}
    vert_centroids: {"T1":(3,), ...} world centroids
    surfaces      : {"sternum": verts, "costal_cartilages": verts,
                     "vertebrae_T1": verts, ...}  vertex arrays (may be missing)
    """
    results = []
    sternum = surfaces.get("sternum")
    cartilage = surfaces.get("costal_cartilages")

    for (side, n), cl in sorted(centerlines.items()):
        vc = vert_centroids.get(f"T{n}")
        if vc is not None:
            results.append(check_costo_vertebral(n, side, cl, vc, cfg))
        if n in cfg.true_ribs:
            results.append(check_costo_sternal(n, side, cl, sternum, cartilage, cfg))
        if n in cfg.floating_ribs:
            results.append(check_floating_rib(n, side, cl, sternum, cfg))

    for n in range(1, 12):
        va = surfaces.get(f"vertebrae_T{n}")
        vb = surfaces.get(f"vertebrae_T{n+1}")
        results.append(check_intervertebral(n, va, vb, cfg))
    return results
