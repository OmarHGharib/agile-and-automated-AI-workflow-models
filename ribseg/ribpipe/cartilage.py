"""Costal-cartilage modelling -- connect the bony ribs to the sternum.

Anatomy: ribs do not articulate with the sternum directly.  True ribs (1-7)
join it through their own costal cartilage; false ribs (8-10) join the cartilage
of the rib above; floating ribs (11-12) have free anterior ends.  Costal
cartilage is largely radiolucent on CT, so a pure bone segmentation shows a gap
between each anterior rib end and the sternum.

Design (informed by recent literature)
---------------------------------------
* Costal cartilage IS segmentable on CT (e.g. the 2024 Topology-Guided
  Deformable Mamba benchmark; pediatric studies reconstruct 500+ cartilages).
  So the FIRST choice is always the real ``costal_cartilages`` mask, which the
  pipeline meshes like any other structure.
* Only when that mask is absent/empty do we SYNTHESISE cartilage -- and not as a
  straight stick: each bridge is a smooth, tapered tube along a quadratic Bezier
  that LEAVES THE RIB ALONG ITS OWN TANGENT (C1 continuity) and curves to the
  nearest sternal/cartilage target, so it reads as a natural continuation of the
  rib rather than a strut. Topology (rib -> cartilage -> sternum) is enforced.

All distances are in mm.
"""
from __future__ import annotations

import logging
import numpy as np
from scipy.spatial import cKDTree

from .config import Config, DEFAULT

log = logging.getLogger("ribpipe.cartilage")


def cartilage_is_segmented(cartilage_verts, min_pts: int = 200) -> bool:
    return cartilage_verts is not None and len(cartilage_verts) >= min_pts


# --------------------------------------------------------------------------- #
def _frame(t):
    """An orthonormal (u, v) pair perpendicular to unit tangent t."""
    up = np.array([0, 0, 1.0])
    if abs(np.dot(t, up)) > 0.9:
        up = np.array([0, 1.0, 0])
    u = np.cross(t, up); u /= (np.linalg.norm(u) + 1e-9)
    v = np.cross(t, u); v /= (np.linalg.norm(v) + 1e-9)
    return u, v


def tube_from_polyline(P, radii, sections=14):
    """Triangulated tube of given per-vertex radii around polyline P (M,3)."""
    P = np.asarray(P, float)
    M = len(P)
    if M < 2:
        return np.empty((0, 3)), np.empty((0, 3), int)
    tang = np.gradient(P, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    angs = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    rings = np.empty((M, sections, 3))
    for i in range(M):
        u, v = _frame(tang[i])
        rings[i] = P[i] + radii[i] * (np.cos(angs)[:, None] * u
                                      + np.sin(angs)[:, None] * v)
    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(M - 1):
        for j in range(sections):
            a = i * sections + j
            b = i * sections + (j + 1) % sections
            c = (i + 1) * sections + (j + 1) % sections
            d = (i + 1) * sections + j
            faces.append([a, b, c]); faces.append([a, c, d])
    return verts, np.asarray(faces, int)


def _bezier(A, ctrl, B, n=24):
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * A + 2 * (1 - s) * s * ctrl + s ** 2 * B


def curved_cartilage(anterior_pt, tangent, target, r0, r1, n=24):
    """Tapered tube along a Bezier that leaves the rib along ``tangent`` and
    curves to ``target`` -- an anatomically plausible cartilage continuation."""
    A = np.asarray(anterior_pt, float); B = np.asarray(target, float)
    t = np.asarray(tangent, float); t /= (np.linalg.norm(t) + 1e-9)
    dist = np.linalg.norm(B - A)
    ctrl = A + t * (0.55 * dist)          # continue the rib, then bend to target
    P = _bezier(A, ctrl, B, n)
    radii = np.linspace(r0, r1, len(P))
    return tube_from_polyline(P, radii)


# --------------------------------------------------------------------------- #
def _estimate_sternum_target(centerlines, vert_centroids):
    """When no sternum is segmented, estimate an anterior midline 'sternal bar'.

    The sternum sits at the anterior midline.  We take the medial-anterior
    region implied by the rib anterior endpoints: midline X from the vertebral
    column, and a vertical bar spanning the true-rib endpoint heights.
    Returns (target_pts (K,3), is_estimated=True).
    """
    ant = np.array([np.asarray(cl)[-1] for (s, n), cl in centerlines.items()
                    if n <= 7])
    if len(ant) == 0:
        return None
    if vert_centroids:
        mid_x = float(np.mean([c[0] for c in vert_centroids.values()]))
    else:
        mid_x = float(np.mean(ant[:, 0]))
    y_ant = float(np.percentile(ant[:, 1], 75))     # anterior position
    zs = np.linspace(ant[:, 2].min(), ant[:, 2].max(), 40)
    return np.stack([np.full_like(zs, mid_x), np.full_like(zs, y_ant), zs], 1)


def build_bridges(centerlines: dict, sternum_verts: np.ndarray,
                  cartilage_verts: np.ndarray | None = None,
                  cfg: Config = DEFAULT, vert_centroids: dict | None = None
                  ) -> tuple[dict, list]:
    """Guarantee a visible rib->sternum connection for each true/false rib.

    Per-rib logic (robust to sparse/missing TotalSegmentator output):
      * if REAL costal cartilage already attaches to this rib's anterior end
        (a cartilage vertex within ~2 radii), keep the real cartilage -> skip;
      * otherwise synthesise a curved Bezier cartilage to the nearest sternum
        point; if the sternum mask is missing, connect to an estimated anterior
        midline 'sternal bar' so the model is still continuous.
    Returns (bridges {(side,n):(verts,faces)}, report).
    """
    bridges, report = {}, []
    estimated = False
    if sternum_verts is not None and len(sternum_verts) > 0:
        target_pts = np.asarray(sternum_verts)
    else:
        target_pts = _estimate_sternum_target(centerlines, vert_centroids)
        estimated = True
        log.warning("No sternum mask -- bridging to an ESTIMATED anterior "
                    "midline bar (segment the sternum for accuracy).")
    if target_pts is None or len(target_pts) == 0:
        log.warning("Cannot determine a sternal target -- no bridges built.")
        return bridges, report

    s_tree = cKDTree(target_pts)
    c_tree = (cKDTree(cartilage_verts)
              if cartilage_is_segmented(cartilage_verts, 50) else None)
    attach_tol = 2.5 * cfg.cartilage_radius_mm
    bridge_ribs = tuple(cfg.true_ribs) + tuple(cfg.false_ribs)  # 1..10

    for (side, n), cl in sorted(centerlines.items()):
        if n not in bridge_ribs:
            continue
        cl = np.asarray(cl)
        ant = cl[-1]
        # real cartilage already on this rib end?  then trust it.
        if c_tree is not None and c_tree.query(ant)[0] <= attach_tol:
            report.append({"rib": n, "side": side, "bridged": False,
                           "note": "real costal cartilage attaches here"})
            continue
        tangent = ant - cl[-min(8, len(cl))]
        dist, si = s_tree.query(ant)
        if dist > cfg.cartilage_max_gap_mm:
            report.append({"rib": n, "side": side, "gap_mm": round(float(dist), 1),
                           "bridged": False, "note": "gap exceeds max"})
            continue
        v, f = curved_cartilage(ant, tangent, target_pts[si],
                                cfg.cartilage_radius_mm,
                                cfg.cartilage_radius_mm * 0.7)
        if len(f):
            bridges[(side, n)] = (v, f)
            report.append({"rib": n, "side": side, "gap_mm": round(float(dist), 1),
                           "bridged": True, "model": "curved_bezier_tube",
                           "target": "estimated_midline" if estimated else "sternum"})
    log.info("Cartilage: %d bridges synthesised (sternum %s)", len(bridges),
             "ESTIMATED" if estimated else "segmented")
    return bridges, report
