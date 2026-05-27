"""Costovertebral joints -- connect each rib head to the thoracic spine.

Posteriorly, every rib articulates with the vertebral column at the
costovertebral joint: the rib HEAD meets the demifacets on the vertebral
body (rib n typically with Tn and T(n-1)) and the rib TUBERCLE meets the
transverse process (costotransverse joint).  These are synovial joints with
articular cartilage + strong ligaments (radiate, costotransverse).  On CT the
bony rib head and the vertebra are separate masks with a small gap, so a pure
bone model leaves the rib heads floating beside the spine.

This module closes that gap the same way ``cartilage`` closes the front: a
short tapered connector that leaves the rib along its own (posterior) tangent
and meets the nearest surface of the corresponding vertebra, enforcing the
spine -> rib topology.  It is a model element (joint/ligament), clearly named,
so morphometry is unaffected.
"""
from __future__ import annotations

import logging
import numpy as np
from scipy.spatial import cKDTree

from .config import Config, DEFAULT
from .cartilage import curved_cartilage   # reuse the tangent-continuous tube

log = logging.getLogger("ribpipe.joints")


def build_costovertebral_links(centerlines: dict, vertebra_surfaces: dict,
                               vert_centroids: dict | None = None,
                               cfg: Config = DEFAULT) -> tuple[dict, list]:
    """Short connectors from each rib's posterior end to its vertebra surface.

    Parameters
    ----------
    centerlines : {(side, n): (N,3) world centerline (posterior->anterior)}
    vertebra_surfaces : {"vertebrae_T1": verts, ...} surface vertices.
    Returns (links {(side,n):(verts,faces)}, report).
    """
    links, report = {}, []
    # KD-trees per available vertebra surface
    trees = {name: cKDTree(v) for name, v in vertebra_surfaces.items()
             if v is not None and len(v)}
    if not trees:
        log.warning("No vertebra surfaces -- cannot build costovertebral joints.")
        return links, report

    all_names = list(trees.keys())
    all_pts = {name: vertebra_surfaces[name] for name in all_names}

    for (side, n), cl in sorted(centerlines.items()):
        cl = np.asarray(cl)
        post = cl[0]                                   # posterior (vertebral) end
        # rib n articulates mainly with Tn (and T(n-1)); prefer Tn, else nearest
        cand = [f"vertebrae_T{n}", f"vertebrae_T{max(1, n-1)}"]
        cand = [c for c in cand if c in trees] or all_names
        best = None
        for name in cand:
            d, i = trees[name].query(post)
            if best is None or d < best[0]:
                best = (d, all_pts[name][i], name)
        dist, target, vname = best
        if dist > cfg.costovertebral_max_gap_mm:
            report.append({"rib": n, "side": side, "vertebra": vname,
                           "gap_mm": round(float(dist), 1), "linked": False,
                           "note": "gap exceeds max"})
            continue
        tangent = post - cl[min(8, len(cl) - 1)]       # rib exit toward spine
        v, f = curved_cartilage(post, tangent, target,
                                cfg.costovertebral_radius_mm,
                                cfg.costovertebral_radius_mm * 0.85)
        if len(f):
            links[(side, n)] = (v, f)
            report.append({"rib": n, "side": side, "vertebra": vname,
                           "gap_mm": round(float(dist), 1), "linked": True,
                           "model": "costovertebral_connector"})
    log.info("Built %d costovertebral joint connectors", len(links))
    return links, report
